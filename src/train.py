import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
from pathlib import Path
import argparse

from models.siamese_net import SiameseUNet
from data.dataset import ChangeDetectionDataset
from utils.metrics import MetricTracker, print_metrics
from config import Config




def train_epoch(model, dataloader, criterion, optimizer, device, writer, epoch):
    """
    Bir epoch için eğitim döngüsü

    Args:
        model: PyTorch model
        dataloader: Training DataLoader
        criterion: Loss fonksiyonu
        optimizer: Optimizer
        device: cuda veya cpu
        writer: TensorBoard SummaryWriter
        epoch: Epoch numarası

    Returns:
        avg_loss: Ortalama loss değeri
    """
    model.train()
    total_loss = 0

    pbar = tqdm(dataloader, desc=f'Epoch {epoch + 1} [Train]')
    for batch_idx, (img_a, img_b, labels) in enumerate(pbar):
        # GPU'ya taşı
        img_a = img_a.to(device)
        img_b = img_b.to(device)
        labels = labels.to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(img_a, img_b)

        # Loss hesapla
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # TensorBoard'a yaz
        global_step = epoch * len(dataloader) + batch_idx
        writer.add_scalar('Train/Batch_Loss', loss.item(), global_step)

        # Progress bar güncelle
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    # Epoch ortalaması
    avg_loss = total_loss / len(dataloader)
    writer.add_scalar('Train/Epoch_Loss', avg_loss, epoch)

    return avg_loss


def validate(model, dataloader, criterion, device, writer, epoch):
    """
    Validation döngüsü

    Args:
        model: PyTorch model
        dataloader: Validation DataLoader
        criterion: Loss fonksiyonu
        device: cuda veya cpu
        writer: TensorBoard SummaryWriter
        epoch: Epoch numarası

    Returns:
        avg_loss: Ortalama loss
        metrics: Metrik dictionary
    """
    model.eval()
    total_loss = 0

    # Metric tracker
    metric_tracker = MetricTracker()

    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f'Epoch {epoch + 1} [Val]')
        for img_a, img_b, labels in pbar:
            # GPU'ya taşı
            img_a = img_a.to(device)
            img_b = img_b.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(img_a, img_b)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            # Metrikleri güncelle
            metric_tracker.update(outputs, labels)

            # Progress bar güncelle
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    # Ortalama loss
    avg_loss = total_loss / len(dataloader)

    # Metrikleri hesapla
    metrics = metric_tracker.compute()

    # TensorBoard'a yaz
    writer.add_scalar('Val/Loss', avg_loss, epoch)
    writer.add_scalar('Val/F1', metrics['f1'], epoch)
    writer.add_scalar('Val/IoU', metrics['iou'], epoch)
    writer.add_scalar('Val/Precision', metrics['precision'], epoch)
    writer.add_scalar('Val/Recall', metrics['recall'], epoch)
    writer.add_scalar('Val/Accuracy', metrics['accuracy'], epoch)

    return avg_loss, metrics


def main():
    """Ana eğitim fonksiyonu"""

    # Argümanlar
    parser = argparse.ArgumentParser(description='Scene Change Detection Training')
    parser.add_argument('--data_dir', type=str, default=Config.DATA_DIR,
                        help='Veri klasörü yolu')
    parser.add_argument('--batch_size', type=int, default=Config.BATCH_SIZE,
                        help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=Config.NUM_EPOCHS,
                        help='Epoch sayısı')
    parser.add_argument('--lr', type=float, default=Config.LEARNING_RATE,
                        help='Learning rate')
    parser.add_argument('--checkpoint_dir', type=str, default=Config.CHECKPOINT_DIR,
                        help='Checkpoint klasörü')
    parser.add_argument('--log_dir', type=str, default=Config.LOG_DIR,
                        help='TensorBoard log klasörü')
    parser.add_argument('--transform_method', type=str, default=Config.TRANSFORM_METHOD,
                        choices=['albumentations', 'paired', 'minimal'],
                        help='Transform metodu')
    parser.add_argument('--num_workers', type=int, default=Config.NUM_WORKERS,
                        help='DataLoader worker sayısı')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume checkpoint path')

    args = parser.parse_args()


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Klasörleri oluştur
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)

    # TensorBoard writer
    writer = SummaryWriter(args.log_dir)

    # Dataset oluştur
    print("Loading datasets...")
    train_dataset = ChangeDetectionDataset(
        root_dir=args.data_dir,
        mode='train',
        transform_method=args.transform_method
    )

    val_dataset = ChangeDetectionDataset(
        root_dir=args.data_dir,
        mode='val',
        transform_method=args.transform_method
    )

    # DataLoader oluştur[web:98][web:100]
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )

    print(f"Train samples: {len(train_dataset)}, Batches: {len(train_loader)}")
    print(f"Val samples: {len(val_dataset)}, Batches: {len(val_loader)}")

    # Model oluştur
    print("Creating model...")
    model = SiameseUNet(in_channels=3, out_channels=1).to(device)

    # Loss fonksiyonu
    criterion = nn.BCELoss()

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5    )

    # Resume checkpoint varsa yükle
    start_epoch = 0
    best_f1 = 0.0

    if args.resume and Path(args.resume).exists():
        print(f"Loading checkpoint from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_f1 = checkpoint.get('best_f1', 0.0)
        print(f"Resumed from epoch {start_epoch}, best F1: {best_f1:.4f}")

    # Eğitim döngüsü[web:100][web:103]
    print("\nStarting training...")
    print("=" * 70)

    for epoch in range(start_epoch, args.num_epochs):
        print(f"\nEpoch {epoch + 1}/{args.num_epochs}")
        print("-" * 70)

        # Train
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, device, writer, epoch
        )

        # Validate
        val_loss, metrics = validate(
            model, val_loader, criterion, device, writer, epoch
        )

        # Learning rate scheduler
        scheduler.step(val_loss)

        # Sonuçları yazdır
        print(f"\nEpoch {epoch + 1} Results:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss:   {val_loss:.4f}")
        print(f"  F1 Score:   {metrics['f1']:.4f}")
        print(f"  IoU:        {metrics['iou']:.4f}")
        print(f"  Precision:  {metrics['precision']:.4f}")
        print(f"  Recall:     {metrics['recall']:.4f}")
        print(f"  Accuracy:   {metrics['accuracy']:.4f}")

        # Model kaydet (her epoch)
        checkpoint_path = Path(args.checkpoint_dir) / f'checkpoint_epoch_{epoch + 1}.pth'
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'metrics': metrics,
            'best_f1': best_f1
        }, checkpoint_path)

        # En iyi modeli kaydet
        if metrics['f1'] > best_f1:
            best_f1 = metrics['f1']
            best_model_path = Path(args.checkpoint_dir) / 'best_model.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'f1': best_f1,
                'metrics': metrics
            }, best_model_path)
            print(f"  ✓ New best model saved! F1: {best_f1:.4f}")

    print("\n" + "=" * 70)
    print("Training completed!")
    print(f"Best F1 Score: {best_f1:.4f}")

    # TensorBoard'u kapat
    writer.close()


if __name__ == '__main__':
    main()
