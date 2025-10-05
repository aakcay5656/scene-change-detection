import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
import cv2
import argparse
from typing import Tuple, Dict

from models.siamese_net import SiameseUNet
from data.transforms import get_transforms
from utils.metrics import calculate_metrics


class ChangeDetector:
    """Eğitilmiş model ile değişiklik tespiti yapan sınıf"""

    def __init__(self, checkpoint_path: str, device: str = 'cuda'):
        """
        Args:
            checkpoint_path: Eğitilmiş model checkpoint dosyası
            device: 'cuda' veya 'cpu'
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # Model yükle[web:142][web:145]
        self.model = SiameseUNet(in_channels=3, out_channels=1).to(self.device)
        self.load_checkpoint(checkpoint_path)
        self.model.eval()

        # Transform
        self.transform = get_transforms(mode='test', method='minimal')

        print(f"Model loaded successfully from {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str):
        """Checkpoint'ten model yükle[web:136][web:142]"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # State dict yükle
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)

        # Checkpoint bilgilerini göster
        if 'epoch' in checkpoint:
            print(f"  Epoch: {checkpoint['epoch'] + 1}")
        if 'f1' in checkpoint:
            print(f"  Best F1: {checkpoint['f1']:.4f}")
        if 'metrics' in checkpoint:
            metrics = checkpoint['metrics']
            print(f"  Metrics: F1={metrics.get('f1', 0):.4f}, "
                  f"IoU={metrics.get('iou', 0):.4f}")

    @torch.no_grad()
    def predict(self, img_a: Image.Image, img_b: Image.Image,
                threshold: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
        """
        İki görüntü arasındaki değişiklikleri tespit et

        Args:
            img_a: İlk görüntü (referans)
            img_b: İkinci görüntü (sorgu)
            threshold: Binary threshold (0.5)

        Returns:
            change_map: Float değişiklik haritası [0, 1]
            change_binary: Binary değişiklik maskesi {0, 1}
        """
        # Transform uygula
        img_a_tensor, img_b_tensor, _ = self.transform(
            img_a, img_b, Image.new('L', img_a.size)
        )

        # Batch dimension ekle
        img_a_tensor = img_a_tensor.unsqueeze(0).to(self.device)
        img_b_tensor = img_b_tensor.unsqueeze(0).to(self.device)

        # Inference[web:130]
        output = self.model(img_a_tensor, img_b_tensor)

        # Numpy'a çevir
        change_map = output.squeeze().cpu().numpy()
        change_binary = (change_map > threshold).astype(np.uint8)

        return change_map, change_binary

    # src/inference.py - visualize fonksiyonunu güncelleyin

    def visualize(self, img_a: Image.Image, img_b: Image.Image,
                  change_map: np.ndarray, change_binary: np.ndarray,
                  save_path: str = None, show: bool = True):
        """
        Değişiklikleri görselleştir
        """
        # Orijinal görüntü boyutlarını al
        orig_width, orig_height = img_a.size

        # Change map ve binary'yi orijinal boyuta resize et
        if change_map.shape != (orig_height, orig_width):
            print(f"  Resizing predictions for visualization...")
            print(f"    From {change_map.shape} to ({orig_height}, {orig_width})")

            change_map_resized = cv2.resize(
                change_map,
                (orig_width, orig_height),
                interpolation=cv2.INTER_LINEAR  # Float map için linear
            )
            change_binary_resized = cv2.resize(
                change_binary.astype(np.uint8),
                (orig_width, orig_height),
                interpolation=cv2.INTER_NEAREST  # Binary için nearest
            )
            change_map = change_map_resized
            change_binary = change_binary_resized

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # Görüntü A
        axes[0, 0].imshow(img_a)
        axes[0, 0].set_title('Referans Görüntü (A)', fontsize=14, fontweight='bold')
        axes[0, 0].axis('off')

        # Görüntü B
        axes[0, 1].imshow(img_b)
        axes[0, 1].set_title('Sorgu Görüntü (B)', fontsize=14, fontweight='bold')
        axes[0, 1].axis('off')

        # Değişiklik haritası (heatmap)
        im = axes[0, 2].imshow(change_map, cmap='hot', vmin=0, vmax=1)
        axes[0, 2].set_title('Değişiklik Olasılığı', fontsize=14, fontweight='bold')
        axes[0, 2].axis('off')
        plt.colorbar(im, ax=axes[0, 2], fraction=0.046)

        # Binary maske
        axes[1, 0].imshow(change_binary, cmap='gray')
        axes[1, 0].set_title('Binary Değişiklik Maskesi', fontsize=14, fontweight='bold')
        axes[1, 0].axis('off')

        # Overlay - A üzerine
        img_a_np = np.array(img_a)
        overlay_a = img_a_np.copy()
        overlay_a[change_binary > 0] = [255, 0, 0]  # Kırmızı
        axes[1, 1].imshow(cv2.addWeighted(img_a_np, 0.6, overlay_a, 0.4, 0))
        axes[1, 1].set_title('Tespit Edilen Değişiklikler (A)', fontsize=14, fontweight='bold')
        axes[1, 1].axis('off')

        # Overlay - B üzerine
        img_b_np = np.array(img_b)
        overlay_b = img_b_np.copy()
        overlay_b[change_binary > 0] = [255, 0, 0]  # Kırmızı
        axes[1, 2].imshow(cv2.addWeighted(img_b_np, 0.6, overlay_b, 0.4, 0))
        axes[1, 2].set_title('Tespit Edilen Değişiklikler (B)', fontsize=14, fontweight='bold')
        axes[1, 2].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

    def analyze_changes(self, change_binary: np.ndarray) -> Dict:
        """
        Değişiklikleri analiz et

        Returns:
            İstatistikler dictionary
        """
        total_pixels = change_binary.size
        changed_pixels = np.sum(change_binary > 0)
        change_percentage = (changed_pixels / total_pixels) * 100

        # Connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            change_binary.astype(np.uint8), connectivity=8
        )

        # En büyük 5 değişiklik bölgesi
        areas = stats[1:, cv2.CC_STAT_AREA]  # 0. label background
        top_5_areas = sorted(areas, reverse=True)[:5]

        return {
            'total_pixels': total_pixels,
            'changed_pixels': int(changed_pixels),
            'change_percentage': float(change_percentage),
            'num_change_regions': num_labels - 1,  # Background hariç
            'top_5_region_areas': top_5_areas,  # .tolist() KALDIRILDI
            'mean_region_area': float(np.mean(areas)) if len(areas) > 0 else 0
        }



def test_single_pair(detector: ChangeDetector, img_a_path: str, img_b_path: str,
                     output_dir: str = 'results', label_path: str = None):
    """
    Tek görüntü çifti test et
    """
    print(f"\nTesting pair:")
    print(f"  A: {img_a_path}")
    print(f"  B: {img_b_path}")

    # Görüntüleri yükle
    img_a = Image.open(img_a_path).convert('RGB')
    img_b = Image.open(img_b_path).convert('RGB')

    # Inference
    change_map, change_binary = detector.predict(img_a, img_b)

    # Analiz
    stats = detector.analyze_changes(change_binary)
    print(f"\nChange Analysis:")
    print(f"  Changed pixels: {stats['changed_pixels']:,} / {stats['total_pixels']:,}")
    print(f"  Change percentage: {stats['change_percentage']:.2f}%")
    print(f"  Number of regions: {stats['num_change_regions']}")
    print(f"  Mean region area: {stats['mean_region_area']:.1f} pixels")

    # Ground truth varsa metrik hesapla
    if label_path and Path(label_path).exists():
        label_pil = Image.open(label_path).convert('L')

        print(f"\nBoyut kontrolü:")
        print(f"  Prediction shape: {change_binary.shape}")
        print(f"  Label size: {label_pil.size}")

        # Label'ı prediction boyutuna resize et (PIL ile)
        if label_pil.size != (change_binary.shape[1], change_binary.shape[0]):
            print(f"  Resizing label to match prediction...")
            label_pil = label_pil.resize(
                (change_binary.shape[1], change_binary.shape[0]),
                Image.NEAREST  # Binary mask için nearest neighbor
            )
            print(f"  New label size: {label_pil.size}")

        label = np.array(label_pil)
        if label.max() > 1:
            label = label / 255.0

        metrics = calculate_metrics(change_binary, label)
        print(f"\nMetrics:")
        print(f"  F1 Score:  {metrics['f1']:.4f}")
        print(f"  IoU:       {metrics['iou']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")

    # Görselleştir
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_path = Path(output_dir) / f"{Path(img_a_path).stem}_result.png"

    detector.visualize(img_a, img_b, change_map, change_binary,
                       save_path=str(output_path), show=False)


def test_batch(detector: ChangeDetector, test_dir: str, output_dir: str = 'results'):
    """
    Batch test

    Args:
        detector: ChangeDetector instance
        test_dir: Test klasörü (A/, B/, label/ içermeli)
        output_dir: Çıktı klasörü
    """
    from utils.metrics import MetricTracker
    from tqdm import tqdm

    test_path = Path(test_dir)
    img_a_dir = test_path / 'A'
    img_b_dir = test_path / 'B'
    label_dir = test_path / 'label'

    # Tüm görüntüleri listele
    img_pairs = sorted(img_a_dir.glob('*.png'))

    if not img_pairs:
        print(f"No images found in {img_a_dir}")
        return None

    print(f"\nBatch testing {len(img_pairs)} image pairs...")
    print(f"Output directory: {output_dir}")

    # Output klasörünü oluştur
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    metric_tracker = MetricTracker()
    processed_count = 0
    error_count = 0

    # Progress bar
    pbar = tqdm(img_pairs, desc="Processing")

    for img_path in pbar:
        img_name = img_path.name
        img_a_path = img_a_dir / img_name
        img_b_path = img_b_dir / img_name
        label_path = label_dir / img_name

        try:
            # Görüntüleri yükle
            img_a = Image.open(img_a_path).convert('RGB')
            img_b = Image.open(img_b_path).convert('RGB')

            # Inference
            change_map, change_binary = detector.predict(img_a, img_b)

            # Metrik hesapla (label varsa)
            if label_path.exists():
                label = np.array(Image.open(label_path).convert('L'))
                if label.max() > 1:
                    label = label / 255.0

                # Boyut uyumu kontrolü ve düzeltme
                if change_binary.shape != label.shape:
                    label = cv2.resize(
                        label,
                        (change_binary.shape[1], change_binary.shape[0]),
                        interpolation=cv2.INTER_NEAREST
                    )

                # Metrikleri güncelle
                metric_tracker.update(
                    torch.from_numpy(change_binary).unsqueeze(0).float(),
                    torch.from_numpy(label).unsqueeze(0).float()
                )

            # Görselleştir ve kaydet
            output_path = Path(output_dir) / f"{img_path.stem}_result.png"
            detector.visualize(img_a, img_b, change_map, change_binary,
                               save_path=str(output_path), show=False)

            processed_count += 1
            pbar.set_postfix({'processed': processed_count, 'errors': error_count})

        except Exception as e:
            error_count += 1
            pbar.set_postfix({'processed': processed_count, 'errors': error_count})
            print(f"\nError processing {img_name}: {str(e)}")
            continue

    pbar.close()

    # Sonuçları yazdır
    print("\n" + "=" * 70)
    print("BATCH TEST RESULTS")
    print("=" * 70)
    print(f"Total images:     {len(img_pairs)}")
    print(f"Processed:        {processed_count}")
    print(f"Errors:           {error_count}")
    print(f"Success rate:     {processed_count / len(img_pairs) * 100:.2f}%")

    if processed_count > 0:
        # Ortalama metrikler
        metrics = metric_tracker.compute()
        print("\nPerformance Metrics:")
        print("-" * 70)
        print(f"  F1 Score:       {metrics['f1']:.4f}")
        print(f"  IoU:            {metrics['iou']:.4f}")
        print(f"  Precision:      {metrics['precision']:.4f}")
        print(f"  Recall:         {metrics['recall']:.4f}")
        print(f"  Accuracy:       {metrics['accuracy']:.4f}")
        print(f"  Specificity:    {metrics['specificity']:.4f}")
        print(f"  Kappa:          {metrics['kappa']:.4f}")
        print(f"  Dice:           {metrics['dice']:.4f}")
        print("-" * 70)
        print(f"\nResults saved to: {output_dir}/")
        print("=" * 70)

        return metrics
    else:
        print("\nNo images were successfully processed.")
        return None



def test_video(detector: ChangeDetector, video_path: str,
               output_path: str = 'output_video.mp4', fps: int = 30):
    """
    Video üzerinde frame-by-frame test
    """
    import os

    # Dosya kontrolü
    if not os.path.exists(video_path):
        print(f"✗ ERROR: Video file not found: {video_path}")
        print(f"  Current directory: {os.getcwd()}")
        print(f"  Absolute path: {os.path.abspath(video_path)}")
        return

    print(f"✓ Video file exists: {video_path}")
    print(f"  File size: {os.path.getsize(video_path) / (1024 * 1024):.2f} MB")

    # Video aç
    cap = cv2.VideoCapture(video_path)

    # Açıldı mı kontrol et
    if not cap.isOpened():
        print(f"\n✗ ERROR: Cannot open video file!")
        print(f"  Possible reasons:")
        print(f"    1. Codec not supported (install opencv-contrib-python)")
        print(f"    2. Video file is corrupted")
        print(f"    3. Format not supported (.mp4, .avi, .mov)")

        # Backend bilgisi
        backend = cap.getBackendName()
        print(f"  OpenCV Backend: {backend}")

        # Çözüm önerileri
        print(f"\n  Solutions:")
        print(f"    pip uninstall opencv-python -y")
        print(f"    pip install opencv-contrib-python")
        return

    # Video özellikleri
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"\n✓ Video opened successfully!")
    print(f"  Resolution: {width}x{height}")
    print(f"  Total frames: {total_frames}")
    print(f"  FPS: {video_fps}")

    # Frame sayısı 0 ise (codec sorunu)
    if total_frames == 0:
        print(f"\n⚠ WARNING: Frame count is 0 - codec issue likely")
        print(f"  Will process until end of stream...")

    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width * 2, height))

    if not out.isOpened():
        print(f"✗ ERROR: Cannot create output video writer!")
        cap.release()
        return

    print(f"\nProcessing video...")

    # İlk frame'i referans olarak al
    ret, prev_frame = cap.read()
    if not ret:
        print("✗ Error reading first frame")
        cap.release()
        out.release()
        return

    frame_idx = 0
    processed_frames = 0

    # Progress bar
    from tqdm import tqdm
    pbar = tqdm(total=total_frames if total_frames > 0 else None,
                desc="Processing")

    while True:
        ret, curr_frame = cap.read()
        if not ret:
            break

        try:
            # PIL Image'a çevir
            prev_pil = Image.fromarray(cv2.cvtColor(prev_frame, cv2.COLOR_BGR2RGB))
            curr_pil = Image.fromarray(cv2.cvtColor(curr_frame, cv2.COLOR_BGR2RGB))

            # Inference
            change_map, change_binary = detector.predict(prev_pil, curr_pil)

            # Görselleştirme
            change_binary_3ch = cv2.cvtColor(
                (change_binary * 255).astype(np.uint8),
                cv2.COLOR_GRAY2BGR
            )

            # Side-by-side
            combined = np.hstack([curr_frame, change_binary_3ch])
            out.write(combined)

            processed_frames += 1
            pbar.update(1)

        except Exception as e:
            print(f"\n✗ Error processing frame {frame_idx}: {e}")
            break

        prev_frame = curr_frame
        frame_idx += 1

    pbar.close()
    cap.release()
    out.release()

    print(f"\n✓ Video processing completed!")
    print(f"  Processed frames: {processed_frames}")
    print(f"  Output saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Change Detection Inference')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Model checkpoint path')
    parser.add_argument('--mode', type=str, choices=['single', 'batch', 'video'],
                        default='single', help='Test mode')
    parser.add_argument('--img_a', type=str, help='First image path (single mode)')
    parser.add_argument('--img_b', type=str, help='Second image path (single mode)')
    parser.add_argument('--label', type=str, help='Ground truth label (optional)')
    parser.add_argument('--test_dir', type=str, help='Test directory (batch mode)')
    parser.add_argument('--video', type=str, help='Video path (video mode)')
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Output directory')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'], help='Device')

    args = parser.parse_args()

    # Detector oluştur
    detector = ChangeDetector(args.checkpoint, device=args.device)

    # Mode'a göre test
    if args.mode == 'single':
        if not args.img_a or not args.img_b:
            raise ValueError("--img_a and --img_b required for single mode")
        test_single_pair(detector, args.img_a, args.img_b,
                         args.output_dir, args.label)

    elif args.mode == 'batch':
        if not args.test_dir:
            raise ValueError("--test_dir required for batch mode")
        test_batch(detector, args.test_dir, args.output_dir)

    elif args.mode == 'video':
        if not args.video:
            raise ValueError("--video required for video mode")
        test_video(detector, args.video,
                   Path(args.output_dir) / 'output_video.mp4')


if __name__ == '__main__':
    main()
