import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from pathlib import Path


class ChangeDetectionDataset(Dataset):
    def __init__(self, root_dir, mode='train', transform_method='albumentations'):
        """
        Args:
            root_dir: Veri klasörü yolu
            mode: 'train', 'val', 'test'
            transform_method: 'albumentations', 'paired', 'minimal'
        """
        self.root_dir = Path(root_dir)
        self.mode = mode

        # Transform'u import et
        from data.transforms import get_transforms
        self.transform = get_transforms(mode=mode, method=transform_method)

        self.image_pairs = []
        self.load_data()

    def load_data(self):
        """Veri yollarını yükle"""
        img_dir_a = self.root_dir / self.mode / 'A'
        img_dir_b = self.root_dir / self.mode / 'B'
        label_dir = self.root_dir / self.mode / 'label'

        # Tüm görüntüleri listele
        for img_path in sorted(img_dir_a.glob('*.png')):
            img_name = img_path.name
            self.image_pairs.append({
                'img_a': img_dir_a / img_name,
                'img_b': img_dir_b / img_name,
                'label': label_dir / img_name
            })

        print(f"Loaded {len(self.image_pairs)} image pairs for {self.mode}")

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        """
        Returns:
            img_a: torch.FloatTensor (3, H, W)
            img_b: torch.FloatTensor (3, H, W)
            label: torch.FloatTensor (1, H, W)
        """
        pair = self.image_pairs[idx]

        # Görüntüleri yükle
        img_a = Image.open(pair['img_a']).convert('RGB')
        img_b = Image.open(pair['img_b']).convert('RGB')
        label = Image.open(pair['label']).convert('L')

        # Transform uygula
        if self.transform:
            img_a, img_b, label = self.transform(img_a, img_b, label)

        # Son kontrol - label float olmalı
        if not isinstance(label, torch.Tensor):
            label = torch.from_numpy(label)

        if label.dtype != torch.float32:
            label = label.float()

        # [0, 1] aralığında olduğundan emin ol
        if label.max() > 1.0:
            label = label / 255.0

        return img_a, img_b, label


