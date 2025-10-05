import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import random
from PIL import Image


class AlbumentationsTransform:
    """
    Albumentations kütüphanesini kullanarak gelişmiş augmentasyon
    """

    def __init__(self, mode='train'):
        self.mode = mode

        if mode == 'train':
            self.transform = A.Compose([
                A.Resize(256, 256),

                # Geometrik transformasyonlar
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),

                # Affine kullan (ShiftScaleRotate yerine)
                A.Affine(
                    scale=(0.9, 1.1),
                    translate_percent=(-0.1, 0.1),
                    rotate=(-45, 45),
                    shear=(-10, 10),
                    p=0.5
                ),

                # Spatial transformasyonlar
                A.OneOf([
                    A.ElasticTransform(alpha=120, sigma=120 * 0.05, p=0.5),
                    A.GridDistortion(p=0.5),
                    # OpticalDistortion düzeltildi - shift_limit kaldırıldı
                    A.OpticalDistortion(distort_limit=0.5, p=0.5),
                ], p=0.3),

                # Renk ve ışık augmentasyonları
                A.OneOf([
                    A.RandomBrightnessContrast(p=0.5),
                    A.RandomGamma(p=0.5),
                    A.HueSaturationValue(p=0.5),
                ], p=0.3),

                # Blur ve noise
                A.OneOf([
                    A.GaussianBlur(p=0.5),
                    A.MedianBlur(blur_limit=5, p=0.5),
                    A.GaussNoise(p=0.5),
                ], p=0.2),

                # Weather effects (optional)
                A.OneOf([
                    A.RandomRain(p=0.3),
                    A.RandomFog(p=0.3),
                    A.RandomShadow(p=0.3),
                ], p=0.1),

                # Normalization
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
                ToTensorV2()
            ], additional_targets={'image2': 'image', 'mask': 'mask'})

        else:  # val/test
            self.transform = A.Compose([
                A.Resize(256, 256),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
                ToTensorV2()
            ], additional_targets={'image2': 'image', 'mask': 'mask'})

    def __call__(self, img_a, img_b, mask):
        """
        Args:
            img_a: PIL Image veya numpy array
            img_b: PIL Image veya numpy array
            mask: PIL Image veya numpy array

        Returns:
            img_a_tensor: torch.FloatTensor (C, H, W)
            img_b_tensor: torch.FloatTensor (C, H, W)
            mask_tensor: torch.FloatTensor (1, H, W) - BCELoss için Float!
        """
        # PIL Image'ı numpy'a çevir
        if isinstance(img_a, Image.Image):
            img_a = np.array(img_a)
        if isinstance(img_b, Image.Image):
            img_b = np.array(img_b)
        if isinstance(mask, Image.Image):
            mask = np.array(mask)

        # Mask'i 2D yap
        if len(mask.shape) == 3:
            mask = mask[:, :, 0]

        # Mask'i [0, 1] aralığına normalize et
        if mask.max() > 1.0:
            mask = mask / 255.0

        # Transform uygula
        transformed = self.transform(
            image=img_a,
            image2=img_b,
            mask=mask
        )

        # Mask'i float'a çevir ve doğru boyuta getir[web:82]
        mask_tensor = transformed['mask']

        # Eğer mask uint8 ise float'a çevir
        if mask_tensor.dtype == torch.uint8:
            mask_tensor = mask_tensor.float() / 255.0

        # Eğer mask float değilse çevir
        if mask_tensor.dtype != torch.float32:
            mask_tensor = mask_tensor.float()

        # Mask'in boyutunu kontrol et
        if mask_tensor.dim() == 2:
            mask_tensor = mask_tensor.unsqueeze(0)  # (H, W) -> (1, H, W)
        elif mask_tensor.dim() == 3 and mask_tensor.size(0) != 1:
            mask_tensor = mask_tensor.unsqueeze(0)  # (H, W, 1) -> (1, H, W, 1) -> squeeze -> (1, H, W)

        return transformed['image'], transformed['image2'], mask_tensor


class PairedRandomTransform:
    """
    PyTorch native transforms ile paired transformation
    """

    def __init__(self, mode='train'):
        self.mode = mode

    def __call__(self, img_a, img_b, mask):
        # Resize
        img_a = TF.resize(img_a, [256, 256])
        img_b = TF.resize(img_b, [256, 256])
        mask = TF.resize(mask, [256, 256], interpolation=T.InterpolationMode.NEAREST)

        if self.mode == 'train':
            # Random Horizontal Flip
            if random.random() > 0.5:
                img_a = TF.hflip(img_a)
                img_b = TF.hflip(img_b)
                mask = TF.hflip(mask)

            # Random Vertical Flip
            if random.random() > 0.5:
                img_a = TF.vflip(img_a)
                img_b = TF.vflip(img_b)
                mask = TF.vflip(mask)

            # Random Rotation
            angle = random.choice([0, 90, 180, 270])
            if angle != 0:
                img_a = TF.rotate(img_a, angle)
                img_b = TF.rotate(img_b, angle)
                mask = TF.rotate(mask, angle)

        # ToTensor
        img_a = TF.to_tensor(img_a)
        img_b = TF.to_tensor(img_b)
        mask = TF.to_tensor(mask)

        # Normalize (sadece görüntüler için)
        img_a = TF.normalize(img_a, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        img_b = TF.normalize(img_b, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        # Mask'i float'a çevir ve [0, 1] aralığına getir[web:78][web:79]
        mask = mask.float()
        if mask.max() > 1.0:
            mask = mask / 255.0

        return img_a, img_b, mask


class MinimalTransform:
    """
    En basit transformasyon - test amaçlı
    """

    def __init__(self, size=256):
        self.size = size

    def __call__(self, img_a, img_b, mask):
        # Resize
        img_a = TF.resize(img_a, [self.size, self.size])
        img_b = TF.resize(img_b, [self.size, self.size])
        mask = TF.resize(mask, [self.size, self.size], interpolation=T.InterpolationMode.NEAREST)

        # ToTensor
        img_a = TF.to_tensor(img_a)
        img_b = TF.to_tensor(img_b)
        mask = TF.to_tensor(mask)

        # Normalize
        img_a = TF.normalize(img_a, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        img_b = TF.normalize(img_b, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        # Mask float ve [0, 1] aralığı
        mask = mask.float()
        if mask.max() > 1.0:
            mask = mask / 255.0

        return img_a, img_b, mask


def get_transforms(mode='train', method='albumentations'):
    """
    Transform factory fonksiyonu

    Args:
        mode: 'train', 'val', 'test'
        method: 'albumentations', 'paired', 'minimal'
    """
    if method == 'albumentations':
        return AlbumentationsTransform(mode=mode)
    elif method == 'paired':
        return PairedRandomTransform(mode=mode)
    elif method == 'minimal':
        return MinimalTransform()
    else:
        raise ValueError(f"Unknown transform method: {method}")


class DeNormalize:
    """Görselleştirme için denormalize"""

    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor
