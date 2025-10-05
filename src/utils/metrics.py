# src/utils/metrics.py

import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import confusion_matrix, cohen_kappa_score
from typing import Dict, Optional, Tuple
import warnings


class ChangeDetectionMetrics:
    """
    Sahne değişikliği tespiti için kapsamlı metrik hesaplama sınıfı[web:60][web:66].
    Binary segmentation için optimize edilmiştir.
    """

    def __init__(self, threshold: float = 0.5, eps: float = 1e-7):
        """
        Args:
            threshold: Binary threshold değeri (0.5)
            eps: Bölme hatalarını önlemek için küçük değer
        """
        self.threshold = threshold
        self.eps = eps
        self.reset()

    def reset(self):
        """Tüm metrikleri sıfırla"""
        self.tp = 0  # True Positives
        self.tn = 0  # True Negatives
        self.fp = 0  # False Positives
        self.fn = 0  # False Negatives

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        """
        Metrik değerlerini güncelle

        Args:
            preds: Tahmin edilen değerler (B, 1, H, W) veya (B, H, W)
            targets: Gerçek değerler (B, 1, H, W) veya (B, H, W)
        """
        # Tensor'ları numpy'a çevir
        if isinstance(preds, torch.Tensor):
            preds = preds.detach().cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.detach().cpu().numpy()

        # Boyutları düzenle
        preds = preds.squeeze()
        targets = targets.squeeze()

        # Binary threshold uygula
        preds_binary = (preds > self.threshold).astype(np.uint8)
        targets_binary = (targets > self.threshold).astype(np.uint8)

        # Confusion matrix elemanlarını hesapla
        self.tp += np.sum((preds_binary == 1) & (targets_binary == 1))
        self.tn += np.sum((preds_binary == 0) & (targets_binary == 0))
        self.fp += np.sum((preds_binary == 1) & (targets_binary == 0))
        self.fn += np.sum((preds_binary == 0) & (targets_binary == 1))

    def compute(self) -> Dict[str, float]:
        """
        Tüm metrikleri hesapla[web:63][web:64][web:67]

        Returns:
            Dict içinde metrik değerleri
        """
        # Precision (Positive Predictive Value)
        precision = self.tp / (self.tp + self.fp + self.eps)

        # Recall (Sensitivity, True Positive Rate)
        recall = self.tp / (self.tp + self.fn + self.eps)

        # F1-Score (Dice Coefficient)
        f1 = 2 * precision * recall / (precision + recall + self.eps)

        # IoU (Jaccard Index)
        iou = self.tp / (self.tp + self.fp + self.fn + self.eps)

        # Accuracy (Overall Accuracy)
        accuracy = (self.tp + self.tn) / (self.tp + self.tn + self.fp + self.fn + self.eps)

        # Specificity (True Negative Rate)
        specificity = self.tn / (self.tn + self.fp + self.eps)

        # Cohen's Kappa
        total = self.tp + self.tn + self.fp + self.fn
        p_observed = (self.tp + self.tn) / (total + self.eps)
        p_expected = ((self.tp + self.fp) * (self.tp + self.fn) +
                      (self.fn + self.tn) * (self.fp + self.tn)) / (total * total + self.eps)
        kappa = (p_observed - p_expected) / (1 - p_expected + self.eps)

        return {
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'iou': float(iou),
            'accuracy': float(accuracy),
            'specificity': float(specificity),
            'kappa': float(kappa),
            'tp': int(self.tp),
            'tn': int(self.tn),
            'fp': int(self.fp),
            'fn': int(self.fn)
        }


def calculate_metrics(preds: np.ndarray,
                      targets: np.ndarray,
                      threshold: float = 0.5) -> Dict[str, float]:
    """
    Tek seferlik metrik hesaplama fonksiyonu[web:66][web:12]

    Args:
        preds: Tahmin edilen değerler
        targets: Gerçek değerler
        threshold: Binary threshold

    Returns:
        Metrik değerleri dictionary
    """
    metrics = ChangeDetectionMetrics(threshold=threshold)
    metrics.update(preds, targets)
    return metrics.compute()


class IoUMetric:
    """
    Intersection over Union (IoU) metriği[web:71][web:74][web:76]
    """

    def __init__(self, num_classes: int = 2, eps: float = 1e-7):
        self.num_classes = num_classes
        self.eps = eps
        self.reset()

    def reset(self):
        self.intersection = np.zeros(self.num_classes)
        self.union = np.zeros(self.num_classes)

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        """
        Args:
            preds: (B, C, H, W) veya (B, H, W)
            targets: (B, C, H, W) veya (B, H, W)
        """
        if isinstance(preds, torch.Tensor):
            preds = preds.detach().cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.detach().cpu().numpy()

        preds = preds.squeeze()
        targets = targets.squeeze()

        # Binary durumu için
        if self.num_classes == 2:
            preds_binary = (preds > 0.5).astype(np.uint8)
            targets_binary = (targets > 0.5).astype(np.uint8)

            for cls in range(self.num_classes):
                pred_cls = (preds_binary == cls)
                target_cls = (targets_binary == cls)

                self.intersection[cls] += np.sum(pred_cls & target_cls)
                self.union[cls] += np.sum(pred_cls | target_cls)

    def compute(self) -> Dict[str, float]:
        """IoU değerlerini hesapla"""
        iou_per_class = self.intersection / (self.union + self.eps)
        mean_iou = np.mean(iou_per_class)

        return {
            'mean_iou': float(mean_iou),
            'iou_class_0': float(iou_per_class[0]),
            'iou_class_1': float(iou_per_class[1]) if self.num_classes > 1 else 0.0
        }


class F1Score:
    """
    F1-Score hesaplama sınıfı[web:63][web:64]
    """

    def __init__(self, threshold: float = 0.5, eps: float = 1e-7):
        self.threshold = threshold
        self.eps = eps
        self.reset()

    def reset(self):
        self.tp = 0
        self.fp = 0
        self.fn = 0

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        if isinstance(preds, torch.Tensor):
            preds = preds.detach().cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.detach().cpu().numpy()

        preds = (preds > self.threshold).astype(np.uint8).flatten()
        targets = (targets > self.threshold).astype(np.uint8).flatten()

        self.tp += np.sum((preds == 1) & (targets == 1))
        self.fp += np.sum((preds == 1) & (targets == 0))
        self.fn += np.sum((preds == 0) & (targets == 1))

    def compute(self) -> float:
        precision = self.tp / (self.tp + self.fp + self.eps)
        recall = self.tp / (self.tp + self.fn + self.eps)
        f1 = 2 * precision * recall / (precision + recall + self.eps)
        return float(f1)


class ConfusionMatrix:
    """
    Confusion Matrix hesaplama ve görselleştirme[web:65]
    """

    def __init__(self, num_classes: int = 2):
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        self.matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        if isinstance(preds, torch.Tensor):
            preds = preds.detach().cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.detach().cpu().numpy()

        preds = (preds > 0.5).astype(np.uint8).flatten()
        targets = (targets > 0.5).astype(np.uint8).flatten()

        cm = confusion_matrix(targets, preds, labels=range(self.num_classes))
        self.matrix += cm

    def compute(self) -> np.ndarray:
        return self.matrix

    def get_metrics_from_matrix(self) -> Dict[str, float]:
        """Confusion matrix'ten metrik hesapla"""
        tp = self.matrix[1, 1]
        tn = self.matrix[0, 0]
        fp = self.matrix[0, 1]
        fn = self.matrix[1, 0]

        eps = 1e-7
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)
        accuracy = (tp + tn) / (tp + tn + fp + fn + eps)

        return {
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'accuracy': float(accuracy)
        }


class DiceCoefficient:
    """
    Dice Coefficient (F1-Score ile aynı)[web:66][web:68]
    """

    def __init__(self, smooth: float = 1.0):
        self.smooth = smooth
        self.reset()

    def reset(self):
        self.dice_sum = 0.0
        self.count = 0

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        """
        Args:
            preds: (B, 1, H, W)
            targets: (B, 1, H, W)
        """
        if not isinstance(preds, torch.Tensor):
            preds = torch.from_numpy(preds)
        if not isinstance(targets, torch.Tensor):
            targets = torch.from_numpy(targets)

        preds = (preds > 0.5).float()
        targets = (targets > 0.5).float()

        batch_size = preds.size(0)

        for i in range(batch_size):
            pred = preds[i].flatten()
            target = targets[i].flatten()

            intersection = (pred * target).sum()
            dice = (2. * intersection + self.smooth) / (
                    pred.sum() + target.sum() + self.smooth
            )

            self.dice_sum += dice.item()
            self.count += 1

    def compute(self) -> float:
        if self.count == 0:
            return 0.0
        return self.dice_sum / self.count


# Metric aggregator
class MetricTracker:
    """
    Birden fazla metriği takip eden sınıf[web:60]
    """

    def __init__(self, metrics_list: Optional[list] = None):
        """
        Args:
            metrics_list: Takip edilecek metrik isimleri
                         ['f1', 'iou', 'precision', 'recall', 'accuracy']
        """
        if metrics_list is None:
            metrics_list = ['f1', 'iou', 'precision', 'recall', 'accuracy']

        self.metrics_list = metrics_list
        self.cd_metrics = ChangeDetectionMetrics()
        self.iou_metric = IoUMetric()
        self.dice_metric = DiceCoefficient()

    def reset(self):
        """Tüm metrikleri sıfırla"""
        self.cd_metrics.reset()
        self.iou_metric.reset()
        self.dice_metric.reset()

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        """Tüm metrikleri güncelle"""
        self.cd_metrics.update(preds, targets)
        self.iou_metric.update(preds, targets)
        self.dice_metric.update(preds, targets)

    def compute(self) -> Dict[str, float]:
        """Tüm metrikleri hesapla ve döndür"""
        cd_results = self.cd_metrics.compute()
        iou_results = self.iou_metric.compute()
        dice_result = self.dice_metric.compute()

        results = {
            'f1': cd_results['f1'],
            'precision': cd_results['precision'],
            'recall': cd_results['recall'],
            'iou': cd_results['iou'],
            'accuracy': cd_results['accuracy'],
            'specificity': cd_results['specificity'],
            'kappa': cd_results['kappa'],
            'dice': dice_result,
            'mean_iou': iou_results['mean_iou']
        }

        return results

    def get_summary(self) -> str:
        """Metrik özetini string olarak döndür"""
        metrics = self.compute()

        summary = "Metrics Summary:\n"
        summary += f"  F1-Score:    {metrics['f1']:.4f}\n"
        summary += f"  IoU:         {metrics['iou']:.4f}\n"
        summary += f"  Precision:   {metrics['precision']:.4f}\n"
        summary += f"  Recall:      {metrics['recall']:.4f}\n"
        summary += f"  Accuracy:    {metrics['accuracy']:.4f}\n"
        summary += f"  Kappa:       {metrics['kappa']:.4f}\n"

        return summary


# Loss fonksiyonları ile birleştirilmiş metrikler
class DiceLoss(nn.Module):
    """
    Dice Loss - 1 - Dice Coefficient[web:68]
    """

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        preds = torch.sigmoid(preds) if preds.min() < 0 else preds

        preds = preds.view(-1)
        targets = targets.view(-1)

        intersection = (preds * targets).sum()
        dice = (2. * intersection + self.smooth) / (
                preds.sum() + targets.sum() + self.smooth
        )

        return 1 - dice


class IoULoss(nn.Module):
    """
    IoU Loss - 1 - IoU[web:74][web:76]
    """

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        preds = torch.sigmoid(preds) if preds.min() < 0 else preds

        preds = preds.view(-1)
        targets = targets.view(-1)

        intersection = (preds * targets).sum()
        union = preds.sum() + targets.sum() - intersection

        iou = (intersection + self.smooth) / (union + self.smooth)

        return 1 - iou


class CombinedLoss(nn.Module):
    """
    BCE + Dice Loss kombinasyonu (change detection için popüler)[web:60]
    """

    def __init__(self, alpha: float = 0.5, beta: float = 0.5):
        """
        Args:
            alpha: BCE weight
            beta: Dice weight
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.bce = nn.BCELoss()
        self.dice = DiceLoss()

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(preds, targets)
        dice_loss = self.dice(preds, targets)

        return self.alpha * bce_loss + self.beta * dice_loss


# Utility fonksiyonlar
def print_metrics(metrics: Dict[str, float], prefix: str = ""):
    """Metrikleri güzel formatta yazdır"""
    print(f"\n{prefix} Metrics:")
    print("=" * 50)
    for key, value in metrics.items():
        if isinstance(value, (int, np.integer)):
            print(f"  {key:15s}: {value:>10d}")
        else:
            print(f"  {key:15s}: {value:>10.4f}")
    print("=" * 50)


def compute_batch_metrics(preds: torch.Tensor,
                          targets: torch.Tensor) -> Dict[str, float]:
    """
    Batch için hızlı metrik hesaplama

    Args:
        preds: (B, 1, H, W)
        targets: (B, 1, H, W)

    Returns:
        Metrik dictionary
    """
    tracker = MetricTracker()
    tracker.update(preds, targets)
    return tracker.compute()
