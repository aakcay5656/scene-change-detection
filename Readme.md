TR
# Sahne Değişikliği Tespit Sistemi

## Proje Hakkında

Bu proje, iki görüntü arasındaki değişiklikleri otomatik olarak tespit eden ve sınıflandıran bir derin öğrenme modelidir. Siamese U-Net mimarisi kullanılarak geliştirilmiştir ve PyTorch framework'ü ile implement edilmiştir.

### Özellikler

- Siamese U-Net mimarisi ile yüksek performanslı değişiklik tespiti
- GPU (CUDA) desteği ile hızlı eğitim ve inference
- TensorBoard entegrasyonu ile eğitim takibi
- Kapsamlı metrik değerlendirme (F1, IoU, Precision, Recall, Accuracy)
- Docker desteği ile kolay deployment
- Batch inference ve video işleme desteği
- Albumentations ile gelişmiş data augmentation

## Gereksinimler

### Sistem Gereksinimleri

- Python 3.11+
- CUDA 12.1+ (GPU kullanımı için)
- NVIDIA GPU (önerilen: RTX 3060 veya üzeri, minimum 8GB VRAM)
- 16GB RAM (önerilen)
- 50GB boş disk alanı

### Python Kütüphaneleri

```
torch==2.1.0+cu121
torchvision==0.16.0+cu121
torchaudio==2.1.0+cu121
albumentations==1.4.0
opencv-contrib-python==4.8.1.78
matplotlib==3.8.2
tensorboard==2.15.0
numpy==1.24.3
Pillow==10.1.0
scikit-learn==1.3.2
tqdm==4.66.1
pytest==7.4.3
```

## Kurulum

### Manuel Kurulum

```
# Repository'yi klonlayın
git clone https://github.com/aakcay5656/scene-change-detection.git
cd scene-change-detection

# Virtual environment oluşturun
python -m venv .venv

# Virtual environment'ı aktif edin
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# Bağımlılıkları kurun
pip install -r requirements.txt
```

### Docker ile Kurulum

```
# Docker image'ı oluşturun
docker-compose build

# Eğitim için çalıştırın
docker-compose up training

# Inference için çalıştırın
docker-compose up inference
```

## Veri Yapısı

Veri setinizi aşağıdaki yapıda organize edin:

```
data/
├── processed/
│   ├── train/
│   │   ├── A/              # Referans görüntüler
│   │   ├── B/              # Sorgu görüntüler
│   │   └── label/          # Ground truth maskeler
│   ├── val/
│   │   ├── A/
│   │   ├── B/
│   │   └── label/
│   └── test/
│       ├── A/
│       ├── B/
│       └── label/
```

Görüntü formatları: PNG, JPG
Maske formatları: PNG (binary: 0 veya 255)

## Kullanım

### Model Eğitimi

```
# Temel eğitim
python src/train.py

# Özelleştirilmiş parametrelerle
python src/train.py \
    --data_dir data/processed \
    --batch_size 16 \
    --num_epochs 100 \
    --lr 0.0001 \
    --checkpoint_dir checkpoints \
    --log_dir logs/experiment_1 \
    --transform_method albumentations \
    --num_workers 4
```

### TensorBoard ile Eğitim Takibi

```
# TensorBoard'u başlatın
tensorboard --logdir=logs

# Tarayıcıda açın
# http://localhost:6006
```

### Inference

#### Tek Görüntü Çifti

```
python src/inference.py \
    --checkpoint checkpoints/best_model.pth \
    --mode single \
    --img_a data/processed/test/A/image_001.png \
    --img_b data/processed/test/B/image_001.png \
    --label data/processed/test/label/image_001.png \
    --output_dir results/single
```

#### Batch Inference

```
python src/inference.py \
    --checkpoint checkpoints/best_model.pth \
    --mode batch \
    --test_dir data/processed/test \
    --output_dir results/batch
```

#### Video İşleme

```
python src/inference.py \
    --checkpoint checkpoints/best_model.pth \
    --mode video \
    --video input_video.mp4 \
    --output_dir results/video
```

## Docker Kullanımı

### Eğitim

```
# Eğitimi başlat
docker-compose up training

# Arka planda çalıştır
docker-compose up -d training

# Logları takip et
docker-compose logs -f training
```

### Inference

```
# Batch inference
docker-compose up inference

# GPU kullanımını kontrol et
docker-compose exec training nvidia-smi
```

### TensorBoard

```
# TensorBoard'u başlat
docker-compose up tensorboard

# Tarayıcıda aç: http://localhost:6006
```

## Proje Yapısı

```
scene-change-detection/
├── src/
│   ├── models/
│   │   ├── __init__.py
│   │   └── siamese_net.py          # Siamese U-Net modeli
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py              # Dataset sınıfı
│   │   └── transforms.py           # Data augmentation
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── metrics.py              # Değerlendirme metrikleri
│   │   └── visualize.py            # Görselleştirme araçları
│   ├── train.py                    # Eğitim scripti
│   ├── inference.py                # Inference scripti
│   └── config.py                   # Konfigürasyon
├── data/
│   ├── raw/                        # Ham veri
│   └── processed/                  # İşlenmiş veri
├── checkpoints/                    # Model checkpoint'leri
├── logs/                           # TensorBoard logları
├── results/                        # Inference sonuçları
├── tests/                          # Unit testler
├── notebooks/                      # Jupyter notebook'lar
├── Dockerfile                      # Docker konfigürasyonu
├── docker-compose.yml              # Docker Compose konfigürasyonu
├── requirements.txt                # Python bağımlılıkları
├── .gitignore                      # Git ignore kuralları
└── README.md                       # Bu dosya
```

## Model Mimarisi

### Siamese U-Net

Model, paylaşımlı ağırlıklı (shared weights) Siamese encoder ve U-Net decoder yapısından oluşur:

- Encoder: İki görüntüyü paralel olarak işler
- Feature Extraction: Her seviyede fark özellikleri hesaplanır
- Decoder: U-Net yapısı ile upsampling ve segmentasyon
- Output: Binary change mask (256x256)

### Hiperparametreler

- Input Size: 256x256
- Batch Size: 8-16
- Learning Rate: 1e-4
- Optimizer: Adam
- Loss Function: Binary Cross Entropy
- Scheduler: ReduceLROnPlateau

## Değerlendirme Metrikleri

Model performansı aşağıdaki metriklerle değerlendirilir:

- F1-Score: Precision ve Recall'un harmonik ortalaması
- IoU (Intersection over Union): Jaccard Index
- Precision: Pozitif tahminlerin doğruluk oranı
- Recall: Gerçek pozitiflerin tespit oranı
- Accuracy: Genel doğruluk oranı
- Kappa: Cohen's Kappa katsayısı

## Sonuçlar

### Örnek Performans

```
Validation Set Results:
  F1 Score:    0.6114
  IoU:         0.4403
  Precision:   0.6523
  Recall:      0.5782
  Accuracy:    0.8856
```

### Görselleştirme

Inference sonuçları 6 panelden oluşur:

1. Referans Görüntü (A)
2. Sorgu Görüntü (B)
3. Değişiklik Olasılığı (Heatmap)
4. Binary Değişiklik Maskesi
5. Tespit Edilen Değişiklikler (A üzerine overlay)
6. Tespit Edilen Değişiklikler (B üzerine overlay)

## Test

```
# Unit testleri çalıştır
pytest tests/

# Coverage raporu
pytest --cov=src tests/

# Belirli bir test
pytest tests/test_model.py
```

## GPU Kullanımı

### GPU Kontrolü

```
# NVIDIA driver kontrolü
nvidia-smi

# PyTorch CUDA kontrolü
python -c "import torch; print(torch.cuda.is_available())"
```

### GPU Bellek Optimizasyonu

- Mixed Precision Training: Bellek kullanımını yarıya indirir
- Gradient Accumulation: Küçük batch size ile büyük effective batch size
- Pin Memory: DataLoader'da GPU transfer hızını artırır

## Sorun Giderme

### CUDA Out of Memory

```
# Batch size'ı azaltın
python src/train.py --batch_size 4

# Veya gradient accumulation kullanın
```

### Video Okuma Hatası

```
# opencv-contrib-python kurun
pip uninstall opencv-python -y
pip install opencv-contrib-python
```

### TensorBoard Açılmıyor

```
# Port çakışması varsa farklı port kullanın
tensorboard --logdir=logs --port=6007
```

## Geliştirme

### Yeni Özellik Ekleme

1. Feature branch oluşturun: `git checkout -b feature/yeni-ozellik`
2. Değişikliklerinizi yapın
3. Test ekleyin: `tests/test_yeni_ozellik.py`
4. Commit edin: `git commit -m "Add: Yeni özellik açıklaması"`
5. Push edin: `git push origin feature/yeni-ozellik`
6. Pull request oluşturun

### Kod Standartları

- PEP 8 uyumlu kod yazın
- Docstring ekleyin (Google style)
- Type hints kullanın
- Unit test yazın

## Lisans

Bu proje MIT Lisansı ile lisanslanmıştır.

## İletişim

Sorularınız için:

- Email: aakcay5656@gmail.com
- GitHub Issues: https://github.com/aakcay5656/scene-change-detection/issues

## Referanslar

- PyTorch Documentation: https://pytorch.org/docs/
- Albumentations: https://albumentations.ai/
- U-Net Paper: https://arxiv.org/abs/1505.04597
- Siamese Networks: https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf

## Değişiklik Geçmişi

### v1.0.0 (2025-10-05)

- İlk sürüm
- Siamese U-Net model implementasyonu
- Docker desteği
- TensorBoard entegrasyonu
- Batch ve video inference desteği
- Kapsamlı dokümantasyon

## Teşekkürler

- PyTorch topluluğuna
- Açık kaynak kütüphane geliştiricilerine

-----------------------------------------------------------------------
EN
# Scene Change Detection System

## About the Project

This project is a deep learning model that automatically detects and classifies changes between two images. It is developed using Siamese U-Net architecture and implemented with PyTorch framework.

### Features

- High-performance change detection with Siamese U-Net architecture
- Fast training and inference with GPU (CUDA) support
- Training monitoring with TensorBoard integration
- Comprehensive metric evaluation (F1, IoU, Precision, Recall, Accuracy)
- Easy deployment with Docker support
- Batch inference and video processing capabilities
- Advanced data augmentation with Albumentations

## Requirements

### System Requirements

- Python 3.11+
- CUDA 12.1+ (for GPU usage)
- NVIDIA GPU (recommended: RTX 3060 or higher, minimum 8GB VRAM)
- 16GB RAM (recommended)
- 50GB free disk space

### Python Libraries

```
torch==2.1.0+cu121
torchvision==0.16.0+cu121
torchaudio==2.1.0+cu121
albumentations==1.4.0
opencv-contrib-python==4.8.1.78
matplotlib==3.8.2
tensorboard==2.15.0
numpy==1.24.3
Pillow==10.1.0
scikit-learn==1.3.2
tqdm==4.66.1
pytest==7.4.3
```

## Installation

### Manual Installation

```bash
# Clone the repository
git clone https://github.com/aakcay5656/scene-change-detection.git
cd scene-change-detection

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Docker Installation

```bash
# Build Docker image
docker-compose build

# Run for training
docker-compose up training

# Run for inference
docker-compose up inference
```

## Data Structure

Organize your dataset in the following structure:

```
data/
├── processed/
│   ├── train/
│   │   ├── A/              # Reference images
│   │   ├── B/              # Query images
│   │   └── label/          # Ground truth masks
│   ├── val/
│   │   ├── A/
│   │   ├── B/
│   │   └── label/
│   └── test/
│       ├── A/
│       ├── B/
│       └── label/
```

Image formats: PNG, JPG
Mask formats: PNG (binary: 0 or 255)

## Usage

### Model Training

```bash
# Basic training
python src/train.py

# Training with custom parameters
python src/train.py \
    --data_dir data/processed \
    --batch_size 16 \
    --num_epochs 100 \
    --lr 0.0001 \
    --checkpoint_dir checkpoints \
    --log_dir logs/experiment_1 \
    --transform_method albumentations \
    --num_workers 4
```

### Training Monitoring with TensorBoard

```bash
# Start TensorBoard
tensorboard --logdir=logs

# Open in browser
# http://localhost:6006
```

### Inference

#### Single Image Pair

```bash
python src/inference.py \
    --checkpoint checkpoints/best_model.pth \
    --mode single \
    --img_a data/processed/test/A/image_001.png \
    --img_b data/processed/test/B/image_001.png \
    --label data/processed/test/label/image_001.png \
    --output_dir results/single
```

#### Batch Inference

```bash
python src/inference.py \
    --checkpoint checkpoints/best_model.pth \
    --mode batch \
    --test_dir data/processed/test \
    --output_dir results/batch
```

#### Video Processing

```bash
python src/inference.py \
    --checkpoint checkpoints/best_model.pth \
    --mode video \
    --video input_video.mp4 \
    --output_dir results/video
```

## Docker Usage

### Training

```bash
# Start training
docker-compose up training

# Run in background
docker-compose up -d training

# Follow logs
docker-compose logs -f training
```

### Inference

```bash
# Batch inference
docker-compose up inference

# Check GPU usage
docker-compose exec training nvidia-smi
```

### TensorBoard

```bash
# Start TensorBoard
docker-compose up tensorboard

# Open in browser: http://localhost:6006
```

## Project Structure

```
scene-change-detection/
├── src/
│   ├── models/
│   │   ├── __init__.py
│   │   └── siamese_net.py          # Siamese U-Net model
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py              # Dataset class
│   │   └── transforms.py           # Data augmentation
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── metrics.py              # Evaluation metrics
│   │   └── visualize.py            # Visualization tools
│   ├── train.py                    # Training script
│   ├── inference.py                # Inference script
│   └── config.py                   # Configuration
├── data/
│   ├── raw/                        # Raw data
│   └── processed/                  # Processed data
├── checkpoints/                    # Model checkpoints
├── logs/                           # TensorBoard logs
├── results/                        # Inference results
├── tests/                          # Unit tests
├── notebooks/                      # Jupyter notebooks
├── Dockerfile                      # Docker configuration
├── docker-compose.yml              # Docker Compose configuration
├── requirements.txt                # Python dependencies
├── .gitignore                      # Git ignore rules
└── README.md                       # This file
```

## Model Architecture

### Siamese U-Net

The model consists of a Siamese encoder with shared weights and a U-Net decoder:

- Encoder: Processes both images in parallel
- Feature Extraction: Calculates difference features at each level
- Decoder: Upsampling and segmentation with U-Net structure
- Output: Binary change mask (256x256)

### Hyperparameters

- Input Size: 256x256
- Batch Size: 8-16
- Learning Rate: 1e-4
- Optimizer: Adam
- Loss Function: Binary Cross Entropy
- Scheduler: ReduceLROnPlateau

## Evaluation Metrics

Model performance is evaluated with the following metrics:

- F1-Score: Harmonic mean of Precision and Recall
- IoU (Intersection over Union): Jaccard Index
- Precision: Accuracy rate of positive predictions
- Recall: Detection rate of true positives
- Accuracy: Overall accuracy rate
- Kappa: Cohen's Kappa coefficient

## Results

### Sample Performance

```
Validation Set Results:
  F1 Score:    0.6114
  IoU:         0.4403
  Precision:   0.6523
  Recall:      0.5782
  Accuracy:    0.8856
```

### Visualization

Inference results consist of 6 panels:

1. Reference Image (A)
2. Query Image (B)
3. Change Probability (Heatmap)
4. Binary Change Mask
5. Detected Changes (overlay on A)
6. Detected Changes (overlay on B)

## Testing

```bash
# Run unit tests
pytest tests/

# Coverage report
pytest --cov=src tests/

# Specific test
pytest tests/test_model.py
```

## GPU Usage

### GPU Check

```bash
# Check NVIDIA driver
nvidia-smi

# Check PyTorch CUDA
python -c "import torch; print(torch.cuda.is_available())"
```

### GPU Memory Optimization

- Mixed Precision Training: Reduces memory usage by half
- Gradient Accumulation: Large effective batch size with small batch size
- Pin Memory: Increases GPU transfer speed in DataLoader

## Troubleshooting

### CUDA Out of Memory

```bash
# Reduce batch size
python src/train.py --batch_size 4

# Or use gradient accumulation
```

### Video Reading Error

```bash
# Install opencv-contrib-python
pip uninstall opencv-python -y
pip install opencv-contrib-python
```

### TensorBoard Not Opening

```bash
# Use different port if port conflict
tensorboard --logdir=logs --port=6007
```

## Development

### Adding New Features

1. Create feature branch: `git checkout -b feature/new-feature`
2. Make your changes
3. Add tests: `tests/test_new_feature.py`
4. Commit: `git commit -m "Add: New feature description"`
5. Push: `git push origin feature/new-feature`
6. Create pull request

### Code Standards

- Write PEP 8 compliant code
- Add docstrings (Google style)
- Use type hints
- Write unit tests

## License

This project is licensed under the MIT License.

## Contact

For questions:

- Email: aakcay5656@gmail.com
- GitHub Issues: https://github.com/aakcay5656/scene-change-detection/issues

## References

- PyTorch Documentation: https://pytorch.org/docs/
- Albumentations: https://albumentations.ai/
- U-Net Paper: https://arxiv.org/abs/1505.04597
- Siamese Networks: https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf

## Changelog

### v1.0.0 (2025-10-05)

- Initial release
- Siamese U-Net model implementation
- Docker support
- TensorBoard integration
- Batch and video inference support
- Comprehensive documentation

## Acknowledgments

- PyTorch community
- Open source library developers