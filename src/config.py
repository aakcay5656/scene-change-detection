class Config:
    """Eğitim konfigürasyonu"""

    # Data
    DATA_DIR = 'data/processed'
    TRANSFORM_METHOD = 'albumentations'  # 'albumentations', 'paired', 'minimal'

    # Training
    BATCH_SIZE = 8
    NUM_EPOCHS = 30
    LEARNING_RATE = 1e-4
    NUM_WORKERS = 4

    # Model
    IN_CHANNELS = 3
    OUT_CHANNELS = 1

    # Paths
    CHECKPOINT_DIR = 'checkpoints'
    LOG_DIR = 'logs'

    # Scheduler
    SCHEDULER_PATIENCE = 5
    SCHEDULER_FACTOR = 0.5

    # Device
    USE_CUDA = True