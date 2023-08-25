CONFIG = {
    'model': 'microsoft/deberta-v3-base',
    'dropout': 0.5,
    'max_length': 512,
    'batch_size': 2,  # anything more results in CUDA OOM [for unfreezed encoder] on Kaggle GPU
    'epochs': 1,
    'lr': 3e-4,
    'enable_scheduler': True,
    'scheduler': 'CosineAnnealingWarmRestarts',
    'gradient_accumulation_steps': 16,
    'adam_eps': 1e-6,  # 1e-8 default
    'freeze_encoder': True,
    'keys': ['prompt_question', 'text'],
    'folds': 5
}