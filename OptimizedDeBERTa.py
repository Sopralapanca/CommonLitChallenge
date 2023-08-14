from utils.prepare_dataset import pipeline
from utils.create_model import RegressorModel, Trainer
from accelerate import Accelerator
import torch

# Token Length more than maxlen: 867

config = {
    'model': 'microsoft/deberta-v3-base',
    'dropout': 0.5,
    'max_length': 1024,
    'batch_size': 4,  # anything more results in CUDA OOM [for unfreezed encoder] on Kaggle GPU
    'epochs': 1,
    'lr': 3e-4,
    'enable_scheduler': True,
    'scheduler': 'CosineAnnealingWarmRestarts',
    'gradient_accumulation_steps': 16,
    'adam_eps': 1e-6,  # 1e-8 default
    'freeze_encoder': True
}

train_loader, valid_loader, test_loader, tokenizer = pipeline(config, split=0.2)

accelerator = Accelerator(gradient_accumulation_steps=config['gradient_accumulation_steps'])

model = RegressorModel(config).to(device=accelerator.device)
trainer = Trainer(model, (train_loader, valid_loader), config, accelerator)
trainer.fit()