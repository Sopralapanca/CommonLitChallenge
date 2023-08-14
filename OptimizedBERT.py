from utils.prepare_dataset import pipeline
from utils.create_model import RegressorModel, Trainer
from utils.utils import plot_and_save_graph, save_results
from accelerate import Accelerator
import torch
import time

# Token Length more than maxlen: 867

config = {
    'model': 'bert-base-cased',
    'dropout': 0.5,
    'max_length': 512,
    'batch_size': 8,  # anything more results in CUDA OOM [for unfreezed encoder] on Kaggle GPU
    'epochs': 10,
    'lr': 3e-4,
    'enable_scheduler': True,
    'scheduler': 'CosineAnnealingWarmRestarts',
    'gradient_accumulation_steps': 2,
    'adam_eps': 1e-6,  # 1e-8 default
    'freeze_encoder': True
}

train_loader, valid_loader, test_loader, tokenizer = pipeline(config, split=0.2)

accelerator = Accelerator(gradient_accumulation_steps=config['gradient_accumulation_steps'])

model = RegressorModel(config).to(device=accelerator.device)
trainer = Trainer(model, (train_loader, valid_loader), config, accelerator)

start_time = time.time()
train_losses, val_losses = trainer.fit()
elapsed_time = time.time() - start_time

plot_and_save_graph(config["epochs"], "BERT", train_losses, val_losses)

save_results("BERT", train_losses, val_losses, elapsed_time)
