from utils.prepare_dataset import pipeline
from utils.create_model import RegressorModel, Trainer
from utils.utils import plot_and_save_graph, save_results
from accelerate import Accelerator
import time

config = {
    'model': 'microsoft/deberta-v3-base',
    'name': 'DeBERTa-newdataset',
    'max_length': 868,  # "text" field length at max is 867
    'batch_size': 4,  # anything more results in CUDA OOM [for unfreezed encoder] on Kaggle GPU
    'epochs': 20,
    'lr': 3e-4,
    'enable_scheduler': True,
    'scheduler': 'CosineAnnealingWarmRestarts',
    'gradient_accumulation_steps': 16,
    'adam_eps': 1e-6,  # 1e-8 default
    'freeze_encoder': True
}

input_cols = ["fixed_summary_text"]
target_cols = ["content", "wording"]
features = ["length_ratio", "normalized_text_length", "karp_tfidf_scores",
            "normalized_text_misspelled_counter", "normalized_corrected_misspelled_counter", "normalized_2grams-cooccurrence-count",
            "normalized_2grams-correct-count", "normalized_3grams-correct-count", "normalized_4grams-correct-count",
            "normalized_3grams-cooccurrence-count", "normalized_4grams-cooccurrence-count",
            "semantic_similarity"]
feature_dim = len(features)
train_loader, valid_loader, test_loader, tokenizer = pipeline(config, input_cols=input_cols, target_cols=target_cols,
                                                                           features=features, dynamic_padding=True, split=0.2)

accelerator = Accelerator(gradient_accumulation_steps=config['gradient_accumulation_steps'])

model = RegressorModel(config, pooling='mean-pooling', features_dim=feature_dim).to(device=accelerator.device)

trainer = Trainer(model, (train_loader, valid_loader), config, accelerator)

start_time = time.time()
train_losses, val_losses = trainer.fit()
elapsed_time = time.time() - start_time

plot_and_save_graph(config["epochs"], config["name"], train_losses, val_losses)

save_results(config["name"], train_losses, val_losses, elapsed_time)
