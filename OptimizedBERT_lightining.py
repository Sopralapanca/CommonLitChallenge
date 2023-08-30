from utils.prepare_dataset import pipeline
from utils.lightning_model import RegressorModel, MetricsCallback
from utils.utils import plot_and_save_graph, save_results, get_least_utilized_gpu
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
import time
import lightning.pytorch as pl
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Token Length: 867
config = {
    'model': 'bert-base-cased',
    'name': 'bert-lightining-100epochs',
    'max_length': 512,
    'batch_size': 4,  # anything more results in CUDA OOM [for unfreezed encoder] on Kaggle GPU
    'epochs': 100,
    'lr': 0.0005,  # prima era 3e-4 = 0.0003
    'enable_scheduler': True,
    'scheduler': 'CosineAnnealingWarmRestarts',
    'gradient_accumulation_steps': 16,
    'adam_eps': 1e-6,  # 1e-8 default
    'freeze_encoder': True
}

input_cols = ["text"]
target_cols = ["content", "wording"]
features = ["length_ratio", "normalized_text_length", "karp_tfidf_scores",
            "normalized_text_misspelled_counter", "normalized_corrected_misspelled_counter",
            "normalized_2grams-cooccurrence-count",
            "normalized_2grams-correct-count", "normalized_3grams-correct-count", "normalized_4grams-correct-count",
            "normalized_3grams-cooccurrence-count", "normalized_4grams-cooccurrence-count",
            "semantic_similarity"]
feature_dim = len(features)

train_loader, valid_loader, test_loader, tokenizer = pipeline(config, input_cols=input_cols, features=features, oversample=True,
                                                              target_cols=target_cols, dynamic_padding=True, split=0.2)

device = get_least_utilized_gpu()
model = RegressorModel(config, pooling='mean-pooling', features_dim=feature_dim).to(device=device)
metrics = MetricsCallback()
trainer = pl.Trainer(callbacks=[EarlyStopping(monitor="val_loss", mode="min", patience=5),
                                metrics],
                     accumulate_grad_batches=config['gradient_accumulation_steps'],
                     devices=1,
                     max_epochs=config["epochs"],
                     enable_checkpointing=False,
                     )

start_time = time.time()
trainer.fit(model, train_loader, valid_loader)
elapsed_time = time.time() - start_time

train_loss = metrics.metrics["train_loss"]
val_loss = metrics.metrics["valid_loss"][1:]  # remove the first value since it is a sanity check that has been performed

plot_and_save_graph(len(train_loss), config["name"], train_loss, val_loss)
save_results(config["name"], train_loss, val_loss, elapsed_time)
