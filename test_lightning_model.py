from utils.prepare_dataset import pipeline
from utils.lightning_model import RegressorModel
from utils.utils import plot_and_save_graph, save_results, get_least_utilized_gpu
from accelerate import Accelerator
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
import lightning.pytorch as pl
from finetuning_scheduler import FinetuningScheduler
import time

# Token Length: 867
config = {
    'model': 'bert-base-cased',
    'name': 'bert-dp-mp-nd-f-ss-increaselr2-regressiondropout-bn',
    'max_length': 512,
    'batch_size': 4,  # anything more results in CUDA OOM [for unfreezed encoder] on Kaggle GPU
    'epochs': 20,
    'lr': 0.001,   # prima era 3e-4 = 0.0003
    'enable_scheduler': True,
    'scheduler': 'CosineAnnealingWarmRestarts',
    'gradient_accumulation_steps': 16,
    'adam_eps': 1e-6,  # 1e-8 default
    'freeze_encoder': True
}

input_cols = ["text"]
target_cols = ["content", "wording"]
train_loader, valid_loader, test_loader, tokenizer, feature_dim = pipeline(config, input_cols=input_cols, target_cols=target_cols,
                                                              dynamic_padding=True, split=0.2)

device = get_least_utilized_gpu()
model = RegressorModel.load_from_checkpoint("./lightning_logs/version_1/checkpoints/epoch=161-step=11664.ckpt", config=config, features_dim=feature_dim)
model.eval()
# predict with the model
y_hat = model(valid_loader)
print(y_hat)