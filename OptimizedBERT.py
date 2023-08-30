from utils.prepare_dataset import pipeline
from utils.create_model import RegressorModel, Trainer
from utils.utils import plot_and_save_graph, save_results
from accelerate import Accelerator
import time
import os


os.environ["TOKENIZERS_PARALLELISM"] = "false"


# Token Length: 867

config = {
    'model': 'bert-base-cased',
    'name': 'bert-2models-stratified-oversampled',
    'max_length': 512,
    'batch_size': 4,  # anything more results in CUDA OOM [for unfreezed encoder] on Kaggle GPU
    'epochs': 20,
    'lr': 0.0005,   # prima era 3e-4 = 0.0003
    'enable_scheduler': True,
    'scheduler': 'CosineAnnealingWarmRestarts',
    'gradient_accumulation_steps': 16,
    'adam_eps': 1e-6,  # 1e-8 default
    'freeze_encoder': True
}

features = ["length_ratio", "normalized_text_length", "karp_tfidf_scores",
            "normalized_text_misspelled_counter", "normalized_corrected_misspelled_counter", "normalized_2grams-cooccurrence-count",
            "normalized_2grams-correct-count", "normalized_3grams-correct-count", "normalized_4grams-correct-count",
            "normalized_3grams-cooccurrence-count", "normalized_4grams-cooccurrence-count",
            "semantic_similarity"]
feature_dim = len(features)



#ic = [["fixed_summary_text"],["text"]]
#tc = [["content"],["wording"]]
multioutput = True

#train_results = []
#valid_results = []
#total_time = 0

#for input_cols, target_cols in zip(ic, tc):
input_cols = ["fixed_summary_text"]
target_cols = ["content","wording"]

train_loader, valid_loader, test_loader, tokenizer = pipeline(config, input_cols=input_cols, target_cols=target_cols,
                                                            features=features, dynamic_padding=True, split=0.2, oversample=True)
accelerator = Accelerator(gradient_accumulation_steps=config['gradient_accumulation_steps'])

model = RegressorModel(config, pooling='mean-pooling', features_dim=feature_dim, target_cols=target_cols).to(device=accelerator.device)

trainer = Trainer(model, (train_loader, valid_loader), config, accelerator)

start_time = time.time()
train_losses, val_losses = trainer.fit(multioutput)
elapsed_time = time.time() - start_time
#total_time += elapsed_time

plot_and_save_graph(config["epochs"], config["name"], train_losses["loss"], val_losses["loss"])

save_results(config["name"], train_losses, val_losses, elapsed_time)

#train_results.append(train_losses["loss"][-1])
#valid_results.append(val_losses["loss"][-1])


"""training_info = {
        'model-name': config["name"],
        'train-loss': sum(train_results) / len(train_results),
        'valid-loss': sum(valid_results) / len(valid_results),
        'train-content-loss': 0,
        'valid-content-loss': 0,
        'train-wording-loss': 0,
        'valid-wording-loss': 0,
        'elapsed-time': total_time,
    }

csv_filename = 'training_info2.csv'

with open(csv_filename, 'a', newline='') as csvfile:
    fieldnames = training_info.keys()
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    # If the file is empty, write the header row
    if csvfile.tell() == 0:
        writer.writeheader()

    # Write the training information to the CSV file
    writer.writerow(training_info)"""