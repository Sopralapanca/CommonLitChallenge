from utils.prepare_dataset import pipeline
from utils.create_model import RegressorModel, Trainer
from accelerate import Accelerator
from utils.utils import plot_and_save_graph
import time
import os
import numpy as np
import itertools as it
import csv
import random
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def model_selection(config, model_name, max_epochs, features_dim, target_cols):
    keys = config.keys()
    combinations = it.product(*(config[key] for key in keys))
    param_list = list(combinations)

    # Specify the number of random combinations you want to select
    num_random_combinations = 30  # You can change this to the desired number

    # Select random combinations without replacement
    random_combinations = random.sample(param_list, num_random_combinations)


    for i, elem in enumerate(random_combinations):
        results_row = {}
        for k, v in zip(keys, elem):
            results_row[k] = v

        accelerator = Accelerator(gradient_accumulation_steps=16)

        model = RegressorModel(
            name=model_name, fflayers=results_row["fflayers"], ffdropout=results_row["ffdropout"],
            activation_function=results_row["activation_function"], features_dim=features_dim,
            target_cols=target_cols
        ).to(device=accelerator.device)

        trainer = Trainer(model, (train_loader, valid_loader), max_epochs, accelerator, results_row["lr"])

        start_time = time.time()
        train_losses, val_losses = trainer.fit(verbose=False)
        elapsed_time = time.time() - start_time

        plot_and_save_graph(len(train_losses["loss"]), f"model ID: {i+20}",
                            train_losses["loss"], val_losses["loss"])

        results_row['id'] = i+20
        results_row['model-name'] = model_name
        results_row['train-loss'] = train_losses["loss"][-1]
        results_row['valid-loss'] = val_losses["loss"][-1]
        results_row['train-content-loss'] = train_losses["content"][-1].item()
        results_row['valid-content-loss'] = val_losses["content"][-1].item()
        results_row['train-wording-loss'] = train_losses["wording"][-1].item()
        results_row['valid-wording-loss'] = val_losses["wording"][-1].item()
        results_row['elapsed-time'] = elapsed_time
        results_row['epochs'] = len(train_losses["loss"])

        csv_filename = 'model_selection_results.csv'

        with open(csv_filename, 'a', newline='') as csvfile:
            fieldnames = results_row.keys()
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            # If the file is empty, write the header row
            if csvfile.tell() == 0:
                writer.writeheader()

            # Write the training information to the CSV file
            writer.writerow(results_row)

    print("model selection completed!")


# Token Length: 867

config = {
    'model': 'bert-base-cased',
    'max_length': 512,
    'batch_size': 4,  # anything more results in CUDA OOM [for unfreezed encoder] on Kaggle GPU
}

features = ["length_ratio", "normalized_text_length", "karp_tfidf_scores",
            "normalized_text_misspelled_counter", "normalized_corrected_misspelled_counter",
            "normalized_2grams-cooccurrence-count",
            "normalized_2grams-correct-count", "normalized_3grams-correct-count", "normalized_4grams-correct-count",
            "normalized_3grams-cooccurrence-count", "normalized_4grams-cooccurrence-count",
            "semantic_similarity"]
feature_dim = len(features)
multioutput = True

input_cols = ["fixed_summary_text"]
target_cols = ["content", "wording"]

train_loader, valid_loader, test_loader, tokenizer = pipeline(config, input_cols=input_cols, target_cols=target_cols,
                                                              features=features, dynamic_padding=True, split=0.2,
                                                              oversample=True)

model_name = 'bert-base-cased'
max_epochs = 100

model_selection_config = {
    'lr': np.arange(1e-4, 5e-3, 0.0005).tolist(),
    'fflayers': [1, 2, 3, 4],
    'ffdropout': [0.2, 0.3, 0.4, 0.5],
    'activation_function': ["relu", "leaky-relu"],
}
model_selection(model_selection_config, model_name, max_epochs, features_dim=feature_dim, target_cols=target_cols)
