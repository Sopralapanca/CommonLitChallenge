# -*- coding: utf-8 -*-
import more_itertools
import pandas as pd
import numpy as np
import itertools
import json
import copy
import gc
import os

import torch
import torch.nn as nn
from torch.utils.data import Sampler, Dataset, DataLoader

from transformers import AutoTokenizer, AutoModel, AutoConfig
from accelerate import Accelerator

from trainer_utils import oversample_df, SmartBatchingDataset, RegressorModel, Trainer

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="3"
os.environ["TOKENIZERS_PARALLELISM"]="true"
train_path = "./CommonLitChallenge/data/dataset_bf.zip"
train_data = pd.read_csv(train_path, compression='zip', sep=',', index_col=0)

config = {
    'model': 'bert-base-cased', # name of the model to be downloaded with huggingsface transformers
    'name': 'bert cased', # name of the model to be displayed on the results
    'max_length': 512,
    'batch_size': 8,  # si può provare ad aumentare ma occhio agli out of memory error
    'epochs': 40,
    'lr': 7e-4,
    'gradient_accumulation_steps': 16,
    'adam_eps': 1e-6,  # 1e-8 default
    'freeze_encoder': True,
    'fflayers': 3,
    'ffdropout': 0.05,
    'activation_function': 'relu',
    'weight_decay': 0.001
}
accelerator = Accelerator(gradient_accumulation_steps=config['gradient_accumulation_steps'])
tokenizer = AutoTokenizer.from_pretrained(config['model'])

features = [   
    'text_word_cnt', 
    'text_length', 
    'text_stopword_cnt', 
    'text_punct_cnt', 
    'text_different_word_cnt', 
    'text_misspelled_cnt', 
    'text_word_ratio', 
    'text_length_ratio', 
    'text_stopword_ratio', 
    'text_punct_ratio', 
    'text_different_word_ratio', 
    'karp_tfidf_scores', 
    '2grams_cnt', 
    '3grams_cnt', 
    '4grams_cnt']
feature_dim = len(features)

checkpoint_threshold = 1
best_validation = 1

multioutput = True
target_cols = ["content", "wording"]
input_cols = ['prompt_title', 'text']
columns_used = features + input_cols + target_cols

restricted_train_data = train_data[columns_used].copy()
restricted_train_data.reset_index(inplace=True)
prompt_ids = restricted_train_data['prompt_id'].unique()
n_folds = len(restricted_train_data['prompt_id'].unique())
del train_data
gc.collect();

f = open('features_log.txt', 'a')

while feature_dim > 4:
  ablation_info = []
  # Model Construction pooling options == combination , mean-pooling
  model = RegressorModel(name=config["model"], fflayers=config["fflayers"], pooling='combination',
                  ffdropout=config["ffdropout"],activation_function=config["activation_function"],
                  features_dim=feature_dim-1, target_cols=target_cols).to(device=accelerator.device)
  for i in range(0, feature_dim):
    abl_features = features.copy()
    removed_feat = abl_features.pop(i)
    ablation_dim = len(abl_features)
    folds_info = []

    info = f"TESTING WITHOUT: {removed_feat}\nACTUAL FEATURE DIMENSION: {ablation_dim}\n-----------\n"
    f.write(info)
    print(info)

    for fold, prompt_id in enumerate(prompt_ids):
      info = f"\nFOLD: {fold + 1}\n-----------\n"
      f.write(info)
      print(info)
      # Resetting weigths
      model.reset_weights()

      # Selection of the training set and oversample
      validation_df = restricted_train_data.loc[restricted_train_data.prompt_id == prompt_id]
      training_df = restricted_train_data.loc[restricted_train_data.prompt_id != prompt_id]
      # training_df = oversample_df(training_df)

      train_set = SmartBatchingDataset(training_df, tokenizer, input_cols, target_cols, features_cols=abl_features)
      valid_set = SmartBatchingDataset(validation_df, tokenizer, input_cols, target_cols, features_cols=abl_features)
      train_loader = train_set.get_dataloader(batch_size=config['batch_size'], max_len=config["max_length"],
                                              pad_id=tokenizer.pad_token_id)
      valid_loader = valid_set.get_dataloader(batch_size=config['batch_size'], max_len=config["max_length"],
                                              pad_id=tokenizer.pad_token_id)

      trainer = Trainer(model, (train_loader, valid_loader), epochs=config['epochs'], accelerator=accelerator,
                        lr=config["lr"], weight_decay=config["weight_decay"])
      train_losses, val_losses = trainer.fit(multioutput)
      
      del trainer, training_df, validation_df, train_set, valid_set, train_loader, valid_loader
      gc.collect();

      if checkpoint_threshold < min(val_losses["loss"]):
        checkpoint_threshold = min(val_losses["loss"])

      fold_info = {
            'train-loss': train_losses["loss"][-1],
            'valid-loss': val_losses["loss"][-1],
            'valid-content-loss': val_losses["content"][-1].item(),
            'valid-wording-loss': val_losses["wording"][-1].item(),
            'best-valid-loss': min(val_losses["loss"]),
        }
      folds_info.append(fold_info)
      print(fold_info)
      f.write(json.dumps(fold_info))

    abl_info = {
        'feature_set': abl_features,
        'removed_feature': removed_feat,
        'train-loss': np.sum([info['train-loss'] for info in folds_info])/n_folds,
        'valid-loss': np.sum([info['valid-loss'] for info in folds_info])/n_folds,
        'valid-content-loss': np.sum([info['valid-content-loss'] for info in folds_info])/n_folds,
        'valid-wording-loss': np.sum([info['valid-wording-loss'] for info in folds_info])/n_folds,
        'best-valid-loss': np.sum([info['best-valid-loss'] for info in folds_info])/n_folds,
    }
    print(abl_info)
    ablation_info.append(abl_info)
    f.write(json.dumps(abl_info))

  actual_combination = min(ablation_info, key=lambda x:x['best-valid-loss'])
  if actual_combination['best-valid-loss'] < best_validation:
    features = actual_combination['feature_set'].copy()
    info = f'\nReached a better score {actual_combination["best-valid-loss"]} (actual) VS {best_validation} (old)\nRemoving from the experiment: {actual_combination["removed_feature"]}\nActual feature set: {features}\n-----------\n'
    f.write(info)
    print(info)
    best_validation = actual_combination['best-valid-loss']
    feature_dim = len(features)
  else:
    info = f'\nDid not reached a better score {best_validation} (old) VS {actual_combination["best-valid-loss"]} (actual)\nThe best feature set is still {features}\nExiting...\n'
    f.write(info)
    print(info)
    break
f.close()