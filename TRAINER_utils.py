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

class SmartBatchingDataset(Dataset):
    def __init__(self, df, tokenizer, input_cols, target_cols, features_cols):
        super(SmartBatchingDataset, self).__init__()

        input_df = copy.deepcopy(df[input_cols])
        features_df = copy.deepcopy(df[features_cols])

        # Combine strings from multiple columns with [CLS], [SEP], and [SEP] separators
        input_df['combined_col'] = input_df.apply(
            lambda row: tokenizer.cls_token + ' ' + f' {tokenizer.sep_token} '.join(row) + f' {tokenizer.sep_token}',
            axis=1)

        self._data = [
            input_df['combined_col'].apply(tokenizer.tokenize).apply(tokenizer.convert_tokens_to_ids).to_list(),
            features_df.values.tolist()]
        self._targets = df[target_cols].values.tolist() if target_cols is not None else None
        self.sampler = None

    def __len__(self):
        return len(self._data)

    def __getitem__(self, item):
        if self._targets is not None:
            return [self._data[0][item], self._data[1][item]], self._targets[item]
        else:
            return [self._data[0][item], self._data[1][item]]

    def get_dataloader(self, batch_size, max_len, pad_id):

        self.sampler = SmartBatchingSampler(
            data_source=self._data[0],
            batch_size=batch_size
        )
        collate_fn = SmartBatchingCollate(
            targets=self._targets,
            max_length=max_len,
            pad_token_id=pad_id
        )
        dataloader = DataLoader(
            dataset=self,
            batch_size=batch_size,
            sampler=self.sampler,
            collate_fn=collate_fn,
            pin_memory=True,
            num_workers=1 # calcolare grafico con i tempi impiegati per capire qual è il numero di worker migliore da utilizzare
        )
        return dataloader


class SmartBatchingSampler(Sampler): #aggiungere al loader la possibilità di selezionare l'indice del fold
    def __init__(self, data_source, batch_size):
        super(SmartBatchingSampler, self).__init__(data_source)
        self.len = len(data_source)
        sample_lengths = [len(seq) for seq in data_source]
        argsort_inds = np.argsort(sample_lengths)
        self.batches = list(more_itertools.chunked(argsort_inds, n=batch_size))
        self._backsort_inds = None

    def __iter__(self):
        if self.batches:
            last_batch = self.batches.pop(-1)
            np.random.shuffle(self.batches)
            self.batches.append(last_batch)
        self._inds = list(more_itertools.flatten(self.batches))
        yield from self._inds

    def __len__(self):
        return self.len

    @property
    def backsort_inds(self):
        if self._backsort_inds is None:
            self._backsort_inds = np.argsort(self._inds)
        return self._backsort_inds

class SmartBatchingCollate:
    def __init__(self, targets, max_length, pad_token_id):
        self._targets = targets
        self._max_length = max_length
        self._pad_token_id = pad_token_id

    def __call__(self, batch):
        if self._targets is not None:
            sequences, targets = list(zip(*batch))
        else:
            sequences = list(batch)

        ids = []
        features = []
        for i in range(len(sequences)):
            ids.append(sequences[i][0])
            features.append(sequences[i][1])

        input_ids, attention_mask = self.pad_sequence(
            ids,
            max_sequence_length=self._max_length,
            pad_token_id=self._pad_token_id
        )

        if self._targets is not None:
            output = [input_ids, features], attention_mask, torch.tensor(targets)
        else:
            output = [input_ids, features], attention_mask
        return output

    def pad_sequence(self, sequence_batch, max_sequence_length, pad_token_id):
        max_batch_len = max(len(sequence) for sequence in sequence_batch)
        max_len = min(max_batch_len, max_sequence_length)
        padded_sequences, attention_masks = [[] for i in range(2)]
        attend, no_attend = 1, 0
        for sequence in sequence_batch:



            # As discussed above, truncate if exceeds max_len
            new_sequence = list(sequence[:max_len])


            attention_mask = [attend] * len(new_sequence)
            pad_length = max_len - len(new_sequence)

            new_sequence.extend([pad_token_id] * pad_length)
            attention_mask.extend([no_attend] * pad_length)

            padded_sequences.append(new_sequence)
            attention_masks.append(attention_mask)

        padded_sequences = torch.tensor(padded_sequences)
        attention_masks = torch.tensor(attention_masks)

        return padded_sequences, attention_masks

"""# Model Definition"""
class RegressorModel(nn.Module):
    def __init__(self, name, fflayers, ffdropout,
                 features_dim, target_cols, activation_function,
                 freeze_encoder=True,pooling="mean-pooling", dropoutLLM=False):
        super(RegressorModel, self).__init__()

        """
        :param name:                strings, model name to be downloaded with huggingsface transformers
        :param fflayers:            int, number of layers of the feedforward network
        :param ffdropout:           float, percentage of dropout between LLM embeddings and feedforward net
        :param features_dim:        int, dimension of the feature extracted
        :param target_cols:         list of strings, columns of the dataframe that represents the target
        :param activation_function: string, activation function to use
        :param freeze_encoder:      boolean, if True the LLM is freezed and will be not trained
        :param pooling:             strings, pooling method to user
        :param dropoutLLM:          boolean, if False dropout percentage of LLM will be setted to 0
        """


        self.model_name = name
        self.pooling = pooling
        self.model_config = AutoConfig.from_pretrained(self.model_name)
        if pooling == 'combination':
            self.model_config.update({'output_hidden_states':True})
        self.target_cols = target_cols
        self.drop = nn.Dropout(p=ffdropout)
        self.fflayers = fflayers


        if not dropoutLLM:
            self.model_config.hidden_dropout_prob = 0.0
            self.model_config.attention_probs_dropout_prob = 0.0

        self.encoder = AutoModel.from_pretrained(f"{name}", config=self.model_config)

        if freeze_encoder:
            for param in self.encoder.base_model.parameters():
                param.requires_grad = False
        
        size = self.encoder.config.hidden_size + features_dim
        if self.pooling == 'combination':
          size = self.encoder.config.hidden_size*4 + features_dim
        # The output layer that takes the last hidden layer of the BERT model
        self.cls_layer1 = nn.Linear(size, 2*size)

        self.ff_hidden_layers = nn.ModuleList()

        size = 2*size
        for _ in range(self.fflayers):
            out_size = int(size/2)
            self.ff_hidden_layers.append(nn.Linear(size, out_size))
            size = out_size

        self.act = None
        if activation_function == 'relu':
            self.act = nn.ReLU()

        if activation_function == 'leaky-relu':
            self.act = nn.LeakyReLU()

        # last layer
        self.output_layer = nn.Linear(size, len(self.target_cols))

    def reset_weights(self):
      self.cls_layer1.reset_parameters()
      for layer in self.ff_hidden_layers:
        layer.reset_parameters()
      self.output_layer.reset_parameters()

    def forward(self, inputs, attention_mask):

        input_ids = inputs[0]
        features = inputs[1]

        features = torch.tensor(features).float().to(input_ids.device)

        if self.pooling == 'cls':
          with torch.no_grad():
            outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask,
                                output_hidden_states=False)
          # Obtain the representations of [CLS] heads
          logits = outputs.last_hidden_state[:, 0, :]

        # Feed the input to Bert model to obtain contextualized representations
        if self.pooling == 'combination': 
          with torch.no_grad():
            outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
          all_hidden_states = torch.stack(outputs[2])
          input_mask_expanded = attention_mask.unsqueeze(-1).expand(all_hidden_states[-1].size()).float()
          sum_embeddings = torch.sum(all_hidden_states[-1] * input_mask_expanded, 1)
          sum_mask = input_mask_expanded.sum(1)
          sum_mask = torch.clamp(sum_mask, min=1e-9)
          logits_1 = sum_embeddings / sum_mask
          input_mask_expanded = attention_mask.unsqueeze(-1).expand(all_hidden_states[-2].size()).float()
          sum_embeddings = torch.sum(all_hidden_states[-2] * input_mask_expanded, 1)
          sum_mask = input_mask_expanded.sum(1)
          sum_mask = torch.clamp(sum_mask, min=1e-9)
          logits_2 = sum_embeddings / sum_mask
          input_mask_expanded = attention_mask.unsqueeze(-1).expand(all_hidden_states[-3].size()).float()
          sum_embeddings = torch.sum(all_hidden_states[-3] * input_mask_expanded, 1)
          sum_mask = input_mask_expanded.sum(1)
          sum_mask = torch.clamp(sum_mask, min=1e-9)
          logits_3 = sum_embeddings / sum_mask
          input_mask_expanded = attention_mask.unsqueeze(-1).expand(all_hidden_states[-4].size()).float()
          sum_embeddings = torch.sum(all_hidden_states[-4] * input_mask_expanded, 1)
          sum_mask = input_mask_expanded.sum(1)
          sum_mask = torch.clamp(sum_mask, min=1e-9)
          logits_4 = sum_embeddings / sum_mask
          logits = torch.cat((logits_1, logits_2, logits_3, logits_4),-1)

        #   logits  = concatenate_pooling[:, 0]

        if self.pooling == 'mean-pooling':
          with torch.no_grad():
            outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask,
                                  output_hidden_states=False)
          last_hidden_state = outputs.last_hidden_state
          input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
          sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
          sum_mask = input_mask_expanded.sum(1)
          sum_mask = torch.clamp(sum_mask, min=1e-9)
          logits = sum_embeddings / sum_mask


        combined_features = torch.cat((logits, features), dim=1) # combine bert embeddings with features extracted from the dataset

        output = self.drop(combined_features)
        output = self.act(self.cls_layer1(output))


        for hl in self.ff_hidden_layers:
            output = self.act(hl(output))

        output = self.output_layer(output)
        return output
    
"""# Training Loop"""
class EarlyStopper:
    def __init__(self, checkpoint_path="", patience=5, min_delta=0.003, general_min_validation=np.inf):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf
        self.checkpoint_path = checkpoint_path
        self.general_min_validation = general_min_validation

    def early_stop(self, validation_loss, epoch, optimizer, model):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
            name = f"checkpoint.pt"
            full_path = self.checkpoint_path + name
            print('New best local validation loss')

            if validation_loss < self.general_min_validation:
                self.general_min_validation = validation_loss
                self.counter = 0
                name = f"checkpoint.pt"
                full_path =  self.checkpoint_path + name

                print(f"New best validation loss, saving checkpoint at {full_path}")

                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': validation_loss,
                }, full_path)
              
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

class Trainer:
    def __init__(self, model, loaders, epochs, accelerator, lr, weight_decay):

        """
        :param model:       PyTorch model to train
        :param loaders:     tuple of DataLoaders, (train loader, valid loader)
        :param epochs:      int, max epochs to train a model
        :param accelerator: PyTorch device, device for gradient step accumulation
        :param lr:          float, learning rate
        """

        self.model = model
        self.train_loader, self.val_loader = loaders
        self.weight_decay = weight_decay


        self.epochs = epochs

        self.accelerator = accelerator

        self.lr = lr

        self.optim = self._get_optim()

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optim, T_0=5, eta_min=1e-7)

        self.train_losses = {
            "loss": [],
            "content": [],
            "wording": []
        }
        self.val_losses = {
            "loss": [],
            "content": [],
            "wording": []
        }

    def prepare(self):
        self.model, self.optim, self.train_loader, self.val_loader, self.scheduler = self.accelerator.prepare(
            self.model,
            self.optim,
            self.train_loader,
            self.val_loader,
            self.scheduler
        )

    def _get_optim(self):
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': self.weight_decay},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.lr, eps=1e-6)
        return optimizer

    def loss_fn(self, outputs, targets, multioutput=True):
        colwise_rmse = torch.sqrt(torch.mean((targets - outputs) ** 2, dim=0))
        if multioutput:
            content_loss = colwise_rmse[0]
            wording_loss = colwise_rmse[1]
        else:
            content_loss = 0.
            wording_loss = 0.

        loss = torch.mean(colwise_rmse, dim=0)

        return loss, content_loss, wording_loss

    def train_one_epoch(self, epoch, multioutput, outer_progress="", verbose=True):

        running_loss = 0.
        c_loss = 0.
        w_loss = 0.

        inner_iterations = len(self.train_loader)

        idx = 0
        for el in self.train_loader:
            input_ids, attention_mask, target = el

            with self.accelerator.accumulate(self.model):
                output = self.model(inputs=input_ids, attention_mask=attention_mask)

                loss, content_loss, wording_loss = self.loss_fn(output, target, multioutput)
                running_loss += loss.item()
                c_loss += content_loss
                w_loss += wording_loss

                self.accelerator.backward(loss)

                self.optim.step()

                self.scheduler.step(epoch - 1 + idx / len(self.train_loader))

                self.optim.zero_grad()

                del input_ids, attention_mask, target, loss

                if verbose:
                    inner_progress = f"Training Batch: [{idx}/{inner_iterations}]"
                    print(f"\r{outer_progress} {inner_progress}", end="")

            idx += 1

        train_loss = running_loss / len(self.train_loader)
        c_loss = c_loss / len(self.train_loader)
        w_loss = w_loss / len(self.train_loader)
        self.train_losses["loss"].append(train_loss)
        self.train_losses["content"].append(c_loss)
        self.train_losses["wording"].append(w_loss)

        if verbose:
            inner_progress = f"Training Batch:[{idx}/{inner_iterations}] training loss: {self.train_losses['loss'][-1]}"
            print(f"\r{outer_progress} {inner_progress}", end="")

    @torch.no_grad()
    def valid_one_epoch(self, epoch, multioutput, outer_progress="", verbose=True):

        running_loss = 0.
        c_loss = 0.
        w_loss = 0.

        inner_iterations = len(self.val_loader)

        idx = 0
        for input_ids, attention_mask, target in self.val_loader:
            output = self.model(inputs=input_ids, attention_mask=attention_mask)

            loss, content_loss, wording_loss = self.loss_fn(output, target, multioutput)
            running_loss += loss.item()
            c_loss += content_loss
            w_loss += wording_loss

            del input_ids, attention_mask, target, loss

            if verbose:
                inner_progress = f"Validation Batch: [{idx}/{inner_iterations}] training loss: {self.train_losses['loss'][-1]}"
                print(f"\r{outer_progress} {inner_progress}", end="")

            idx += 1

        val_loss = running_loss / len(self.val_loader)
        c_loss = c_loss / len(self.val_loader)
        w_loss = w_loss / len(self.val_loader)
        self.val_losses["loss"].append(val_loss)
        self.val_losses["content"].append(c_loss)
        self.val_losses["wording"].append(w_loss)

        if verbose:
            inner_progress = f"Validation Batch:[{idx}/{inner_iterations}] training loss: {self.train_losses['loss'][-1]} validation loss: {self.val_losses['loss'][-1]}"
            print(f"\r{outer_progress} {inner_progress}", end="")

    def fit(self, multioutput=True, min_delta=0.003, verbose=True, best_validation=np.inf):

        self.prepare()

        # Define the number of iterations for both loops
        outer_iterations = self.epochs

        early_stopper = EarlyStopper(patience=5, min_delta=min_delta, general_min_validation=best_validation)
        outer_progress = ""

        # Create outer progress bar
        for epoch in range(1, outer_iterations + 1):
            if verbose:
                outer_progress = f"Epoch: {epoch}/{outer_iterations}"

            self.model.train()

            self.train_one_epoch(epoch, multioutput, outer_progress=outer_progress, verbose=verbose)
            self.clear()

            self.model.eval()
            self.valid_one_epoch(epoch, multioutput, outer_progress=outer_progress, verbose=verbose)

            self.clear()

            if early_stopper.early_stop(validation_loss=self.val_losses["loss"][-1],
                                        epoch=epoch, optimizer=self.optim, model=self.model):
                break

            if verbose:
                print()

        if verbose:
            print("\nTraining completed!")

        return self.train_losses, self.val_losses

    def clear(self):
        gc.collect()
        torch.cuda.empty_cache()

def oversample_df(df):
  """
  :param df: Dataframe to be overampled based on prompt_id
  :return: Dataframe oversampled
  """
  classes = df["prompt_id"].value_counts().to_dict()
  most = max(classes.values())
  classes_list = []
  for key in classes:
    classes_list.append(df[df["prompt_id"] == key])
  classes_sample = []
  for i in range(1, len(classes_list)):
    classes_sample.append(classes_list[i].sample(most, replace=True))
  df_maybe = pd.concat(classes_sample)
  final_df = pd.concat([df_maybe, classes_list[0]], axis=0)

  return final_df