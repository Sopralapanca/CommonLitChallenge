import torch.nn as nn
from transformers import AutoModel, AutoConfig
import torch
from lightning.pytorch.callbacks import Callback
import gc
import numpy as np
import copy
import lightning.pytorch as pl


class MetricsCallback(Callback):
    """PyTorch Lightning metric callback."""

    def __init__(self):
        super().__init__()
        self.metrics = {"train_loss": [],
                        "valid_loss": []
                        }

    def on_validation_epoch_end(self, trainer, pl_module):
        self.metrics["valid_loss"].append(trainer.callback_metrics["val_loss"].item())

    def on_train_epoch_end(self, trainer, pl_module):
        self.metrics["train_loss"].append(trainer.callback_metrics["train_loss"].item())



class RegressorModel(pl.LightningModule):
    def __init__(self, config, features_dim, ffdroput=0.1, fflayers=2, lr=0.0003, pooling="cls", nodropout=True):
        super(RegressorModel, self).__init__()
        self.model_name = config['model']
        self.pooling = pooling
        self.model_config = AutoConfig.from_pretrained(self.model_name)
        self.config = config
        self.drop = nn.Dropout(p=ffdroput)
        self.batch_size = config["batch_size"]

        if nodropout:
            self.model_config.hidden_dropout_prob = 0.0
            self.model_config.attention_probs_dropout_prob = 0.0

        self.encoder = AutoModel.from_pretrained(self.model_name, config=self.model_config)

        self.freeze = config['freeze_encoder']
        if self.freeze:
            for param in self.encoder.base_model.parameters():
                param.requires_grad = False

        # The output layer that takes the [CLS] representation and gives an output
        size = self.encoder.config.hidden_size + features_dim
        self.cls_layer1 = nn.Linear(size, 2*size)
        self.hidden_ff_layers = nn.ModuleList()

        self.relu1 = nn.LeakyReLU()
        self.ff1 = nn.Linear(128, 2)

    def loss_fn(self, outputs, targets):
        colwise_mse = torch.mean(torch.square(targets - outputs), dim=0)
        loss = torch.mean(torch.sqrt(colwise_mse), dim=0)
        return loss

    def compute_step(self, batch):
        inputs = batch[0]
        attention_mask = batch[1]
        target = batch[2]

        input_ids = inputs[0]
        features = inputs[1]

        features = torch.tensor(features).float().to(input_ids.device)

        # Feed the input to Bert model to obtain contextualized representations
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask,
                               output_hidden_states=False)  # returns a BaseModelOutput object

        if self.pooling == 'cls':
            # Obtain the representations of [CLS] heads
            logits = outputs.last_hidden_state[:, 0, :]

        if self.pooling == 'mean-pooling':
            last_hidden_state = outputs.last_hidden_state
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
            sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
            sum_mask = input_mask_expanded.sum(1)
            sum_mask = torch.clamp(sum_mask, min=1e-9)
            logits = sum_embeddings / sum_mask

        combined_features = torch.cat((logits, features), dim=1)

        output = self.drop(combined_features)
        output = self.cls_layer1(output)
        output = self.relu1(output)
        output = self.ff1(output)

        loss = self.loss_fn(output, target)
        return loss

    def forward(self, batch, batch_idx):
        loss = self.compute_step(batch)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.compute_step(batch)
        self.log("train_loss", loss, batch_size=self.batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.compute_step(batch)
        self.log("val_loss", loss, batch_size=self.batch_size)
        return loss

    def test_step(self, batch, batch_idx):
        loss = self.compute_step(batch)
        self.log("test_loss", loss, batch_size=self.batch_size)
        return loss

    def configure_optimizers(self):
        no_decay = ['bias', 'LayerNorm.weight']

        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': 0.01},
            {'params': [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.config['lr'], eps=self.config['adam_eps'])

        scheduler = {'scheduler': torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, eta_min=1e-7, verbose=False),
                     'interval': 'step'}

        return [optimizer], [scheduler]
