import torch.nn as nn
from transformers import AutoModel
import torch
from tqdm.notebook import tqdm
import gc


class RegressorModel(nn.Module):
    def __init__(self, config):
        super(RegressorModel, self).__init__()
        self.model_name = config['model']
        # The output layer that takes the [CLS] representation and gives an output

        self.freeze = config['freeze_encoder']

        self.encoder = AutoModel.from_pretrained(self.model_name)
        if self.freeze:
            for param in self.encoder.base_model.parameters():
                param.requires_grad = False

        self.cls_layer1 = nn.Linear(self.encoder.config.hidden_size, 128)
        self.relu1 = nn.LeakyReLU()
        self.ff1 = nn.Linear(128, 64)
        self.ff2 = nn.Linear(64, 32)
        self.ff3 = nn.Linear(32, 2)

    def forward(self, input_ids, attention_mask):
        # Feed the input to Bert model to obtain contextualized representations
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # Obtain the representations of [CLS] heads
        logits = outputs.last_hidden_state[:, 0, :]
        output = self.cls_layer1(logits)
        output = self.relu1(output)
        output = self.ff1(output)
        output = self.relu1(output)
        output = self.ff2(output)
        output = self.relu1(output)
        output = self.ff3(output)
        return output


class Trainer:
    def __init__(self, model, loaders, config, accelerator):
        self.model = model
        self.train_loader, self.val_loader = loaders
        self.config = config
        # self.input_keys = ['input_ids', 'token_type_ids', 'attention_mask']
        self.accelerator = accelerator

        self.optim = self._get_optim()

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optim, T_0=5, eta_min=1e-7)

        self.train_losses = []
        self.val_losses = []

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
             'weight_decay': 0.01},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.config['lr'], eps=self.config['adam_eps'])
        return optimizer

    def loss_fn(self, outputs, targets):
        colwise_mse = torch.mean(torch.square(targets - outputs), dim=0)
        loss = torch.mean(torch.sqrt(colwise_mse), dim=0)
        return loss

    def train_one_epoch(self, epoch, outer_progress, outer_iterations):

        running_loss = 0.
        inner_iterations = len(self.train_loader)

        idx = 0
        for input_ids, attention_mask, target in self.train_loader:
            with self.accelerator.accumulate(self.model):
                output = self.model(input_ids=input_ids, attention_mask=attention_mask)

                loss = self.loss_fn(output, target)
                running_loss += loss.item()

                self.accelerator.backward(loss)

                self.optim.step()

                if self.config['enable_scheduler']:
                    self.scheduler.step(epoch - 1 + idx / len(self.train_loader))

                self.optim.zero_grad()

                del input_ids, attention_mask, target, loss

                inner_progress = f"Training Batch: [{idx}/{inner_iterations}]"
                print(f"\r{outer_progress} {inner_progress}", end="")

            idx += 1

        train_loss = running_loss / len(self.train_loader)
        self.train_losses.append(train_loss)
        inner_progress = f"Training Batch:[{idx}/{inner_iterations}] training loss: {self.train_losses[-1]}"
        print(f"\r{outer_progress} {inner_progress}", end="")

    @torch.no_grad()
    def valid_one_epoch(self, epoch, outer_progress, outer_iterations):

        running_loss = 0.
        inner_iterations = len(self.val_loader)

        idx = 0
        for input_ids, attention_mask, target in self.val_loader:
            output = self.model(input_ids=input_ids, attention_mask=attention_mask)

            loss = self.loss_fn(output, target)
            running_loss += loss.item()

            del input_ids, attention_mask, target, loss

            inner_progress = f"Validation Batch: [{idx}/{inner_iterations}] training loss: {self.train_losses[-1]}"
            print(f"\r{outer_progress} {inner_progress}", end="")

            idx += 1

        val_loss = running_loss / len(self.val_loader)
        self.val_losses.append(val_loss)
        inner_progress = f"Validation Batch:[{idx}/{inner_iterations}] training loss: {self.train_losses[-1]} validation loss: {self.val_losses[-1]}"
        print(f"\r{outer_progress} {inner_progress}", end="")

    def test(self, test_loader):

        preds = []
        for (inputs) in test_loader:
            outputs = self.model(inputs)
            preds.append(outputs.detach().cpu())

        preds = torch.concat(preds)
        return preds

    def fit(self):

        self.prepare()

        # Define the number of iterations for both loops
        outer_iterations = self.config['epochs']

        # Create outer progress bar
        for epoch in range(1, outer_iterations + 1):
            outer_progress = f"Epoch: {epoch}/{outer_iterations}"
            self.model.train()

            self.train_one_epoch(epoch, outer_progress, outer_iterations)
            self.clear()

            self.model.eval()
            self.valid_one_epoch(epoch, outer_progress, outer_iterations)

            self.clear()

            print()

        print("\nTraining completed!")
        return self.train_losses, self.val_losses

    def clear(self):
        gc.collect()
        torch.cuda.empty_cache()
