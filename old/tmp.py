from transformers import AutoModel
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import gc
from spacy.lang.en import English


from accelerate import Accelerator

config = {
    'model': 'bert-base-cased',
    'dropout': 0.5,
    'max_length': 512,
    'batch_size': 8,  # anything more results in CUDA OOM [for unfreezed encoder] on Kaggle GPU
    'epochs': 7,
    'lr': 3e-4,
    'enable_scheduler': True,
    'scheduler': 'CosineAnnealingWarmRestarts',
    'gradient_accumulation_steps': 2,
    'adam_eps': 1e-6,  # 1e-8 default
    'freeze_encoder': True
}

tokenizer = AutoTokenizer.from_pretrained(config['model'])

summaries_train_path = "../data/summaries_train.csv"
prompt_train_path = "../data/prompts_train.csv"

summaries_test_path = "../data/summaries_test.csv"
prompt_test_path = "../data/prompts_test.csv"

train_data = pd.read_csv(summaries_train_path, sep=',', index_col=0)
prompt_data = pd.read_csv(prompt_train_path, sep=',', index_col=0)

training_data = train_data.merge(prompt_data, on='prompt_id')
training_data.head()




def preprocessText(text):
    try:
        # replace newline with space
        text = text.replace("\n", " ")
        # split text
        words = text.split()

        # stop word removal
        stop_words = spacy.lang.en.stop_words.STOP_WORDS
        words = [w for w in words if not w in stop_words]
        # return pre-processed paragraph text
        text = ' '.join(words)
        return text
    except:
        return text


string_columns = ["text", "prompt_question", "prompt_title", "prompt_text"]

for col in string_columns:
    # apply preprocessText function to each text column in the dfTrain dataframe
    training_data[col] = training_data[col].apply(lambda x: preprocessText(x))

summary_word_count = training_data['text'].apply(lambda x: len(x.split()))
prompt_word_count = training_data['prompt_text'].apply(lambda x: len(x.split()))
prompt_question_word_count = training_data['prompt_question'].apply(lambda x: len(x.split()))
prompt_title_word_count = training_data['prompt_title'].apply(lambda x: len(x.split()))

print(len(summary_word_count))
print(len(prompt_word_count))
print(len(prompt_question_word_count))
print(len(prompt_title_word_count))


from torch.utils.data import Dataset, DataLoader
import torch


class CommonLitDataset(Dataset):
    def __init__(self, data, maxlen, tokenizer, target_cols):
        # Store the contents of the file in a pandas dataframe
        self.df = data.reset_index()
        # Initialize the tokenizer for the desired transformer model
        self.tokenizer = tokenizer
        # Maximum length of the tokens list to keep all the sequences of fixed size
        self.maxlen = maxlen

        # list of target columns
        self.target_cols = target_cols

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        # Select the sentence and label at the specified index in the data frame
        text = self.df.loc[index, "text"]
        prompt_question = self.df.loc[index, "prompt_question"]
        prompt_title = self.df.loc[index, "prompt_title"]
        prompt_text = self.df.loc[index, "prompt_text"]

        #full_text = prompt_title + " " + self.tokenizer.sep_token + " " + prompt_text + " " + self.tokenizer.sep_token + " " + prompt_question + " " + self.tokenizer.sep_token + " " + text
        full_text = text
        # Preprocess the text to be suitable for the transformer
        tokens = self.tokenizer.tokenize(full_text)
        tokens = ['[CLS]'] + tokens + ['[SEP]']
        if len(tokens) < self.maxlen:
            tokens = tokens + ['[PAD]' for _ in range(self.maxlen - len(tokens))]
        else:
            tokens = tokens[:self.maxlen - 1] + ['[SEP]']

        # Obtain the indices of the tokens in the BERT Vocabulary
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_ids = torch.tensor(input_ids)
        # Obtain the attention mask i.e a tensor containing 1s for no padded tokens and 0s for padded ones
        attention_mask = (input_ids != 0).long()

        try:
            target = self.df.loc[index, self.target_cols]
        except Exception as e:
            raise e

        target = torch.tensor(target, dtype=torch.float32)

        return input_ids, attention_mask, target


from sklearn.model_selection import train_test_split

train, validation = train_test_split(training_data, test_size=0.2)
print(train.shape, validation.shape)

target_cols = ["content", "wording"]
train_set = CommonLitDataset(data=train, maxlen=config['max_length'], tokenizer=tokenizer, target_cols=target_cols)
valid_set = CommonLitDataset(data=validation, maxlen=config['max_length'], tokenizer=tokenizer, target_cols=target_cols)

train_loader = DataLoader(dataset=train_set, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
valid_loader = DataLoader(dataset=valid_set, batch_size=config['batch_size'], shuffle=True, pin_memory=True)


class RMSELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps

    def forward(self, yhat, y):
        loss = torch.sqrt(self.mse(yhat, y) + self.eps)
        return loss


class MCRMSELoss(nn.Module):
    def __init__(self, num_scored=2):
        super().__init__()
        self.rmse = RMSELoss()
        self.num_scored = num_scored

    def forward(self, yhat, y):
        score = 0
        for i in range(self.num_scored):
            score += self.rmse(yhat[:, i], y[:, i]) / self.num_scored

        return score

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from transformers import BertPreTrainedModel, BertModel
class BertRegresser(nn.Module):
    def __init__(self, config):
        super(BertRegresser, self).__init__()
        self.model_name = config['model']
        #The output layer that takes the [CLS] representation and gives an output

        self.freeze = config['freeze_encoder']

        self.encoder = AutoModel.from_pretrained(self.model_name)
        if self.freeze:
            for param in self.encoder.base_model.parameters():
                param.requires_grad = False

        self.cls_layer1 = nn.Linear(self.encoder.config.hidden_size, 128)
        self.relu1 = nn.ReLU()
        self.ff1 = nn.Linear(128, 2)

    def forward(self, input_ids, attention_mask):
        #Feed the input to Bert model to obtain contextualized representations
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        #Obtain the representations of [CLS] heads
        logits = outputs.last_hidden_state[:,0,:]
        output = self.cls_layer1(logits)
        output = self.relu1(output)
        output = self.ff1(output)
        return output


class Trainer:
    def __init__(self, model, loaders, config, accelerator):
        self.model = model
        self.train_loader, self.val_loader = loaders
        self.config = config
        self.input_keys = ['input_ids', 'token_type_ids', 'attention_mask']
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

    def train_one_epoch(self, epoch):

        running_loss = 0.
        progress = tqdm(self.train_loader, total=len(self.train_loader))

        for idx, (input_ids, attention_mask, target) in enumerate(progress):
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

        train_loss = running_loss / len(self.train_loader)
        self.train_losses.append(train_loss)

    @torch.no_grad()
    def valid_one_epoch(self, epoch):

        running_loss = 0.
        progress = tqdm(self.val_loader, total=len(self.val_loader))

        for idx, (input_ids, attention_mask, target) in enumerate(progress):
            output = self.model(input_ids=input_ids, attention_mask=attention_mask)

            loss = self.loss_fn(output, target)
            running_loss += loss.item()

            del input_ids, attention_mask, target, loss

        val_loss = running_loss / len(self.val_loader)
        self.val_losses.append(val_loss)

    def test(self, test_loader):

        preds = []
        for (inputs) in test_loader:
            outputs = self.model(inputs)
            preds.append(outputs.detach().cpu())

        preds = torch.concat(preds)
        return preds

    def fit(self):

        self.prepare()

        fit_progress = tqdm(
            range(1, self.config['epochs'] + 1),
            leave=True,
            desc="Training..."
        )

        for epoch in fit_progress:
            self.model.train()
            fit_progress.set_description(f"EPOCH {epoch} / {self.config['epochs']} | training...")
            self.train_one_epoch(epoch)
            self.clear()

            self.model.eval()
            fit_progress.set_description(f"EPOCH {epoch} / {self.config['epochs']} | validating...")
            self.valid_one_epoch(epoch)
            self.clear()

            print(f"{'➖️' * 10} EPOCH {epoch} / {self.config['epochs']} {'➖️' * 10}")
            print(f"train loss: {self.train_losses[-1]}")
            print(f"valid loss: {self.val_losses[-1]}\n\n")

    def clear(self):
        gc.collect()
        torch.cuda.empty_cache()

accelerator = Accelerator(gradient_accumulation_steps=config['gradient_accumulation_steps'])
model = BertRegresser(config).to(device=accelerator.device)
trainer = Trainer(model, (train_loader, valid_loader), config, accelerator)
trainer.fit()