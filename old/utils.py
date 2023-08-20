import spacy
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, Trainer
from tqdm.notebook import tqdm
import gc


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


class CommonLitDataset(Dataset):
    def __init__(self, encodings, labels):
        # Initialize the tokenizer for the desired transformer model
        self.encodings = encodings
        # list of target columns
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])

        return item


class RegressorModel(nn.Module):
    def __init__(self, config):
        super(RegressorModel, self).__init__()
        self.model_name = config['model']

        self.freeze = config['freeze_encoder']

        # The output layer that takes the [CLS] representation and gives an output
        self.encoder = AutoModel.from_pretrained(self.model_name)
        if self.freeze:
            for param in self.encoder.base_model.parameters():
                param.requires_grad = False

        self.cls_layer1 = nn.Linear(self.encoder.config.hidden_size, 128)
        self.relu1 = nn.ReLU()
        self.ff1 = nn.Linear(128, 2)

    def forward(self, input_ids, attention_mask):
        # Feed the input to Bert model to obtain contextualized representations
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # Obtain the representations of [CLS] heads
        logits = outputs.last_hidden_state[:, 0, :]
        output = self.cls_layer1(logits)
        output = self.relu1(output)
        output = self.ff1(output)
        return output


class CustomTrainer(Trainer):
    def __int__(self, *args, **kwargs):
        super().__int__(*args, **kwargs)
    def compute_loss(self, model, inputs, return_outputs=False):
        print(inputs)
        targets = inputs.pop("labels")
        outputs = model(**inputs)
        colwise_mse = torch.mean(torch.square(targets - outputs), dim=0)
        loss = torch.mean(torch.sqrt(colwise_mse), dim=0)
        return (loss, outputs) if return_outputs else loss
