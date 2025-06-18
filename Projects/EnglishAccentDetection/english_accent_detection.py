import torch as t
import torch.nn as nn
import torch.nn.functional as F

import pandas as pd
import numpy as np
import random
from lightning.pytorch.tuner import Tuner
from lightning.pytorch import loggers as pl_loggers
from lightning.pytorch import Trainer, LightningModule, seed_everything
from lightning.pytorch.callbacks.early_stopping import EarlyStopping


from datasets import load_dataset
from encodec import EncodecModel
from encodec.utils import convert_audio
from torch.utils.data import Dataset, default_collate
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader
from functools import lru_cache
from torcheval.metrics import MulticlassAccuracy, MulticlassAUROC, MulticlassRecall, MulticlassPrecision

class EnglishAccentDataset(Dataset):

    # List of accents, DO NOT CHANGE THE ORDER (taken from huggingface)
    accents = ['Dutch', 'German', 'Czech', 'Polish', 'French', 'Hungarian', 'Finnish', 'Romanian', 'Slovak', 'Spanish', 'Italian', 'Estonian', 'Lithuanian', 'Croatian', 'Slovene', 'English', 'Scottish', 'Irish', 'NorthernIrish', 'Indian', 'Vietnamese', 'Canadian', 'American']

    encodec = EncodecModel.encodec_model_24khz()
    encodec.eval()
    encodec.set_target_bandwidth(6.0)
    # Frozen Encodec Model
    for parameter in encodec.parameters():
        parameter.requires_grad_(False)

    def __init__(self, split = None, train_length = 1000):
        super().__init__()
        if split:
            assert split in ['train', 'validation', 'test']
        self.hf_dataset = load_dataset("westbrook/English_Accent_DataSet", split=split).with_format('torch')
        self.train_length = train_length

    def __len__(self):
        return len(self.hf_dataset)
    
    def __getitem__(self, index):
        with t.no_grad():
            wav, sr = self.hf_dataset[index]['audio']['array'], self.hf_dataset[index]['audio']['sampling_rate'].item()
            target = self.hf_dataset[index]['accent']
            if t.numel(wav) == 0:
                print(f"Error: index -> {index}")
                print(f"accent: {target}")
                return None
            wav = wav.unsqueeze(0).unsqueeze(0)
            wav = convert_audio(wav, sr, EnglishAccentDataset.encodec.sample_rate, EnglishAccentDataset.encodec.channels)
            frames = EnglishAccentDataset.encodec.encode(wav)
            codes = frames[0][0]
            codes = codes.squeeze()
            codes = codes[:, :self.train_length]
            padded_codes, mask = self.pad_codes(codes)
        return padded_codes.cpu(), mask, target

    def pad_codes(self, codes):
        K, T = codes.shape
        ret = t.zeros([K, self.train_length])
        ret = t.fill(ret, value=1024)
        ret[:, :T] = codes
        mask = ret != 1024
        return ret, mask

    @lru_cache(maxsize=1)
    def get_class_weights(self):
        ret = {}
        for row in self.hf_dataset:
            accent = EnglishAccentDataset.get_accent_from_label(row['accent'].item())
            ret[accent] = ret.get(accent, 0) + 1
        return ret
    
    def get_examples_from_class(self, accent):
        index = EnglishAccentDataset.get_label_from_accent(accent)
        examples = [i for i, row in enumerate(self.hf_dataset) if row['accent'].item() == index]
        return examples


    def get_accent_from_label(label: int):
        return EnglishAccentDataset.accents[label]

    def get_label_from_accent(accent: str):
        return EnglishAccentDataset.accents.index(accent)


class AccentRecogniser(LightningModule):
    def __init__(self, input_dim, num_classes, num_heads=16, num_layers=12, ff_dim=512, dropout=0.2):
        super().__init__()
        self.learning_rate = 0.000001
        self.batch_size = 16
        self.num_classes = num_classes

        self.input_dim = input_dim
        self.embedders = nn.ModuleList([nn.Embedding(1024 + 1, input_dim, padding_idx=1024) for _ in range(8)])
        
        # Transformer Encoder Layer
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim, nhead=num_heads, dim_feedforward=ff_dim, dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer, num_layers=num_layers
        )
        
        # Classifier head (fully connected layers)
        self.fc = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, num_classes)
        )

        self.train_dataset = EnglishAccentDataset('train')
        # Getting class Weights
        class_weights = self.train_dataset.get_class_weights()
        s = sum(class_weights.values())
        class_weights = [s / (len(EnglishAccentDataset.accents) * class_weights[c]) for c in EnglishAccentDataset.accents]
        self.loss_fn = t.nn.CrossEntropyLoss(weight=t.Tensor(class_weights))

    def forward(self, x, masks):
        x = x.long()
        B, K, T = x.shape
        masks = masks[:, 0, :]
        y = t.zeros([B, K, T, self.input_dim]).to(x.device)
        for i in range(len(self.embedders)):
            y[:, i] = self.embedders[i](x[:, i])
        x = y # [B, K, T, input_dim]

        # Remove Codebook Dimension
        x = t.sum(x, dim=1) # [B, T, input_dim]

        x = self.transformer_encoder(x, src_key_padding_mask = t.logical_not(masks)) # [B, T, input_dim]

        # Selecting the last element means we have the embedding that corresponds to the whole time series.
        x = x[:, -1, :] # [B, input_dim]

        x = self.fc(x) # [B, num_classes]

        return x
    

    def configure_optimizers(self):
        return t.optim.Adam(self.parameters(), lr = self.learning_rate or self.lr)
    
    def _collate_fn(self, batch):
        # Remove None items from batch
        batch = [item for item in batch if item is not None]
        return default_collate(batch)

    def train_dataloader(self):
        train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4, collate_fn=self._collate_fn)
        return train_dataloader

    def val_dataloader(self):
        val_dataset = EnglishAccentDataset('validation')
        val_dataloader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4, collate_fn=self._collate_fn)
        return val_dataloader

    def test_dataloader(self):
        test_dataset = EnglishAccentDataset('test')
        test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4, collate_fn=self._collate_fn)
        return test_dataloader

    def _step(self, batch, batch_idx):
        x, masks, targets = batch
        x = self(x, masks)
        loss = self.loss_fn(x, targets)
        return loss, x


    def training_step(self, batch, batch_idx):
        loss, logits = self._step(batch, batch_idx)
        self.log("train_loss", loss, prog_bar=True, logger=True, on_step=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss, x = self._step(batch, batch_idx)
        self.log("val_loss", loss, prog_bar=True, logger=True, on_step=True)

    def test_step(self, batch, batch_idx):
        x, masks, targets = batch
        metrics =  {
            'accuracy': MulticlassAccuracy(num_classes=self.num_classes),
            'precision' : MulticlassPrecision(num_classes=self.num_classes),
            'recall': MulticlassRecall(num_classes=self.num_classes),
            'AUC': MulticlassAUROC(num_classes=self.num_classes),
            }
        loss, logits = self._step(batch, batch_idx)
        for _, metric in metrics.items():
            metric.update(logits, targets)
        self.log_dict({'test_loss': loss, 'test_acc': metrics['accuracy'].compute(), 'test_prec': metrics['precision'].compute(), 'test_recall': metrics['recall'].compute(), 'test_AUC': metrics['AUC'].compute()})


if __name__ == '__main__':
    seed_everything(42, workers=True)
    t.set_float32_matmul_precision('medium')
    model = AccentRecogniser(1024, num_classes=len(EnglishAccentDataset.accents))

    default_dir = "/home/andreacacioli/Documents/github/MachineLearning/Projects/EnglishAccentDetection"

    # Change the logging directory
    logger = pl_loggers.TensorBoardLogger(save_dir=default_dir)

    trainer = Trainer(accelerator='gpu', profiler='simple', max_epochs=50, logger = logger, callbacks=[EarlyStopping(monitor="val_loss", mode='min', verbose=True)], default_root_dir=default_dir)

    #tuner = Tuner(trainer)
    #lr_finder = tuner.lr_find(model)
    #print(lr_finder.results)

    ## Plot
    #fig = lr_finder.plot(suggest=True)
    #fig.show(block = True)


    #tuner.scale_batch_size(model, mode='binsearch')

    trainer.fit(model)

    trainer.test(model)
