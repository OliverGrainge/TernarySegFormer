import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import SegformerForSemanticSegmentation, AutoImageProcessor, get_cosine_schedule_with_warmup
import torch 
from torch.utils.data import Dataset
from datasets import load_dataset
import numpy as np
import torchvision.transforms as T


class ImageNetDataset(Dataset):
    def __init__(self, split, processor, data_dir='./imagenet'):
        # Load the dataset from Hugging Face
        self.dataset = load_dataset("ILSVRC/imagenet-1k", cache_dir=data_dir, split=split)
        self.convert_rgb = T.Lambda(lambda x: x if x.mode == 'RGB' else x.convert('RGB'))
        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Get image and label from the dataset
        sample = self.dataset[idx]
        image = sample['image']
        label = sample['label']
        image = self.convert_rgb(image)
        
        # Apply the processor to the image
        processed_image = self.processor(image, return_tensors="pt")['pixel_values'].squeeze(0)
        return processed_image, label

class ImageNetClassifier(pl.LightningModule):
    def __init__(self, segformer_model, processor, data_dir='./imagenet', num_classes=1000, learning_rate=1e-3, num_workers=0, batch_size=32, warmup_steps=1000, max_steps=None):
        super(ImageNetClassifier, self).__init__()
        self.save_hyperparameters(ignore=['segformer_model', 'processor'])
        self.data_dir = data_dir
        # Use the provided model and add a classification head
        self.segformer_model = segformer_model
        self.processor = processor

        self.global_avg_pool = nn.AdaptiveAvgPool2d((2, 2))
        self.classifier = nn.Linear(1024, num_classes)
        
        # Define loss function
        self.criterion = nn.CrossEntropyLoss()
        self.learning_rate = learning_rate
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps

    def forward(self, x):
        # Forward pass through feature extractor and classifier
        features = self.segformer_model.segformer.encoder(x)
        features = features.last_hidden_state 
        features = self.global_avg_pool(features).flatten(start_dim=1)
        logits = self.classifier(features)
        return logits

    def training_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        self.log('train_loss', loss, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        acc = (outputs.argmax(dim=1) == labels).float().mean()
        self.log('val_loss', loss, prog_bar=True, logger=True)
        self.log('val_acc', acc, prog_bar=True, logger=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        
        if self.max_steps is None:
            # Calculate max_steps if not provided
            dataset_size = len(self.train_dataloader().dataset)
            self.max_steps = (dataset_size // self.batch_size) * self.trainer.max_epochs
            
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=self.max_steps
        )
        
        return [optimizer], [{'scheduler': scheduler, 'interval': 'step'}]

    def train_dataloader(self):
        train_dataset = ImageNetDataset(data_dir=self.data_dir, split='train', processor=self.processor)
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=self.num_workers, 
            pin_memory=True
        )
        return train_loader

    def val_dataloader(self):
        val_dataset = ImageNetDataset(data_dir=self.data_dir, split='validation', processor=self.processor)
        val_loader = DataLoader(
            val_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=self.num_workers, 
            pin_memory=torch.cuda.is_available()
        )
        return val_loader

# Example usage
if __name__ == "__main__":
    model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
    processor = AutoImageProcessor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
    
    model = ImageNetClassifier(model, processor, data_dir='../imagenet')
    trainer = pl.Trainer(max_epochs=10, fast_dev_run=True)
    trainer.fit(model)
