import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import os
import json
from PIL import Image
import pdb

def dataset_steup(local_dir='coco', split='train'):
    annotations_path = os.path.join(local_dir,'annotations', f'instances_{split}2017.json')
    with open(annotations_path, 'r') as f:
        annotations = json.load(f)

    images_info = annotations['images']


class COCODataset(Dataset):
    def __init__(self, split, processor, local_dir='coco'):
        # Load the COCO dataset from local directory
        self.split = split
        self.local_dir = local_dir
        self.processor = processor

        # Load annotations
        annotations_path = os.path.join(local_dir,'annotations', f'instances_{split}2017.json')
        with open(annotations_path, 'r') as f:
            annotations = json.load(f)
        
        self.images_info = annotations['images']
        pdb.set_trace()
        self.annotations = {ann['image_id']: ann for ann in annotations['annotations']}
        self.image_dir = os.path.join(local_dir, 'images', f'{split}2017')

    def __len__(self):
        return len(self.images_info)

    def __getitem__(self, idx):
        # Get image info and annotation
        image_info = self.images_info[idx]
        image_id = image_info['id']
        image_path = os.path.join(self.image_dir, image_info['file_name'])
        
        # Load image
        image = Image.open(image_path).convert("RGB")
        
        # Load segmentation label
        label = self.annotations[image_id]['segmentation']
        
        # Apply the processor to the image
        processed_image = self.processor(image, return_tensors="pt")['pixel_values'].squeeze(0)
        return processed_image, label


class COCOSegmenter(pl.LightningModule):
    def __init__(self, segformer_model, processor, learning_rate=1e-3, num_workers=0, batch_size=8):
        super(COCOSegmenter, self).__init__()
        self.save_hyperparameters()
        
        # Use the provided model
        self.segformer_model = segformer_model
        self.processor = processor
        
        # Define loss function
        self.criterion = nn.CrossEntropyLoss()
        self.learning_rate = learning_rate
        self.num_workers = num_workers
        self.batch_size = batch_size

    def forward(self, x):
        # Forward pass through the full model
        return self.segformer_model(x).logits

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
        self.log('val_loss', loss, prog_bar=True, logger=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def train_dataloader(self):
        train_dataset = COCODataset(split='train', processor=self.processor)
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=self.num_workers, 
            pin_memory=True
        )
        return train_loader

    def val_dataloader(self):
        val_dataset = COCODataset(split='val', processor=self.processor)
        val_loader = DataLoader(
            val_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=self.num_workers, 
            pin_memory=torch.cuda.is_available()
        )
        return val_loader

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        return self.segformer_model.state_dict(destination, prefix, keep_vars)

# Example usage
if __name__ == "__main__":
    from transformers import SegformerForSemanticSegmentation, AutoImageProcessor

    model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
    processor = AutoImageProcessor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
    
    model = COCOSegmenter(model, processor)
    trainer = pl.Trainer(max_epochs=10, fast_dev_run=True)
    trainer.fit(model)
