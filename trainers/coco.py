import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import os
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from pycocotools.coco import COCO
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation, SegformerConfig



class CocoSegmentationDataset(Dataset):
    def __init__(self, root, split="train"):
        self.split = split
        self.root = root
        self.coco = COCO(os.path.join(root, f'annotations/stuff_{split}2017.json'))
        self.image_indexes = list(self.coco.imgs.keys())
        self.max_category_id = max(cat['id'] for cat in self.coco.loadCats(self.coco.getCatIds()))
        self.image_processor = AutoImageProcessor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512", return_tensors="pt")
        self.mask_transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.Lambda(lambda x: torch.from_numpy(np.array(x, dtype=np.uint8)).contiguous()),
        ])

    def __len__(self):
        return len(self.image_indexes)
    
    def __getitem__(self, index): 
        img_id = self.image_indexes[index] 
        print("img_id", img_id)
        img_id = 25
        annotations = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id))
        image_info = self.coco.loadImgs(img_id)[0]
        file_name = f"./coco/images/{self.split}2017/{image_info['file_name']}"
        image = Image.open(file_name).convert('RGB')

        mask = np.zeros((image_info['height'], image_info['width']), dtype=np.uint8)
        for ann in annotations:
            binary_mask = self.coco.annToMask(ann)
            mask = np.maximum(mask, binary_mask * ann['category_id']) 

        image = self.image_processor(image)
        mask = torch.from_numpy(mask).unsqueeze(0)
        mask = Image.fromarray(mask.numpy().squeeze(), mode='L')
        mask = self.mask_transform(mask)
        mask = torch.from_numpy(np.array(mask, dtype=np.uint8)).contiguous()

        image_tensor = torch.from_numpy(image['pixel_values'][0]).contiguous()
        mask = mask.contiguous()
        return image_tensor, mask


class COCOSegmenter(pl.LightningModule):
    def __init__(self, segformer_model, learning_rate=1e-3, num_workers=0, batch_size=8, data_root='coco'):
        super(COCOSegmenter, self).__init__()
        self.save_hyperparameters()
        
        self.segformer_model = segformer_model
        self.criterion = nn.CrossEntropyLoss()
        self.learning_rate = learning_rate
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.data_root = data_root

    def forward(self, x):
        return self.segformer_model(x).logits

    def training_step(self, batch, batch_idx):
        images, masks = batch
        masks = masks.squeeze(1).long()  # Ensure correct shape for CrossEntropyLoss
        outputs = self(images)

        # Reshape outputs and masks
        print(f"outputs.shape: {outputs.shape}, masks.shape: {masks.shape}")
        
        loss = self.criterion(outputs, masks)
        self.log('train_loss', loss, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, masks = batch
        masks = masks.squeeze(1).long()
        outputs = self(images)
        # Use reshape instead of view
        print(f"outputs.shape: {outputs.shape}, masks.shape: {masks.shape}")
        loss = self.criterion(outputs, masks)
        self.log('val_loss', loss, prog_bar=True, logger=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def train_dataloader(self):
        train_dataset = CocoSegmentationDataset(root=self.data_root, split='train')
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=self.num_workers, 
            pin_memory=torch.cuda.is_available()
        )
        return train_loader

    def val_dataloader(self):
        val_dataset = CocoSegmentationDataset(root=self.data_root, split='val')
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
    from transformers import SegformerForSemanticSegmentation

    segformer_model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b0-finetuned-ade-512-512"
    )

    model = COCOSegmenter(segformer_model, data_root="coco")
    trainer = pl.Trainer(max_epochs=10, fast_dev_run=True)
    trainer.fit(model)



