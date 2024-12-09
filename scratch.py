import os
from torch.utils.data import Dataset
from pycocotools.coco import COCO
import numpy as np
from PIL import Image
from torchvision import transforms
import torch 

from transformers import AutoImageProcessor, SegformerForSemanticSegmentation, SegformerConfig

class CocoSegmentationDataset(Dataset):
    def __init__(self, root, split="train"):
        self.split = split
        self.root = root
        self.coco = COCO(os.path.join(root, f'annotations/stuff_{split}2017.json'))
        self.image_indexes = list(self.coco.imgs.keys())
        self.max_category_id = max(cat['id'] for cat in self.coco.loadCats(self.coco.getCatIds()))
        self.image_processor = AutoImageProcessor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
        self.mask_transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.Lambda(lambda x: torch.from_numpy(np.array(x, dtype=np.uint8))),
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
        return torch.from_numpy(image['pixel_values'][0]), mask


# Example usage
if __name__ == "__main__":
    import torch
    import torch.nn as nn

    # Example tensors
    logits = torch.randn(8, 150, 128, 128)  # Network output
    mask = torch.randint(0, 150, (8, 128, 128))  # Ground truth mask
    print(logits.shape, mask.shape)

    # Define the loss function
    criterion = nn.CrossEntropyLoss()

    # Compute the loss
    loss = criterion(logits, mask)
    print(loss.item())
    

    
    

    