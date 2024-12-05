"""
import os
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset
from pycocotools.coco import COCO
from PIL import Image
import numpy as np

class CocoSegmentationDataset(Dataset):
    def __init__(self, root, annotation_file, transform=None):

        self.root = root
        self.coco = COCO(annotation_file)
        self.img_ids = list(self.coco.imgs.keys())
        self.transform = transform

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, index):
        # Load image and corresponding annotations
        img_id = self.img_ids[index]
        ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=False)
        anns = self.coco.loadAnns(ann_ids)
        
        # Load the image
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.root, img_info['file_name'])
        image = Image.open(img_path).convert('RGB')
        
        # Create the segmentation mask
        mask = np.zeros((img_info['height'], img_info['width']), dtype=np.uint8)
        for ann in anns:
            mask = np.maximum(mask, self.coco.annToMask(ann) * ann['category_id'])
        mask = Image.fromarray(mask)
        
        # Apply transformations
        if self.transform:
            image = self.transform(image)
            mask = T.functional.to_tensor(mask)  # Convert mask to tensor
        else:
            image = T.functional.to_tensor(image)
            mask = T.functional.to_tensor(mask)
        
        return image, mask

# Example usage
if __name__ == "__main__":
    data_root = "coco/train2017"
    annotation_path = "coco/annotations/instances_train2017.json"

    # Define transformations
    transforms = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor(),
    ])

    # Create dataset
    dataset = CocoSegmentationDataset(root=data_root, annotation_file=annotation_path, transform=transforms)

    # Test dataset
    image, mask = dataset[0]
    print("Image shape:", image.shape)
    print("Mask shape:", mask.shape)
"""
import torch

from transformers import SegformerForSemanticSegmentation, AutoImageProcessor

x = torch.randn(10, 3, 512, 512)
segformer_model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
global_avg_pool = torch.nn.AdaptiveAvgPool2d((2, 2))
classifier = torch.nn.Linear(1024, 1000)

features = segformer_model.segformer.encoder(x)
features = features.last_hidden_state 
print(features.shape)
f = global_avg_pool(features).flatten(start_dim=1)
print(f.shape)
logits = classifier(f)
print(logits.shape)