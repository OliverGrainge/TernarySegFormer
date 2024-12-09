
import os
from torch.utils.data import Dataset
from pycocotools.coco import COCO
import torchvision.transforms as T
import numpy as np
from PIL import Image

class CocoSegmentationDataset(Dataset):
    def __init__(self, root, transform=None):

        self.root = root
        self.coco = COCO(os.path.join(root, 'annotations/stuff_val2017.json'))
        self.image_indexes = list(self.coco.imgs.keys())
        self.transform = transform

    def __len__(self):
        return len(self.img_ids)
    
    def __getitem__(self, index): 
        img_id = self.image_indexes[index] 
        annotations = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id))
        image_info = self.coco.loadImgs(img_id)[0]
        file_name = f"./coco/images/train2017/{image_info[0]['file_name']}"
        image = Image.open(file_name).convert('RGB')

        mask = np.zeros((image_info[0]['height'], image_info[0]['width']), dtype=np.uint8)
        for ann in annotations:
            rle = self.coco.annToRLE(ann)
            binary_mask = self.coco.annToMask(ann)
            mask = np.maximum(mask, binary_mask * ann['category_id']) 

        if self.transform: 
            image = self.transform(image)
            mask = T.functional.to_tensor(mask)
        #else: 
        #    image = T.functional.to_tensor(image)
        #    mask = T.functional.to_tensor(mask)

        return image, mask



# Example usage
if __name__ == "__main__":
    data_root = "./coco/"
    dataset = CocoSegmentationDataset(data_root)
    for i in range(1000):
        try:
            image, mask = dataset[i]
            print(image.shape, mask.shape)
            
        except Exception as e:
            pass
    

    
    

    