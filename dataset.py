import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
from pycocotools.coco import COCO
from PIL import Image
import os

class COCOSegmentationDataset(Dataset):
    def __init__(self, root, annFile, transform=None):
        """
        Args:
            root (str): Root directory where images are downloaded.
            annFile (str): Path to the COCO annotation file.
            transform (callable, optional): A function/transform to apply to both the image and mask.
        """
        self.root = root
        self.coco = COCO(annFile)
        self.ids = list(self.coco.imgs.keys())
        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id, iscrowd=False)
        anns = coco.loadAnns(ann_ids)

        # Load image
        img_info = coco.loadImgs(img_id)[0]
        path = img_info['file_name']
        image = Image.open(os.path.join(self.root, path)).convert('RGB')

        # Load segmentation mask
        mask = Image.new('L', (img_info['width'], img_info['height']))
        for ann in anns:
            mask = Image.composite(Image.fromarray(coco.annToMask(ann) * 255, 'L'), mask, Image.fromarray(coco.annToMask(ann)))

        # Apply transformations
        if self.transform is not None:
            image, mask = self.transform(image, mask)

        return image, mask

# Define the transformations
class SegmentationTransform:
    def __init__(self, image_transform=None, mask_transform=None):
        self.image_transform = image_transform
        self.mask_transform = mask_transform

    def __call__(self, image, mask):
        if self.image_transform is not None:
            image = self.image_transform(image)
        if self.mask_transform is not None:
            mask = self.mask_transform(mask)
        return image, mask

# Example usage
if __name__ == "__main__":
    # Paths
    image_root = "path/to/coco/images"
    annotation_file = "path/to/coco/annotations/instances_train2017.json"

    # Transformations
    image_transform = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor(),
    ])
    mask_transform = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor(),
    ])
    transform = SegmentationTransform(image_transform=image_transform, mask_transform=mask_transform)

    # Dataset and DataLoader
    coco_dataset = COCOSegmentationDataset(root=image_root, annFile=annotation_file, transform=transform)
    coco_dataloader = DataLoader(coco_dataset, batch_size=4, shuffle=True, num_workers=2)

    # Iterate through data
    for images, masks in coco_dataloader:
        print(images.shape, masks.shape)
        break
