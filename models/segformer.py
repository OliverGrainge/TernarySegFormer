from transformers import AutoImageProcessor, SegformerForSemanticSegmentation, SegformerConfig
import torch
from PIL import Image
import requests
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import os 


os.sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '../NeuroCompress/'))

import NeuroPress as NP

def get_segformer_model_and_processor(pretrained=False):
    if pretrained: 
        model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b3-finetuned-ade-512-512")
        processor = AutoImageProcessor.from_pretrained("nvidia/segformer-b3-finetuned-ade-512-512")
    else: 
        configuration = SegformerConfig()
        configuration.num_labels = 150
        model = SegformerForSemanticSegmentation(configuration)
        processor = AutoImageProcessor.from_pretrained("nvidia/segformer-b3-finetuned-ade-512-512")

    return model, processor

if __name__ == "__main__":
    # Load the model and processor
    model, processor = get_segformer_model_and_processor(pretrained=True)

    print(model.segformer.encoder)
    feature_extractor = model.segformer.encoder

    img = torch.randn(1, 3, 512, 512)
    out = feature_extractor(img)
    

    # Load and process the image

    urls = [
        "http://images.cocodataset.org/val2017/000000000139.jpg",
        "http://images.cocodataset.org/val2017/000000000285.jpg",
        "http://images.cocodataset.org/val2017/000000000632.jpg",
        "http://images.cocodataset.org/val2017/000000000724.jpg",
        "http://images.cocodataset.org/val2017/000000000776.jpg", 
        "http://images.cocodataset.org/val2017/000000000785.jpg",
        "http://images.cocodataset.org/val2017/000000000802.jpg",
        "http://images.cocodataset.org/val2017/000000000872.jpg",
        "http://images.cocodataset.org/val2017/000000000885.jpg",
        "http://images.cocodataset.org/val2017/000000001000.jpg",
        "http://images.cocodataset.org/val2017/000000001268.jpg",
    ]
    for url in urls:
        image = Image.open(requests.get(url, stream=True).raw)
        inputs = processor(images=image, return_tensors="pt")

        # Get model predictions
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits  # Shape: (batch_size, num_labels, height/4, width/4)
        print(logits.shape, inputs['pixel_values'].shape)

        # Get the predicted segmentation mask
        predicted_mask = torch.argmax(logits, dim=1)  # Shape: (batch_size, height/4, width/4)
        
        # Upscale the predicted mask to match the original image size (512, 512)
        predicted_mask = F.interpolate(predicted_mask.unsqueeze(1).float(), size=(512, 512), mode='nearest').squeeze().cpu().numpy()

        # Resize the original image to 512x512 for visualization
        image = image.resize((512, 512))

        # Visualize the results
        plt.figure(figsize=(10, 5))
        
        # Original image
        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.axis('off')
        plt.title('Original Image')

        # Image with segmentation mask
        plt.subplot(1, 2, 2)
        plt.imshow(image)
        plt.imshow(predicted_mask, alpha=0.5, cmap='jet')  # Overlay the mask with transparency
        plt.axis('off')
        plt.title('Image with Segmentation Mask')

        plt.show()
