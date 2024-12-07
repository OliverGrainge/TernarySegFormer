import torch
from transformers import SegformerForSemanticSegmentation, AutoImageProcessor
import pytorch_lightning as pl
from trainers.imagenet import ImageNetClassifier
from models.ternary_segformer import get_ternary_segformer_model_and_processor
from models.segformer import get_segformer_model_and_processor
import argparse
import wandb
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

def parse_args():
    parser = argparse.ArgumentParser(description="Train a model for semantic segmentation.")
    parser.add_argument('--training_method', type=str, required=True, choices=['imagenet'], help='Training method to use.')
    parser.add_argument('--model', type=str, required=True, choices=['segformer', 'ternary_segformer'], help='Model type to use.')
    return parser.parse_args()

args = parse_args()

if args.training_method == "imagenet": 
    if args.model == "segformer": 
        model, processor = get_segformer_model_and_processor(pretrained=False)
    elif args.model == "ternary_segformer": 
        model, processor = get_ternary_segformer_model_and_processor(pretrained=False)
        print(model)

    model = ImageNetClassifier(model, processor, data_dir='./imagenet', batch_size=48, num_workers=12, warmup_steps=200)
    wandb_logger = WandbLogger(
        project="SegFomer-" + args.training_method,
        log_model=True
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='step')
    
    checkpoint_callback = ModelCheckpoint(
        dirpath='checkpoints/imagenet/',
        filename=f'{args.model}-{{epoch:02d}}-{{val_loss:.2f}}',
        save_top_k=3,
        monitor='val_loss',
        mode='min'
    )
    
    trainer = pl.Trainer(
        max_epochs=10, 
        precision='bf16', 
        accelerator='auto', 
        logger=wandb_logger,
        callbacks=[checkpoint_callback, lr_monitor]
    )
    trainer.fit(model)


elif args.training_method == "coco": 
    raise NotImplementedError("COCO training not implemented yet")
