from datasets import load_dataset

SAVE_DIR="./imagenet"
dataset = load_dataset("ILSVRC/imagenet-1k", cache_dir=SAVE_DIR)
