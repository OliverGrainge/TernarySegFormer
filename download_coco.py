import os
import requests
import zipfile
from tqdm import tqdm

def download_coco(dest_folder="coco"):
    """
    Downloads the COCO 2017 dataset (train images and annotations).

    Args:
        dest_folder (str): The destination folder to store the downloaded files.
    """
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    # URLs for the COCO 2017 dataset
    urls = {
        "annotations": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
        "train_images": "http://images.cocodataset.org/zips/train2017.zip",
        "val_images": "http://images.cocodataset.org/zips/val2017.zip",
    }

    for key, url in urls.items():
        filename = os.path.join(dest_folder, url.split("/")[-1])

        # Download the file if it doesn't exist
        if not os.path.exists(filename):
            print(f"Downloading {key}...")
            response = requests.get(url, stream=True)
            total_size = int(response.headers.get('content-length', 0))
            block_size = 1024
            with open(filename, "wb") as file, tqdm(total=total_size, unit='B', unit_scale=True, desc=key) as pbar:
                for data in response.iter_content(block_size):
                    file.write(data)
                    pbar.update(len(data))
            print(f"Finished downloading {key}.")

        # Extract the downloaded file
        with zipfile.ZipFile(filename, 'r') as zip_ref:
            print(f"Extracting {key}...")
            zip_ref.extractall(dest_folder)
            print(f"Finished extracting {key}.")

if __name__ == "__main__":
    download_coco()
