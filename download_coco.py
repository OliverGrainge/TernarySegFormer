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

        # Get the size of the partially downloaded file
        downloaded_size = os.path.getsize(filename) if os.path.exists(filename) else 0

        # Get the total file size from the server
        response = requests.head(url)
        total_size = int(response.headers.get('content-length', 0))

        if downloaded_size < total_size:
            print(f"Downloading {key}...")
            headers = {"Range": f"bytes={downloaded_size}-"}  # Request the remaining bytes
            with requests.get(url, headers=headers, stream=True) as response, open(filename, "ab") as file:
                with tqdm(
                    total=total_size,
                    initial=downloaded_size,
                    unit='B',
                    unit_scale=True,
                    desc=key,
                ) as pbar:
                    for chunk in response.iter_content(1024):
                        if chunk:  # Filter out keep-alive chunks
                            file.write(chunk)
                            pbar.update(len(chunk))
            print(f"Finished downloading {key}.")
        else:
            print(f"{key} is already fully downloaded.")

        # Extract the downloaded file
        with zipfile.ZipFile(filename, 'r') as zip_ref:
            print(f"Extracting {key}...")
            zip_ref.extractall(dest_folder)
            print(f"Finished extracting {key}.")

if __name__ == "__main__":
    download_coco()
