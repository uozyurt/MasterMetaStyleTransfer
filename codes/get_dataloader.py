import torch
import torch.utils
import torch.utils.data
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2
import glob
import os

# TODO: check if normalization with mean and std is needed
transform = transforms.Compose([
    transforms.ToPILImage(), # -> PIL image
    transforms.Resize((512, 512)), # -> resize to 512x512
    transforms.RandomCrop((256,256)) , # random crop to 256x256
    transforms.ToTensor()
])

class coco_train_dataset(Dataset):
    # initialize the dataset
    def __init__(self, project_absolute_path, coco_dataset_relative_path = "datasets/coco_train_dataset/train2017"):
        # get the absolute path of the dataset
        dataset_ablsolute_path = os.path.join(project_absolute_path, coco_dataset_relative_path)

        print(dataset_ablsolute_path)

        # load coco dataset paths from local directory
        self.coco_dataset_images_paths = glob.glob(os.path.join(dataset_ablsolute_path, "*.jpg"))

    # return the length of the dataset
    def __len__(self):
        return len(self.coco_dataset_images_paths)

    # return the image at the given index
    def __getitem__(self, id):
        # load image
        img = cv2.imread(self.coco_dataset_images_paths[id])

        # apply transformations
        img = transform(img)

        return img
    
class wikiart_dataset(Dataset):
    # initialize the dataset
    def __init__(self, project_absolute_path, wikiart_dataset_relative_path = "datasets/wikiart/**"):
        # get the absolute path of the dataset
        dataset_ablsolute_path = os.path.abspath(os.path.join(project_absolute_path, wikiart_dataset_relative_path))
        
        # load wikiart dataset paths from local directory
        self.wikiart_dataset_images_paths = glob.glob(os.path.join(dataset_ablsolute_path, "*.jpg"))

    # return the length of the dataset
    def __len__(self):
        return len(self.wikiart_dataset_images_paths)

    # return the image at the given index
    def __getitem__(self, id):
        # load image
        img = cv2.imread(self.wikiart_dataset_images_paths[id])

        # apply transformations
        img = transform(img)

        return img


if __name__ == "__main__":
    # get the absolute path of the project
    project_absolute_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    # get the relative paths of the datasets
    coco_dataset_relative_path = "datasets/coco_train_dataset/train2017"
    wikiart_dataset_relative_path = "datasets/wikiart/**"

    # create the datasets
    coco_dataset_instance = coco_train_dataset(project_absolute_path = project_absolute_path,
                                                coco_dataset_relative_path = coco_dataset_relative_path)

    wikiart_dataset_instance = wikiart_dataset(project_absolute_path = project_absolute_path,
                                                wikiart_dataset_relative_path = wikiart_dataset_relative_path)
    
    print(f"Number of images in COCO dataset: {len(coco_dataset_instance)}")
    print(f"Number of images in Wikiart dataset: {len(wikiart_dataset_instance)}")


    # determine the batch size
    BATCH_SIZE_coco = 4
    BATCH_SIZE_wikiart = 1

    # determine the number of workerss
    NUM_WORKERS_coco = 4
    NUM_WORKERS_wikiart = 4

    # determine if the data should be shuffled
    SHUFFLE_coco = True
    SHUFFLE_wikiart = True

    # determine pin memory (for faster data transfer to GPU)
    PIN_MEMORY_coco = False
    PIN_MEMORY_wikiart = False

    # determine drop last
    DROP_LAST_coco = True
    DROP_LAST_wikiart = True

    # create the dataloaders
    coco_dataloader = DataLoader(coco_dataset_instance,
                                 batch_size=BATCH_SIZE_coco,
                                 shuffle=SHUFFLE_coco,
                                 num_workers=NUM_WORKERS_coco,
                                 pin_memory=PIN_MEMORY_coco,
                                 drop_last=DROP_LAST_coco)

    wikiart_dataloader = DataLoader(wikiart_dataset_instance,
                                    batch_size=BATCH_SIZE_wikiart,
                                    shuffle=SHUFFLE_wikiart,
                                    num_workers=NUM_WORKERS_wikiart,
                                    pin_memory=PIN_MEMORY_wikiart,
                                    drop_last=DROP_LAST_wikiart)

    # iterate over the dataloaders
    for i, data in enumerate(coco_dataloader):
        print(f"COCO dataloader iteration: {i}, shape: {data.shape}")

        if i == 5:
            break

    for i, data in enumerate(wikiart_dataloader):
        print(f"Wikiart dataloader iteration: {i}, shape: {data.shape}")

        if i == 5:
            break