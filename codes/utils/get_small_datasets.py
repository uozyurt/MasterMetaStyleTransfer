import os
import shutil
import random
import glob

if(__name__ == "__main__"):
    # set the sample size
    SAMPLE_SIZE = 1000

    # set the seed for reproducibility
    random.seed(42)


    # get current absolute project path
    absolute_project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

    # get the paths of the datasets
    coco_path = os.path.join(absolute_project_path, "datasets/coco_train_dataset/train2017")
    if not os.path.exists(coco_path):
        raise Exception("COCO dataset is not downloaded, please download the dataset first")
    
    wikiart_path = os.path.join(absolute_project_path, "datasets/wikiart")
    if not os.path.exists(wikiart_path):
        raise Exception("Wikiart dataset is not downloaded, please download the dataset first")


    # set the destination paths
    dest_path_coco = os.path.join(absolute_project_path, "datasets/coco_train_dataset_small")
    dest_path_wikiart = os.path.join(absolute_project_path, "datasets/wikiart_small")


    # create a new directory for the small dataset
    if not os.path.exists(dest_path_coco):
        os.makedirs(dest_path_coco)

    if not os.path.exists(dest_path_wikiart):
        os.makedirs(dest_path_wikiart)



    # get image names
    coco_images = sorted(os.listdir(coco_path))
    wikiart_images = sorted(glob.glob(wikiart_path + "/*/*"))


    # get random samples
    coco_samples = random.sample(coco_images, SAMPLE_SIZE)
    wikiart_samples = random.sample(wikiart_images, SAMPLE_SIZE)

    # copy the samples to the new directory
    for image_path in wikiart_samples:
        shutil.copyfile(image_path, os.path.join(dest_path_wikiart, image_path.split("/")[-1]))


    for image in coco_samples:
        shutil.copyfile(os.path.join(coco_path, image), os.path.join(dest_path_coco, image))

