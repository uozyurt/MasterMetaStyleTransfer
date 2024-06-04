#!/bin/bash


# Create datasets directory if it doesn't exist
mkdir -p ./datasets

# Download COCO dataset using wget
echo "Downloading COCO dataset..."
wget -P ./datasets http://images.cocodataset.org/zips/train2017.zip

# Download WikiArt dataset using wget
echo "aria2c is not installed, downloading WikiArt dataset using wget..."
wget -P ./datasets http://web.fsktm.um.edu.my/~cschan/source/ICIP2017/wikiart.zip

echo "Download completed!"