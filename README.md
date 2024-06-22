# Flower_Identification

## Group Members:
1. Pragya Harsh
2. Reshika Vedicherla

## Description
A simple classification based ML model that can be used to predict the species of a flower out of 102 species by uploading its image. A Wikipedia link for the same is generated as well. 

It uses VGG16: VGG16 is a convolutional neural network model that's used for image recognition. It has 16 layers that have weights. It's considered one of the best vision model architectures.

Rest of the details are mentioned in the .ipynb file.

## Dataset Used: 
Oxford 102 Category Flower Dataset

Link: https://www.kaggle.com/datasets/haseeb85/oxford-102-category-flower-dataset

## Directory Structure:
- Dataset.rar : contains the dataset
- flower102save : contains the trained model
- Prediction : images to be predicted by the model
- app.py : Web Interface to upload images and generate a Wikipedia Link for the predicted flower
- dataset_mapping_to_name.json : maps numbers to flower names from the dataset
- flower.py : helping file for app.py
- flower_identification_102classes_final.ipynb : Python Notebook associated with the project
