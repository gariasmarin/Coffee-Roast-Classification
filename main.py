import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.models as models
from pathlib import Path

import data_setup

# setup paths and directories
data_path = Path("coffee-bean-dataset/")
image_path = data_path / "roast_levels"
train_dir = image_path / "train"
test_dir = image_path / "test"


# first test with EfficientNet B0
weights = models.EfficientNet_B0_Weights.DEFAULT
# get transforms to create pretrained weights
auto_transforms = weights.transforms()

# Create training and testing DataLoaders as well as get a list of class names
train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(train_dir=train_dir,
                                                                               test_dir=test_dir,
                                                                               transform=auto_transforms, # perform same data transforms on our own data as the pretrained model
                                                                               batch_size=32) # set mini-batch size to 32

print("class names:", class_names)



