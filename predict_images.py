import torch
import torchinfo
import torchvision
import torchvision.models as models
from torch import nn
from torchvision import transforms
from pathlib import Path

from predict_image_driver import predict_image_driver


def predict_images(m=str, custom_img=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_path = Path("coffee-bean-dataset/")
    image_path = data_path / "roast_levels"
    train_dir = image_path / "train"
    test_dir = image_path / "test"

    class_names = ['dark', 'green', 'light', 'medium']
    output_shape = len(class_names)

    if "efficientnet" in m:
        if "b0" in m:
            weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
            model = torchvision.models.efficientnet_b0(weights=weights).to(device)
            model.classifier = torch.nn.Sequential(
                torch.nn.Dropout(p=0.2, inplace=True),
                torch.nn.Linear(in_features=1280,
                                out_features=output_shape,  # same number of output units as our number of classes
                                bias=True)).to(device)
            model.load_state_dict(torch.load("models/efficientnet_b0.pth"))

        elif "b1" in m:
            weights = torchvision.models.EfficientNet_B1_Weights.DEFAULT
            model = torchvision.models.efficientnet_b1(weights=weights).to(device)
            model.classifier = torch.nn.Sequential(
                torch.nn.Dropout(p=0.2, inplace=True),
                torch.nn.Linear(in_features=1280,
                                out_features=output_shape,  # same number of output units as our number of classes
                                bias=True)).to(device)
            model.load_state_dict(torch.load("models/efficientnet_b1.pth"))

    if "convnext" in m:
        weights = torchvision.models.ConvNeXt_Tiny_Weights.DEFAULT
        model = torchvision.models.convnext_tiny(weights=weights).to(device)
        model.classifier = model.classifier = torch.nn.Sequential(
            torchvision.models.convnext.LayerNorm2d(768),
            torch.nn.Flatten(1),
            torch.nn.Linear(in_features=768,
                            out_features=output_shape,  # same number of output units as our number of classes
                            bias=True)).to(device)
        model.load_state_dict((torch.load("models/convnext_tiny.pth")))

    import random
    num_images_to_plot = 8
    test_image_path_list = list(Path(test_dir).glob("*/*.png"))  # get list all image paths from test data
    test_image_path_sample = random.sample(population=test_image_path_list,  # go through all the test image paths
                                           k=num_images_to_plot)  # randomly select 'k' image paths to pred and plot
    for image_path in test_image_path_sample:
        predict_image_driver(model=model,
                             image_path=image_path,
                             class_names=class_names,
                             # transform=weights.transforms(), # optionally pass in a specified transform from our pretrained model weights
                             image_size=(224, 224),
                             device=device)
