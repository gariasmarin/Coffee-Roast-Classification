import torch
import torchinfo
import torchvision
from torch import nn
from torchvision.models.convnext import LayerNorm2d
from functools import partial
from torchvision import transforms
from pathlib import Path

import data_setup
import engine
from plot_curves import plot_loss_curves
from save_model import save_model


def transfer_model(m=str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("number of threads used:", torch.get_num_threads())
    # setup paths and directories
    data_path = Path("coffee-bean-dataset/")
    image_path = data_path / "roast_levels"
    train_dir = image_path / "train"
    test_dir = image_path / "test"

    # first test with EfficientNet B0
    # weights = models.EfficientNet_B0_Weights.DEFAULT
    # get transforms to create pretrained weights
    # auto_transforms = weights.transforms()
    manual_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        # 1. Reshape all images to 224x224 (though some models may require different sizes)
        transforms.ToTensor(),  # 2. Turn image values to between 0 & 1
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             # 3. A mean of [0.485, 0.456, 0.406] (across each colour channel)
                             std=[0.229, 0.224, 0.225])
        # 4. A standard deviation of [0.229, 0.224, 0.225] (across each colour channel),
    ])

    # Create training and testing DataLoaders as well as get a list of class names
    train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(train_dir=str(train_dir),
                                                                                   test_dir=str(test_dir),
                                                                                   transform=manual_transforms,
                                                                                   # perform same data transforms on our own data as the pretrained model
                                                                                   batch_size=32)  # set mini-batch size to 32

    print("class names:", class_names)

    # Instantiate model. We are using EfficientNet_B0 for the first run
    model = torchvision.models.get_model(m, weights="DEFAULT")


    # Must freeze all base feature layers of model. Object is to only train last layer using Transfer Learning
    for param in model.features.parameters():
        param.requires_grad = False

    # Get the length of class_names (one output unit for each class)
    output_shape = len(class_names)

    # Recreate the classifier layer and seed it to the target device
    # from summary(), we see classifier layer of model is 2 layers: Dropout and Linear
    if "efficientnet" in m:
        model.classifier = torch.nn.Sequential(
            torch.nn.Dropout(p=0.2, inplace=True),
            torch.nn.Linear(in_features=1280,
                            out_features=output_shape,  # same number of output units as our number of classes
                            bias=True)).to(device)
    if "convnext" in m:
        model.classifier = torch.nn.Sequential(
            LayerNorm2d(768),
            torch.nn.Flatten(1),
            torch.nn.Linear(in_features=768,
                            out_features=output_shape,  # same number of output units as our number of classes
                            bias=True)).to(device)

    torchinfo.summary(model=model,
                      input_size=(32, 3, 224, 224),
                      col_names=["input_size", "output_size", "num_params", "trainable"],
                      col_width=20,
                      row_settings=["var_names"]
                      )

    # Define loss and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Set the manual seeds
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    # Start the timer
    from timeit import default_timer as timer
    start_time = timer()

    # Setup training and save the results
    results = engine.train(model=model,
                           train_dataloader=train_dataloader,
                           test_dataloader=test_dataloader,
                           optimizer=optimizer,
                           loss_fn=loss_fn,
                           epochs=10,
                           device=device)
    plot_loss_curves(results, m)

    # End the timer and print out how long it took
    end_time = timer()
    print(f"[INFO] Total training time: {end_time - start_time:.3f} seconds")

    save_model(model=model,
               target_dir="models",
               model_name=(str(m) + ".pth")
               )
