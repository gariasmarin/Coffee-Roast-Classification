import matplotlib.pyplot as plt
import torch
import torchinfo
import torchvision
import torchvision.models as models
from torch import nn
from torchvision import transforms
from pathlib import Path

import data_setup
import engine


def plot_loss_curves(results):
    """Plots training curves of a results dictionary.

    Args:
        results (dict): dictionary containing list of values, e.g.
            {"train_loss": [...],
             "train_acc": [...],
             "test_loss": [...],
             "test_acc": [...]}
    """

    # Get the loss values of the results dictionary (training and test)
    loss = results['train_loss']
    test_loss = results['test_loss']

    # Get the accuracy values of the results dictionary (training and test)
    accuracy = results['train_acc']
    test_accuracy = results['test_acc']

    # Figure out how many epochs there were
    epochs = range(len(results['train_loss']))

    # Setup a plot
    plt.figure(figsize=(15, 7))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label='train_loss')
    plt.plot(epochs, test_loss, label='test_loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.legend()


    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, label='train_accuracy')
    plt.plot(epochs, test_accuracy, label='test_accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.legend()
    plt.show();


def main():
    # Setup device, CUDA or CPU (I have CPU)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("number of threads used:", torch.get_num_threads())
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
    train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(train_dir=str(train_dir),
                                                                                   test_dir=str(test_dir),
                                                                                   transform=auto_transforms,
                                                                                   # perform same data transforms on our own data as the pretrained model
                                                                                   batch_size=32)  # set mini-batch size to 32

    print("class names:", class_names)
    print("train_dataloader:", train_dataloader)
    print("test_dataloader:", test_dataloader)

    # Instantiate model. We are using EfficientNet_B0 for the first run
    model = models.efficientnet_b0(weights=weights).to(device)

    # Must freeze all base feature layers of model. Object is to only train last layer using Transfer Learning
    for param in model.features.parameters():
        param.requires_grad = False

    # Get the length of class_names (one output unit for each class)
    output_shape = len(class_names)

    # Recreate the classifier layer and seed it to the target device
    # from summary(), we see classifier layer of model is 2 layers: Dropout and Linear
    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(p=0.2, inplace=True),
        torch.nn.Linear(in_features=1280,
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
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0008)

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

    # End the timer and print out how long it took
    end_time = timer()
    print(f"[INFO] Total training time: {end_time - start_time:.3f} seconds")

    plot_loss_curves(results)


if __name__ == "__main__":
    main()
