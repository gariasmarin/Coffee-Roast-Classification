from typing import List, Tuple

import matplotlib.pyplot as plt
from PIL import Image
import torch
import torchvision
from torchvision import transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def predict_image_driver(model: torch.nn.Module,
                  image_path: str,
                  class_names: List[str],
                  image_size: Tuple[int, int] = (224, 224),
                  transform: torchvision.transforms = None,
                  device: torch.device = device):
    # open the image
    img = Image.open(image_path)

    # transform the image for model
    if transform is not None:
        image_transform = transform
    else:  # perform a manual transform (model training performs an auto transform)
        image_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],  # values come from ImageNet means and std
                                 std=[0.229, 0.224, 0.225])
        ])

        ## Begin predict on the image ##

        model.to(device)

        model.eval()
        with (torch.inference_mode()):
            # perform the transform on the image
            transformed_image = image_transform(img).unsqueeze(dim=0)

            # make the prediction with the model
            target_image_pred = model(transformed_image.to(device))

            # find prediction probabilities using softmax() for multi-class classification
            target_image_pred_probs = torch.softmax(target_image_pred, dim=1)

            # convert probabilities to labels
            target_image_pred_label = torch.argmax(target_image_pred_probs, dim=1)

            # plot the image with predicted label
            plt.figure()
            plt.imshow(img)
            plt.title(f"Pred: {class_names[target_image_pred_label]} | Prob: {target_image_pred_probs.max():.3f}")
            plt.axis(False)
            plt.show()
