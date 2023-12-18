import torch
from predict_images import predict_images
from transfer_model import transfer_model


def main():
    # Setup device, CUDA or CPU (I have CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # transfer_model("efficientnet_b0")
    # transfer_model("efficientnet_b1")
    # transfer_model("convnext_tiny")

    predict_images("efficientnet_b1")


if __name__ == "__main__":
    main()
