# Coffee-Roast-Classification
Image classifier that will determine the roast level of a coffee bean (Unroasted, Light, Medium, Dark). 

To run, download the repository to local system and install dependencies. Optionally, download pretrained models in the "models/" directory.

# 1. Training
Trainable models include EfficientNet and ConvNeXt models.
The transfer_model() script provides the method to train a model, and takes a string as an input.

**Example:**

To train the EfficientNet-B0 model, use:

```transfer_model("efficientnet_b0")```

To train the ConvNeXt_Tiny model, use:

```transfer_model("convnext_tiny")```

The trained models will be saved to the "models/" directory as a .pth file

# 2. Classification

To classify images from the test set, use the predict_images() script.

**Example:**

To predict images using the EfficientNet-B1 model (assuming it was trained), use:

```predict_images("efficientnet_b1")```

A figure with an image and a probability score will appear. To continue viewing the test images, close the current figure, and the next figure will show.

Here is an example of what an image classification will look like:

![medium1](https://github.com/gariasmarin/Coffee-Roast-Classification/assets/58484718/d3551b23-1f83-4b9d-908c-d32da1f3a129)




# Future work
Current work is to implement using custom images with the classifier.

The main goal of this classifier is to use it as the label classifier for an object detection model. 
This object detection model will be used to determine how accurate the roast level of coffee beans are compared to specified roast level on the sold bag.

About 10 to 20 coffee beans will be laid out in an image, and the model will use bounding boxes to separate each bean and classify its roast level.
This process should give a good look into how truthful or accurate common coffee roasters are when labeling a product, showing which brands are worth buying for those who care about the quality of their coffee.
