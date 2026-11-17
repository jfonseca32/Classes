import os
import sys

# add the project root to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
from controllers.perception_controller.CONSTANTS import TRAINED_MODEL_PATH
from controllers.perception_controller.networks import ImageClassifier, LeNet


def test_lenet_model_architecture():
    """
    Test the architecture of the LeNet model.
    """
    model = LeNet()
    input = torch.randn(1, 1, 32, 32)
    output = model(input)
    if output.shape != (1, 10):
        print(f"Output shape incorrect: {output.shape}")
        return False
    print(f"Output shape: {output.shape}")

    # Check for convolutional layers
    conv_layers = [module for module in model.modules() if isinstance(module, torch.nn.Conv2d)]
    relu_layers = [module for module in model.modules() if isinstance(module, torch.nn.ReLU)]
    avgpool_layers = [
        module for module in model.modules() if isinstance(module, torch.nn.AvgPool2d)
    ]
    linear_layers = [module for module in model.modules() if isinstance(module, torch.nn.Linear)]

    if len(conv_layers) != 3:
        print(f"Expected 3 conv layers, got {len(conv_layers)}")
        return False
    if len(linear_layers) != 2:
        print(f"Expected 2 linear layers, got {len(linear_layers)}")
        return False

    # Will not check avg pool and ReLU layers b/c you can define one and reuse it in forward pass.

    print(
        "Layer counts - conv:",
        len(conv_layers),
        "relu:",
        len(relu_layers),
        "avgpool:",
        len(avgpool_layers),
        "linear:",
        len(linear_layers),
    )

    return True


def test_image_classifier_preprocess_image():
    """
    Test the preprocess_image method of the ImageClassifier class.
    """
    image_classifier = ImageClassifier(model_path=TRAINED_MODEL_PATH)
    image = np.random.randint(0, 255, (32, 32, 3))
    preprocessed_image = image_classifier.preprocess_image(image)
    if preprocessed_image.shape != (1, 1, 32, 32):
        print(f"Preprocessed image shape incorrect: {preprocessed_image.shape}")
        return False
    print(f"Preprocessed image shape: {preprocessed_image.shape}")

    return True


def test_image_classifier_class_to_character():
    """
    Test the class_to_character method of the ImageClassifier class.
    """
    image_classifier = ImageClassifier(model_path=TRAINED_MODEL_PATH)
    class_index = np.random.randint(0, 10)
    character = image_classifier.class_to_character(class_index)
    if character is None:
        print("Character is None")
        return False
    print(f"Character: {character} for class index: {class_index}")
    return True


def test_image_classifier_single_inference():
    """
    Test the single_inference method of the ImageClassifier class.
    """

    image_classifier = ImageClassifier(model_path=TRAINED_MODEL_PATH)
    image = np.random.randint(0, 255, (32, 32, 3))  # run on random image
    predicted_class_index = image_classifier.single_inference(image)
    if predicted_class_index is None:
        print("Prediction failed.  Predicted class index is None")
        return False
    elif predicted_class_index < 0 or predicted_class_index > 9:
        print(f"Prediction failed.  Predicted class index is out of range: {predicted_class_index}")
        return False
    else:
        print(f"Prediction successful.  Predicted class index: {predicted_class_index}")

    return True
