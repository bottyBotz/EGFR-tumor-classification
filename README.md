# PyTorch Lightning Project Template

This repo serves as a starter template for PyTorch Lightning projects. The example project uses a 3D CNN to classify scans as having a EFGR mutation or not.

## Placeholder - How to Install / Run
TODO: Add description


## PlaceHolder - Structure 

### src
TODO: Add description

#### config.py
TODO: Add description

#### conv3d.py
TODO: Add description

#### dataset.py
TODO: Add description


# Description of the Approach For Example Project

## Model Design/Selection

The chosen model is a 3D Convolutional Neural Network (3D CNN). These are particularly suited for 3D image data. The network consists of three convolutional layers followed by two fully connected layers. Each convolutional layer is followed by a ReLU activation and a Batch Normalization layer to aid in the training process. A max-pooling layer is also used after each batch normalization to reduce the spatial dimensions and consequently the number of parameters for computational efficiency. Dropout is used before the fully connected layers to mitigate overfitting. This vanilla architecture is a good starting point for 3D classification tasks.

# Data Splitting Mechanism
The data is split into training, validation, and test sets using a stratified shuffle split, ensuring that each set has a balanced representation of the different classes. I used 70:15:15 split for training, validation, and test sets respectively.

# Objective Function

The objective function for this binary classification task is a weighted binary cross-entropy loss. It uses normalized weights to account for the class imbalance in the dataset which were determined empirically to be [0.1966, 0.8034].

# Optimization Method

The optimization method used is the Adam optimizer. Weight Decay uses the default value, and should be tuned subsequently.

# Hyperparameters

- `Learning Rate`: 3e-4 is used. This is a popular choice for Adam optimizer. See https://twitter.com/karpathy/status/801621764144971776?lang=en for more details.
- `Batch Size`: 32. May be lowered for regularization. Small sample size + class imbalance -> used larger batch size to ensure metric stability.
- `Number of Epochs`: Determined through early stopping. I set to 100.
- `Dropout Rate`: 0.3 is used to mitigate overfitting.
- `Kernel Size`: (2, 2, 2) in all convolutional layers. Lower than 3 due to larger pooling size used.
- `Kernel Stride`: 1 for all convolutional layers, keeping the spatial dimensions intact.
- `Kernel Padding`: (1, 1, 1) ensures that the spatial dimensions are preserved after convolution.
- `Max Pooling Size`: (3, 3, 3) in all max pooling layers. Larger receptive field hopefully will see tumors.
- `Patience`: Set to 20. Would like to tune.




