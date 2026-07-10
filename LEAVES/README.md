
<div align="center">
<h1 align="center">
<img src="https://cdn-icons-png.flaticon.com/128/10172/10172835.png" width="100" />
<br>LEAVES</h1>
<h3>◦ This directory contains the example codes for the LEAVES.

LEAVES: Learning Views for Time-Series Data in Contrastive Learning

Han Yu, Huiyuan Yang, Akane Sano

Machine Learning for Healthcare 2025

https://arxiv.org/abs/2210.07340

The major contribution component is the LEAVES module with the differentiable augmentation methods, which programmed in ./models/auto_aug.py and ./utils/differentiable_augs.py, respectively.
</h3>

</div>

---

## 📖 Table of Contents
- [📖 Table of Contents](#-table-of-contents)
- [📍 Abstract](#-abstract)
- [📂 Repository Structure](#-repository-structure)
- [⚙️ Codes](#️-modules)
- [🚀 Getting Started](#-getting-started)
  - [🔧 Installation](#-installation)
  - [🤖 Running LEAVES](#-running-leaves)
---


## 📍 Abstract

The repository "LEAVES" contains code for efficient tuning learning views for various self-superivsed learning methods using algorithms such as SimCLR and BYOL in time-series data. It includes functionality for data loading, model creation, training, and evaluation. The code provides implementations of ResNet encoders, auto-augmentation techniques, and linear evaluation models. It also includes custom distribution classes, such as "MixtureSameFamily" and "StableNormal", for working with reparameterized distributions in PyTorch. 

---

## 📂 Repository Structure

```sh
└── LEAVES/
    ├── configs.py
    ├── distribution/
    │   ├── mixture_same_family.py
    │   └── stable_nromal.py
    ├── main.py
    ├── models/
    │   ├── .ipynb_checkpoints/
    │   │   └── SimCLR-checkpoint.py
    │   ├── BYOL.py
    │   ├── SimCLR.py
    │   ├── auto_aug.py
    │   ├── linear_evaluation.py
    │   ├── resnet.py
    │   ├── resnet_1d.py
    │   └── viewmaker.py
    ├── train.py
    └── utils/
        ├── augmentation.py
        ├── data_utils.py
        ├── dataset.py
        └── differentiable_augs.py

```

---


## ⚙️ Codes

<details closed><summary>Root</summary>

| File                                                                       | Summary                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| ---                                                                        | ---                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| [train.py](https://github.com/comp-well-org/LEAVES/blob/main/train.py)     | The code provided contains several functions for training various models. The `trainSimCLR` and `trainSimCLR_` functions train a model using the SimCLR algorithm, optimizing both the encoder and the view maker. It computes the encoder and view maker loss and updates the respective optimizer.The `trainBYOL` and `trainBYOL_` functions train a model using the BYOL algorithm. Similar to SimCLR, it optimizes both the encoder and the view maker. It computes the encoder and view maker loss and updates the respective optimizer.The `trainLinearEvalution` function trains a model using linear evaluation. It optimizes only the encoder and uses a BCEWithLogitsLoss criterion. It trains the model on the given training data, computes the loss, and updates the optimizer.All functions save the model periodically, log the loss values, and print evaluation metrics (such as accuracy and confusion matrix) during training. |
| [main.py](https://github.com/comp-well-org/LEAVES/blob/main/main.py)       | The code above is a script that performs training and evaluation on a deep learning model. It imports various modules and functions from different files within the directory tree. The main function creates data loaders, creates a model, and trains or evaluates the model based on the configuration settings. The script supports different frameworks like SimCLR and BYOL. The create_model function creates the model and initializes it with pre-trained weights if specified. Finally, the main function calls the appropriate training or evaluation function based on the configuration settings.                                                                                                                                                                                                                                                                                                                                    |
| [configs.py](https://github.com/comp-well-org/LEAVES/blob/main/configs.py) | The code is a configuration file that sets up various parameters for training a model using the BYOL or SimCLR framework. It includes data configurations such as file paths and the number of classes, augmentation configurations like noise and warp sigma, model configurations such as input channel size and projection size, and dual modal configurations. It also sets specific parameters for the "LEAVES" experiment, including the framework, use of leaves, number of channels, and view bounds.                                                                                                                                                                                                                                                                                                                                                                                                                       |

</details>

<details closed><summary>Distribution</summary>

| File                                                                                                            | Summary                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| ---                                                                                                             | ---                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| [mixture_same_family.py](https://github.com/comp-well-org/LEAVES/blob/main/distribution/mixture_same_family.py) | The code defines a custom distribution class called `MixtureSameFamily` that implements a mixture distribution where all components are from different parameterizations of the same distribution type. It takes in a `Categorical` distribution for selecting the components and a component distribution. The code also includes some utility functions and imports necessary libraries for working with distributions in PyTorch.                                                                         |
| [stable_nromal.py](https://github.com/comp-well-org/LEAVES/blob/main/distribution/stable_nromal.py)             | The code provides an implementation of the StableNormal distribution in PyTorch, which adds stable cumulative distribution functions (CDF) and log-CDF to the standard Normal distribution. It includes functions for ndtr (standard Gaussian CDF), log_ndtr (standard Gaussian log-CDF), and log_ndtr_series (asymptotic series expansion of the log of normal CDF). The code also includes some test code to compare the results with SciPy's ndtr implementation for both float32 and float64 data types. |

</details>

<details closed><summary>Models</summary>

| File                                                                                                  | Summary                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| ---                                                                                                   | ---                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| [viewmaker.py](https://github.com/comp-well-org/LEAVES/blob/main/models/viewmaker.py)                 | The code defines a ViewMaker class that represents a neural network used for stochastic mapping of a multichannel 2D input to an output of the same size. The network consists of convolutional layers, residual blocks, and upsampling layers. It allows for control over various parameters such as the number of channels, distortion budget, activation function, clamping of outputs, frequency domain perturbation, downsampling, and number of residual blocks. The network's forward method applies the necessary transformations and returns the output. Key components of the network include ConvLayer, ResidualBlock, and UpsampleConvLayer.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| [linear_evaluation.py](https://github.com/comp-well-org/LEAVES/blob/main/models/linear_evaluation.py) | The code defines a class called LinearEvaResNet, which is a neural network model for linear evaluation using a ResNet encoder. The model takes as input an image and passes it through the encoder, which consists of a series of convolutional layers. The output of the encoder is then flattened and passed through fully connected layers to produce the final classification output. The model also includes dropout regularization.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| [BYOL.py](https://github.com/comp-well-org/LEAVES/blob/main/models/BYOL.py)                           | The code defines the BYOL (Bootstrap Your Own Latent) model, which is a self-supervised learning algorithm for representation learning. The BYOL model consists of an encoder network that maps input data to a latent space, a predictor network that predicts features from the encoded data, and an exponential moving average (EMA) mechanism for target network updates. The main functionality of the code includes:-Helper functions for default values, flattening tensors, caching, gradient requirements, and loss calculation-A class for random augmentation, implementing augmentation with a given probability-A class for exponential moving average (EMA), used to update the target encoder network-MLP (Multi-Layer Perceptron) architectures for the projector and predictor networks-The main BYOL class, which initializes the encoder, encoder target, predictor, and other parameters-Methods for creating the encoder, target encoder, and resetting/updating the target encoder-The forward method, which performs the forward pass of the BYOL model, including encoding, projection, and loss calculation                                                                                                           |
| [resnet_1d.py](https://github.com/comp-well-org/LEAVES/blob/main/models/resnet_1d.py)                 | The code represents a ResNet model architecture for 1D signal data. It includes the definition of a BasicBlock and two ResNet models: model_ResNet and model_ResNet_dualmodal. The BasicBlock is a building block for the ResNet models, consisting of convolutional layers, batch normalization, and residual connections. The model_ResNet is a single-modality ResNet model, while the model_ResNet_dualmodal is a dual-modality ResNet model that takes input from two different channels and concatenates them. Both models have forward methods to process input data and produce output.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| [auto_aug.py](https://github.com/comp-well-org/LEAVES/blob/main/models/auto_aug.py)                   | The code represents a module for auto augmentation, which is a technique used in training neural networks. The module includes functions for various types of data augmentation, such as jitter, scaling, rotation, time distortion, permutation, magnitude warp, and frequency depression. These augmentations are applied to the input data to enhance the performance and robustness of the neural network. The module also includes an attention mechanism for focusing on important features during augmentation.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| [resnet.py](https://github.com/comp-well-org/LEAVES/blob/main/models/resnet.py)                       | The code represents a ResNet encoder implementation in PyTorch, specifically for 1D convolutional neural networks. It includes custom implementation of the `MyConv1dPadSame` and `MyMaxPool1dPadSame` classes to support "SAME" padding. These classes extend the corresponding PyTorch classes to provide consistent padding behavior.The `BasicBlock` class represents a basic block of the ResNet architecture, consisting of two convolutional layers with batch normalization, rectified linear unit (ReLU) activation, and optional dropout. The block performs residual connections and downsampling if specified.The `ResNetEncoder` class implements the ResNet encoder architecture by stacking the basic blocks. It starts with a first block that applies a convolutional layer, followed by a specified number of residual blocks. The output of the final residual block is passed through average pooling before being fed into a final batch normalization layer and ReLU activation. The architecture also includes an adaptive average pooling layer to ensure consistent output dimensions.Overall, the code provides a flexible and customizable implementation of a ResNet encoder for 1D convolutional neural networks. |
| [SimCLR.py](https://github.com/comp-well-org/LEAVES/blob/main/models/SimCLR.py)                       | The code defines a SimCLR model for contrastive learning. It contains several classes and functions related to the SimCLR objective and loss calculation. The main SimCLR class has a forward method that takes in two sets of input data and outputs the embeddings for each set. It uses a ResNet encoder and a fully connected layer to generate the embeddings. The contrastive loss is calculated based on the embeddings using the l2_normalize function and other operations. The model also includes functionality for handling dual-modal inputs and using a viewmaker for data augmentation.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |

</details>



<details closed><summary>Utils</summary>

| File                                                                                                     | Summary                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| ---                                                                                                      | ---                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| [augmentation.py](https://github.com/comp-well-org/LEAVES/blob/main/utils/augmentation.py)               | The code in `augmentation.py` provides functions for data augmentation, specifically for time series data. The implemented functions include `jitter`, which adds random noise to the data, `scaling`, which scales the data by a random factor, `rotation`, which randomly rotates the features of the data, and `permutation`, which randomly splits the data into segments. These functions aim to increase the diversity of the dataset and improve the robustness of machine learning models trained on the data.                                     |
| [data_utils.py](https://github.com/comp-well-org/LEAVES/blob/main/utils/data_utils.py)                   | The code in `utils/data_utils.py` provides functions for data preprocessing and conversion. It includes a function `normalize_data()` that normalizes a given dataset, `Catergorical2OneHotCoding()` which converts categorical data to one-hot encoding, `Logits2Binary()` which applies sigmoid function and returns the index of the maximum value, `logits_2_multi_label()` which converts logits to multi-label predictions, and `test()` which demonstrates the usage of these functions.                                                            |
| [dataset.py](https://github.com/comp-well-org/LEAVES/blob/main/utils/dataset.py)                         | The code defines several classes that extend the `torch.utils.data.Dataset` class to handle different types of datasets. The datasets include `TransDataset`, `SleepEDFE_Dataset`, `SemiSupDatasetSMILE`, and `SupervisedDataset`. Each dataset class has its own `__init__`, `__len__`, and `__getitem__` methods to load, preprocess, and return the data. These datasets are designed for tasks such as data transformation, sleep electroencephalography (EEG) signal classification, semi-supervised learning, and supervised learning.               |
| [differentiable_augs.py](https://github.com/comp-well-org/LEAVES/blob/main/utils/differentiable_augs.py) | The code in `utils/differentiable_augs.py` provides functions for various data augmentation techniques. These include jittering, scaling, rotation, time distortion, permutation, magnitude warping, and frequency depression. These techniques can be used to augment data for tasks such as image classification or time series analysis. The code also includes custom autograd functions for differentiable rounding and converting tensors to floats. Overall, this code provides a set of functions for differentiable data augmentation operations. |

</details>

---

## 🚀 Getting Started

### 🔧 Installation

1. Clone the LEAVES repository:
```sh
git clone https://github.com/comp-well-org/LEAVES
```

2. Change to the project directory:
```sh
cd LEAVES
```

3. Install the dependencies:

The python environment used in this work can be found in environment.yml
### 🤖 Running LEAVES

Config your training in the ```configs.py ``` file including the data config and LEAVES config.

```sh
python main.py
```

4. Datasets:

The datasets used in this study:
- Apnea-ECG: https://physionet.org/content/apnea-ecg/1.0.0/
- Sleep-EDFE: https://www.physionet.org/content/sleep-edfx/1.0.0/
- PAMAP2: https://archive.ics.uci.edu/ml/datasets/pamap2+physical+activity+monitoring
- PTB-XL: https://physionet.org/content/ptb-xl/1.0.2/

## License
LEAVES is released under the GPL 3.0. See [LICENSE](https://github.com/comp-well-org/LEAVES/blob/main/LICENSE.txt) for details.

