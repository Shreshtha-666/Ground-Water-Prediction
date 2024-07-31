# Ground-Water-Prediction
# Groundwater Prediction using ANN

This project aims to predict groundwater availability using an Artificial Neural Network (ANN) model. The dataset contains various features related to groundwater availability, and the model is built using TensorFlow and Keras.

## Table of Contents

- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Dataset

The dataset used for this project is `Dynamic_2017_2_0.csv`. It contains various features relevant to groundwater availability, such as geographical and environmental factors.

## Installation

To run this project, you need to have Python installed on your machine. Additionally, you need to install the following libraries:

- pandas
- scikit-learn
- tensorflow
- matplotlib


pip install pandas scikit-learn tensorflow matplotlib

##Usage
Clone the repository:
git clone https://github.com/your-username/groundwater-prediction.git
cd groundwater-prediction

Place the dataset:
Ensure that the 'Dynamic_2017_2_0.csv' file is in the same directory as the script.

Run the script:
python groundwater_prediction.py
The script will load the data, preprocess it, build, train, and evaluate the ANN model. It will also plot the training and validation loss and mean absolute error graphs.

##Model Architecture
The ANN model used in this project has the following architecture:

Input layer: Number of neurons equal to the number of features
Hidden layer 1: 64 neurons, ReLU activation
Hidden layer 2: 32 neurons, ReLU activation
Output layer: 1 neuron (for the regression output)
The model is compiled using the Adam optimizer and mean squared error loss function. Early stopping is used to prevent overfitting.

##Results
The model's performance is evaluated using the following metrics:

Mean Squared Error (MSE)
Mean Absolute Error (MAE)
R-squared Score (R²)
The script will print these metrics and plot the training and validation loss and MAE over the epochs.


##Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

##License
This project is licensed under the MIT License. See the LICENSE file for more details.

### Notes:
1. **Repository URL**: Update the repository URL in the "Clone the repository" section with your actual GitHub repository link.
2. **License**: Update the license section if you use a different license.
