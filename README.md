# Bayesian Neural Networks for Active and Continual Learning

## Table of Contents
- [Introduction](#introduction)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Acknowledgements](#acknowledgements)

## Introduction
This repository contains code for implementing Bayesian Neural Networks (BNNs) for active and continual learning tasks. The code is organized into different modules, each focusing on specific functionalities required for these tasks.

## Project Structure

- **Data/**: Directory containing datasets used for training and testing. Here the data generating scripts are also found. Note, no data from chapter 4 is made publicly available 
- **Models/**: Directory containing different Bayesian neural network models.
- **NeuralLayers/**: Custom neural network layers. Here the closed form derived from variational dropout by Rakesh & Jain (2021) is found.
- **ThesisPlots/**: Scripts and notebooks to generate plots for the thesis. The .png files used in the thesis are also contained here.
- **Utils/**: Utility functions and helpers. Among other dataloader scripts are found here.
- **activeLearningClassification.py**: Script for active learning classification tasks. This is the script all the classification models in chapter 2 and 3 of the thesis are implemented with.
- **howToRunInstances.py**: Instructions on how to run different instances. This is the regression equivalent to "activeLearningClassification.py".
- **runALClassification.py**: Script to run active learning classification. This should be removed as it is an old file
- **runBNNClassification.py**: Script to run BNN classification. This only runs a single instance of the model. Code outputs are reported in appendix.
- **runBNNRegression.py**: Script to run BNN regression. This only runs a single instance of the model. Code outputs are reported in appendix.
- **timeseriesClassification.py**: Script for time series classification. This is the script used to classify WAI scores based on dyadic HRV Synchrony
- **timeseriesRegression.py**: Script for time series regression. This is the script that tests timeseries correlations in patient therapist dyads.

## Installation
To set up the environment and install the necessary dependencies, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/kristianseverin/BNNs-for-Active-Learning.git
   cd BNNs-for-Active-Learning
   pip install -r requirements.txt
   ```

## Usage
All the models can be run from terminal by calling the belonging script, e.g., python3 activeLearningClassification.py
All hyperparameters should be set manually, but defaults are provided. 

## Acknowledgements
The repo and the code owes a great deal to the publicly available github repo of Rakesh & Jain (2021): Rakesh, V., & Jain, S. (2021). Efficacy of Bayesian Neural Networks in Active
Learning, 2601–2609. https://doi.org/10.1109/CVPRW53098.2021.
00294
