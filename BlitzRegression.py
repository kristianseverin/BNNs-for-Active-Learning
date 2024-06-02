import torch
import GPUtil
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader, TensorDataset
from Utils import custom_data_loader, preprocess_data
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
from Models.simpleFFBNN import SimpleFFBNN
from Models.denseBBBRegression import DenseBBBRegression
from Models.denseRegression import DenseRegressor
from Models.BlitzRegression import BayesianRegressor
import argparse

def get_device():
    """Function to get the device to be used for training the model
    """
    cuda = torch.cuda.is_available()
    print("CUDA Available: ", cuda)

    if cuda:
        gpu = GPUtil.getFirstAvailable()
        print("GPU Available: ", gpu)
        torch.cuda.set_device(gpu)
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print("Device: ", device)
    return device

# import data
dataloader_train, dataloader_test, dataloader_val = preprocess_data(pd.read_csv('/Users/kristian/Documents/Skole/9. Semester/Thesis Preparation/Code/BNNs/Data/quality_of_food.csv'))

def evaluate_regression(regressor,
                        X,
                        y,
                        samples = 100,
                        std_multiplier = 2):
    preds = [regressor(X) for i in range(samples)]
    preds = torch.stack(preds)
    means = preds.mean(axis=0)
    stds = preds.std(axis=0)
    ci_upper = means + (std_multiplier * stds)
    ci_lower = means - (std_multiplier * stds)
    ic_acc = (ci_lower <= y) * (ci_upper >= y)
    ic_acc = ic_acc.float().mean()
    return ic_acc, (ci_upper >= y).float().mean(), (ci_lower <= y).float().mean()

device = get_device()

regressor = BayesianRegressor(4, 1).to(device)
optimizer = optim.Adam(regressor.parameters(), lr=0.01)
criterion = nn.MSELoss()

test_loss = []
val_loss = []
accuracy = []
upper_ci = []
lower_ci = []
iteration = 0
for epoch in range(1000):
    for i, (datapoints, labels) in enumerate(dataloader_train):
        optimizer.zero_grad()

        loss = regressor.sample_elbo(inputs=datapoints.to(device),
                                     labels=labels.to(device),
                                     criterion=criterion,
                                     sample_nbr=3,
                                     complexity_cost_weight=1/len(dataloader_train))

        loss.backward()
        optimizer.step()

        iteration += 1
        if iteration % 100 == 0:
            ic_acc, under_ci_upper, over_ci_lower = evaluate_regression(regressor, 
                                                                dataloader_test.dataset.tensors[0].to(device),
                                                                 dataloader_test.dataset.tensors[1].to(device),
                                                                 samples = 25,
                                                                 std_multiplier = 2)


            print("CI acc: {:.2f}, CI upper acc: {:.2f}, CI lower acc: {:.2f}".format(ic_acc, under_ci_upper, over_ci_lower))
            print("Loss: {:.4f}".format(loss))

            # save loss for plotting
            test_loss.append(loss.item())
            accuracy.append(ic_acc)
            upper_ci.append(under_ci_upper)
            lower_ci.append(over_ci_lower)


            # save to /Users/kristian/Documents/Skole/9. Semester/Thesis Preparation/Code/BNNs/ThesisPlots/LearningCurves
            np.save("/Users/kristian/Documents/Skole/9. Semester/Thesis Preparation/Code/BNNs/ThesisPlots/LearningCurves/test_loss_blitz.npy", test_loss)
            np.save("/Users/kristian/Documents/Skole/9. Semester/Thesis Preparation/Code/BNNs/ThesisPlots/LearningCurves/accuracy_blitz.npy", accuracy)
            np.save("/Users/kristian/Documents/Skole/9. Semester/Thesis Preparation/Code/BNNs/ThesisPlots/LearningCurves/upper_ci_blitz.npy", upper_ci)
            np.save("/Users/kristian/Documents/Skole/9. Semester/Thesis Preparation/Code/BNNs/ThesisPlots/LearningCurves/lower_ci_blitz.npy", lower_ci)



# test the model on the validation set
for epoch in range(1000):
    for i, (datapoints, labels) in enumerate(dataloader_val):
        loss = regressor.sample_elbo(inputs=datapoints.to(device),
                                    labels=labels.to(device),
                                    criterion=criterion,
                                    sample_nbr=3,
                                    complexity_cost_weight=1/len(dataloader_val))

    val_loss.append(loss.item())
np.save("/Users/kristian/Documents/Skole/9. Semester/Thesis Preparation/Code/BNNs/ThesisPlots/LearningCurves/val_loss_blitz.npy", val_loss)

regressor.eval()
with torch.no_grad():
    preds = regressor(dataloader_val.dataset.tensors[0].to(device))
    loss = criterion(preds, dataloader_val.dataset.tensors[1].to(device))
    print("Validation Loss: ", loss.item())
    print("R2 Score: ", r2_score(dataloader_val.dataset.tensors[1].cpu().numpy(), preds.cpu().numpy())) 

    # save predictions
    np.save("/Users/kristian/Documents/Skole/9. Semester/Thesis Preparation/Code/BNNs/ThesisPlots/preds_blitz.npy", preds.cpu().numpy())
    np.save("/Users/kristian/Documents/Skole/9. Semester/Thesis Preparation/Code/BNNs/ThesisPlots/true_blitz.npy", dataloader_val.dataset.tensors[1].cpu().numpy())
