import torch
from torch.distributions import Categorical
import pandas as pd
import numpy as np
import GPUtil
import argparse
import matplotlib.pyplot as plt
from Utils import custom_data_loader_classification, preprocess_classification_data, preprocess_classification_activeL_data
from Utils import preprocess_classification_data
from Models.largeFFBNNClassification import LargeFFBNNClassification
from Models.simpleFFBNNClassification import SimpleFFBNNClassification
from runBNNClassification import get_device, arg_inputs, custom_data_loader_classification, runBNNClassification
from sklearn.model_selection import train_test_split


def splitActiveData(df):
    '''This function splits the data into a training set and an active set. 
       The training set is used to train the seed model and the active set is used in the active learning process.
       The function saves the datasets to csv files. The active df does not contain the target values.
       
       Args:
       df (pd.DataFrame): The dataframe with the data.
       '''
    df, df_active = train_test_split(df, test_size=0.9, random_state=42)
    df.to_csv('/Users/kristian/Documents/Skole/9. Semester/Thesis Preparation/Code/BNNs/Activelearning/Data/df.csv', index = False)
    # remove the target values from the active data
    df_active = df_active.drop(columns = ['target'])
    df_active.to_csv('/Users/kristian/Documents/Skole/9. Semester/Thesis Preparation/Code/BNNs/Activelearning/Data/df_active.csv', index = False)
    return df

def trainSeedModel():
    '''This function trains the seed model on the data that is not used in the active learning process. 
       The seed model is used to predict the target values of the samples with the highest uncertainty.'''
    
    # read data
    df = splitActiveData(pd.read_csv('/Users/kristian/Documents/Skole/9. Semester/Thesis Preparation/Code/BNNs/Data/quality_of_food_int.csv'))

    # preprocess data
    dataloader_train, dataloader_test, dataloader_val = preprocess_classification_data(df)

    # get the device
    device = get_device()

    # define the model
    model = SimpleFFBNNClassification(4, 5)
    run = runBNNClassification(model, dataloader_train, dataloader_test, dataloader_val, device, 450, 0.0001, torch.nn.CrossEntropyLoss(), torch.optim.SGD, False)

    # train the model
    test_loss_seed, val_loss_seed, accuracy_seed = run.train()

    # visualize the loss
    run.visualizeLoss()

    # save the model
    torch.save(model, '/Users/kristian/Documents/Skole/9. Semester/Thesis Preparation/Code/BNNs/Activelearning/SeedModels/simple_model.pth')

    return test_loss_seed, val_loss_seed, accuracy_seed

def activeLearning(max_rounds = 10):
    '''This function implements the active learning process. The function uses the seed model to predict the target values of the samples with the highest uncertainty.
       The function then simulates the target values for the samples with the highest uncertainty using the same logic that was used, when the data was generated in the first place AKA the oracle.
       The function then adds the samples with the highest uncertainty to the training set and retrains the model. The process is repeated until the model has been trained on all the data.

       TOdo: use train the model for x rounds and then evaluate the model on the val set. If the model performs better when the samples with the highest uncertainty are added to the training set, then save the best model (measured by best accuracy).
        repeat the process for desired number of rounds. Save the model with the highest accuracy on the val set. 
       '''
        
    # if model exists, load it, else train it
    try:
        model = torch.load('/Users/kristian/Documents/Skole/9. Semester/Thesis Preparation/Code/BNNs/Activelearning/SeedModels/simple_model.pth')
    except:
        test_loss_seed, val_loss_seed, accuracy_seed = trainSeedModel()
    
    # read data
    df = pd.read_csv('/Users/kristian/Documents/Skole/9. Semester/Thesis Preparation/Code/BNNs/Activelearning/Data/df_active.csv')

    # get the device
    device = get_device()

    # preprocess data
    dataloader_train, dataloader_test = preprocess_classification_activeL_data(df)
    
    # learning curves for the active learning process are saved in these lists for visualization
    accuracy_curves = []
    train_loss_curves = []
    test_loss_curves = []
    # loop through the active learning process (process is stopped if the new model is not better than the seed model)
    for r in range(max_rounds):
        print(f'Round: {r}')

        # if a simple_model_best.pth exists, load it, else load the simple_model.pth
        try :
            model = torch.load('/Users/kristian/Documents/Skole/9. Semester/Thesis Preparation/Code/BNNs/Activelearning/SeedModels/simple_model_best.pth')
        except:
            model = torch.load('/Users/kristian/Documents/Skole/9. Semester/Thesis Preparation/Code/BNNs/Activelearning/SeedModels/simple_model.pth')

        # use the model to predict the target values
        model.eval()
        predictions = []
        uncertainty = []
        with torch.no_grad():
            for X, y in dataloader_train:
                X, y = X.to(device), y.to(device)
                outputs = model(X)
                _, predicted = torch.max(outputs, 1)
                predictions.append(predicted)
                uncertainty.append(outputs)

        # get the uncertainty
        for i in range(len(uncertainty)):
            uncertainty[i] = torch.nn.functional.softmax(uncertainty[i], dim=1)
        
        def acquisitionFunction(uncertainty):
            '''This function calculates the entropy and variation ratio of the model predictions.
            More uncertainty means higher entropy and variation ratio. More acquisition functions can be added.

            Args:
            uncertainty (list): A list of tensors with the model predictions as probabilities.
            '''
            uncertainty = torch.cat(uncertainty, dim=0)
            entropy = Categorical(probs=uncertainty).entropy()

            #entropy = uncertainty.entropy()
            variation_ratio = 1 - torch.max(uncertainty, dim=1).values
            return entropy, variation_ratio
            
        # get the uncertainty of the model predictions using the acquisition function    
        entropy, variation_ratio = acquisitionFunction(uncertainty)

        # get the indices of the samples with the highest uncertainty
        n = 100
        indices_entropy = np.argsort(entropy)[:n]
        indices_variation_ratio = np.argsort(variation_ratio)[:n]

        try:
            df = pd.read_csv('/Users/kristian/Documents/Skole/9. Semester/Thesis Preparation/Code/BNNs/Activelearning/Data/df_new_active.csv')
        except:
            df = pd.read_csv('/Users/kristian/Documents/Skole/9. Semester/Thesis Preparation/Code/BNNs/Activelearning/Data/df_active.csv')

        # get the samples with the highest uncertainty
        df_entropy = df.iloc[indices_entropy]
        df_variation_ratio = df.iloc[indices_variation_ratio]

        def simulateOracle(df):
            '''This function simulates the target values for the samples with the highest uncertainty using the same logic that was used, when the data was generated in the first place.
            The function is used to simulate the oracle/human expert in the active learning process.

            Args:
            df (pd.DataFrame): The dataframe with the samples with the highest uncertainty.

            '''

            # get the features
            income = df['monthly_income']
            time = df['time_of_month']
            savings = df['savings']
            guests = df['guests']

            quality_based_on_income = np.where(
                income >= 7000, 5, 
                np.where(income >= 4000, 4, 
                np.where(income >= 3000, 3, 
                np.where(income >= 2000, 2, 1)))
                )

            # Determine quality based on time of month
            quality_based_on_time = np.where(time >= 16, 1, 5)


            # Determine quality based on size of savings
            quality_based_on_savings = np.where(savings == 'high', 5, np.where(savings == 'medium', 3, 1))

            # Determine quality based on number of guests
            quality_based_on_guests = np.where(
                guests == 0, 3, 5
            )

            # Calculate the overall quality of the food exactly like the data was generated in the first place but no noise
            quality_of_food = (quality_based_on_income * 0.4 + quality_based_on_time * 0.1 + quality_based_on_savings * 0.2 + quality_based_on_guests * 0.3) / 1

            # add the target to the dataframe
            df['target'] = quality_of_food.astype(int)

            # make the target 0-indexed
            df['target'] = df['target'] - 1
            
            return df

        # simulate the target values for the samples with the highest uncertainty
        df_entropy = simulateOracle(df_entropy)

        # add the samples with the highest uncertainty to the training set
        df = pd.read_csv('/Users/kristian/Documents/Skole/9. Semester/Thesis Preparation/Code/BNNs/Activelearning/Data/df.csv')
        df = pd.concat([df, df_entropy], ignore_index=True)

        # save the new training set
        df.to_csv('/Users/kristian/Documents/Skole/9. Semester/Thesis Preparation/Code/BNNs/Activelearning/Data/df.csv', index = False)

        # save the new active set without the 100 samples with the highest uncertainty
        try:
            df_new_active = pd.read_csv('/Users/kristian/Documents/Skole/9. Semester/Thesis Preparation/Code/BNNs/Activelearning/Data/df_new_active.csv')
        except:
            df_new_active = pd.read_csv('/Users/kristian/Documents/Skole/9. Semester/Thesis Preparation/Code/BNNs/Activelearning/Data/df_active.csv')
        
        # remove the samples with the highest uncertainty from the active set because they have been added to the training set
        df_new_active_iloc = df_new_active.iloc[indices_entropy]
        df_new_active = df_new_active.drop(df_new_active_iloc.index)
        df_new_active.to_csv('/Users/kristian/Documents/Skole/9. Semester/Thesis Preparation/Code/BNNs/Activelearning/Data/df_new_active.csv', index = False)

        def trainNewModel():
            '''This function trains the model on the new training set that includes the samples with the highest uncertainty. 
            it also compares the accuracy of the seed model with the accuracy of the model after the active learning process.'''

            try:
                model = torch.load('/Users/kristian/Documents/Skole/9. Semester/Thesis Preparation/Code/BNNs/Activelearning/SeedModels/simple_model_best.pth')
            except:
                model = torch.load('/Users/kristian/Documents/Skole/9. Semester/Thesis Preparation/Code/BNNs/Activelearning/SeedModels/simple_model.pth')

            # preprocess data
            dataloader_train_new, dataloader_test_new, dataloader_val_new = preprocess_classification_data(df)

            # finetune the model with the new data
            run = runBNNClassification(model, dataloader_train_new, dataloader_test_new, dataloader_val_new, device, 450, 0.0001, torch.nn.CrossEntropyLoss(), torch.optim.SGD, False)

            # train the model
            test_loss_new, val_loss_new, accuracy_new = run.train()

            # visualize the loss
            run.visualizeLoss()

            # save the model
            torch.save(model, '/Users/kristian/Documents/Skole/9. Semester/Thesis Preparation/Code/BNNs/Activelearning/SeedModels/simple_model_AL.pth')

            return test_loss_new, val_loss_new, accuracy_new

            #print(f'Accuracy new model: {accuracy_new[-1]}')

        # train the new model after the active learning process and append the learning curves to lists
        test_loss_new, val_loss_new, accuracy_new = trainNewModel()
        accuracy_curves.append(accuracy_new)
        print(f'Accuracy Curves: {accuracy_curves}')
        train_loss_curves.append(test_loss_new)
        test_loss_curves.append(val_loss_new)

        def compareModels():
            '''This function compares the accuracy of the seed model with the accuracy of the model after the active learning process.'''

            try:
                model_seed = torch.load('/Users/kristian/Documents/Skole/9. Semester/Thesis Preparation/Code/BNNs/Activelearning/SeedModels/simple_model_best.pth')
            except:
                model_seed = torch.load('/Users/kristian/Documents/Skole/9. Semester/Thesis Preparation/Code/BNNs/Activelearning/SeedModels/simple_model.pth')

            # load the new model
            model_new = torch.load('/Users/kristian/Documents/Skole/9. Semester/Thesis Preparation/Code/BNNs/Activelearning/SeedModels/simple_model_AL.pth')

            # read data
            df = pd.read_csv('/Users/kristian/Documents/Skole/9. Semester/Thesis Preparation/Code/BNNs/Activelearning/Data/df.csv')

            # get the accuracy of the seed model
            model_seed.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for X, y in dataloader_test:
                    X, y = X.to(device), y.to(device)
                    outputs = model_seed(X)
                    _, predicted = torch.max(outputs, 1)
                    total += y.size(0)
                    correct += (predicted == y).sum().item()
            accuracy_seed = 100 * correct / total

            # get the accuracy of the new model
            model_new.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for X, y in dataloader_test:
                    X, y = X.to(device), y.to(device)
                    outputs = model_new(X)
                    _, predicted = torch.max(outputs, 1)
                    total += y.size(0)
                    correct += (predicted == y).sum().item()
            accuracy_new = 100 * correct / total

            print(f'Accuracy seed model: {accuracy_seed}')
            print(f'Accuracy new model: {accuracy_new}')

            if accuracy_new > accuracy_seed:
                print('The new model is better than the seed model.')
                torch.save(model_new, '/Users/kristian/Documents/Skole/9. Semester/Thesis Preparation/Code/BNNs/Activelearning/SeedModels/simple_model_best.pth')

            return accuracy_new, accuracy_seed

        # compare the models
        accuracy_new[r], accuracy_seed[r] = compareModels()

        # if the new model is not better than the seed twice in a row, break the loop
        if accuracy_new[r] < accuracy_seed[r] and accuracy_new[r-1] < accuracy_seed[r-1]:
            print('The new model has not improved the seed model two rounds in a row, so the active learning process is stopped.')
            break

    return accuracy_curves, train_loss_curves, test_loss_curves, accuracy_seed
        
def main():
    args = arg_inputs()
    device = get_device()

    # train the model on the new data and get the learning curves
    accuracy_curves, train_loss_curves, test_loss_curves, accuracy_curve_seed = activeLearning()

    # save the learning curves
    np.save('/Users/kristian/Documents/Skole/9. Semester/Thesis Preparation/Code/BNNs/ThesisPlots/LearningCurves/accuracy_curves.npy', accuracy_curves)
    np.save('/Users/kristian/Documents/Skole/9. Semester/Thesis Preparation/Code/BNNs/ThesisPlots/LearningCurves/train_loss_curves.npy', train_loss_curves)
    np.save('/Users/kristian/Documents/Skole/9. Semester/Thesis Preparation/Code/BNNs/ThesisPlots/LearningCurves/test_loss_curves.npy', test_loss_curves)
    np.save('/Users/kristian/Documents/Skole/9. Semester/Thesis Preparation/Code/BNNs/ThesisPlots/LearningCurves/accuracy_curve_seed.npy', accuracy_curve_seed)

    # these are the arguments that are used to train the model
    print(f'args: {args}')

if __name__ == '__main__':
    main()
    
# run from terminal with: python runALClassification.py (--model SimpleFFBNNClassification --epochs 150 --lr 0.0001 --criterion CrossEntropyLoss --optimizer SGD --savemodel False)
