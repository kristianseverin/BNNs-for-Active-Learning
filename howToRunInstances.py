# import
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from itertools import repeat

import numpy as np
import pandas as pd
from Utils import custom_data_loader, preprocess_data, preprocess_activeL_data, preprocess_activeL_all_data, custom_data_loader_EPICLE, preprocess_activeL_EPICLE_data
from Utils.SummaryWriter import LogSummary
from Models.simpleFFBNN import SimpleFFBNN
from Models.denseRegression import DenseRegressor
from Models.paperModel import SimpleFFBNNPaper
from Models.densePaper import DenseRegressorPaper
from Models.denseBBBRegression import DenseBBBRegression

from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import SubsetRandomSampler

from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
import os
from scipy.stats import entropy
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

device = get_device()

#model = SimpleFFBNNPaper(4, 1)
#model = DenseBBBRegression(4, 1)
model = DenseRegressorPaper(4, 1)

class SaveOutput():
    def __init__(self, instances, batch_size, rounds):
        self.T = instances
        self.batch_size = batch_size
        self.outputs = []
        self.rounds = rounds
        self.counter = 0


    def __call__(self, module, module_in, module_out):
        if self.counter < 3:
            sample_data = np.random.randint(self.batch_size)
            #outs = module_out.view(self.batch_size, -1)
            outs = module_out.view(self.T, self.batch_size, -1)[:, 0, :]
            layer_size = outs.shape[1]

            
            write_summary.per_round_layer_output(layer_size, outs, self.rounds)
            
            
            self.counter += 1


    def clear(self):
        self.outputs = []
        

#dataset_train, dataset_test, dataset_activeL, df_custom = preprocess_activeL_all_data()
#dataset_train, dataset_test, dataset_activeL, df_custom = preprocess_activeL_data()
dataset_train, dataset_test, dataset_activeL, df_custom = preprocess_activeL_EPICLE_data()

class runActiveLearning():
    def __init__(self, model_name, model, top_unc, dataloader_train, dataloader_test, dataset_active_l, epochs, rounds, learning_rate, 
    batch_size, instances, seed_sample, retrain, resume_round, optimizer, df_custom):
        self.model_name = model_name
        self.model = model
        self.top_unc = top_unc
        self.dataloader_train = dataloader_train
        self.dataloader_test = dataloader_test
        self.dataset_active_l = dataset_active_l
        self.epochs = epochs
        self.rounds = rounds
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.instances = instances
        self.seed_sample = seed_sample
        self.optimizer = optimizer
        self.df_custom = df_custom
        self.retrain = retrain
        

        # a set of lists to store the selected indices with highest uncertainty
        self.selected_data = set([])
        # unexplored data
        self.unexplored_data = set(range(len(dataloader_train)))

        # make sure sklearn.metrics.r2_score is imported
        #self.r2_score = r2_score


    
    def objective(self, output, target, kl, beta):
        '''Objective function to calculate the loss function / KL divergence'''
        loss_fun = nn.functional.mse_loss
        discrimination_error = loss_fun(output.view(-1), target)
        variational_bound = discrimination_error + beta * kl
        return variational_bound, discrimination_error, kl

    def get_entropy(self, y):
        '''Function to calculate the entropy of the ensemble outputs
        y: the ensemble outputs (shape: 30, 64, 1)'''
        # calculate the entropy of the ensemble outputs using pytorch
        flattened_y = y.view(y.size(0), -1)
        probs = F.softmax(flattened_y, dim=1)
        entropy = -(probs * torch.log(probs)).sum(dim=1)
        return entropy

    def get_variation_ratio(self, y):
        standard_deviation = torch.std(y, dim=0).view(-1)
        return standard_deviation

    def get_validation_data(self, is_validation):
        if not is_validation:
            # train sampler randomly samples data from the selected data set
            train_sampler = SubsetRandomSampler(list(self.selected_data))
            # train loader will load the data from the train sampler
            self.train_loader = DataLoader(self.dataloader_train, batch_size=self.batch_size, sampler=train_sampler, num_workers=1)

        indices = list(self.unexplored_data)
        np.random.shuffle(indices)
        split = int(np.floor(0.1 * len(indices)))  # this line is to split the training_data into 90% training and 10% validation
        validation_idx = np.random.choice(indices, size = split) # this line is to randomly select 10% of the data for validation
        train_sampler = SubsetRandomSampler(list(self.selected_data))
        validation_sampler = SubsetRandomSampler(validation_idx)
        self.train_loader = DataLoader(self.dataloader_train, batch_size=self.batch_size, sampler=train_sampler, num_workers=1)
        self.validation_loader = DataLoader(self.dataloader_train, batch_size=self.batch_size, sampler=validation_sampler, num_workers=1)

    def random_data(self, rounds):
        if rounds == 0:    
            # randomly select data
            self.selected_data = set(range(self.dataloader_train))  # seed sample in Rakeesh & Jain paper
        else:
            minimum_index = np.random.choice(list(self.unexplored_data), self.top_unc)
            self.selected_data = self.selected_data.union(minimum_index)
            self.unexplored_data = self.unexplored_data.difference(self.selected_data)



    def activeDataSelection(self, rounds):
        
        if rounds == 1:
            self.selected_data = set(range(len(self.dataloader_train)))
            self.unexplored_data = self.selected_data
        else:
            self.all_data = DataLoader(self.dataloader_train, batch_size=self.batch_size, shuffle=False, num_workers=1)
            correct = 0
            metrics = []
            hook_handles = []
            save_output = SaveOutput(self.instances, self.batch_size, self.rounds)
            self.model.eval()
            for layer in self.model.kl_layers:
                handle = layer.register_forward_hook(save_output)
                hook_handles.append(handle)

            with torch.no_grad():
                for batch_index, (X, y) in enumerate(self.all_data):
                    batch_size = X.shape[0]
                    save_output.batch_size = batch_size
                    X = X.repeat(self.instances, 1)
                    y = y.squeeze()
                    y = y.repeat(self.instances)
                    X, y = X.to(device), y.to(device)
                    y_pred = self.model(X)
       
                    ensemble_outputs = y_pred.reshape(self.instances, batch_size, 1)
                    #entropy = self.get_entropy(ensemble_outputs)
                    variation_ratio = self.get_variation_ratio(ensemble_outputs)
                    metrics.append(variation_ratio)
                    
                save_output.clear()
                save_output.counter = 0
                for handle in hook_handles:
                    handle.remove()

                metrics = torch.cat(metrics)
                new_indices = torch.argsort(metrics, descending=True).tolist()
                new_indices = [n for n in new_indices if n not in self.selected_data]
            
                self.selected_data =  set(new_indices[:self.top_unc])
                self.unexplored_data = self.unexplored_data.difference(self.selected_data)
                
    def annotateSelectedData(self, rounds):
        
        indices = list(self.selected_data)
        data_to_annotate = [self.all_data.dataset[i] for i in indices]
        # remove the selected data from the all data
        x_all = [x for x, y in self.all_data.dataset]
        y_all = [y for x, y in self.all_data.dataset]
        x_all = [x for i, x in enumerate(x_all) if i not in indices]
        y_all = [y for i, y in enumerate(y_all) if i not in indices]

        # create a new dataset from the remaining data
        self.dataloader_train = TensorDataset(torch.stack(x_all), torch.stack(y_all))

        def refit_and_rescale(data):
            
            data_to_fit_X = self.df_custom.X
            data_to_fit_y = self.df_custom.y

            scaler = StandardScaler().fit(data_to_fit_X)
            # get the x_values from the data to be annotated
            
            x_values = [x for x, y in data]
            y_values = [y for x, y in data]

            x_arrays =[x.numpy() for x in x_values]
            y_arrays = [y.numpy() for y in y_values]
            
            x_descaled = [torch.tensor(scaler.inverse_transform(x.reshape(1, -1))) for x in x_arrays]

            # get the x_values in numpy format
            x_np = [x.numpy() for x in x_descaled]
            x_flattened = [arr.flatten() for arr in x_np]

            # create a dataframe from the x_values
            x_df = pd.DataFrame(x_flattened, columns = ['income', 'time', 'savings', 'guests'])

            '''due to the way the data is transformed and inverse transformed 0 value are not exactly 0, but very close to 0.
            Therefore, the values close to 0 are replaced with 0'''
            tolerance = 1e-5
            x_df = x_df.mask(x_df.abs() < tolerance, 0)
            return x_df
        
        df = refit_and_rescale(data_to_annotate)


        def determine_quality(data):

            '''quality of food is determined by the income, time, savings and guests 
            following the approach the data was originally generated with.
            This function is therefore acting as the oracle.
            args:
            data: the data to be annotated in a pandas dataframe format with the columns income, time, savings and guests
            ''' 
            quality_based_on_income = np.where(data['income'] >= 7000, 5,
            np.where(data['income'] >= 4000, 4,
            np.where(data['income'] >= 3000, 3,
            np.where(data['income'] >= 2000, 2, 1))))
            quality_based_on_time = np.where(data['time'] >= 16, 1, 5)
            quality_based_on_savings = np.where(data['savings'] == 2, 5,
            np.where(data['savings'] == 1, 3, 1))
            quality_based_on_guests = np.where(data['guests'] == 0, 3, 5) 
            quality_of_food = (quality_based_on_income * 0.4 + quality_based_on_time * 0.1 + quality_based_on_savings * 0.2 + quality_based_on_guests * 0.3) / 1
            noise = np.random.normal(scale = 1, size = len(data))
            quality_of_food += noise
            quality_of_food = np.clip(quality_of_food, 1, 5)

            # make the quality of food y in the data
            data['quality_of_food'] = quality_of_food
            return data
        
        annotated_data = determine_quality(df)
    
        # scale the newly annotated data
        x_scaler = StandardScaler().fit(self.df_custom.X)
        x = annotated_data.drop('quality_of_food', axis = 1)
        y = annotated_data['quality_of_food']
        x_scaled = x_scaler.transform(x)
        x_scaled = torch.tensor(x_scaled.astype(np.float32))   
        y_scaler = StandardScaler().fit(self.df_custom.y.reshape(-1, 1))
        y_scaled = y_scaler.transform(y.values.reshape(-1, 1))
        y_scaled = torch.tensor(y_scaled.astype(np.float32))
        y_scaled = torch.tensor(y_scaled)

        if rounds == 3: # seems arbitrary, but the first 2 rounds are not annotated (starts at 1 training seed model, then finds the top_uncertain data)
            x_already_annotated = [x for x, y in self.dataset_active_l]
            y_already_annotated = [y for x, y in self.dataset_active_l]

            # make all tensors dtype(np.float32)
        else:
            x_already_annotated = [x for x, y in self.combined_dataset]
            y_already_annotated = [y for x, y in self.combined_dataset]
  
        
        x_alr_numpy = [x.numpy() for x in x_already_annotated]
        x_alr_tensors = torch.tensor(x_alr_numpy)

        y_alr_numpy = [y.numpy() for y in y_already_annotated]
        y_alr_tensors = torch.tensor(y_alr_numpy)
 
        x_all_annotated = torch.cat((x_scaled, x_alr_tensors), 0)
        y_all_annotated = torch.cat((y_scaled, y_alr_tensors), 0)

        self.combined_dataset = TensorDataset(x_all_annotated, y_all_annotated)    
        self.all_annotated_data = DataLoader(self.combined_dataset, shuffle = False, num_workers=1) # for the next round of active learning
        

    def TrainModel(self, rounds, epochs, is_validation):
        '''This function trains the seed model for the active learning process
        '''
         
        if self.retrain == True:
            # load the model with no training to retrain from scratch
            #self.model = SimpleFFBNNPaper(4, 1)
            self.model = DenseRegressorPaper(4, 1)

        if device.type == 'cpu':
            self.model = self.model.to(device)



        t_total, v_total = 0, 0
        t_r2_scores = []
        if epochs == 1:
            self.get_validation_data(is_validation)
        self.model.train()
        t_loss, v_loss = [], []
        t_likelihood, v_likelihood = [], []
        t_kl, v_kl = [], []
        self.model.train()
        # if rounds smaller or equal to 3, the data to be trained on is the active data set
        
        if rounds <= 3:
            m = len(self.dataset_active_l)
        else:
            m = len(self.all_annotated_data)
        

        if rounds <= 3:
            self.running_accumulation_annots = self.dataset_active_l
            
        else:
            self.running_accumulation_annots = self.all_annotated_data
            for X, y in self.running_accumulation_annots:
                # shape is now torch.Size([1, 4]), needs to be torch.Size([4])
                X = X.squeeze()
                # shape is now torch.Size([1, 1]), needs to be torch.Size([1])
                y = y.reshape(-1)
            
                
        for batch_index, (inputs, targets) in enumerate(self.running_accumulation_annots):
            X = inputs.repeat(self.instances,1) # (number of mcmc samples, input size)
            
            if rounds <= 3:
                Y = targets.repeat(self.instances) # (number of mcmc samples, output size)
            else:
                Y = targets.squeeze().repeat(self.instances)
            
            X, Y = X.to(device), Y.to(device)
    
            self.optimizer.zero_grad()
            outputs = self.model(X)

            if self.model_name == 'Simple' or self.model_name == 'Dense':
                loss, log_likelihood, kl = self.objective(outputs, Y, self.model.kl_divergence(), 2 ** (m - (batch_index + 1)) / (2 ** m - 1))
                t_likelihood.append(log_likelihood.item())
                t_kl.append(kl.item())
                t_total += targets.size(0)
            else:
                loss = self.model.sample_elbo(inputs = X, 
                                    labels = Y, 
                                    criterion = nn.MSELoss(),
                                    sample_nbr = 3,
                                    complexity_cost_weight = 2 ** (m - (batch_index + 1)) / (2 ** m - 1))

                log_likelihood, kl = 0, 0
                t_likelihood.append(log_likelihood)
                t_kl.append(kl)


            # calculate r2 score manually
            r2_score_value = 1 - (np.sum((outputs.detach().cpu().numpy() - targets.detach().cpu().numpy()) ** 2) / np.sum((targets.detach().cpu().numpy() - np.mean(targets.detach().cpu().numpy())) ** 2))
            t_r2_scores.append(r2_score_value)
            
            t_loss.append(loss.item())
            loss.backward()

            # define the optimizer
            optimizer = self.optimizer

            optimizer.step()
            if self.model_name == 'Simple' or self.model_name == 'Dense':
                for layer in self.model.kl_layers:
                    layer.clip_variances()

            # save training loss
        np.save('/Users/kristian/Documents/Skole/9. Semester/Thesis Preparation/Code/BNNs/ThesisPlots/Results/EPICLE/MP15/trainloss' + str(rounds) + '.npy', t_loss)
    
        if is_validation:
            m_val = len(self.validation_loader)
            self.model.eval()
            for batch_index, (inputs, targets) in enumerate(self.validation_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = self.model(inputs)

                if self.model_name == 'Simple' or self.model_name == 'Dense':

                    loss_val, log_likelihood_val, kl_val = self.objective(outputs, targets, self.model.kl_divergence(), 1 / m_val)
                    v_total += targets.size(0)
                    v_loss.append(loss_val.item())
                    v_likelihood.append(log_likelihood_val.item())
                    v_kl.append(kl_val.item())
                else:
                    loss_val = self.model.sample_elbo(inputs = inputs, 
                                            labels = targets, 
                                            criterion = nn.MSELoss(),
                                            sample_nbr = 3,
                                            complexity_cost_weight = 1 / m_val)
                    
                    val_likelihood, val_kl = torch.tensor(0), torch.tensor(0)
                    v_loss.append(loss_val.item())
                    v_likelihood.append(val_likelihood.item())
                    v_kl.append(val_kl.item())

            
            avg_v_loss = np.average(v_loss)
            avg_t_loss = np.average(t_loss)
            avg_v_likelihood = np.average(v_likelihood)
            avg_t_likelihood = np.average(t_likelihood)
            avg_v_kl = np.average(v_kl)
            avg_t_kl = np.average(t_kl)


            print(
                'epochs: {}, train loss: {}, train likelihood: {}, train kl: {}'.format(
                    epochs, avg_t_loss, \
                    avg_t_likelihood, avg_t_kl))

            print(
                'epochs: {}, validation loss: {}, validation likelihood: {}, validation kl: {}'.format(
                    epochs, avg_v_loss, \
                    avg_v_likelihood, avg_v_kl))

            return avg_v_loss

        else:
            avg_t_loss = np.average(t_loss)
            avg_t_likelihood = np.average(t_likelihood)
            avg_t_kl = np.average(t_kl)
            avg_t_r2 = np.average(t_r2_scores)

            print(
                'epochs: {}, train loss: {}, train likelihood: {}, train kl: {}, train_avg_R2: {}'.format(
                    epochs, avg_t_loss, \
                    avg_t_likelihood, avg_t_kl, avg_t_r2))

            return avg_t_loss, avg_t_r2


    
    def TestModel(self, rounds):
        if device.type == 'cpu':
            state = torch.load(self.train_weight_path, map_location=torch.device('cpu'))
        else:
            state = torch.load(self.train_weight_path)

        self.model.load_state_dict(state['weights'])

        self.model.eval()
        predictions = []
        actual = []
        mse_scores = []
        test_loss = []
        with torch.no_grad():
            for batch_index, (inputs, targets) in enumerate(self.dataloader_test):
                X, Y = inputs.to(device), targets.to(device)
                    
                outputs = self.model(inputs)

                # Calculate the MSE loss for the batch

                mse_scores.append(nn.MSELoss(outputs.view(-1).mean(), Y))
                
                # Get the loss for the batch
                loss, log_likelihood, kl = self.objective(outputs, Y, self.model.kl_divergence(), 1 / len(self.dataloader_test))
                test_loss.append(loss.item())

                predictions.append(torch.mean(outputs, 0).detach().cpu().numpy()) # mean of the outputs
                actual.append(targets.numpy())
            # save the loss as a pandas dataframe
            df_loss = pd.DataFrame(test_loss, columns = ['TestLoss'])
            # save the loss
            df_loss.to_csv('/Users/kristian/Documents/Skole/9. Semester/Thesis Preparation/Code/BNNs/ThesisPlots/Results/EPICLE/MP15/TestLoss' + str(rounds) + '.csv', mode ="a" ,index=False)
            # without overwriting
            #np.save('/Users/kristian/Documents/Skole/9. Semester/Thesis Preparation/Code/BNNs/ThesisPlots/Results/Regression/SimpleFFBNN/ActiveLearning/Loss/TestLoss' + str(rounds) + '.npy', test_loss, allow_pickle=True)
            
        if self.model_name == 'Simple' or self.model_name == 'Dense':
            predictions = np.concatenate(predictions)
        else:
            predictions = predictions
        actual = np.concatenate(actual)

        df = pd.DataFrame(data = {'Actual': actual, 'Predictions': predictions})
        df['R2'] = r2_score(df.Actual, df.Predictions)
        df['MSE'] = mean_squared_error(df.Actual, df.Predictions)
        df['RMSE'] = np.sqrt(df['MSE'])
        df['Round'] = rounds
        print(f'This is the dataframe: {df.head()}')
        print(f'This is the R2 score: {df["R2"][0]}')
        print(f'This is the MSE score: {df["MSE"][0]}')
        print(f'This is the RMSE score: {df["RMSE"][0]}')
        # adjusted r2 score
        n = len(df)
        p = 4
        r2 = r2_score(df.Actual, df.Predictions)
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
        print(f'This is the adjusted R2 score: {adj_r2}')
        df['Adj_R2'] = adj_r2


        # save the results
        #df.to_csv('ThesisPlots/Results/Regression/SimpleFFBNN/ActiveLearning/results' + str(rounds) + '.csv', index=False)
        # save the results without overwriting
        df.to_csv('/Users/kristian/Documents/Skole/9. Semester/Thesis Preparation/Code/BNNs/ThesisPlots/Results/EPICLE/MP15/MP15r1e100' + str(rounds) + '.csv', mode='a', header=False, index=False)
        
                

    def getTrainedModel(self, rounds, epochs):
        # path to save the trained model
        self.train_weight_path = 'trainedModels/trained_weights/MP15r1e100/' + self.model_name + '_' + 'e' + str(epochs) + '_' + '-r' + str(rounds) + '-b' + str(self.batch_size) + '.pkl'
        return (self.model, self.train_weight_path)


    def saveModel(self, model, optimizer, path_to_save):
        state = {
            'rounds': self.rounds,
            'weights': model.state_dict(),
            'selected_data': self.selected_data,
            'optimizer': self.optimizer.state_dict()
            }

        path_to_save = path_to_save

        torch.save(state, path_to_save)
        
if __name__ == '__main__':
    if not os.path.isdir('trainedModels/trained_weights/MP15r1e100'):
        os.makedirs('trainedModels/trained_weights/MP15r1e100')


    # use the class to run the active learning
    active_learning = runActiveLearning(model_name='Dense', model=model, dataloader_train=dataset_train, top_unc = 100, dataloader_test=dataset_test, 
    dataset_active_l= dataset_activeL, epochs=100, rounds=1, learning_rate=0.001, batch_size=1, instances = 30, 
    seed_sample=1, retrain=False, resume_round=False, optimizer= torch.optim.SGD(model.parameters(), lr=0.0001), df_custom = df_custom)

    write_summary = LogSummary('active_learning')

    # get data to train the model
    active_learning.get_validation_data(is_validation=True)

    # train just the seed model
    for e in range(1, 1):
        active_learning.TrainModel(1, active_learning.epochs, True)

        # get the trained model
        model, path = active_learning.getTrainedModel(1, 1)

        # save the model
        active_learning.saveModel(model, active_learning.optimizer, 'trainedModels/trained_weights/MP15r1e100/' + 'Dense' + '_' + 'e' + str(e) + '_' + '-r' + str(1) + '-b' + str(active_learning.batch_size) + '.pkl')

    # run the active learning process
    for r in range(1, active_learning.rounds+1):
        print(f'Round: {r}')
        #model, path = active_learning.getTrainedModel(r)
                 
        print(f'Training model in round: {r}')
        active_learning.activeDataSelection(r) 
        print(f'Annotating selected data in round: {r}')

        if r == 1 or r == 2:
            pass

        else:
            active_learning.annotateSelectedData(r)

        for e in range(1,active_learning.epochs+1):
            active_learning.TrainModel(r, active_learning.epochs, False)
            print(f'Training model in round: {r} and epoch: {e}')
            model, path = active_learning.getTrainedModel(r, e)
            active_learning.saveModel(model, active_learning.optimizer, 'trainedModels/trained_weights/MP15r1e100/' + 'Dense' + '_' + 'e' + str(e) + '_' + '-r' + str(r) + '-b' + str(active_learning.batch_size) + '.pkl')

            active_learning.TestModel(active_learning.rounds)
    