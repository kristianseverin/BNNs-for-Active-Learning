# import
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from itertools import repeat

import numpy as np
import pandas as pd
from Utils import custom_data_loader_classification, preprocess_activeL_data_classification, preprocess_activeL_data_classification_alldata, preprocess_activeL_data_classification_EPICLE
from Utils.SummaryWriter import LogSummary
from Models.simpleClassificationPaper import SimpleFFBNNClassificationPaper
from Models.LargeClassification import LargeFFBNNClassificationPaper
from Models.BBBClassification import BBBClassification

from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import SubsetRandomSampler

from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from sklearn.preprocessing import StandardScaler
import os
from scipy.stats import entropy
import argparse

from netcal.metrics import ECE
from netcal.scaling import TemperatureScaling



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

#model = SimpleFFBNNClassificationPaper(4, 5)
model = LargeFFBNNClassificationPaper(4, 2)
#model = BBBClassification(4, 5)


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
        

#dataset_train, dataset_test, dataset_activeL, df_custom = preprocess_activeL_data_classification()
dataset_train, dataset_test, dataset_activeL, df_custom = preprocess_activeL_data_classification_EPICLE()


class runActiveLearning():
    def __init__(self, model_name, model, top_unc, dataloader_train, dataloader_test, dataset_active_l, epochs, rounds, learning_rate, 
    batch_size, instances, seed_sample, retrain, resume_round, optimizer, df_custom, noisy_data):
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
        self.noisy_data = noisy_data
        
        # a set of lists to store the selected indices with highest uncertainty
        self.selected_data = set([])
        # unexplored data
        self.unexplored_data = set(range(len(dataloader_train)))

        # make sure sklearn.metrics.r2_score is imported
        #self.r2_score = r2_score
    
    def objective(self, output, target, kl, beta):
        '''Objective function to calculate the loss function / KL divergence'''
        loss_fun = nn.functional.cross_entropy
        discrimination_error = loss_fun(output, target)
        variational_bound = discrimination_error + beta * kl
        return variational_bound, discrimination_error, kl

    def get_entropy(self, y):
        '''Function to calculate the entropy of the ensemble outputs
        y: the ensemble outputs (shape: 30, 64, 1)'''
        # calculate the entropy of the ensemble outputs using pytorch
        ensemble_probs = y.mean(0)
        entropy = Categorical(probs=ensemble_probs).entropy()
        return entropy

    def get_variation_ratio(self, y):
        predicts = []
        n_instances = y.size(0)
        for out in y:
            out = out.squeeze(dim=1)
            _, predicted = out.max(1)
            predicts.append(predicted.unsqueeze(0))
        predicts = torch.cat(predicts, dim=0)
        #m, l = predicts.mode(dim=0)
    
        #mode_count = predicts.eq(m).sum(dim=0)
        #variation_ratio = 1 - torch.div(mode_count, n_instances)
        variation_ratio = torch.var(predicts.double(), dim =0)
        return variation_ratio

    def get_validation_data(self, is_validation):
        if not is_validation:
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
                    
                    ensemble_outputs = torch.unsqueeze(F.softmax(self.model(X), dim=1), 0)
                    print(f'this is the ensemble outputs: {ensemble_outputs}')
                    ensemble_outputs = ensemble_outputs.reshape(self.instances, batch_size, -1)
                    entropy = self.get_entropy(ensemble_outputs)
                    metrics.append(entropy)
                    #variation_ratio = self.get_variation_ratio(ensemble_outputs)
                    #metrics.append(variation_ratio)
                    
                save_output.clear()
                save_output.counter = 0
                for handle in hook_handles:
                    handle.remove()

                metrics = torch.cat(metrics)
                new_indices = torch.argsort(metrics, descending=True).tolist()
                #print(f'this is the new indices in round {rounds}: {new_indices}, length: {len(new_indices)}')
                #print(f'this is the selected data in round {rounds}: {self.selected_data}, length: {len(self.selected_data)}')
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
            x_df = pd.DataFrame(x_flattened, columns = ['100Lags', '25Lags', '5Lags', '1Lag'])
            y_df = pd.DataFrame(y_arrays, columns = ['P_HRV_RR'])

            '''due to the way the data is transformed and inverse transformed 0 value are not exactly 0, but very close to 0.
            Therefore, the values close to 0 are replaced with 0'''
            tolerance = 1e-5
            x_df = x_df.mask(x_df.abs() < tolerance, 0)
            df = pd.concat([x_df, y_df], axis = 1)
            return df
        
        annotated_data = refit_and_rescale(data_to_annotate)


        # scale the newly annotated data
        x_scaler = StandardScaler().fit(self.df_custom.X)
        x = annotated_data.drop('P_HRV_RR', axis = 1)
        y = annotated_data['P_HRV_RR']
        x_scaled = x_scaler.transform(x)
        x_scaled = torch.tensor(x_scaled.astype(np.float32))   
        #y_scaler = StandardScaler().fit(self.df_custom.y.reshape(-1, 1))
        #y_scaled = y_scaler.transform(y.values.reshape(-1, 1))
        #y_scaled = torch.tensor(y_scaled.astype(np.float32))
        #y_scaled = torch.tensor(y_scaled)
        # y is int and not scaled as it is a classification problem
        y_scaled = torch.tensor(y.values.astype(np.int64))
        
        # y_scaled should be tensor of lists to give it 2 dimensions
        y_scaled = y_scaled.reshape(-1, 1)


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
        y_alr_numpy = [y.reshape(-1) for y in y_alr_numpy]
        #y_alr_tensors_concatenated = np.concatenate(y_alr_numpy)
        y_alr_tensors = torch.tensor(y_alr_numpy)

        x_all_annotated = torch.cat((x_scaled, x_alr_tensors), 0)
        y_all_annotated = torch.cat((y_scaled, y_alr_tensors), 0)

        # make the datatype Long for the y values
        y_all_annotated = y_all_annotated.long()

        self.combined_dataset = TensorDataset(x_all_annotated, y_all_annotated)    
        self.all_annotated_data = DataLoader(self.combined_dataset, shuffle = False, num_workers=1) # for the next round of active learning
        

    def TrainModel(self, rounds, epochs, is_validation):
        '''This function trains the seed model for the active learning process
        '''
         
        if self.retrain == True:
            # load the model with no training to retrain from scratch
            #self.model = SimpleFFBNNPaper(4, 1)
            self.model = SimpleFFBNNClassificationPaper(4, 5)

        if device.type == 'cpu':
            self.model = self.model.to(device)


        t_correct, v_correct = 0, 0
        t_total, v_total = 0, 0
        t_accuracy_scores = []
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
            for X, y in self.running_accumulation_annots:
                # shape is now torch.Size([1, 4]), needs to be torch.Size([4])
                X = X.squeeze()
                # shape is now torch.Size([1, 1]), needs to be torch.Size([1])   
            
        else:
            self.running_accumulation_annots = self.all_annotated_data
            for X, y in self.running_accumulation_annots:
                # shape is now torch.Size([1, 4]), needs to be torch.Size([4])
                X = X.squeeze()
                # shape is now torch.Size([1, 1]), needs to be torch.Size([1])
                #y = y.reshape(-1)
            
                
        for batch_index, (inputs, targets) in enumerate(self.running_accumulation_annots):
            X = inputs.repeat(self.instances, 1)
            
            if rounds <= 3:
                Y = targets.repeat(self.instances)
            else:
                Y = targets.squeeze().repeat(self.instances)
            
            X, Y = X.to(device), Y.to(device)
            self.optimizer.zero_grad()
            outputs = self.model(X)            

            if self.model_name == 'Simple' or self.model_name == 'Dense':
                loss, log_likelihood, kl = self.objective(outputs, Y, self.model.kl_divergence(), 2 ** (m - (batch_index + 1)) / (2 ** m - 1))
                t_likelihood.append(log_likelihood.item())
                t_kl.append(kl.item())
                
                _, predicted = F.softmax(outputs, dim=1).max(1)
                t_total += targets
                t_correct += predicted.eq(targets).sum().item()


            else:
                loss = self.model.sample_elbo(inputs = X, 
                                    labels = Y, 
                                    criterion = nn.CrossEntropyLoss(),
                                    sample_nbr = 3,
                                    complexity_cost_weight = 2 ** (m - (batch_index + 1)) / (2 ** m - 1))

                _, predicted = F.softmax(outputs, dim=1).max(1)
                t_total += targets
                t_correct += predicted.eq(targets).sum().item()
                log_likelihood, kl = 0, 0
                t_likelihood.append(log_likelihood)
                t_kl.append(kl)


            # predicted_ = to the value that occurs the most in predicted
            #predicted = torch.mode(predicted, 0).values
            #print(f'this is the target: {targets}, these are the predictions: {predicted}')
            
            
            # calculate accuracy
            #t_accuracy_scores.append(accuracy_score(targets, predicted))
            
            t_loss.append(loss.item())
            loss.backward()

            # define the optimizer
            optimizer = self.optimizer

            optimizer.step()
            if self.model_name == 'Simple' or self.model_name == 'Dense':
                for layer in self.model.kl_layers:
                    layer.clip_variances()

            # save training loss
        np.save('/Users/kristian/Documents/Skole/9. Semester/Thesis Preparation/Code/BNNs/ThesisPlots/Results/timeseriesAL/trainloss' + str(rounds) + '.npy', t_loss)
    
        if is_validation:
            m_val = len(self.validation_loader)
            self.model.eval()
            for batch_index, (inputs, targets) in enumerate(self.validation_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = self.model(inputs)

                if self.model_name == 'Simple' or self.model_name == 'Dense':

                    loss_val, log_likelihood_val, kl_val = self.objective(outputs, targets, self.model.kl_divergence(), 1 / m_val)
                    _, predicted = F.softmax(outputs, dim=1).max(1)
                    v_total += targets.size(0)
                    v_correct += predicted.eq(targets).sum().item()
                    v_loss.append(loss_val.item())
                    v_likelihood.append(log_likelihood_val.item())
                    v_kl.append(kl_val.item())




                else:
                    loss_val = self.model.sample_elbo(inputs = inputs, 
                                            labels = targets, 
                                            criterion = nn.CrossEntropyLoss(),
                                            sample_nbr = 3,
                                            complexity_cost_weight = 1 / m_val)

                    
                    _, predicted = F.softmax(outputs, dim=1).max(1)
                    v_total += targets.size(0)
                    v_correct += predicted.eq(targets).sum().item()
                    
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
            avg_t_accuracy = np.average(t_accuracy_scores)

            print(
                'epochs: {}, train loss: {}, train likelihood: {}, train kl: {}, train_avg_accuracy: {}'.format(
                    epochs, avg_t_loss, \
                    avg_t_likelihood, avg_t_kl, avg_t_accuracy))

            return avg_t_loss, avg_t_accuracy


    
    def TestModel(self, rounds):
        if device.type == 'cpu':
            state = torch.load(self.train_weight_path, map_location=torch.device('cpu'))
        else:
            state = torch.load(self.train_weight_path)

        self.model.load_state_dict(state['weights'])
        
        if self.model_name == 'Simple' or self.model_name == 'Dense':

            self.model.eval()
            predictions = []
            actual = []
            test_loss = []
            accuracy = []
            with torch.no_grad():
                for batch_index, (inputs, targets) in enumerate(self.dataloader_test):
                    X, Y = inputs.to(device), targets.to(device)
                    outputs = self.model(X)
                    outputs = F.softmax(outputs, dim=1)
                    outputs = torch.mean(outputs, 0)
                    
                    # Get the loss for the batch
                    loss, log_likelihood, kl = self.objective(outputs, Y, self.model.kl_divergence(), 1 / len(self.dataloader_test))
                    test_loss.append(loss.item())

                    predictions.append(outputs.argmax(dim=0).cpu().numpy())
                    actual.append(targets.numpy())

        else:   

            #n_bins = 10
            #temperature = TemperatureScaling()
            #ece = ECE(n_bins)
            #confidence = []
            #targets_list = []
            actual = []
            predictions = []
            test_loss = []
            self.model.eval()
            with torch.no_grad():
                for batch_index, (inputs, targets) in enumerate(self.dataloader_test):
                    X, Y = inputs.to(device), targets.to(device)
                    outputs = self.model(X.repeat(self.instances, 1))
                    # ensemble_outputs = torch.unsqueeze(F.softmax(self.model(X), dim=1), 0)
                    outputs = F.softmax(outputs, dim=1)
                    #confidence.append(outputs)
                    #targets_list.append(Y)
                # get the mean of the confidence probabilities current shape is (30, 5) i.e. 30 instances and 5 classes, so get the mean of the 30 instances
                #confidence = torch.mean(torch.stack(confidence), 1)

                    actual.append(targets.numpy())
                    preds = torch.argmax(outputs, dim=1)

                    # the one that occurs the most
                    preds = torch.mode(preds, 0).values

                  
                
                    predictions.append(preds.cpu().numpy())

               

                # get the one that is most 
                #confidence = torch.argmax(confidence, dim=1)
                    loss = self.model.sample_elbo(inputs = X.repeat(3, 1),
                                        labels = Y.repeat(3),
                                        criterion = nn.CrossEntropyLoss(),
                                        sample_nbr = 3,
                                        complexity_cost_weight = 1 / len(self.dataloader_test))

                    test_loss.append(loss.item())


            # save the loss as a pandas dataframe
        df_loss = pd.DataFrame(test_loss, columns = ['TestLoss'])
            # save the loss
        df_loss.to_csv('ThesisPlots/Results/timeseriesAL/TestLoss' + str(rounds) + '.csv', mode ="a" ,index=False)
            # without overwriting
            #np.save('/Users/kristian/Documents/Skole/9. Semester/Thesis Preparation/Code/BNNs/ThesisPlots/Results/Regression/SimpleFFBNN/ActiveLearning/Loss/TestLoss' + str(rounds) + '.npy', test_loss, allow_pickle=True)
               
        #if self.model_name == 'Simple' or self.model_name == 'Dense':
            #predictions = np.concatenate(predictions)
         #   pass
        #else:
         #   pass
          #  predictions = predictions
        #actual = np.concatenate(actual)

        df = pd.DataFrame(data = {'Actual': actual, 'Predictions': predictions})
        df['Actual'] = df['Actual'].astype(int)
        df['Predictions'] = df['Predictions'].astype(int)
        df['accuracy'] = accuracy_score(df.Actual, df.Predictions)
        df['precision'], df['recall'], df['f1_score'], _ = precision_recall_fscore_support(df.Actual, df.Predictions, average='weighted')
        print(f'This is the dataframe: {df.head()}')
        print(f'This is the accuracy: {df["accuracy"][0]}')
        print(f'This is the precision: {df["precision"][0]}')
        print(f'This is the recall: {df["recall"][0]}')
        print(f'This is the f1_score: {df["f1_score"][0]}')
        # save the results
        #df.to_csv('ThesisPlots/Results/Regression/SimpleFFBNN/ActiveLearning/results' + str(rounds) + '.csv', index=False)
        # save the results without overwriting
        df.to_csv('ThesisPlots/Results/timeseriesAL/results' + str(rounds) + '.csv', mode='a', index=False)
        # save with the column names
        #df.to_csv('ThesisPlots/Results/Regression/SimpleFFBNN/ActiveLearning/results' + str(rounds) + '.csv', index=False)

        '''model is tested at all rounds and epochs, so the best f1_score and accuracy 
        achieved is printed out here
        '''
        df = pd.read_csv('ThesisPlots/Results/timeseriesAL/results' + str(0) + '.csv', names = ['Actual', 'Predictions', 'accuracy', 'precision', 'recall', 'f1_score'], header = 0)
        print(f'This is the dataframe: {df.head()}')
        df['f1_score'] = pd.to_numeric(df['f1_score'], errors='coerce')
        df['accuracy'] = pd.to_numeric(df['accuracy'], errors='coerce')
        print(f'This is the best f1_score: {df["f1_score"].max()}')
        print(f'This is the best accuracy: {df["accuracy"].max()}')


    def Test_ensemble(self, rounds, is_sample=True):

        if device.type == 'cpu':
            state = torch.load(self.train_weight_path, map_location=torch.device('cpu'))
        else:
            state = torch.load(self.train_weight_path)
        self.model.load_state_dict(state['weights'])
        self.model.eval()

        self.test_loader = DataLoader(self.dataloader_test, batch_size=self.batch_size, shuffle=True, num_workers=1)

        correct = 0
        # number of ensemble samples for testing
        test_samples = 6
        corrects = np.zeros(test_samples, dtype=int)
        predictions = []
        targets = []
        n_bins = 10
        temperature = TemperatureScaling()
        ece = ECE(n_bins)
        confidence = []
        with torch.no_grad():
            for data, target in self.test_loader:
                targets.append(target.cpu().numpy())
                data, target = data.to(device), target.to(device)
                outputs = torch.zeros(test_samples, 1, 2).to(device)
                for i in range(test_samples):
                    # for each sample we obtain the output from our trained NN. Here we set sample=True,
                    # because we need different weights for each prediction.
                    # give self.model(data, sample=True) for ensemble testing, the default is not to sample from weight distribution

                    #outputs = self.model(data)
                    outputs = F.softmax(self.model(data), dim=1)
                    outputs = outputs.unsqueeze(0)
                    #outputs[i] = F.softmax(self.model(data), dim=1)

                # given a batch and the predicted probabilities for C classes, get the mean probability across K test_samples for each class
                output_probs = outputs.mean(0)
                confidence.append(output_probs)
                # predict the output class with highest probability across K test_samples
                preds = outputs.max(2, keepdim=True)[1]  # get the index of the max probability
                # From the mean probability, get class with max probability for each d \in test batch
                pred = output_probs.max(1, keepdim=True)[1]  # index of max log-probability
                predictions.append(pred.cpu().numpy().flatten())
                # view as basically reshapes the tensor into the shape of a target tensor, here target is a 1-D tensor, while pred is a 5*1 tensor
                corrects += preds.eq(target.view_as(pred)).sum(dim=1).squeeze().cpu().numpy()
                #correct += pred.eq(target.view_as(pred)).sum().item()

        print('Ensemble Accuracy: {}/{}'.format(correct, len(self.dataloader_test)))
        confidence = torch.cat(confidence).cpu().numpy()
        targets = np.concatenate(targets)
        temperature.fit(confidence, targets)
        calibrated = temperature.transform(confidence)
        uncaliberated_error = ece.measure(confidence,targets)
        calibrated_error = ece.measure(calibrated, targets)
        predictions = np.concatenate(predictions)
        report = classification_report(predictions, targets, output_dict=True)
        df = pd.DataFrame(report).transpose()
        # add the expected calibration error to the dataframe
        df.loc["ece"] = [uncaliberated_error]*len(df.columns)
        df.loc["ece_calibrated"] = [calibrated_error]*len(df.columns)
        df['rounds'] = rounds
        
        # save the results
        df.to_csv('ThesisPlots/Results/timeseriesAL/ensemble_results' + str(rounds) + '.csv', mode='a', index=True)
        print(df)

        
    def getTrainedModel(self, rounds, epochs):
        # path to save the trained model
        self.train_weight_path = 'trainedModels/trained_weights/timeseriesAL/' + self.model_name + '_' + 'e' + str(epochs) + '_' + '-r' + str(rounds) + '-b' + str(self.batch_size) + '.pkl'
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
    if not os.path.isdir('trainedModels/trained_weights/timeseriesAL'):
        os.makedirs('trainedModels/trained_weights/timeseriesAL')


    # use the class to run the active learning
    active_learning = runActiveLearning(model_name='Dense', model=model, dataloader_train=dataset_train, top_unc = 1, 
    dataloader_test=dataset_test, dataset_active_l= dataset_activeL, epochs=20, rounds=2, learning_rate=0.0001, batch_size=1, instances = 30, seed_sample=1, 
    retrain=False, resume_round=False, optimizer= torch.optim.Adam(model.parameters(), lr=0.0001), df_custom = df_custom, noisy_data = True)

    write_summary = LogSummary('active_learning')

    # get data to train the model
    active_learning.get_validation_data(is_validation=True)

    # train just the seed model
    """Seed model is trained for 100 epochs and saved
    """
    for e in range(1, 1):
        active_learning.TrainModel(1, 1, True)  # could change to active_learning.epochs

        # get the trained model
        model, path = active_learning.getTrainedModel(1, e)

        # save the model
        active_learning.saveModel(model, active_learning.optimizer, 'trainedModels/trained_weights/timeseriesAL/' + 'Dense' + '_' + 'e' + str(e) + '_' + '-r' + str(1) + '-b' + str(active_learning.batch_size) + '.pkl')

        # test the model
        active_learning.TestModel(1)

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
            active_learning.saveModel(model, active_learning.optimizer, 'trainedModels/trained_weights/timeseriesAL/' + 'Dense' + '_' + 'e' + str(e) + '_' + '-r' + str(r) + '-b' + str(active_learning.batch_size) + '.pkl')

        #print(f'Testing model in round: {r}')
        #active_learning.TestModel(active_learning.rounds)
            active_learning.Test_ensemble(r)
    