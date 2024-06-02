from Utils import custom_data_loader_classification, preprocess_classification_data
import torch
import torch.nn as nn
import GPUtil
import pandas as pd
from Models.simpleFFBNNClassification import SimpleFFBNNClassification
from Models.largeFFBNNClassification import LargeFFBNNClassification
from Models.BBBClassification import BBBClassification
import matplotlib.pyplot as plt
import argparse
import torch.nn.functional as F
import numpy as np

def arg_inputs():
    # initiate the parser
    parser = argparse.ArgumentParser()
    # add the arguments
    parser.add_argument("--model", 
                        "-m", 
                        help="The model to be used", 
                        type=str,
                        default = None)

    parser.add_argument("--epochs",
                        "-e",
                        help="Number of epochs",
                        type=int,
                        default=1000)

    parser.add_argument("--lr",
                        "-l",
                        help="Learning rate",
                        type=float,
                        default=0.0001)
                    
    parser.add_argument("--criterion",
                        "-c",
                        help="Criterion",
                        type=str,
                        default="nn.CrossEntropyLoss()")

    parser.add_argument("--savemodel", 
                        "-s",
                        help="Save the model",
                        type=bool,
                        default=False)


   # parse the arguments
    args = parser.parse_args()
    return args


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

# get the device
device = get_device()

# read data and preprocess
df = pd.read_csv('/Users/kristian/Documents/Skole/9. Semester/Thesis Preparation/Code/BNNs/Data/quality_of_food_int.csv')
dataloader_train, dataloader_test, dataloader_val = preprocess_classification_data(df, 64)

# define the model
class runBNNClassification:
    def __init__(self, model, dataloader_train, dataloader_test, dataloader_val, device, epochs, lr, criterion, optimize, savemodel):
        self.model = model
        self.dataloader_train = dataloader_train
        self.dataloader_test = dataloader_test
        self.dataloader_val = dataloader_val
        self.device = device
        self.epochs = epochs
        self.lr = lr
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.model.to(device)
        self.train_loss = []
        self.test_loss = []
        self.val_loss = []
        self.train_accuracy = []
        self.test_accuracy = []
        self.val_accuracy = []
        self.savemodel = savemodel
        
    def objective(self, output, target, kl, beta):
        loss_fun = nn.CrossEntropyLoss()
        discrimination_error = loss_fun(output, target)
        variational_bound = discrimination_error + beta * kl
        return variational_bound, discrimination_error, kl


    def train(self):
        for epoch in range(self.epochs):
            self.model.train()
            running_loss = 0.0
            for X, y in self.dataloader_train:
                X, y = X.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(X)
                Trainloss = self.model.sample_elbo(inputs=X,
                                               labels=y,
                                               criterion=self.criterion,
                                               sample_nbr=3,
                                               complexity_cost_weight=1/len(self.data_train))

                Trainloss.backward()
                self.optimizer.step()
                running_loss += Trainloss.item()
            print(f'Epoch: {epoch}, Loss: {running_loss}')




            # get test loss
            self.model.eval()
            correct = 0
            total = 0
            test_loss = 0
            with torch.no_grad():
                for X, y in self.dataloader_test:
                    X, y = X.to(self.device), y.to(self.device)
                    outputs = self.model(X)
                    _, predicted = torch.max(outputs, 1)
                    total += y.size(0)
                    correct += (predicted == y).sum().item()
                    Testloss = self.model.sample_elbo(inputs=X,
                                               labels=y,
                                               criterion=self.criterion,
                                               sample_nbr=3,
                                               complexity_cost_weight=1/len(self.dataloader_test))
                    test_loss += Testloss.item()
            print(f'Epoch: {epoch}, Test Loss: {test_loss}')
            print(f'Accuracy: {100 * correct / total}')



            # get validation loss
            self.model.eval()
            correct = 0
            total = 0
            val_loss = 0
            with torch.no_grad():
                for X, y in self.dataloader_val:
                    X, y = X.to(self.device), y.to(self.device)
                    outputs = self.model(X)
                    _, predicted = torch.max(outputs, 1)
                    total += y.size(0)
                    correct += (predicted == y).sum().item()
                    Valloss = self.model.sample_elbo(inputs=X,
                                               labels=y,
                                               criterion=self.criterion,
                                               sample_nbr=3,
                                               complexity_cost_weight=1/len(self.dataloader_val))
                    val_loss += Valloss.item()
            


            accuracy_per = 100 * correct / total


            
            self.test_loss.append(test_loss)
            self.val_loss.append(val_loss)
            self.accuracy.append(100 * correct / total)

        print('Finished Training')

    def trainBBBClassification(self):
        correct = 0
        total = 0
        for epoch in range(self.epochs):
            self.model.train()
            for X, y in self.dataloader_train:
                X, y = X.to(self.device), y.to(self.device)
                self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9)
                outputs = self.model(X)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == y).sum().item()
                total += y.size(0)
                TrainLoss = self.model.sample_elbo(inputs=X,
                                               labels=y,
                                               criterion=self.criterion,
                                               sample_nbr=3,
                                               complexity_cost_weight=1/len(self.dataloader_train))

                accuracy_train = 100 * correct / total
            self.train_accuracy.append(accuracy_train)

            TrainLoss.backward()
            self.optimizer.step()

            # get the test loss
            correct = 0
            total = 0
            self.model.eval()
            with torch.no_grad():
                for X, y in self.dataloader_test:
                    X, y = X.to(self.device), y.to(self.device)
                    outputs = self.model(X)
                    _, predicted = torch.max(outputs, 1)
                    correct += (predicted == y).sum().item()
                    total += y.size(0)
                    TestLoss = self.model.sample_elbo(inputs=X,
                                               labels=y,
                                               criterion=self.criterion,
                                               sample_nbr=3,
                                               complexity_cost_weight=1/len(self.dataloader_test))

                accuracy_test = 100 * correct / total
            self.test_accuracy.append(accuracy_test)
                    


            # get the validation loss
            correct = 0
            total = 0
            self.model.eval()
            with torch.no_grad():
                for X, y in self.dataloader_val:
                    X, y = X.to(self.device), y.to(self.device)
                    outputs = self.model(X)
                    _, predicted = torch.max(outputs, 1)
                    correct += (predicted == y).sum().item()
                    total += y.size(0)
                    ValLoss = self.model.sample_elbo(inputs=X,
                                               labels=y,
                                               criterion=self.criterion,
                                               sample_nbr=3,
                                               complexity_cost_weight=1/len(self.dataloader_val.dataset))

                accuracy_val = 100 * correct / total
            self.val_accuracy.append(accuracy_val)
                    
            

            
            self.train_loss.append(TrainLoss)
            self.test_loss.append(TestLoss)
            self.val_loss.append(ValLoss)

            #self.train_accuracy = train_accuracy
            #self.test_accuracy = test_accuracy
            #self.val_accuracy = val_accuracy

            print(f'Epoch: {epoch + 1}, trainloss: {self.train_loss[-1]}, testloss: {self.test_loss[-1]}, valloss: {self.val_loss[-1]}')
            print(f'Epoch: {epoch + 1}, trainaccuracy: {self.train_accuracy[-1]}, testaccuracy: {self.test_accuracy[-1]}, valaccuracy: {self.val_accuracy[-1]}')       



        print('Finished Training')


    def trainClosedFormClassification(self):
        train_loss_closed, test_loss_closed, val_loss_closed = [], [], []
        train_log_likelihood_closed, test_log_likelihood_closed, val_log_likelihood_closed = [], [], []
        train_kl_closed, test_kl_closed, val_kl_closed = [], [], []
        m = len(self.dataloader_train.dataset)  # number of samples
        train_accuracy, test_accuracy, val_accuracy = [], [], []
        

        for epoch in range(self.epochs):
            
            outputs = self.model(self.dataloader_train.dataset.tensors[0].to(self.device))
            loss, log_like, scaled_kl = self.objective(outputs, self.dataloader_train.dataset.tensors[1].to(self.device), self.model.kl_divergence(), 1/  m)
            train_loss_closed.append(loss)
            train_log_likelihood_closed.append(log_like)
            train_kl_closed.append(scaled_kl)

            # get the accuracy
            _, predicted = torch.max(outputs, 1)
            correct = (predicted == self.dataloader_train.dataset.tensors[1].to(self.device)).sum().item()
            accuracy = 100 * correct / m
            train_accuracy.append(accuracy)
            print(f'Epoch: {epoch}, Train Loss: {loss}, Accuracy: {accuracy}')

            loss.backward()
            self.optimizer.step()

            for layer in self.model.kl_layers:
                layer.clip_variances()

            # get the test loss
            outputs = self.model(self.dataloader_test.dataset.tensors[0].to(self.device))
            loss, log_like, scaled_kl = self.objective(outputs, self.dataloader_test.dataset.tensors[1].to(self.device), self.model.kl_divergence(), 1/  m)
            test_loss_closed.append(loss)
            test_log_likelihood_closed.append(log_like)
            test_kl_closed.append(scaled_kl)

            # get the accuracy
            _, predicted = torch.max(outputs, 1)
            correct = (predicted == self.dataloader_test.dataset.tensors[1].to(self.device)).sum().item()
            accuracy = 100 * correct / len(self.dataloader_test.dataset)
            test_accuracy.append(accuracy)
            print(f'Epoch: {epoch}, Test Loss: {loss}, Accuracy: {accuracy}')

            # get the validation loss
            outputs = self.model(self.dataloader_val.dataset.tensors[0].to(self.device))
            loss, log_like, scaled_kl = self.objective(outputs, self.dataloader_val.dataset.tensors[1].to(self.device), self.model.kl_divergence(), 1/  m)
            val_loss_closed.append(loss)
            val_log_likelihood_closed.append(log_like)
            val_kl_closed.append(scaled_kl)

            # get the accuracy
            _, predicted = torch.max(outputs, 1)
            correct = (predicted == self.dataloader_val.dataset.tensors[1].to(self.device)).sum().item()
            accuracy = 100 * correct / len(self.dataloader_val.dataset)
            val_accuracy.append(accuracy)
            print(f'Epoch: {epoch}, Validation Loss: {loss}, Accuracy: {accuracy}')


            # define loss and accuracy for the training process to be used in the visualization
            self.train_loss = train_loss_closed
            self.test_loss = test_loss_closed
            self.val_loss = val_loss_closed
            self.train_accuracy = train_accuracy
            self.test_accuracy = test_accuracy
            self.val_accuracy = val_accuracy


        print('Finished Training')


    def visualizeLoss(self):

        # tensor.detach().numpy() converts the tensor to numpy
        train_loss = torch.tensor(self.train_loss).detach().numpy()
        test_loss = torch.tensor(self.test_loss).detach().numpy()
        val_loss = torch.tensor(self.val_loss).detach().numpy()

        # plot the training loss
        plt.plot(train_loss, label='Training Loss')
        plt.plot(test_loss, label='Test Loss')
        plt.plot(val_loss, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Loss Curves')
        plt.legend()
        plt.show()

        # accuracy curves
        plt.plot(self.train_accuracy, label='Training Accuracy')
        plt.plot(self.test_accuracy, label='Test Accuracy')
        plt.plot(self.val_accuracy, label='Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title('Accuracy Curves')
        plt.legend()
        plt.show()

    def saveMetrics(self, path):
    
        train_loss = torch.tensor(self.train_loss).detach().numpy()
        test_loss = torch.tensor(self.test_loss).detach().numpy()
        val_loss = torch.tensor(self.val_loss).detach().numpy()

        # save the metrics
        np.save(f'{path}/train_loss.npy', train_loss)
        np.save(f'{path}/test_loss.npy', test_loss)
        np.save(f'{path}/val_loss.npy', val_loss)
        np.save(f'{path}/train_accuracy.npy', self.train_accuracy)
        np.save(f'{path}/test_accuracy.npy', self.test_accuracy)
        np.save(f'{path}/val_accuracy.npy', self.val_accuracy)



    def get_uncertainty(self):
        '''
        Function to get the uncertainty of the model predictions
        '''
        self.model.eval()
        uncertainty = []
        with torch.no_grad():
            for X, y in self.dataloader_val:
                X, y = X.to(self.device), y.to(self.device)
                outputs = self.model(X)
                uncertainty.append(outputs)

        # rescale the uncertainty to be between 0 and 1
        for i in range(len(uncertainty)):
            uncertainty[i] = torch.nn.functional.softmax(uncertainty[i], dim=1)

        return uncertainty


    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
    
def main():
    args = arg_inputs()

    if args.model == "SimpleFFBNNClassification":
        model = SimpleFFBNNClassification(4, 5)

        run = runBNNClassification(model, dataloader_train, dataloader_test, dataloader_val, device, args.epochs, args.lr, args.criterion, torch.optim.Adam, args.savemodel)
        run.trainClosedFormClassification()
        run.visualizeLoss()
        #uncertainty_simple_model = run.get_uncertainty()
        #for i in range(len(uncertainty_simple_model)):
         #   for j in range(len(uncertainty_simple_model[i])):
          #      print(f'Max Uncertainty: {torch.max(uncertainty_simple_model[i][j])}')
           #     print(f'Min Uncertainty: {torch.min(uncertainty_simple_model[i][j])}')
        kl = run.model.kl_divergence()
        print(f'KL Divergence: {kl}')

        # save the metrics
        run.saveMetrics('/Users/kristian/Documents/Skole/9. Semester/Thesis Preparation/Code/BNNs/ThesisPlots/Results/Classification/SimpleClassification')

        if args.savemodel:
            run.save_model('/Users/kristian/Documents/Skole/9. Semester/Thesis Preparation/Code/BNNs/trainedModels/simple_model.pth')

        # run from terminal with the following command:


        # run the simple model from terminal with the following command:
        # python runBNNClassification.py -m SimpleFFBNNClassification -e 1000 -l 0.0001 -c nn.CrossEntropyLoss() -s True

    elif args.model == "BBBClassification":
        model = BBBClassification(4, 5)
        run = runBNNClassification(model, dataloader_train, dataloader_test, dataloader_val, device, args.epochs, args.lr, args.criterion, torch.optim.Adam, args.savemodel)
        run.trainBBBClassification()
        run.visualizeLoss()
        #uncertainty_bbb_model = run.get_uncertainty()
        #for i in range(len(uncertainty_bbb_model)):
         #   for j in range(len(uncertainty_bbb_model[i])):
          #      print(f'Max Uncertainty: {torch.max(uncertainty_bbb_model[i][j])}')
           #     print(f'Min Uncertainty: {torch.min(uncertainty_bbb_model[i][j])}')
        kl = run.model.kl_divergence()
        print(f'KL Divergence: {kl}')

        # save the metrics
        run.saveMetrics('/Users/kristian/Documents/Skole/9. Semester/Thesis Preparation/Code/BNNs/ThesisPlots/Results/Classification/BBB')

        if args.savemodel:
            run.save_model('/Users/kristian/Documents/Skole/9. Semester/Thesis Preparation/Code/BNNs/trainedModels/bbb_model.pth')

        # run the bbb model from terminal with the following command:
        # python runBNNClassification.py -m BBBClassification -e 1000 -l 0.0001 -c nn.CrossEntropyLoss() -s True

    else:
        model = LargeFFBNNClassification(4, 5)
        print(model)
        run = runBNNClassification(model, dataloader_train, dataloader_test, dataloader_val, device, args.epochs, args.lr, args.criterion, torch.optim.Adam, args.savemodel) 
        run.trainClosedFormClassification()
        run.visualizeLoss()
        #uncertainty_large_model = run.get_uncertainty()
        #for i in range(len(uncertainty_large_model)):
         #   for j in range(len(uncertainty_large_model[i])):
          #      print(f'Max Uncertainty: {torch.max(uncertainty_large_model[i][j])}')
           #     print(f'Min Uncertainty: {torch.min(uncertainty_large_model[i][j])}')

        kl = run.model.kl_divergence()
        print(f'KL Divergence: {kl}')

        # save the metrics
        run.saveMetrics('/Users/kristian/Documents/Skole/9. Semester/Thesis Preparation/Code/BNNs/ThesisPlots/Results/Classification/DenseClassification')

        if args.savemodel:
            run.save_model('/Users/kristian/Documents/Skole/9. Semester/Thesis Preparation/Code/BNNs/trainedModels/large_model.pth')
        
    

if __name__ == "__main__":
    main()

# run the large model from terminal with the following command:
# python runBNNClassification.py -m LargeFFBNNClassification -e 1000 -l 0.0001 -c nn.CrossEntropyLoss() -s True