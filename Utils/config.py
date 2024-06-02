import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset


class custom_data_loader(torch.utils.data.Dataset):

  def __init__(self, df,is_normalize = False):
    self.X = df.loc[:, df.columns != 'target']

    # make 'savings' numeric
    self.X['savings'] = np.where(self.X['savings'] == 'low', 0, np.where(self.X['savings'] == 'medium', 1, 2))

    if is_normalize:
        self.X = (self.X-self.X.mean())/self.X.std()
    
    #self.y = df.loc[:, df.columns == 'target']
    self.X = torch.FloatTensor(self.X.values.astype('float32'))
    
    # make y a tensor
    self.y = torch.FloatTensor(df.target.values)
    
    self.shape = self.X.shape

  def __getitem__(self, idx):
    return self.X[idx], self.y[idx]

  def __len__(self):
    return len(self.X)


class custom_data_loader_classification(torch.utils.data.Dataset):
  
    def __init__(self, df,is_normalize = False):
      self.X = df.loc[:, df.columns != 'target']
  
      # make 'savings' numeric
      self.X['savings'] = np.where(self.X['savings'] == 'low', 0, np.where(self.X['savings'] == 'medium', 1, 2))
  
      if is_normalize:
          self.X = (self.X-self.X.mean())/self.X.std()
      
      #self.y = df.loc[:, df.columns == 'target']
      self.X = torch.FloatTensor(self.X.values.astype('float32'))
      
      # make y a tensor
      self.y = torch.LongTensor(df.target.values)
      
      self.shape = self.X.shape
  
    def __getitem__(self, idx):
      return self.X[idx], self.y[idx]
  
    def __len__(self):
      return len(self.X)

class custom_data_loader_EPICLE(torch.utils.data.Dataset):

  def __init__(self, df,is_normalize = False):
    self.X = df.loc[:, df.columns != 'target']

    if is_normalize:
        self.X = (self.X-self.X.mean())/self.X.std()
    
    #self.y = df.loc[:, df.columns == 'target']
    self.X = torch.FloatTensor(self.X.values.astype('float32'))
    
    # make y a tensor
    self.y = torch.FloatTensor(df.target.values)
    
    self.shape = self.X.shape

  def __getitem__(self, idx):
    return self.X[idx], self.y[idx]

  def __len__(self):
    return len(self.X)

# a function that preprocesses the data
def preprocess_data(df, batch_size):
    df_custom = pd.read_csv('/Users/kristian/Documents/Skole/9. Semester/Thesis Preparation/Code/BNNs/Data/quality_of_food.csv')
    df_custom = custom_data_loader(df_custom, is_normalize=True)
    scaler = StandardScaler()
    X = df_custom.X
    y = df_custom.y
    X = scaler.fit_transform(X)
    y = scaler.fit_transform(y.reshape(-1,1))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=11)
    X_train, y_train = torch.FloatTensor(X_train), torch.FloatTensor(y_train)
    X_test, y_test = torch.FloatTensor(X_test), torch.FloatTensor(y_test)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=11)
    dataset_train = TensorDataset(X_train, y_train)
    dataset_test = TensorDataset(X_test, y_test)
    dataset_val = TensorDataset(X_val, y_val)
    dataloader_train = DataLoader(dataset_train, batch_size= batch_size, shuffle=True)
    dataloader_test = DataLoader(dataset_test, batch_size= batch_size, shuffle=True)
    dataloader_val = DataLoader(dataset_val, batch_size= batch_size, shuffle=True)
    return dataloader_train, dataloader_test, dataloader_val, dataset_train, dataset_test, dataset_val

def preprocess_activeL_data():
  '''df_custom is used to scale and rescale with in the annotation function'''
  
  #df_custom = pd.read_csv('/Users/kristian/Documents/Skole/9. Semester/Thesis Preparation/Code/BNNs/Data/quality_of_food.csv')
  df_custom = pd.read_csv('/Users/kristian/Documents/Skole/9. Semester/Thesis Preparation/Code/BNNs/Data/quality_of_food_reg_balanced.csv')
  df_custom = custom_data_loader(df_custom, is_normalize=False)
  scaler = StandardScaler()
  X = df_custom.X
  y = df_custom.y
  X = scaler.fit_transform(X)
  y = scaler.fit_transform(y.reshape(-1,1))
  # split into data to train the seed model and data to be used for active learning
  X_seed, X_activeL, y_seed, y_activeL = train_test_split(X, y, test_size=0.1, random_state=8)
  X_seed, y_seed = torch.FloatTensor(X_seed), torch.FloatTensor(y_seed)
  X_activeL, y_activeL = torch.FloatTensor(X_activeL), torch.FloatTensor(y_activeL)
  # create dataset for training the seed model in active learning 
  dataset_activeL = TensorDataset(X_activeL, y_activeL)
  # split seed data into training and test data
  X_train, X_test, y_train, y_test = train_test_split(X_seed, y_seed, test_size=0.3, random_state=8)
  X_train, y_train = torch.FloatTensor(X_train), torch.FloatTensor(y_train)
  X_test, y_test = torch.FloatTensor(X_test), torch.FloatTensor(y_test)
  dataset_train = TensorDataset(X_train, y_train)
  dataset_test = TensorDataset(X_test, y_test)
  # return train and test for the seed model and the active learning dataset 
  return dataset_train, dataset_test, dataset_activeL, df_custom

def preprocess_activeL_EPICLE_data():
  '''df_custom is used to scale and rescale with in the annotation function'''
  
  #df_custom = pd.read_csv('/Users/kristian/Documents/Skole/9. Semester/Thesis Preparation/Code/BNNs/Data/HHC07_S06_20190403_RR.csv')
  #df_custom = pd.read_csv('/Users/kristian/Documents/Skole/9. Semester/Thesis Preparation/Code/BNNs/Data/LSM13_S03_20170919_P_RR.csv') # naming convention is wrong, but it is the correct file
  df_custom = pd.read_csv('/Users/kristian/Documents/Skole/9. Semester/Thesis Preparation/Code/BNNs/Data/MPI5_S11_20170518.csv')
  df_custom = custom_data_loader_EPICLE(df_custom, is_normalize=False)
  scaler = StandardScaler()
  X = df_custom.X
  y = df_custom.y
  X = scaler.fit_transform(X)
  y = scaler.fit_transform(y.reshape(-1,1))
  # split into data to train the seed model and data to be used for active learning
  X_seed, X_activeL, y_seed, y_activeL = train_test_split(X, y, test_size=0.1, random_state=8)
  X_seed, y_seed = torch.FloatTensor(X_seed), torch.FloatTensor(y_seed)
  X_activeL, y_activeL = torch.FloatTensor(X_activeL), torch.FloatTensor(y_activeL)
  # create dataset for training the seed model in active learning 
  dataset_activeL = TensorDataset(X_activeL, y_activeL)
  # split seed data into training and test data
  X_train, X_test, y_train, y_test = train_test_split(X_seed, y_seed, test_size=0.3, random_state=8)
  X_train, y_train = torch.FloatTensor(X_train), torch.FloatTensor(y_train)
  X_test, y_test = torch.FloatTensor(X_test), torch.FloatTensor(y_test)
  dataset_train = TensorDataset(X_train, y_train)
  dataset_test = TensorDataset(X_test, y_test)
  # return train and test for the seed model and the active learning dataset 
  return dataset_train, dataset_test, dataset_activeL, df_custom



def preprocess_activeL_all_data():
  '''df_custom is used to scale and rescale with in the annotation function'''
  #df_custom = pd.read_csv('/Users/kristian/Documents/Skole/9. Semester/Thesis Preparation/Code/BNNs/Data/quality_of_food.csv')
  df_custom = pd.read_csv('/Users/kristian/Documents/Skole/9. Semester/Thesis Preparation/Code/BNNs/Data/quality_of_food_reg_balanced.csv')
  df_custom = custom_data_loader(df_custom, is_normalize=False)
  scaler = StandardScaler()
  X = df_custom.X
  y = df_custom.y
  X = scaler.fit_transform(X)
  y = scaler.fit_transform(y.reshape(-1,1))

  # split into test and train data
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=8)  
  X_train, y_train = torch.FloatTensor(X_train), torch.FloatTensor(y_train)
  X_test, y_test = torch.FloatTensor(X_test), torch.FloatTensor(y_test)
  dataset_train = TensorDataset(X_train, y_train)
  dataset_test = TensorDataset(X_test, y_test)

  """The following is a hack to make the pipeline work
  However, it is not  good practice to make dataset_activeL the same as dataset_train
  If time permits, this should be fixed
  """
    
  # make dataset_activeL the same as dataset_train
  dataset_activeL = dataset_train  


  # return train and test for the seed model and the active learning dataset 
  return dataset_train, dataset_test, dataset_activeL, df_custom


def preprocess_activeL_data_classification():
    #df_custom = pd.read_csv('/Users/kristian/Documents/Skole/9. Semester/Thesis Preparation/Code/BNNs/Data/quality_of_food_int.csv')
    #df_custom = pd.read_csv('/Users/kristian/Documents/Skole/9. Semester/Thesis Preparation/Code/BNNs/Data/quality_of_food_nonoise.csv')
    #df_custom = pd.read_csv('/Users/kristian/Documents/Skole/9. Semester/Thesis Preparation/Code/BNNs/Data/quality_of_food_int_balanced.csv')
    df_custom = pd.read_csv('/Users/kristian/Documents/Skole/9. Semester/Thesis Preparation/Code/BNNs/Data/quality_of_food_nonoise_balanced.csv') # easiest to work with

    df_custom = custom_data_loader_classification(df_custom, is_normalize=False)
    scaler = StandardScaler()
    X = df_custom.X
    y = df_custom.y

    def make_zero_based(y):

      """Zero base the target variable"""
      for i in range(len(y)):
        y[i] = y[i] - 1
      return y
    y = make_zero_based(y)

    X = scaler.fit_transform(X)
    X = X

    # split into data to train the seed model and data to be used for active learning
    X_seed, X_activeL, y_seed, y_activeL = train_test_split(X, y, test_size=0.1, random_state=8)
    X_seed, y_seed = torch.FloatTensor(X_seed), torch.LongTensor(y_seed)
    X_activeL, y_activeL = torch.FloatTensor(X_activeL), torch.LongTensor(y_activeL)
    # create dataset for training the seed model in active learning 
    dataset_activeL = TensorDataset(X_activeL, y_activeL)
    # split seed data into training and test data
    X_train, X_test, y_train, y_test = train_test_split(X_seed, y_seed, test_size=0.3, random_state=8)
    X_train, y_train = torch.FloatTensor(X_train), torch.LongTensor(y_train)
    X_test, y_test = torch.FloatTensor(X_test), torch.LongTensor(y_test)
    dataset_train = TensorDataset(X_train, y_train)
    dataset_test = TensorDataset(X_test, y_test)
    # return train and test for the seed model and the active learning dataset 
    return dataset_train, dataset_test, dataset_activeL, df_custom

def preprocess_activeL_data_classification_alldata():
    #df_custom = pd.read_csv('/Users/kristian/Documents/Skole/9. Semester/Thesis Preparation/Code/BNNs/Data/quality_of_food_int.csv')
    df_custom = pd.read_csv('/Users/kristian/Documents/Skole/9. Semester/Thesis Preparation/Code/BNNs/Data/quality_of_food_int_balanced.csv')
    #df_custom = pd.read_csv('/Users/kristian/Documents/Skole/9. Semester/Thesis Preparation/Code/BNNs/Data/quality_of_food_nonoise_balanced.csv')
    df_custom = custom_data_loader_classification(df_custom, is_normalize=False)
    scaler = StandardScaler()
    X = df_custom.X
    y = df_custom.y

    def make_zero_based(y):

      """Zero base the target variable"""
      for i in range(len(y)):
        y[i] = y[i] - 1
      return y
    y = make_zero_based(y)

    X = scaler.fit_transform(X)
    
    # split into test and train data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=8)
    X_train, y_train = torch.FloatTensor(X_train), torch.LongTensor(y_train)
    X_test, y_test = torch.FloatTensor(X_test), torch.LongTensor(y_test)
    dataset_train = TensorDataset(X_train, y_train)
    dataset_test = TensorDataset(X_test, y_test)

    """The following is a hack to make the pipeline work
    However, it is not a good practice to make dataset_activeL the same as dataset_train
    If time permits, this should be fixed
    """
    # make dataset_activeL the same as dataset_train
    dataset_activeL = dataset_train  

    return dataset_train, dataset_test, dataset_activeL, df_custom



def preprocess_classification_data(df, batch_size):
  df_custom = pd.read_csv('/Users/kristian/Documents/Skole/9. Semester/Thesis Preparation/Code/BNNs/Data/quality_of_food_int.csv')
  df_custom = custom_data_loader_classification(df_custom, is_normalize=True)
  scaler = StandardScaler()
  X = df_custom.X
  y = df_custom.y
  #X = scaler.fit_transform(X)
  #y = scaler.fit_transform(y.reshape(-1,1))
 
  def make_zero_based(y):
    """Zero base the target variable"""
    for i in range(len(y)):
      y[i] = y[i] - 1
    return y
  y = make_zero_based(y)

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 11)
  X_train, y_train = torch.FloatTensor(X_train), torch.LongTensor(y_train)
  X_test, y_test = torch.FloatTensor(X_test), torch.LongTensor(y_test)
  X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size = 0.5, random_state = 11)

  dataset_train = TensorDataset(X_train, y_train)
  dataset_test = TensorDataset(X_test, y_test)
  dataset_val = TensorDataset(X_val, y_val)
  dataloader_train = DataLoader(dataset_train, batch_size = batch_size, shuffle=True)
  dataloader_test = DataLoader(dataset_test, batch_size = batch_size, shuffle=False)
  dataloader_val = DataLoader(dataset_val, batch_size = batch_size, shuffle=False)
  return dataloader_train, dataloader_test, dataloader_val


def preprocess_classification_activeL_data(df):
    df_custom = pd.read_csv('/Users/kristian/Documents/Skole/9. Semester/Thesis Preparation/Code/BNNs/Data/quality_of_food_int.csv')
    df_custom = custom_data_loader_classification(df_custom, is_normalize=True)
    scaler = StandardScaler()
    X = df_custom.X
    y = df_custom.y

    def make_zero_based(y):
      """Zero base the target variable"""
      for i in range(len(y)):
        y[i] = y[i] - 1
      return y
    y = make_zero_based(y)

    #X = scaler.fit_transform(X)
    #y = scaler.fit_transform(y.reshape(-1,1))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 11)
    X_train, y_train = torch.FloatTensor(X_train), torch.LongTensor(y_train)
    X_test, y_test = torch.FloatTensor(X_test), torch.LongTensor(y_test)
    dataset_train = TensorDataset(X_train, y_train)
    dataset_test = TensorDataset(X_test, y_test)
    dataloader_train = DataLoader(dataset_train, batch_size=64, shuffle=True)
    dataloader_test = DataLoader(dataset_test, batch_size=64, shuffle=False)
    return dataloader_train, dataloader_test
