"""
The code containing neural network part, Skrlj 2019
"""

import torch
torch.manual_seed(123321)
import tqdm
import torch.nn as nn
from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import DataLoader, Dataset
import logging
import numpy as np
np.random.seed(123321)

logging.basicConfig(format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
logging.getLogger().setLevel(logging.INFO)


class E2EDatasetLoader(Dataset):

    """
    A standard dataloader instance, note the csr subsetting.
    """
    
    def __init__(self, features, targets = None, transform=None):
        self.features = features.tocsr()

        if not targets is None:
            self.targets = targets.tocsr()
        else:
            self.targets = targets

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, index):
        instance = torch.from_numpy(self.features[index, :].todense())
        if self.targets is not None:
            target = torch.from_numpy(self.targets[index].todense())
        else:
            target = None
        return instance, target

def to_one_hot(lbx):
    enc = OneHotEncoder(handle_unknown='ignore')
    return enc.fit_transform(lbx.reshape(-1, 1))

class SimpleArch(nn.Module):
    def __init__(self, input_size, dropout = 0.1, hidden_layer_size = 10, output_neurons = 1):

        """
        A simple architecture wrapper -- build with intuitive Sklearn-like API.
        """
        
        super(SimpleArch, self).__init__()    
        self.h1 = nn.Linear(input_size, hidden_layer_size) ## prvi hidden
        self.h2 = nn.Linear(hidden_layer_size, hidden_layer_size) ## drugi hidden -> dimenzija je  ista kot od prvega.
        self.h3 = nn.Linear(hidden_layer_size, 16) ## Tretji hidden ..
        self.h4 = nn.Linear(16, output_neurons)
        self.drop = nn.Dropout(dropout)
        self.act = nn.ELU()
        self.sigma = nn.Sigmoid()

    def forward(self, x):

        """
        The standard forward pass. See the original paper for the formal description of this part of the DRMs. 
        """
        
        out = self.h1(x)
        out = self.drop(out)
        out = self.act(out)
        
        out = self.h2(out)
        out = self.drop(out)
        out = self.act(out)
        
        out = self.h3(out)
        out = self.drop(out)
        out = self.act(out)

        out = self.h4(out)
        out = self.sigma(out)

        return out

class E2EDNN:

    """
    This is the main DRM class. The idea is to have a scikit-learn like interface for construction of ffNNs, capable of handling CSR-like inputs.
    """
    
    def __init__(self, batch_size=8, num_epochs=10, learning_rate=0.0001, stopping_crit=10, hidden_layer_size=30,dropout=0.2, file_type=None, auto_depth=0):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.loss = torch.nn.BCELoss()
        self.auto_depth = auto_depth
        self.dropout = dropout
        self.stopping_crit = stopping_crit
        self.num_epochs = num_epochs
        self.hidden_layer_size = hidden_layer_size
        self.learning_rate = learning_rate
        self.model = None
        self.optimizer = None
        self.num_params = None

    def init_all(self, model, init_func, *params, **kwargs):
        for p in model.parameters():
            init_func(p, *params, **kwargs)

    def fit(self, features, labels, onehot=False):

        nun = len(np.unique(labels))
        logging.info("Found {} unique labels.".format(nun))
        labels = to_one_hot(labels)
        train_dataset = E2EDatasetLoader(features, labels)
        stopping_iteration = 0
        loss = 1
        current_loss = 0
        self.model = SimpleArch(features.shape[1], dropout = self.dropout, hidden_layer_size = self.hidden_layer_size, output_neurons = nun).to(self.device)
#        self.init_all(self.model, torch.nn.init.normal_, mean=0., std=1)
#        self.init_all(self.model, torch.nn.init.uniform_)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.num_params = sum(p.numel() for p in self.model.parameters())
        logging.info("Number of parameters {}".format(self.num_params))
        logging.info("Starting training for {} epochs".format(self.num_epochs))
        for epoch in range(self.num_epochs): ## simple loss-based stopping works OK. TODO -> validation if more data is available.
            if current_loss != loss:
                current_loss = loss
            else:
                stopping_iteration += 1
            if stopping_iteration > self.stopping_crit:
                logging.info("Stopping reached!")
                break
            losses_per_batch = []
            for i, (features, labels) in enumerate(train_dataset):
                features = features.float().to(self.device)
                labels = labels.float().to(self.device)
                self.model.train()
                outputs = self.model.forward(features).view(-1,1)
                labels = labels.view(-1,1)
                loss = self.loss(outputs, labels)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                losses_per_batch.append(float(loss))
            mean_loss = np.mean(losses_per_batch)
            logging.info("epoch {}, mean loss per batch {}".format(epoch, mean_loss))

    def predict(self, features, return_proba = False):

        """
        Classic, sklearn-like predict method.
        """
        
        test_dataset = E2EDatasetLoader(features, None)
        predictions = []
        self.model.eval()
        with torch.no_grad():
            for features, _ in test_dataset:
                features = features.float().to(self.device)
                representation = self.model.forward(features)
                pred = representation.detach().cpu().numpy()[0]                
                predictions.append(pred)
        if not return_proba:
            a = [np.argmax(a_) for a_ in predictions] ## assumes 0 is 0         
        else:
            a = []
            for pred in predictions:
                a.append(pred[1])
            
        return np.array(a).flatten()

    def predict_proba(self, features):

        """
        It is also possible to obtain probabilistic outputs!
        """
        
        test_dataset = E2EDatasetLoader(features, None)
        predictions = []
        self.model.eval()
        with torch.no_grad():
            for features, _ in test_dataset:
                features = features.float().to(self.device)
                representation = self.model.forward(features)
                pred = representation.detach().cpu().numpy()[0]                
                predictions.append(pred)
        a = [a_[1] for  a_ in predictions]
        return np.array(a).flatten()
