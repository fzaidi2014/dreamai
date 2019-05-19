import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import time
from collections import defaultdict
from dreamai.utils import *
from dreamai.model import *
from dreamai.fc import *

class RecommenderNet(Network):
    def __init__(self,
                 num_users=1000,
                 num_items=10000,
                 layers=[512],
                 embedding_dim=100,
                 lr=0.003,
                 optimizer_name='Adam',
                 dropout_p=0.2,
                 embedding_dropout_p=0.02,
                 hidden_non_linearity='relu',
                 output_non_linearity=None,
                 criterion_name='MSELoss',
                 model_type ='recommender',
                 model_name='RecommendNet',
                 best_validation_loss=None,
                 best_model_file ='best_accuracy.pth',
                 chkpoint_file ='chkpoint_file.pth',
                 num_classes=None,
                 best_accuracy=0.,
                 device=None):
        
        super().__init__(device=device)
        
        self.users_embedding = nn.Embedding(num_users,embedding_dim)
        self.items_embedding = nn.Embedding(num_items,embedding_dim)

        class_names = None
        if num_classes is not None:
            fc_model_type = 'classifier'
            outputs = num_classes
            class_names = {k:str(v) for k,v in enumerate(list(range(num_classes)))}
        else:
            fc_model_type = 'recommender'
            outputs = 1

        self.class_names = class_names
        print('class_names = {}'.format(class_names))
        print('0 device = {}'.format(self.device))
        self.set_model_params(
                         criterion_name,
                         optimizer_name,
                         lr,
                         dropout_p,
                         model_name,
                         model_type,
                         best_model_file,
                         chkpoint_file,
                         layers,
                         num_users,
                         num_items,
                         embedding_dim,
                         embedding_dropout_p,
                         hidden_non_linearity,
                         output_non_linearity,
                         best_validation_loss,
                         best_accuracy,
                         num_classes)

        print('1 device = {}'.format(self.device))
        self.model = FC(num_inputs=(2*embedding_dim),
                         num_outputs=outputs,
                         layers= layers,
                         lr=lr,
                         optimizer_name=optimizer_name,
                         dropout_p=dropout_p,
                         hidden_non_linearity=hidden_non_linearity,
                         output_non_linearity=output_non_linearity,
                         criterion_name=criterion_name,
                         model_type = fc_model_type,
                         class_names=class_names,
                         model_name='FC',
                         best_validation_loss=best_validation_loss,
                         best_model_file =best_model_file,
                         chkpoint_file =chkpoint_file,
                         device=self.device)
        
        self.to(self.device)
        print('2 device = {}'.format(self.device))
        self.users_embedding = self.users_embedding.to(self.device)
        self.items_embedding = self.items_embedding.to(self.device)
        self.embedding_dropout =  nn.Dropout(embedding_dropout_p)
        print('3 device = {}'.format(self.device))
        self._init()
        print('4 device = {}'.format(self.device))
        
    def _init(self):
        """
        Setup embeddings and hidden layers with reasonable initial values.
        """         
        for layer in self.model.model:
            if type(layer) == nn.Linear:
                torch.nn.init.xavier_uniform_(layer.weight)
                layer.bias.data.fill_(0.01)

        self.users_embedding.weight.data.uniform_(-0.05, 0.05)
        self.items_embedding.weight.data.uniform_(-0.05, 0.05)


    def set_model_params(self,
                         criterion_name,
                         optimizer_name,
                         lr,
                         dropout_p,
                         model_name,
                         model_type,
                         best_model_file,
                         chkpoint_file,
                         layers,
                         num_users,
                         num_items,
                         embedding_dim,
                         embedding_dropout_p,
                         hidden_non_linearity,
                         output_non_linearity,
                         best_validation_loss,
                         best_accuracy,
                         num_classes):
        
        
        super(RecommenderNet,self).set_model_params(
                              criterion_name,
                              optimizer_name,
                              lr,
                              dropout_p,
                              model_name,
                              model_type,
                              0.,
                              best_validation_loss,                      
                              best_model_file,
                              chkpoint_file
                              )
        
        self.best_validation_loss = best_validation_loss
        self.num_users = num_users
        self.num_items = num_items
        self.layers = layers
        self.embedding_dim = embedding_dim
        self.embedding_dropout_p = embedding_dropout_p
        self.hidden_non_linearity = hidden_non_linearity
        self.output_non_linearity = output_non_linearity
        self.num_classes = num_classes
        self.best_accuracy = best_accuracy
    
    def get_model_params(self):
        params = super(FC, self).get_model_params()
        params['num_users']  =             self.num_users
        params['num_items']  =             self.num_items
        params['layers']   =             self.layers
        params['embedding_dim']   =      self.embedding_dim
        params['embedding_dropout_p'] =  self.embedding_dropout_p
        params['hidden_non_linearity'] = self.hidden_non_linearity
        params['output_non_linearity'] = self.output_non_linearity
        if self.num_classes is not None:
            params['num_classes'] = self.num_classes
        return params
        
    def forward(self,inputs):
        users,items = inputs[:,0],inputs[:,1]
        users = users.to(self.device)
        items = items.to(self.device)
        features = torch.cat([self.users_embedding(users), self.items_embedding(items)], dim=1)
        features = features.to(self.device)
        features = self.embedding_dropout(features)
        return self.model(features.contiguous())

            
    