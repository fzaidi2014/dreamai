import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import time
from collections import defaultdict
from dreamai.utils import *
from dreamai.model import *



class FC(Network):
    def __init__(self,
                 num_inputs=10,
                 num_outputs=10,
                 layers=[],
                 lr=0.003,
                 class_names=None,
                 optimizer_name='Adam',
                 dropout_p=0.2,
                 hidden_non_linearity='relu',
                 output_non_linearity=None,
                 criterion_name='NLLLoss',
                 model_name='FC',
                 model_type ='classifier',
                 best_accuracy=0.,
                 best_validation_loss=None,
                 best_model_file ='best_accuracy.pth',
                 chkpoint_file ='chkpoint_file.pth',
                 device=None):
        
        super().__init__(device=device)
        
        self.hidden_non_linearity = hidden_non_linearity
        
        self.model = nn.Sequential()
        
        if len(layers) > 0:
            self.model.add_module('fc1',nn.Linear(num_inputs,layers[0]))
            self.model.add_module(hidden_non_linearity+'1',nn.ReLU())
            self.model.add_module('dropout1',nn.Dropout(p=dropout_p,inplace=True))

            for i in range(1,len(layers)):
                self.model.add_module('fc'+str(i+1),nn.Linear(layers[i-1],layers[i]))
                self.model.add_module(hidden_non_linearity+str(i+1),nn.ReLU())
                self.model.add_module('dropout'+str(i+1),nn.Dropout(p=dropout_p,
                                                                    inplace=True))

            self.model.add_module('out',nn.Linear(layers[-1],num_outputs))
        else:
            self.model.add_module('out',nn.Linear(num_inputs,num_outputs))
            
        
        if model_type.lower() == 'classifier' and criterion_name.lower() == 'nllloss':
            self.model.add_module('logsoftmax',nn.LogSoftmax(dim=1))   
        elif (model_type.lower() == 'regressor' or model_type.lower() == 'recommender') and output_non_linearity is not None:
            print('output non linearity = {}'.format(output_non_linearity))
            if output_non_linearity.lower() == 'sigmoid':
                self.model.add_module(output_non_linearity,nn.Sigmoid())
                self.output_non_linearity = output_non_linearity

        self.to(self.device)
        self.model.to(self.device)
        
        self.set_model_params(criterion_name,
                              optimizer_name,
                              lr,
                              dropout_p,
                              model_name,
                              model_type,
                              best_accuracy,
                              best_validation_loss,
                              best_model_file,
                              chkpoint_file,
                              num_inputs,
                              num_outputs,
                              layers,class_names)
            
    def forward(self,x):
        return self.model(flatten_tensor(x))
    
    def _get_dropout(self):
        for layer in self.model:
            if type(layer) == torch.nn.modules.dropout.Dropout:
                return layer.p
            
    def _set_dropout(self,p=0.2):
        for layer in self.model:
            if type(layer) == torch.nn.modules.dropout.Dropout:
                print('FC: setting dropout prob to {:.3f}'.format(p))
                layer.p=p
                
    def set_model_params(self,
                         criterion_name,
                         optimizer_name,
                         lr,
                         dropout_p,
                         model_name,
                         model_type,
                         best_accuracy,
                         best_validation_loss,
                         best_model_file,
                         chkpoint_file,
                         num_inputs,
                         num_outputs,
                         layers,
                         class_names):
        
        
        super(FC, self).set_model_params(
                              criterion_name,
                              optimizer_name,
                              lr,
                              dropout_p,
                              model_name,
                              model_type,
                              best_accuracy,
                              best_validation_loss,
                              best_model_file,
                              chkpoint_file
                              )
        
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.layer_dims = layers

        if self.model_type == 'classifier':
            if class_names is not None:
                self.class_names = class_names
            else:
                self.class_names = {k:str(v) for k,v in enumerate(list(range(num_outputs)))}
        
    def get_model_params(self):
        params = super(FC, self).get_model_params()
        params['num_inputs'] = self.num_inputs
        params['num_outputs'] = self.num_outputs
        params['layers'] = self.layer_dims
        params['model_type'] = self.model_type
        if self.model_type == 'classifier':
            params['class_names'] = self.class_names
        params['device'] = self.device
        return params
        
        