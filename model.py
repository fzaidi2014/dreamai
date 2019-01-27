import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import time
from collections import defaultdict
from dreamai.utils import *
import numpy as np


class Classifier():
    def __init__(self,class_names):
        self.class_names = class_names
        self.class_correct = defaultdict(int)
        self.class_totals = defaultdict(int)

    def update_accuracies(self,outputs,labels):
        _, preds = torch.max(torch.exp(outputs), 1)
        correct = np.squeeze(preds.eq(labels.data.view_as(preds)))
        for i in range(labels.shape[0]):
            label = labels.data[i].item()
            self.class_correct[label] += correct[i].item()
            self.class_totals[label] += 1

    def get_final_accuracies(self):
        accuracy = (100*np.sum(list(self.class_correct.values()))/np.sum(list(self.class_totals.values())))
        class_accuracies = [(self.class_names[i],100.0*(self.class_correct[i]/self.class_totals[i])) 
                                 for i in self.class_names.keys() if self.class_totals[i] > 0]
        return accuracy,class_accuracies

    


class Network(nn.Module):
    def __init__(self,device=None):
        super().__init__()
        if device is not None:
            self.device = device
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(self.device)

    def forward(self,x):
        pass
    
    def train_(self,trainloader,criterion,optimizer,print_every):
        self.train()
        t0 = time.time()
        batches = 0
        running_loss = 0.
        for inputs, labels in trainloader:
            batches += 1
            #t1 = time.time()
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            optimizer.zero_grad()
            outputs = self.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            loss = loss.item()
            #print('training this batch took {:.3f} seconds'.format(time.time() - t1))
            running_loss += loss
            
            if batches % print_every == 0:
                print(f"{time.asctime()}.."
                        f"Time Elapsed = {time.time()-t0:.3f}.."
                        f"Batch {batches+1}/{len(trainloader)}.. "
                        f"Average Training loss: {running_loss/(batches):.3f}.. "
                        f"Batch Training loss: {loss:.3f}.. "
                        )
                t0 = time.time()
           
        return running_loss/len(trainloader) 

    def evaluate(self,dataloader,metric='accuracy'):
        
        running_loss = 0.
        classifier = None

        if self.model_type == 'classifier':# or self.num_classes is not None:
           classifier = Classifier(self.class_names)

        self.eval()
        rmse_ = 0.
        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.forward(inputs)
                loss = self.criterion(outputs, labels)
                running_loss += loss.item()
                if classifier is not None and metric == 'accuracy':
                     classifier.update_accuracies(outputs,labels)
                elif metric == 'rmse':
                    rmse_ += rmse(outputs,labels).cpu().numpy()
            
        self.train()

        ret = {}
        print('running_loss = {:.3f}'.format(running_loss))
        print('total rmse = {:.3f}'.format(rmse_))
        ret['final_loss'] = running_loss/len(dataloader)

        if classifier is not None:
            ret['accuracy'],ret['class_accuracies'] = classifier.get_final_accuracies()
        elif metric == 'rmse':
            ret['final_rmse'] = rmse_/len(dataloader)

        return ret
    
   
    def classify(self,inputs,topk=1):
        self.eval()
        self.model.to(self.device)
        with torch.no_grad():
            inputs = inputs.to(self.device)
            outputs = self.forward(inputs)
            ps = torch.exp(outputs)
            p,top = ps.topk(topk, dim=1)
        return p,top

    def predict(self,inputs):
        self.eval()
        self.model.to(self.device)
        with torch.no_grad():
            inputs = inputs.to(self.device)
            outputs = self.forward(inputs)
        return outputs
    
    def fit(self,trainloader,validloader,epochs=2,print_every=10,validate_every=1,save_best_every=1):
           
        for epoch in range(epochs):
            self.model.to(self.device)
            print('epoch {:3d}/{}'.format(epoch+1,epochs))
            epoch_train_loss =  self.train_(trainloader,self.criterion,
                                            self.optimizer,print_every)
                    
            if  validate_every and (epoch % validate_every == 0):
                t2 = time.time()
                eval_dict = self.evaluate(validloader)
                epoch_validation_loss = eval_dict['final_loss']
                
                time_elapsed = time.time() - t2
                print(f"{time.asctime()}--Validation time {time_elapsed:.3f} seconds.."
                      f"Epoch {epoch+1}/{epochs}.. "
                      f"Epoch Training loss: {epoch_train_loss:.3f}.. "
                      f"Epoch validation loss: {epoch_validation_loss:.3f}.. ")

                if self.model_type == 'classifier':# or self.num_classes is not None:
                    epoch_accuracy = eval_dict['accuracy']
                    print("validation accuracy: {:.3f}".format(epoch_accuracy))
                               
                    if self.best_accuracy == 0. or (epoch_accuracy > self.best_accuracy):
                        print('updating best accuracy: previous best = {:.3f} new best = {:.3f}'.format(self.best_accuracy,
                                                                                     epoch_accuracy))
                        self.best_accuracy = epoch_accuracy
                        torch.save(self.state_dict(),self.best_model_file)

                elif (self.model_type.lower() == 'regressor' or self.model_type.lower() == 'recommender') and (epoch % save_best_every == 0):
                    if self.best_validation_loss == None or (epoch_validation_loss < self.best_validation_loss):
                        if self.best_validation_loss is not None:
                            print('updating best validation loss: previous best = {:.7f}'.format(self.best_validation_loss))
                        print('New best loss = {:.7f}'.format(epoch_validation_loss))

                        self.best_validation_loss = epoch_validation_loss
                        torch.save(self.state_dict(),self.best_model_file)
                    
                self.train() # just in case we forgot to put the model back to train mode in validate
                
        print('loading best model')
        self.load_state_dict(torch.load(self.best_model_file))
                
     
                
    def set_criterion(self, criterion):

        if criterion:
            self.criterion = criterion

        # if criterion_obj is None:        
        #     if criterion_name.lower() == 'nllloss':
        #         self.criterion_name = 'NLLLoss'
        #         self.criterion = nn.NLLLoss()
        #     elif criterion_name.lower() == 'crossentropyloss':
        #         self.criterion_name = 'CrossEntropyLoss'
        #         self.criterion = nn.CrossEntropyLoss()
        #     elif criterion_name.lower() == 'mseloss':
        #         self.criterion_name = 'MSELoss'
        #         self.criterion = nn.MSELoss()
        # else:
        #     print(str(criterion_)[:-2])
        #     print(self.criterion_name)
        #     self.criterion_name = criterion_obj.__class__.__name__
        #     self.criterion = criterion_obj
        

    def set_optimizer(self,params,optimizer_name='adam',lr=0.003):
        from torch import optim
        if optimizer_name:
            if optimizer_name.lower() == 'adam':
                print('setting optim Adam')
                self.optimizer = optim.Adam(params,lr=lr)
                self.optimizer_name = optimizer_name
            elif optimizer_name.lower() == 'sgd':
                print('setting optim SGD')
                self.optimizer = optim.SGD(params,lr=lr)
            elif optimizer_name.lower() == 'adadelta':
                print('setting optim Ada Delta')
                self.optimizer = optim.Adadelta(params)       
            
    def set_model_params(self,
                         criterion,
                         optimizer_name,
                         lr,
                         dropout_p,
                         model_name,
                         model_type,
                         best_accuracy,
                         best_validation_loss,
                         best_model_file,
                         chkpoint_file):
        
        self.set_criterion(criterion)
        self.optimizer_name = optimizer_name
        self.set_optimizer(self.parameters(),optimizer_name,lr=lr)
        self.lr = lr
        self.dropout_p = dropout_p
        self.model_name =  model_name
        self.model_type = model_type
        self.best_accuracy = best_accuracy
        self.best_validation_loss = best_validation_loss
        #print('set_model_params: best accuracy = {:.3f}'.format(self.best_accuracy))  
        self.best_model_file = best_model_file
        self.chkpoint_file = chkpoint_file
    
    def get_model_params(self):
        params = {}
        params['device'] = self.device
        params['model_type'] = self.model_type
        params['model_name'] = self.model_name
        params['optimizer_name'] = self.optimizer_name
        params['criterion'] = self.criterion
        params['lr'] = self.lr
        params['dropout_p'] = self.dropout_p
        params['best_accuracy'] = self.best_accuracy
        params['best_validation_loss'] = self.best_validation_loss
        print('get_model_params: best accuracy = {:.3f}'.format(self.best_accuracy))  
        params['best_model_file'] = self.best_model_file
        params['chkpoint_file'] = self.chkpoint_file
        print('get_model_params: chkpoint file = {}'.format(self.chkpoint_file))  
        return params
    
    def save_chkpoint(self):
        saved_model = {}
        saved_model['params'] = self.get_model_params()    
        torch.save(saved_model,self.chkpoint_file)
        print('checkpoint created successfully in {}'.format(self.chkpoint_file))
    
    def freeze(self):
        for param in self.model.parameters():
            param.requires_grad = False
        
        
    def unfreeze(self):
        for param in self.model.parameters():
            param.requires_grad = True


class EnsembleModel(Network):
    def __init__(self,models):
        self.criterion = None
        super().__init__()
        self.models = models
        if sum(model[1] for model in models) != 1.0:
            raise ValueError('Weights of Ensemble must sum to 1')
            
        
    def evaluate(self,dataloader,metric='accuracy'):
        from collections import defaultdict

        if metric == 'accuracy':
           class_names = self.models[0][0].class_names   
           classifier = Classifier(class_names)

        
        with torch.no_grad():
            
            for inputs, labels in dataloader:
                preds_list = []  
                for model in self.models:
                    model[0].eval()
                    model[0].to(model[0].device)
                    inputs, labels = inputs.to(model[0].device), labels.to(model[0].device)
                    outputs = model[0].forward(inputs)
                    if metric == 'accuracy':
                        outputs = torch.exp(outputs)
                    
                    outputs = outputs * model[1] # multiply by model's weight
                    preds_list.append(outputs)
                    
                final_preds = preds_list[0]
                for i in range(1,len(preds_list)):
                    final_preds = final_preds + preds_list[i]
                    
                _, final_preds = torch.max(final_preds, 1)
                #print(final_preds)
                if classifier is not None:
                    classifier.update_accuracies(final_preds,labels)
                      
        ret = {}

        if classifier is not None:
            ret['accuracy'],ret['class_accuracies'] = classifier.get_final_accuracies()

        return ret
                   
    
    def predict(self,inputs,topk=1):
        ps_list = []  
        for model in self.models:
            model[0].eval()
            model[0].to(model[0].device)
            with torch.no_grad():
                inputs = inputs.to(model[0].device)
                outputs = model[0].forward(inputs)
                ps_list.append(torch.exp(outputs)*model[1])
       
        final_ps = ps_list[0]
        for i in range(1,len(ps_list)):
            final_ps = final_ps + ps_list[i]
        
        _,top = final_ps.topk(topk, dim=1)
            
        return top
    
    def forward(self,x):
        outputs = []
        for model in self.models:
             outputs.append(model[0].forward(x))
        return outputs
            
