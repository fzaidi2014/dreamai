import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from dreamai.utils import *
from dreamai.model import *
from dreamai.fc import *
import time

class TransferNetworkImg(Network):
    def __init__(self,
                 model_name='DenseNet',
                 model_type='cv_transfer',
                 lr=0.003,
                 criterion = nn.NLLLoss(),
                 optimizer_name = 'Adam',
                 dropout_p=0.2,
                 pretrained=True,
                 device=None,
                 best_accuracy=0.,
                 best_validation_loss=None,
                 best_model_file ='best_model.pth',
                 chkpoint_file ='chkpoint_file.pth',
                 head = {'num_outputs':10,
                    'layers':[],
                    'model_type':'classifier'
                 },
                 add_extra = True,
                 set_params = True
                 ):

        
        super().__init__(device=device)

        head['criterion'] = criterion
        
        self.set_transfer_model(model_name,pretrained=pretrained,add_extra=add_extra)  
        
        self.set_model_head(model_name = model_name,
                head = head,
                dropout_p = dropout_p,
                device = device
            )
        if set_params:
            self.set_model_params(criterion,
                              optimizer_name,
                              lr,
                              dropout_p,
                              model_name,
                              model_type,
                              best_accuracy,
                              best_validation_loss,
                              best_model_file,
                              chkpoint_file,
                              head)
            
        
    def set_model_params(self,criterion,
                         optimizer_name,
                         lr,
                         dropout_p,
                         model_name,
                         model_type,
                         best_accuracy,
                         best_validation_loss,
                         best_model_file,
                         chkpoint_file,
                         head):
        
        print('Transfer: best accuracy = {:.3f}'.format(best_accuracy))
        
        super(TransferNetworkImg, self).set_model_params(
                                              criterion,
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

        # self.head = head
        self.num_outputs = head['num_outputs']
        self.class_names = {k:str(v) for k,v in enumerate(list(range(head['num_outputs'])))}
        if 'class_names' in head.keys():
            if head['class_names'] is not None:
                if len(head['class_names']) > 0:
                    self.class_names = head['class_names']       
        self.to(self.device)

    def forward(self,x):
        return self.model(x)
        
    def get_model_params(self):
        params = super(TransferNetworkImg, self).get_model_params()
        params['head'] = self.head
        params['device'] = self.device
        return params
    
    def freeze(self,train_classifier=True):
        super(TransferNetworkImg, self).freeze()
        if train_classifier:
            for param in self.model.fc.parameters():
                 param.requires_grad = True

            # if self.model_name.lower() == 'densenet':
            #     for param in self.model.classifier.parameters():
            #         param.requires_grad = True
            # elif self.model_name.lower() == 'resnet34':
            #     for param in self.model.fc.parameters():
            #         param.requires_grad = True
            
                
    def set_transfer_model(self,mname,pretrained=True,add_extra=True):   
        self.model = None
        models_dict = {

            'densenet': {'model':models.densenet121(pretrained=pretrained),'conv_channels':1024},
            'resnet34': {'model':models.resnet34(pretrained=pretrained),'conv_channels':512},
            'resnet50': {'model':models.resnet50(pretrained=pretrained),'conv_channels':2048}

        }
        meta = models_dict[mname.lower()]
        try:
            # self.model = nn.Sequential(*list(models_dict[mname.lower()].modules()))
            model = meta['model']
            for param in model.parameters():
                param.requires_grad = False
            self.model = model    
            print('Set_transfer_model: self.Model set to {}'.format(mname))
        except:
            print('Set_transfer_model: Model {} not supported'.format(mname))            

        # creating and adding extra layers to the model
        dream_model = None
        if add_extra:
            channels = meta['conv_channels']
            dream_model = nn.Sequential(
                nn.Conv2d(channels,channels,3,1,1),
                nn.BatchNorm2d(channels),
                nn.ReLU(True),
                nn.Dropout2d(0.2),
                nn.Conv2d(channels,channels,3,1,1),
                nn.BatchNorm2d(channels),
                nn.ReLU(True),
                nn.Dropout2d(0.2)
#                     nn.Conv2d(channels,channels,3,1,1),
#                     nn.BatchNorm2d(channels),
#                     nn.ReLU(True),
#                     nn.Dropout2d(0.2)
                )        
        self.dream_model = dream_model          
           
    def set_model_head(self,
                        model_name = 'DenseNet',
                        head = {'num_outputs':10,
                                'layers':[],
                                'class_names': None,
                                'model_type':'classifier'
                               },
                        adaptive = True,       
                        dropout_p = 0.2,
                        device = None):

        # models_meta = {
        # 'resnet': {'head_id': -2, 'adaptive_head': [DAI_AvgPool,Flatten()],'normal_head': [nn.AvgPool2d(7,1),Flatten()]},
        # 'densenet': {'head_id': -1,'adaptive_head': [nn.ReLU(inplace=True),DAI_AvgPool,Flatten()]
        #             ,'normal_head': [nn.ReLU(inplace=True),nn.AvgPool2d(7,1),Flatten()]}
        # }

        models_meta = {
        'resnet': {'head_id': -2, 'adaptive_head': [DAI_AvgPool],'normal_head': [nn.AvgPool2d(7,1)]},
        'densenet': {'head_id': -1,'adaptive_head': [nn.ReLU(inplace=True),DAI_AvgPool]
                    ,'normal_head': [nn.ReLU(inplace=True),nn.AvgPool2d(7,1)]}
        }

        name = ''.join([x for x in model_name.lower() if x.isalpha()])
        # name = model_name.lower()
        meta = models_meta[name]
        modules = list(self.model.children())
        l = modules[:meta['head_id']]
        if self.dream_model:
            l+=self.dream_model
        fc = modules[-1]
        in_features =  fc.in_features
        fc = FC(
                num_inputs = in_features,
                num_outputs = head['num_outputs'],
                layers = head['layers'],
                model_type = head['model_type'],
                output_non_linearity = head['output_non_linearity'],
                dropout_p = dropout_p,
                criterion = head['criterion'],
                optimizer_name = None,
                device = device
                )
        if adaptive:
            l += meta['adaptive_head']
        else:
            l += meta['normal_head']
        model = nn.Sequential(*l)
        model.add_module('fc',fc)
        self.model = model
        self.head = head
        
        print('{}: setting head: inputs: {} hidden:{} outputs: {}'.format(model_name,
                                                                          in_features,
                                                                          head['layers'],
                                                                          head['num_outputs']))

    def _get_dropout(self):
        # if self.model_name.lower() == 'densenet':
        #     return self.model.classifier._get_dropout()
        
        # elif self.model_name.lower() == 'resnet50' or self.model_name.lower() == 'resnet34':
        return self.model.fc._get_dropout()
        
            
    def _set_dropout(self,p=0.2):
        
        if self.model.classifier is not None:
                print('{}: setting head (FC) dropout prob to {:.3f}'.format(self.model_name,p))
                self.model.fc._set_dropout(p=p)

        # if self.model_name.lower() == 'densenet':
        #     if self.model.classifier is not None:
        #         print('DenseNet: setting head (FC) dropout prob to {:.3f}'.format(p))
        #         self.model.classifier._set_dropout(p=p)
                
        # elif self.model_name.lower() == 'resnet50' or self.model_name.lower() == 'resnet34':
        #     if self.model.fc is not None:
        #         print('ResNet: setting head (FC) dropout prob to {:.3f}'.format(p))
        #         self.model.fc._set_dropout(p=p)
        

class FacialRec(TransferNetworkImg):
    def __init__(self,
                 model_name='DenseNet',
                 model_type='cv_transfer',
                 lr=0.003,
                 criterion= nn.NLLLoss(),
                 optimizer_name = 'Adam',
                 dropout_p=0.2,
                 pretrained=True,
                 device=None,
                 best_accuracy=0.,
                 best_validation_loss=None,
                 best_model_file ='best_model.pth',
                 chkpoint_file ='chkpoint_file.pth',
                 head = {'num_outputs':10,
                    'layers':[],
                    'model_type':'classifier'
                 },
                 add_extra = True):
        super().__init__(model_name = model_name,
                 model_type = model_type,
                 lr = lr,
                 criterion = criterion,
                 optimizer_name = optimizer_name,
                 dropout_p = dropout_p,
                 pretrained = pretrained,
                 device = device,
                 best_accuracy = best_accuracy,
                 best_validation_loss = best_validation_loss, 
                 best_model_file = best_model_file,
                 chkpoint_file = chkpoint_file,
                 head = head,
                 add_extra=add_extra)

    def forward_once(self, x):
        return self.model(x)

    def forward(self, x):
        # s = x.size()
        # x1 = torch.ones((s[0],s[2],s[3],s[4]))
        # x2 = torch.ones((s[0],s[2],s[3],s[4]))
        # for i,a in enumerate(x):
        #     x1[i] = a[0]
        #     x2[i] = a[1]
        input1,input2 = x[:,0,:,:],x[:,1,:,:]
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        # return (output1, output2)
        dist = F.pairwise_distance(output1,output2)
        return torch.nn.functional.sigmoid(dist)

class FacialRecCenterLoss(TransferNetworkImg):
    def __init__(self,
                 model_name='DenseNet',
                 model_type='cv_transfer',
                 lr=0.003,
                 criterion= nn.NLLLoss(),
                 optimizer_name = 'AdaDelta',
                 dropout_p=0.2,
                 pretrained=True,
                 device=None,
                 best_accuracy=0.,
                 best_validation_loss=None,
                 best_model_file ='best_center_loss_model.pth',
                 chkpoint_file ='center_loss_chkpoint_file.pth',
                 head = {'num_outputs':10,
                    'layers':[],
                    'model_type':'classifier'
                 },
                 add_extra = True,
                 center_features_dim = 512,
                 lamda = 0.03,
                 alpha = 0.5
                 ):
        super().__init__(model_name = model_name,
                 model_type = model_type,
                 lr = lr,
                 criterion = criterion,
                 optimizer_name = optimizer_name,
                 dropout_p = dropout_p,
                 pretrained = pretrained,
                 device = device,
                 best_accuracy = best_accuracy,
                 best_validation_loss = best_validation_loss, 
                 best_model_file = best_model_file,
                 chkpoint_file = chkpoint_file,
                 head = head,
                 add_extra=add_extra,
                 set_params=False)
        self.lamda = lamda         
        self.alpha = alpha
        self.center_features_dim = center_features_dim
        self.centers = ((torch.rand(head['num_outputs'], center_features_dim).to(device) - 0.5) * 2)
        self.add_feature_extractor()
        super(FacialRecCenterLoss, self).set_model_params(
                                              criterion,
                                              optimizer_name,
                                              lr,
                                              dropout_p,
                                              model_name,
                                              model_type,
                                              best_accuracy,
                                              best_validation_loss,
                                              best_model_file,
                                              chkpoint_file,
                                              head
                                              )

    def add_feature_extractor(self):

        modules = list(self.model.children())
        l = modules[:-1]
        head = modules[-1]
        in_features =  list(head.model.children())[0].in_features
        extractor = FC(
                num_inputs = in_features,
                num_outputs = self.center_features_dim,
                model_type = 'extractor',
                criterion = None,
                optimizer_name = None,
                device = self.device
                )
        model = nn.Sequential(*l)
        model.add_module('extractor',extractor)
        model.add_module('fc',head)
        self.model = model

    def forward(self, x):
        
        l = list(self.model.children())
        for m in l[:-2]:
            x = m(x)
        feature = l[-2](x)
        feature_normed = feature.div(torch.norm(feature, p=2, dim=1, keepdim=True).expand_as(feature))
        logits = l[-1](x)
        return logits,feature_normed

    def train_(self,trainloader,criterion,optimizer,print_every):
        self.train()
        t0 = time.time()
        batches = 0
        running_loss = 0.
        running_classifier_loss = 0.
        running_center_loss = 0.
        for inputs, labels in trainloader:
            batches += 1
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            centers = self.centers
            optimizer.zero_grad()
            outputs,features = self.forward(inputs)
            classifier_loss = criterion(outputs, labels)
            center_loss = compute_center_loss(features,centers,labels)
            loss = self.lamda * center_loss + classifier_loss
            loss.backward()
            optimizer.step()
            center_deltas = get_center_delta(features.data, centers, labels, self.alpha, self.device)
            self.centers = centers - center_deltas
            loss = loss.item()
            classifier_loss = classifier_loss.item()
            center_loss = center_loss.item()
            running_loss += loss
            running_classifier_loss += classifier_loss
            running_center_loss += center_loss
            
            if batches % print_every == 0:
                elapsed = time.time()-t0
                if elapsed > 60:
                    elapsed /= 60.
                    measure = 'min'
                else:
                    measure = 'sec'    
                print('+----------------------------------------------------------------------+\n'
                        f"{time.asctime().split()[-2]}\n"
                        f"Time elapsed: {elapsed:.3f} {measure}\n"
                        f"Batch: {batches+1}/{len(trainloader)}\n"
                        f"Average classifier loss: {running_classifier_loss/(batches):.3f}\n"
                        f"Batch classifier loss: {classifier_loss:.3f}\n"    
                        f"Average center loss: {running_center_loss/(batches):.6f}\n"
                        f"Batch center loss: {center_loss:.6f}\n"                                
                        f"Average training loss: {running_loss/(batches):.6f}\n"
                        f"Batch training loss: {loss:.6f}\n"
                      '+----------------------------------------------------------------------+\n'     
                        )
                t0 = time.time()
        return running_loss/len(trainloader)
       
        # return (running_classifier_loss/len(trainloader),
        #         running_center_loss/len(trainloader),running_loss/len(trainloader))

    def evaluate(self,dataloader,metric='accuracy'):
    
        running_loss = 0.
        running_classifier_loss = 0.
        running_center_loss = 0.
        classifier = None

        if self.model_type == 'classifier':# or self.num_classes is not None:
            classifier = Classifier(self.class_names)

        self.eval()
        rmse_ = 0.
        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                centers = self.centers
                outputs, features = self.forward(inputs)
                classifier_loss = self.criterion(outputs, labels)
                center_loss = compute_center_loss(features,centers,labels)
                loss = self.lamda * center_loss + classifier_loss
                running_classifier_loss += classifier_loss.item()
                running_center_loss += center_loss.item()
                running_loss += loss.item()

                if classifier is not None and metric == 'accuracy':
                    classifier.update_accuracies(outputs,labels)
                elif metric == 'rmse':
                    rmse_ += rmse(outputs,labels).cpu().numpy()
            
        self.train()

        ret = {}
        print('Running_classifier_loss: {:.3f}'.format(running_classifier_loss))
        print('Running_center_loss: {:.6f}'.format(running_center_loss))
        print('Running_loss: {:.6f}'.format(running_loss))
        if metric == 'rmse':
            print('Total rmse: {:.3f}'.format(rmse_))
        ret['final_loss'] = running_loss/len(dataloader)

        if classifier is not None:
            ret['accuracy'],ret['class_accuracies'] = classifier.get_final_accuracies()
        elif metric == 'rmse':
            ret['final_rmse'] = rmse_/len(dataloader)

        return ret        


