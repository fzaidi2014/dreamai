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
                }):

        
        super().__init__(device=device)
        
        self.set_transfer_model(model_name,pretrained=pretrained)  
        
        self.set_model_head(model_name = model_name,
                head = head,
                dropout_p = dropout_p,
                device = device
            )
            
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
        self.class_names = np.arange(head['num_outputs'])
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
            
                
    def set_transfer_model(self,mname,pretrained=True):   
        self.model = None
        models_dict = {

            'densenet': models.densenet121(pretrained=pretrained),
            'resnet34': models.resnet34(pretrained=pretrained),
            'resnet50': models.resnet50(pretrained=pretrained),

        }
        try:
            # self.model = nn.Sequential(*list(models_dict[mname.lower()].modules()))
            model = models_dict[mname.lower()]
            for param in model.parameters():
                param.requires_grad = False
            self.model = model    
            print('Set_transfer_model: self.Model set to {}'.format(mname))
        except:
            print('Set_transfer_model: Model {} not supported'.format(mname))            
           
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
        meta = models_meta[name.lower()]
        modules = list(self.model.children())
        l = modules[:meta['head_id']]
        fc = modules[-1]
        in_features =  fc.in_features
        fc = FC(
                num_inputs = in_features,
                num_outputs = head['num_outputs'],
                layers = head['layers'],
                model_type = head['model_type'],
                output_non_linearity = head['output_non_linearity'],
                dropout_p = dropout_p,
                criterion = None,
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
                }):
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
                 head = head)

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


