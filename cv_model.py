from dai_imports import*
from utils import *
from obj_utils import*
from model import *
from fc import *
from darknet import*
import resnet_unet
import resnet_unet_2
import unet_model
import time

class CNNetwork(Network):
    def __init__(self,
                 model = None,
                 model_name = 'custom_CNN',
                 model_type='regressor',
                 lr=0.02,
                 one_cycle_factor = 0.5,
                 criterion = nn.NLLLoss(),
                 optimizer_name = 'sgd',
                 dropout_p=0.45,
                 device=None,
                 best_accuracy=0.,
                 best_validation_loss=None,
                 best_model_file ='best_cnn_model.pth',
                 chkpoint_file ='chkpoint_file.pth',
                 class_names = [],
                 num_classes = None,
                 ):

        super().__init__(device=device)

        self.set_model_params(criterion = criterion,
                            optimizer_name = optimizer_name,
                            lr = lr,
                            one_cycle_factor = one_cycle_factor,
                            dropout_p = dropout_p,
                            model_name = model_name,
                            model_type = model_type,
                            best_accuracy = best_accuracy,
                            best_validation_loss = best_validation_loss,
                            best_model_file = best_model_file,
                            chkpoint_file = chkpoint_file,
                            class_names = class_names,
                            num_classes = num_classes
                            )

        self.model = model.to(device)
        
    def set_model_params(self,criterion,
                         optimizer_name,
                         lr,
                         one_cycle_factor,
                         dropout_p,
                         model_name,
                         model_type,
                         best_accuracy,
                         best_validation_loss,
                         best_model_file,
                         chkpoint_file,
                         head,
                         class_names,
                         num_classes):

        super(CNNetwork, self).set_model_params(
                                              criterion = criterion,
                                              optimizer_name = optimizer_name,
                                              lr = lr,
                                              one_cycle_factor = one_cycle_factor,
                                              dropout_p = dropout_p,
                                              model_name = model_name,
                                              model_type = model_type,
                                              best_accuracy = best_accuracy,
                                              best_validation_loss = best_validation_loss,
                                              best_model_file = best_model_file,
                                              chkpoint_file = chkpoint_file,
                                              class_names = class_names,
                                              num_classes = num_classes
                                              )

    def forward(self,x):
        return self.model(x)
                
class TransferNetworkImg(Network):
    def __init__(self,
                 model_name='DenseNet',
                 model_type='cv_transfer',
                 lr=0.02,
                 one_cycle_factor = 0.5,
                 criterion = nn.CrossEntropyLoss(),
                 optimizer_name = 'Adam',
                 dropout_p=0.45,
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
                 pre_trained_back = None,
                 class_names = [],
                 num_classes = None,
                 add_extra = True,
                 set_params = True,
                 set_head = True
                 ):

        
        super().__init__(device=device)

        self.set_transfer_model(model_name,pretrained=pretrained,add_extra=add_extra,drop_out=dropout_p)

        if set_head:
            self.set_model_head(model_name = model_name,
                    head = head,
                    pre_trained_back = pre_trained_back,
                    dropout_p = dropout_p,
                    criterion = criterion,
                    device = device
                )
        if set_params:
            self.set_model_params(criterion = criterion,
                              optimizer_name = optimizer_name,
                              lr = lr,
                              one_cycle_factor = one_cycle_factor,
                              dropout_p = dropout_p,
                              model_name = model_name,
                              model_type = model_type,
                              best_accuracy = best_accuracy,
                              best_validation_loss = best_validation_loss,
                              best_model_file = best_model_file,
                              chkpoint_file = chkpoint_file,
                              head = head,
                              class_names = class_names,
                              num_classes = num_classes
                              )

        self.model = self.model.to(device)
        
    def set_model_params(self,criterion,
                         optimizer_name,
                         lr,
                         one_cycle_factor,
                         dropout_p,
                         model_name,
                         model_type,
                         best_accuracy,
                         best_validation_loss,
                         best_model_file,
                         chkpoint_file,
                         head,
                         class_names,
                         num_classes):
        
        print('Transfer Learning: current best accuracy = {:.3f}'.format(best_accuracy))
        
        super(TransferNetworkImg, self).set_model_params(
                                              criterion = criterion,
                                              optimizer_name = optimizer_name,
                                              lr = lr,
                                              one_cycle_factor = one_cycle_factor,
                                              dropout_p = dropout_p,
                                              model_name = model_name,
                                              model_type = model_type,
                                              best_accuracy = best_accuracy,
                                              best_validation_loss = best_validation_loss,
                                              best_model_file = best_model_file,
                                              chkpoint_file = chkpoint_file,
                                              class_names = class_names,
                                              num_classes = num_classes
                                              )

        # self.head = head
        if len(class_names) == 0:
            self.class_names = {k:str(v) for k,v in enumerate(list(range(head['num_outputs'])))}
        # if 'class_names' in head.keys():
        #     if head['class_names'] is not None:
        #         if len(head['class_names']) > 0:
        #             self.class_names = head['class_names']       
        # self.to(self.device)

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
            
                
    def set_transfer_model(self,mname,pretrained=True,add_extra=True,drop_out = 0.45):   
        self.model = None
        models_dict = {

            'densenet': {'model':models.densenet121(pretrained=pretrained),'conv_channels':1024,'grid_ceil':False},
            'resnet34': {'model':models.resnet34(pretrained=pretrained),'conv_channels':512,'grid_ceil':True},
            'resnet50': {'model':models.resnet50(pretrained=pretrained),'conv_channels':2048,'grid_ceil':True}

        }
        meta = models_dict[mname.lower()]
        self.grid_ceil = meta['grid_ceil']
        try:
            # self.model = nn.Sequential(*list(models_dict[mname.lower()].modules()))
            model = meta['model']
            for param in model.parameters():
                param.requires_grad = False
            self.model = model    
            print('Setting transfer learning model: self.model set to {}'.format(mname))
        except:
            print('Setting transfer learning model: model name {} not supported'.format(mname))            

        # creating and adding extra layers to the model
        dream_model = None
        if add_extra:
            channels = meta['conv_channels']
            dream_model = nn.Sequential(
                nn.Conv2d(channels,channels,3,1,1),
                # Printer(),
                nn.BatchNorm2d(channels),
                nn.ReLU(True),
                nn.Dropout2d(drop_out),
                nn.Conv2d(channels,channels,3,1,1),
                nn.BatchNorm2d(channels),
                nn.ReLU(True),
                nn.Dropout2d(drop_out),
                nn.Conv2d(channels,channels,3,1,1),
                nn.BatchNorm2d(channels),
                nn.ReLU(True),
                nn.Dropout2d(drop_out)
                )        
        self.dream_model = dream_model          
           
    def set_model_head(self,
                        model_name = 'DenseNet',
                        head = {'num_outputs':10,
                                'layers':[],
                                'class_names': None,
                                'model_type':'classifier'
                               },
                        pre_trained_back = None,
                        criterion = nn.NLLLoss(),  
                        adaptive = True,       
                        dropout_p = 0.45,
                        device = None):

        # models_meta = {
        # 'resnet': {'head_id': -2, 'adaptive_head': [DAI_AvgPool,Flatten()],'normal_head': [nn.AvgPool2d(7,1),Flatten()]},
        # 'densenet': {'head_id': -1,'adaptive_head': [nn.ReLU(inplace=True),DAI_AvgPool,Flatten()]
        #             ,'normal_head': [nn.ReLU(inplace=True),nn.AvgPool2d(7,1),Flatten()]}
        # }

        models_meta = {
        'resnet34': {'conv_channels':512,'head_id': -2, 'adaptive_head': [DAI_AvgPool],'normal_head': [nn.AvgPool2d(7,1)]},
        'resnet50': {'conv_channels':2048,'head_id': -2, 'adaptive_head': [DAI_AvgPool],'normal_head': [nn.AvgPool2d(7,1)]},
        'densenet': {'conv_channels':1024,'head_id': -1,'adaptive_head': [nn.ReLU(inplace=True),DAI_AvgPool]
                    ,'normal_head': [nn.ReLU(inplace=True),nn.AvgPool2d(7,1)]}
        }

        # name = ''.join([x for x in model_name.lower() if x.isalpha()])
        name = model_name.lower()
        meta = models_meta[name]
        if pre_trained_back:
            self.model = pre_trained_back
            self.dream_model = None
        modules = list(self.model.children())
        l = modules[:meta['head_id']]
        if self.dream_model:
            l+=self.dream_model
        if type(head).__name__ != 'dict':
            model = nn.Sequential(*l)
            for layer in head.children():
                if(type(layer).__name__) == 'StdConv':
                    conv_module = layer
                    break
            # temp_conv = head.sconv0.conv
            conv_layer = conv_module.conv
            temp_args = [conv_layer.out_channels,conv_layer.kernel_size,conv_layer.stride,conv_layer.padding]
            temp_args.insert(0,meta['conv_channels'])
            conv_layer = nn.Conv2d(*temp_args)
            conv_module.conv = conv_layer
            # print(head)
            # model.add_module('adaptive_avg_pool',DAI_AvgPool)
            model.add_module('custom_head',head)
        else:
            head['criterion'] = criterion
            if head['model_type'].lower() == 'classifier':
                head['output_non_linearity'] = None
            self.num_outputs = head['num_outputs']
            fc = modules[-1]
            try:
                in_features =  fc.in_features
            except:
                in_features = fc.model.out.in_features    
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
        
        if type(head).__name__ == 'dict':
            print('Model: {}, Setting head: inputs: {} hidden:{} outputs: {}'.format(model_name,
                                                                          in_features,
                                                                          head['layers'],
                                                                          head['num_outputs']))
        else:
            print('Model: {}, Setting head: {}'.format(model_name,type(head).__name__))

    def _get_dropout(self):
        # if self.model_name.lower() == 'densenet':
        #     return self.model.classifier._get_dropout()
        
        # elif self.model_name.lower() == 'resnet50' or self.model_name.lower() == 'resnet34':
        return self.model.fc._get_dropout()
        
            
    def _set_dropout(self,p=0.45):
        
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
                 lr=0.02,
                 one_cycle_factor = 0.5,
                 criterion= nn.NLLLoss(),
                 optimizer_name = 'Adam',
                 dropout_p=0.45,
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
                 one_cycle_factor = one_cycle_factor,
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
                 lr=0.02,
                 one_cycle_factor = 0.5,
                 criterion= nn.NLLLoss(),
                 optimizer_name = 'AdaDelta',
                 dropout_p=0.45,
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
                 one_cycle_factor = one_cycle_factor,
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
                 add_extra = add_extra,
                 set_params = False)
        self.lamda = lamda         
        self.alpha = alpha
        self.center_features_dim = center_features_dim
        self.centers = ((torch.rand(head['num_outputs'], center_features_dim).to(device) - 0.5) * 2)
        self.add_feature_extractor()
        super(FacialRecCenterLoss, self).set_model_params(
                                              criterion,
                                              optimizer_name,
                                              lr,
                                              one_cycle_factor,
                                              dropout_p,
                                              model_name,
                                              model_type,
                                              best_accuracy,
                                              best_validation_loss,
                                              best_model_file,
                                              chkpoint_file,
                                              head
                                              )
        self.model = self.model.to(device)                                

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
        return (logits,feature_normed)

    def compute_loss(self,criterion,outputs,labels):
        centers = self.centers
        out,features = outputs
        classifier_loss = criterion(out, labels)
        center_loss = compute_center_loss(features,centers,labels)
        loss = self.lamda * center_loss + classifier_loss
        return(loss,classifier_loss,center_loss)

    def train_(self,e,trainloader,criterion,optimizer,print_every):

        epoch,epochs = e
        self.train()
        t0 = time.time()
        t1 = time.time()
        batches = 0
        running_loss = 0.
        running_classifier_loss = 0.
        running_center_loss = 0.
        for data_batch in trainloader:
            inputs, labels = data_batch[0],data_batch[1]
            batches += 1
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            centers = self.centers
            optimizer.zero_grad()
            # outputs = self.forward(inputs)
            # out,features = outputs
            # loss,classifier_loss,center_loss = self.compute_loss(criterion,outputs,labels)
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
                elapsed = time.time()-t1
                if elapsed > 60:
                    elapsed /= 60.
                    measure = 'min'
                else:
                    measure = 'sec'
                batch_time = time.time()-t0
                if batch_time > 60:
                    batch_time /= 60.
                    measure2 = 'min'
                else:
                    measure2 = 'sec'   
                print('+----------------------------------------------------------------------+\n'
                        f"{time.asctime().split()[-2]}\n"
                        f"Time elapsed: {elapsed:.3f} {measure}\n"    
                        f"Epoch:{epoch+1}/{epochs}\n"
                        f"Batch: {batches+1}/{len(trainloader)}\n"
                        f"Batch training time: {batch_time:.3f} {measure2}\n"
                        f"Batch classifier loss: {classifier_loss:.3f}\n"    
                        f"Average classifier loss: {running_classifier_loss/(batches):.3f}\n"
                        f"Batch center loss: {center_loss:.6f}\n"
                        f"Average center loss: {running_center_loss/(batches):.6f}\n"
                        f"Batch training loss: {loss:.6f}\n"
                        f"Average training loss: {running_loss/(batches):.6f}\n"
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
            for data_batch in dataloader:
                inputs,labels = data_batch[0],data_batch[1]
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                centers = self.centers
                # outputs = self.forward(inputs)
                # out,features = outputs
                # loss,classifier_loss,center_loss = self.compute_loss(self.criterion,outputs,labels)
                outputs, features = self.forward(inputs)
                out = outputs
                classifier_loss = self.criterion(outputs, labels)
                center_loss = compute_center_loss(features,centers,labels)
                loss = self.lamda * center_loss + classifier_loss
                running_classifier_loss += classifier_loss.item()
                running_center_loss += center_loss.item()
                running_loss += loss.item()

                if classifier is not None and metric == 'accuracy':
                    classifier.update_accuracies(out,labels)
                elif metric == 'rmse':
                    rmse_ += rmse(out,labels).cpu().numpy()
            
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

class SSDObjectDetection(TransferNetworkImg):
    def __init__(self,
                 model_name='resnet34',
                 model_type='obj_detection',
                 lr=0.02,
                 one_cycle_factor = 0.5,
                 criterion= nn.NLLLoss(),
                 optimizer_name = 'Adam',
                 dropout_p=0.45,
                 pretrained=True,
                 device=None,
                 best_accuracy=0.,
                 best_validation_loss=None,
                 best_model_file ='best_ssd.pth',
                 chkpoint_file ='ssd_chkpoint_file.pth',
                 head = {'num_outputs':10,
                    'layers':[],
                    'model_type':'classifier'
                 },
                 add_extra = True,
                 class_names = None,
                 num_classes = None,
                 image_size = (224,224)):
        super().__init__(model_name = model_name,
                 model_type = model_type,
                 lr = lr,
                 one_cycle_factor = one_cycle_factor,
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
                 add_extra = add_extra,
                 set_params = False,
                 class_names = class_names,
                 num_classes = num_classes,
                 set_head = False)

        self.obj = True
        self.image_size = image_size
        self.set_up_object_detection(anc_grids=[4,2,1], anc_zooms=[0.7,1.,1.3], anc_ratios=[(1.,1.),(1.,0.5),(0.5,1.)],
                                    num_classes=num_classes,drop_out=dropout_p)
        super(SSDObjectDetection,self).set_model_head(model_name = model_name, head = self.custom_head)
        super(SSDObjectDetection,self).set_model_params(
                                              criterion = self.ssd_loss,
                                              optimizer_name = optimizer_name,
                                              lr = lr,
                                              one_cycle_factor = one_cycle_factor,
                                              dropout_p = dropout_p,
                                              model_name = model_name,
                                              model_type = model_type,
                                              best_accuracy = best_accuracy,
                                              best_validation_loss = best_validation_loss,
                                              best_model_file = best_model_file,
                                              chkpoint_file = chkpoint_file,
                                              head = head,
                                              class_names = class_names,
                                              num_classes = num_classes
                                              )
        self.model = self.model.to(device)

    def set_up_object_detection(self,anc_grids,anc_zooms,anc_ratios,num_classes,num_colr = 12,drop_out = 0.5):

        # print('Would you like to give your own values for anchor_grids, anchor_zooms,and anchor_ratios? The default values are: {}, {} and {}'
        # .format(anc_grids,anc_zooms,anc_ratios))
        # print('If so, you may call the function "set_up_object_detection" with your own paramteres.')

        cmap = get_cmap(num_colr)
        self.colr_list = [cmap(float(x)) for x in range(num_colr)]
        self.num_colr = num_colr
        self.create_anchors(anc_grids,anc_zooms,anc_ratios)
        self.custom_head = SSD_MultiHead(self.k,num_classes,drop_out,-4.)
        self.loss_f = FocalLoss(num_classes,device=self.device)

    def create_anchors(self,anc_grids,anc_zooms,anc_ratios):
    
        anchor_scales = [(anz*i,anz*j) for anz in anc_zooms for (i,j) in anc_ratios]
        k = len(anchor_scales)
        anc_offsets = [1/(o*2) for o in anc_grids]
        anc_x = np.concatenate([np.repeat(np.linspace(ao, 1-ao, ag), ag)
                                for ao,ag in zip(anc_offsets,anc_grids)])
        anc_y = np.concatenate([np.tile(np.linspace(ao, 1-ao, ag), ag)
                                for ao,ag in zip(anc_offsets,anc_grids)])
        anc_ctrs = np.repeat(np.stack([anc_x,anc_y], axis=1), k, axis=0)
        anc_sizes  =   np.concatenate([np.array([[o/ag,p/ag] for i in range(ag*ag) for o,p in anchor_scales])
                    for ag in anc_grids])
        grid_sizes = torch.tensor(np.concatenate([np.array(
                                [ 1/ag for i in range(ag*ag) for o,p in anchor_scales])
                    for ag in anc_grids])).float().unsqueeze(1).to(self.device)
        anchors = torch.tensor(np.concatenate([anc_ctrs, anc_sizes], axis=1)).float().to(self.device)
        anchor_cnr = hw2corners(anchors[:,:2], anchors[:,2:])
        self.anchors,self.anchor_cnr,self.grid_sizes,self.k = anchors,anchor_cnr,grid_sizes,k

    # def draw_im(im, ann, cats):
    #     ax = img_grid(im, figsize=(16,8))
    #     for b,c in ann:
    #         b = bb_hw(b)
    #         draw_rect(ax, b)
    #         draw_text(ax, b[:2], cats[c], sz=16)

    # def draw_idx(i):
    #     im_a = trn_anno[i]
    # #     im = open_image(IMG_PATH/trn_fns[i])
    #     im = Image.open(IMG_PATH/trn_fns[i]).convert('RGB')
    #     draw_im(im, im_a)


    def show_objects_(self, ax, im, bbox, clas=None, prs=None, thresh=0.3):

        bb = [bb_hw(o) for o in bbox.reshape(-1,4)]
        # print(bb)
        if prs is None:  prs  = [None]*len(bb)
        if clas is None: clas = [None]*len(bb)
        ax = img_grid(im, ax=ax)
        for i,(b,c,pr) in enumerate(zip(bb, clas, prs)):
            if((b[2]>0) and (pr is None or pr > thresh)):
                draw_rect(ax, b, color=self.colr_list[i % self.num_colr])
                txt = f'{i}: '
                if c is not None: txt += ('bg' if c==len(self.class_names) else self.class_names[c])
                if pr is not None: txt += f' {pr:.2f}'
                draw_text(ax, b[:2], txt, color=self.colr_list[i % self.num_colr])
        return ax

    def intersect(self,box_a, box_b):

        max_xy = torch.min(box_a[:, None, 2:], box_b[None, :, 2:])
        min_xy = torch.max(box_a[:, None, :2], box_b[None, :, :2])
        inter = torch.clamp((max_xy - min_xy), min=0)
        return inter[:, :, 0] * inter[:, :, 1]

    def box_sz(self,b): return ((b[:, 2]-b[:, 0]) * (b[:, 3]-b[:, 1]))

    def jaccard(self,box_a, box_b):

        inter = self.intersect(box_a, box_b)
        union = self.box_sz(box_a).unsqueeze(1) + self.box_sz(box_b).unsqueeze(0) - inter
        return inter / union

    def get_y(self, bbox, clas):

        bbox = bbox.view(-1,4)/self.image_size[0]
        bb_keep = ((bbox[:,2]-bbox[:,0])>0).nonzero()[:,0]
        return bbox[bb_keep],clas[bb_keep]

    def actn_to_bb(self, actn):
        actn_bbs = torch.tanh(actn)
        # print(self.grid_sizes.size())
        # print(self.anchors[:,:2].size())
        actn_centers = (actn_bbs[:,:2]/2 * self.grid_sizes) + self.anchors[:,:2]
        actn_hw = (actn_bbs[:,2:]/2+1) * self.anchors[:,2:]
        return hw2corners(actn_centers, actn_hw)

    def map_to_ground_truth(self, overlaps, print_it=False):

        prior_overlap, prior_idx = overlaps.max(1)
    #     if print_it: print(prior_overlap)
        gt_overlap, gt_idx = overlaps.max(0)
        gt_overlap[prior_idx] = 1.99
        for i,o in enumerate(prior_idx): gt_idx[o] = i
        return gt_overlap,gt_idx

    def ssd_1_loss(self, b_c,b_bb,bbox,clas,print_it=False):

        anchor_cnr = hw2corners(self.anchors[:,:2], self.anchors[:,2:])
        bbox,clas = self.get_y(bbox,clas)
        a_ic = self.actn_to_bb(b_bb)
        overlaps = self.jaccard(bbox.data, anchor_cnr.data)
        gt_overlap,gt_idx = self.map_to_ground_truth(overlaps,print_it)
        gt_clas = clas[gt_idx]
        pos = gt_overlap > 0.4
        pos_idx = torch.nonzero(pos)[:,0]
        gt_clas[1-pos] = len(self.class_names)
        gt_bbox = bbox[gt_idx]
        loc_loss = ((a_ic[pos_idx] - gt_bbox[pos_idx]).abs()).mean()
        clas_loss  = self.loss_f(b_c, gt_clas)
        return loc_loss, clas_loss

    def ssd_loss(self, pred, targ, print_it=False):

        lcs,lls = 0.,0.
        for b_c,b_bb,bbox,clas in zip(*pred,*targ):
            loc_loss,clas_loss = self.ssd_1_loss(b_c,b_bb,bbox,clas)
            lls += loc_loss
            lcs += clas_loss
        if print_it: print(f'loc: {lls.data[0]}, clas: {lcs.data[0]}')
        return lls+lcs

    def set_loss(self,loss):
        self.loss_f = loss    

    def show_objects(self, ax, ima, bbox, clas, prs=None, thresh=0.4):

        return self.show_objects_(ax, ima, ((bbox*self.image_size[0]).long()).numpy(),
            (clas).numpy(), (prs).numpy() if prs is not None else None, thresh)

    # def dai_plot_results(self,thresh,loader,model):
    
    #     dai_x,dai_y = next(iter(loader))
    #     dai_x = dai_x.to(self.device)
    #     dai_y = [torch.tensor(l).to(self.device) for l in dai_y]
    #     dai_batch = model(dai_x)
    #     dai_b_clas,dai_b_bb = dai_batch
    #     dai_x = dai_x.cpu()
    #     dai_y = [torch.tensor(l).cpu() for l in dai_y]


    #     fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    #     for idx,ax in enumerate(axes.flat):
    #         ima = dai.denorm_img(dai_x[idx])
    #         bbox,clas = self.get_y(dai_y[0][idx], dai_y[1][idx])
    #         a_ic = self.actn_to_bb(dai_b_bb[idx])
    #         clas_pr, clas_ids = dai_b_clas[idx].max(1)
    #         clas_pr = clas_pr.sigmoid()
    #         self.show_objects(ax, ima, a_ic, clas_ids, clas_pr, clas_pr.max().data[0]*thresh)
    #     plt.tight_layout()  

    def batch_loss(self,model,loader,crit):
        
        data_batch = next(iter(loader))
        dai_x,dai_y = data_batch[0],data_batch[1]
        dai_x = dai_x.to(self.device)
        dai_y = [torch.tensor(l).to(self.device) if type(l).__name__ == 'Tensor' else l.to(device) for l in dai_y]
        dai_batch = model(dai_x)
        return crit(dai_batch,dai_y)

    def show_nms(self,loader = None,num = 10,img = None,score_thresh = 0.25,nms_overlap = 0.1,dp = None):

        if loader:
            data_batch = next(iter(loader))
            x = data_batch[0]
            batch = self.predict(x)
            pred_clas,pred_bbox = batch
            x = x.cpu()

            for i in range(num):
                print(i)
                ima = dp.denorm_img(x[i])
                box_coords = self.actn_to_bb(pred_bbox[i])
                conf_scores = pred_clas[i].sigmoid().t().data
                self.show_nms_(ima,box_coords,conf_scores,score_thresh)
        else:
            x = img
            batch = self.predict(x)
            pred_clas,pred_bbox = batch
            x = x.cpu()
            for i in range(len(x)):
                print(i)
                ima = dp.denorm_img(x[i])
                box_coords = self.actn_to_bb(pred_bbox[i])
                conf_scores = pred_clas[i].sigmoid().t().data
                self.show_nms_(ima,box_coords,conf_scores,score_thresh)

    def show_nms_(self,ima,box_coords,conf_scores,score_thresh = 0.25,nms_overlap = 0.1):

        out1,out2,cc = [],[],[]
        for cl in range(0, len(conf_scores)-1):
            c_mask = conf_scores[cl] > score_thresh
            if c_mask.sum() == 0: continue
            scores = conf_scores[cl][c_mask]
            l_mask = c_mask.unsqueeze(1).expand_as(box_coords)
            boxes = box_coords[l_mask].view(-1, 4)
            ids, count = nms(boxes.data, scores, nms_overlap, 50)
            ids = ids[:count]
            out1.append(scores[ids])
            out2.append(boxes.data[ids])
            cc.append([cl]*count)
        # return(out1,out2)    
        if len(cc)> 0:    
            cc = torch.from_numpy(np.concatenate(cc))
            out1 = torch.cat(out1).cpu()
            out2 = torch.cat(out2).cpu()
            fig, ax = plt.subplots(figsize=(8,8))
            ax = self.show_objects(ax, ima, out2, cc, out1, score_thresh)
            plt.show()

class CustomSSDObjectDetection(TransferNetworkImg):
    def __init__(self,
                 model_name='resnet50',
                 model_type='obj_detection',
                 lr=0.02,
                 one_cycle_factor = 0.5,
                 criterion= nn.NLLLoss(),
                 optimizer_name = 'Adam',
                 dropout_p=0.45,
                 pretrained=True,
                 device=None,
                 best_accuracy=0.,
                 best_validation_loss=None,
                 best_model_file ='best_custom_ssd.pth',
                 chkpoint_file ='cusstom_ssd_chkpoint_file.pth',
                 head = {'num_outputs':10,
                    'layers':[],
                    'model_type':'classifier'
                 },
                 pre_trained_back = None,
                 add_extra = True,
                 class_names = None,
                 num_classes = None,
                 image_size = (224,224)):
        super().__init__(model_name = model_name,
                 model_type = model_type,
                 lr = lr,
                 one_cycle_factor = one_cycle_factor,
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
                 add_extra = add_extra,
                 set_params = False,
                 class_names = class_names,
                 num_classes = num_classes,
                 set_head = False)

        self.obj = True
        self.image_size = image_size
        grids = get_grids(image_size,ceil = self.grid_ceil)
        print('Grids: {}'.format(grids))
        self.set_up_object_detection(anc_grids=grids, anc_zooms=[0.7,1.,1.3], anc_ratios=[(1.,1.),(1.,0.5),(0.5,1.)],
                                    num_classes=num_classes,drop_out=dropout_p)          
        self.set_model_head(model_name = model_name, head = self.custom_head,pre_trained_back = pre_trained_back)
        super(CustomSSDObjectDetection,self).set_model_params(
                                              criterion = self.ssd_loss,
                                              optimizer_name = optimizer_name,
                                              lr = lr,
                                              one_cycle_factor = one_cycle_factor,
                                              dropout_p = dropout_p,
                                              model_name = model_name,
                                              model_type = model_type,
                                              best_accuracy = best_accuracy,
                                              best_validation_loss = best_validation_loss,
                                              best_model_file = best_model_file,
                                              chkpoint_file = chkpoint_file,
                                              head = head,
                                              class_names = class_names,
                                              num_classes = num_classes
                                              )
        self.model = self.model.to(device)

    def set_model_head(self,
                        model_name = 'resnet50',
                        head = {'num_outputs':10,
                                'layers':[],
                                'class_names': None,
                                'model_type':'classifier'
                               },
                        pre_trained_back = None,
                        criterion = nn.NLLLoss(),  
                        adaptive = True,       
                        dropout_p = 0.45,
                        device = None):

        # models_meta = {
        # 'resnet': {'head_id': -2, 'adaptive_head': [DAI_AvgPool,Flatten()],'normal_head': [nn.AvgPool2d(7,1),Flatten()]},
        # 'densenet': {'head_id': -1,'adaptive_head': [nn.ReLU(inplace=True),DAI_AvgPool,Flatten()]
        #             ,'normal_head': [nn.ReLU(inplace=True),nn.AvgPool2d(7,1),Flatten()]}
        # }

        models_meta = {
        'resnet34': {'conv_channels':512,'head_id': -2,'clip_func':resnet_obj_clip,'adaptive_head': [DAI_AvgPool],
                    'normal_head': [nn.AvgPool2d(7,1)]},
        'resnet50': {'conv_channels':2048,'head_id': -2,'clip_func':resnet_obj_clip,'adaptive_head': [DAI_AvgPool],
                    'normal_head': [nn.AvgPool2d(7,1)]},
        'densenet': {'conv_channels':1024,'head_id': -1,'clip_func':densenet_obj_clip,'adaptive_head': [nn.ReLU(inplace=True),DAI_AvgPool]
                    ,'normal_head': [nn.ReLU(inplace=True),nn.AvgPool2d(7,1)]}
        }

        # name = ''.join([x for x in model_name.lower() if x.isalpha()])
        name = model_name.lower()
        meta = models_meta[name]
        if pre_trained_back:
            model = pre_trained_back
        else:    
            modules = list(self.model.children())
            # l = meta['clip_func'](modules,meta['head_id'])
            l = modules[:meta['head_id']]
            if self.dream_model:
                # for layer in self.dream_model.children():
                #     if(type(layer).__name__) == 'Conv2d':
                #         layer.in_channels,layer.out_channels = meta['conv_channels'],meta['conv_channels']
                #     if(type(layer).__name__) == 'BatchNorm2d':
                #         layer.num_features = meta['conv_channels']
                # self.dream_model[0].in_channels = meta['conv_channels']
                # self.dream_model[-4].out_channels = meta['conv_channels']
                # self.dream_model[-3].num_features = meta['conv_channels']
                l+=self.dream_model

            # if type(head).__name__ != 'dict':
            model = nn.Sequential(*l)
        for layer in head.children():
            if(type(layer).__name__) == 'StdConv':
                conv_module = layer
                break
        # temp_conv = head.sconv0.conv
        conv_layer = conv_module.conv
        temp_args = [conv_layer.out_channels,conv_layer.kernel_size,conv_layer.stride,conv_layer.padding]
        temp_args.insert(0,meta['conv_channels'])
        conv_layer = nn.Conv2d(*temp_args)
        conv_module.conv = conv_layer
        # print(head)
        # model.add_module('adaptive_avg_pool',DAI_AvgPool)
        model.add_module('custom_head',head)
        # else:
        #     head['criterion'] = criterion
        #     self.num_outputs = head['num_outputs']
        #     fc = modules[-1]
        #     in_features =  fc.in_features
        #     fc = FC(
        #             num_inputs = in_features,
        #             num_outputs = head['num_outputs'],
        #             layers = head['layers'],
        #             model_type = head['model_type'],
        #             output_non_linearity = head['output_non_linearity'],
        #             dropout_p = dropout_p,
        #             criterion = head['criterion'],
        #             optimizer_name = None,
        #             device = device
        #             )
        #     if adaptive:
        #         l += meta['adaptive_head']
        #     else:
        #         l += meta['normal_head']
        #     model = nn.Sequential(*l)
        #     model.add_module('fc',fc)
        self.model = model
        self.head = head
        
        # if type(head).__name__ == 'dict':
        #     print('{}: setting head: inputs: {} hidden:{} outputs: {}'.format(model_name,
        #                                                                   in_features,
        #                                                                   head['layers'],
        #                                                                   head['num_outputs']))
        # else:
        print('{}: setting head: {}'.format(model_name,type(head).__name__))    

    def set_up_object_detection(self,anc_grids,anc_zooms,anc_ratios,num_classes,num_colr = 12,drop_out = 0.5):

        # print('Would you like to give your own values for anchor_grids, anchor_zooms,and anchor_ratios? The default values are: {}, {} and {}'
        # .format(anc_grids,anc_zooms,anc_ratios))
        # print('If so, you may call the function "set_up_object_detection" with your own paramteres.')

        cmap = get_cmap(num_colr)
        self.colr_list = [cmap(float(x)) for x in range(num_colr)]
        self.num_colr = num_colr
        self.create_anchors(anc_grids,anc_zooms,anc_ratios)
        self.custom_head = CustomSSD_MultiHead(len(anc_grids),self.k,num_classes,drop_out,-4.)
        self.loss_f = FocalLoss(num_classes,device=self.device)

    def create_anchors(self,anc_grids,anc_zooms,anc_ratios):
    
        anchor_scales = [(anz*i,anz*j) for anz in anc_zooms for (i,j) in anc_ratios]
        k = len(anchor_scales)
        anc_offsets = [1/(o*2) for o in anc_grids]
        anc_x = np.concatenate([np.repeat(np.linspace(ao, 1-ao, ag), ag)
                                for ao,ag in zip(anc_offsets,anc_grids)])
        anc_y = np.concatenate([np.tile(np.linspace(ao, 1-ao, ag), ag)
                                for ao,ag in zip(anc_offsets,anc_grids)])
        anc_ctrs = np.repeat(np.stack([anc_x,anc_y], axis=1), k, axis=0)
        anc_sizes  =   np.concatenate([np.array([[o/ag,p/ag] for i in range(ag*ag) for o,p in anchor_scales])
                    for ag in anc_grids])
        grid_sizes = torch.tensor(np.concatenate([np.array(
                                [ 1/ag for i in range(ag*ag) for o,p in anchor_scales])
                    for ag in anc_grids])).float().unsqueeze(1).to(self.device)
        anchors = torch.tensor(np.concatenate([anc_ctrs, anc_sizes], axis=1)).float().to(self.device)
        anchor_cnr = hw2corners(anchors[:,:2], anchors[:,2:])
        self.anchors,self.anchor_cnr,self.grid_sizes,self.k = anchors,anchor_cnr,grid_sizes,k

    # def draw_im(im, ann, cats):
    #     ax = img_grid(im, figsize=(16,8))
    #     for b,c in ann:
    #         b = bb_hw(b)
    #         draw_rect(ax, b)
    #         draw_text(ax, b[:2], cats[c], sz=16)

    # def draw_idx(i):
    #     im_a = trn_anno[i]
    # #     im = open_image(IMG_PATH/trn_fns[i])
    #     im = Image.open(IMG_PATH/trn_fns[i]).convert('RGB')
    #     draw_im(im, im_a)

    def intersect(self,box_a, box_b):

        max_xy = torch.min(box_a[:, None, 2:], box_b[None, :, 2:])
        min_xy = torch.max(box_a[:, None, :2], box_b[None, :, :2])
        inter = torch.clamp((max_xy - min_xy), min=0)
        return inter[:, :, 0] * inter[:, :, 1]

    def box_sz(self,b): return ((b[:, 2]-b[:, 0]) * (b[:, 3]-b[:, 1]))

    def jaccard(self,box_a, box_b):

        inter = self.intersect(box_a, box_b)
        union = self.box_sz(box_a).unsqueeze(1) + self.box_sz(box_b).unsqueeze(0) - inter
        return inter / union

    def get_y(self, bbox, clas):

        bbox = bbox.view(-1,4)/self.image_size[0]
        bb_keep = ((bbox[:,2]-bbox[:,0])>0).nonzero()[:,0]
        return bbox[bb_keep],clas[bb_keep]

    def actn_to_bb(self, actn):
        actn_bbs = torch.tanh(actn)
        # print(self.grid_sizes.size())
        # print(self.anchors[:,:2].size())
        actn_centers = (actn_bbs[:,:2]/2 * self.grid_sizes) + self.anchors[:,:2]
        actn_hw = (actn_bbs[:,2:]/2+1) * self.anchors[:,2:]
        return hw2corners(actn_centers, actn_hw)

    def map_to_ground_truth(self, overlaps, print_it=False):

        prior_overlap, prior_idx = overlaps.max(1)
    #     if print_it: print(prior_overlap)
        gt_overlap, gt_idx = overlaps.max(0)
        gt_overlap[prior_idx] = 1.99
        for i,o in enumerate(prior_idx): gt_idx[o] = i
        return gt_overlap,gt_idx

    def ssd_1_loss(self, b_c,b_bb,bbox,clas,print_it=False):

        anchor_cnr = hw2corners(self.anchors[:,:2], self.anchors[:,2:])
        bbox,clas = self.get_y(bbox,clas)
        a_ic = self.actn_to_bb(b_bb)
        overlaps = self.jaccard(bbox.data, anchor_cnr.data)
        gt_overlap,gt_idx = self.map_to_ground_truth(overlaps,print_it)
        gt_clas = clas[gt_idx]
        pos = gt_overlap > 0.4
        pos_idx = torch.nonzero(pos)[:,0]
        gt_clas[1-pos] = len(self.class_names)
        gt_bbox = bbox[gt_idx]
        loc_loss = ((a_ic[pos_idx] - gt_bbox[pos_idx]).abs()).mean()
        clas_loss  = self.loss_f(b_c, gt_clas)
        return loc_loss, clas_loss

    def ssd_loss(self, pred, targ, print_it=False):

        lcs,lls = 0.,0.
        for b_c,b_bb,bbox,clas in zip(*pred,*targ):
            loc_loss,clas_loss = self.ssd_1_loss(b_c,b_bb,bbox,clas)
            lls += loc_loss
            lcs += clas_loss
        if print_it: print(f'loc: {lls.data[0]}, clas: {lcs.data[0]}')
        return lls+lcs

    def set_loss(self,loss):
        self.loss_f = loss    

    # def dai_plot_results(self,thresh,loader,model):
    
    #     dai_x,dai_y = next(iter(loader))
    #     dai_x = dai_x.to(self.device)
    #     dai_y = [torch.tensor(l).to(self.device) for l in dai_y]
    #     dai_batch = model(dai_x)
    #     dai_b_clas,dai_b_bb = dai_batch
    #     dai_x = dai_x.cpu()
    #     dai_y = [torch.tensor(l).cpu() for l in dai_y]


    #     fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    #     for idx,ax in enumerate(axes.flat):
    #         ima = dai.denorm_img(dai_x[idx])
    #         bbox,clas = self.get_y(dai_y[0][idx], dai_y[1][idx])
    #         a_ic = self.actn_to_bb(dai_b_bb[idx])
    #         clas_pr, clas_ids = dai_b_clas[idx].max(1)
    #         clas_pr = clas_pr.sigmoid()
    #         self.show_objects(ax, ima, a_ic, clas_ids, clas_pr, clas_pr.max().data[0]*thresh)
    #     plt.tight_layout()  

    def batch_loss(self,model,loader,crit):
        
        data_batch = next(iter(loader))
        dai_x,dai_y = data_batch[0],data_batch[1]
        dai_x = dai_x.to(self.device)
        dai_y = [torch.tensor(l).to(self.device) if type(l).__name__ == 'Tensor' else l.to(device) for l in dai_y]
        dai_batch = model(dai_x)
        return crit(dai_batch,dai_y)

    def show_objects(self, ax, ima, bbox, clas, prs=None, thresh=0.4):

        return self.show_objects_(ax, ima, ((bbox*self.image_size[0]).long()).numpy(),
            (clas).numpy(), (prs).numpy() if prs is not None else None, thresh)

    def show_objects_(self, ax, im, bbox, clas=None, prs=None, thresh=0.3):

        bb = [bb_hw(o) for o in bbox.reshape(-1,4)]
        # print(bb)
        if prs is None:  prs  = [None]*len(bb)
        if clas is None: clas = [None]*len(bb)
        ax = img_grid(im, ax=ax)
        for i,(b,c,pr) in enumerate(zip(bb, clas, prs)):
            if((b[2]>0) and (pr is None or pr > thresh)):
                draw_rect(ax, b, color=self.colr_list[i % self.num_colr])
                txt = f'{i}: '
                if c is not None: txt += ('bg' if c==len(self.class_names) else self.class_names[c])
                if pr is not None: txt += f' {pr:.2f}'
                draw_text(ax, b[:2], txt, color=self.colr_list[i % self.num_colr])
        return ax

    def show_nms(self,loader = None,num = 10,img_batch = None,score_thresh = 0.25,nms_overlap = 0.1,dp = None):

        if loader:    
            x = next(iter(loader))[0]
            batch = self.predict(x)
            pred_clas,pred_bbox = batch
            x = x.cpu()
        else:
            x = img_batch
            batch = self.predict(x)
            pred_clas,pred_bbox = batch
            x = x.cpu()
            num = len(x)

        for i in range(num):
            # print(i)
            ima = x[i]
            if dp:
                ima = dp.denorm_img(ima)
            else:
                ima = denorm_img_general(ima)

            box_coords = self.actn_to_bb(pred_bbox[i])
            conf_scores = pred_clas[i].sigmoid().t().data
            self.show_nms_(ima,box_coords,conf_scores,score_thresh)

    def show_nms_(self,ima,box_coords,conf_scores,score_thresh = 0.25,nms_overlap = 0.1):

        out1,out2,cc = [],[],[]
        for cl in range(0, len(conf_scores)-1):
            c_mask = conf_scores[cl] > score_thresh
            if c_mask.sum() == 0: continue
            scores = conf_scores[cl][c_mask]
            l_mask = c_mask.unsqueeze(1).expand_as(box_coords)
            boxes = box_coords[l_mask].view(-1, 4)
            ids, count = nms(boxes.data, scores, nms_overlap, 50)
            ids = ids[:count]
            out1.append(scores[ids])
            out2.append(boxes.data[ids])
            cc.append([cl]*count)
        # return(out1,out2)    
        if len(cc)> 0:    
            cc = torch.from_numpy(np.concatenate(cc))
            out1 = torch.cat(out1).cpu()
            out2 = torch.cat(out2).cpu()
            fig, ax = plt.subplots(figsize=(8,8))
            ax = self.show_objects(ax, ima, out2, cc, out1, score_thresh)
            plt.show()
        else:
            plt.imshow(ima)
            plt.show()

    def predict_objects(self,img_batch,score_thresh = 0.25,nms_overlap = 0.1,dp = None):

        x = img_batch
        batch = self.predict(x)
        pred_clas,pred_bbox = batch
        x = x.cpu()
        for i,img in enumerate(x):
            # print(i)
            if dp:
                img = dp.denorm_img(img)
            else:
                img = denorm_img_general(img)

            box_coords = self.actn_to_bb(pred_bbox[i])
            conf_scores = pred_clas[i].sigmoid().t().data

            out1,out2,cc = [],[],[]
            for cl in range(0, len(conf_scores)-1):
                c_mask = conf_scores[cl] > score_thresh
                if c_mask.sum() == 0: continue
                scores = conf_scores[cl][c_mask]
                l_mask = c_mask.unsqueeze(1).expand_as(box_coords)
                boxes = box_coords[l_mask].view(-1, 4)
                ids, count = nms(boxes.data, scores, nms_overlap, 50)
                ids = ids[:count]
                out1.append(scores[ids])
                out2.append(boxes.data[ids])
                cc.append([cl]*count)
            # return(out1,out2)    
            if len(cc)> 0:    
                clas = np.concatenate(cc)
                prs = torch.cat(out1).cpu().numpy()
                bbox = ((torch.cat(out2).cpu()*self.image_size[0]).long()).numpy()
                # bb = [bb_hw(o) for o in bbox.reshape(-1,4)]
                bb = [[o[1],o[0],o[3],o[2]] for o in bbox.reshape(-1,4)]
                return (bb, ['bg' if c == len(self.class_names) else self.class_names[c] for c in clas])
            else:
                return ([],[])


class CustomSSDObjectDetection_LPR(TransferNetworkImg):
    def __init__(self,
                 model_name='resnet50',
                 model_type='obj_detection',
                 lr=0.02,
                 one_cycle_factor = 0.5,
                 criterion= nn.NLLLoss(),
                 optimizer_name = 'Adam',
                 dropout_p=0.45,
                 pretrained=True,
                 device=None,
                 best_accuracy=0.,
                 best_validation_loss=None,
                 best_model_file ='best_custom_ssd.pth',
                 chkpoint_file ='cusstom_ssd_chkpoint_file.pth',
                 head = {'num_outputs':10,
                    'layers':[],
                    'model_type':'classifier'
                 },
                 pre_trained_back = None,
                 add_extra = True,
                 class_names = None,
                 num_classes = None,
                 image_size = (224,224)):
        super().__init__(model_name = model_name,
                 model_type = model_type,
                 lr = lr,
                 one_cycle_factor = one_cycle_factor,
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
                 add_extra = add_extra,
                 set_params = False,
                 class_names = class_names,
                 num_classes = num_classes,
                 set_head = False)

        self.obj = True
        self.image_size = image_size
        grids = get_grids(image_size,ceil = self.grid_ceil)
        print('Grids: {}'.format(grids))
        self.set_up_object_detection(anc_grids=grids, anc_zooms=[0.7,1.,1.3], anc_ratios=[(1.,1.),(1.,0.5),(0.5,1.)],
                                    num_classes=num_classes,drop_out=dropout_p)          
        self.set_model_head(model_name = model_name, head = self.custom_head,pre_trained_back = pre_trained_back)
        super(CustomSSDObjectDetection_LPR,self).set_model_params(
                                              criterion = self.ssd_loss,
                                              optimizer_name = optimizer_name,
                                              lr = lr,
                                              one_cycle_factor = one_cycle_factor,
                                              dropout_p = dropout_p,
                                              model_name = model_name,
                                              model_type = model_type,
                                              best_accuracy = best_accuracy,
                                              best_validation_loss = best_validation_loss,
                                              best_model_file = best_model_file,
                                              chkpoint_file = chkpoint_file,
                                              head = head,
                                              class_names = class_names,
                                              num_classes = num_classes
                                              )
        self.model = self.model.to(device)

    def set_model_head(self,
                        model_name = 'resnet50',
                        head = {'num_outputs':10,
                                'layers':[],
                                'class_names': None,
                                'model_type':'classifier'
                               },
                        pre_trained_back = None,
                        criterion = nn.NLLLoss(),  
                        adaptive = True,       
                        dropout_p = 0.45,
                        device = None):

        # models_meta = {
        # 'resnet': {'head_id': -2, 'adaptive_head': [DAI_AvgPool,Flatten()],'normal_head': [nn.AvgPool2d(7,1),Flatten()]},
        # 'densenet': {'head_id': -1,'adaptive_head': [nn.ReLU(inplace=True),DAI_AvgPool,Flatten()]
        #             ,'normal_head': [nn.ReLU(inplace=True),nn.AvgPool2d(7,1),Flatten()]}
        # }

        models_meta = {
        'resnet34': {'conv_channels':512,'head_id': -2,'clip_func':resnet_obj_clip,'adaptive_head': [DAI_AvgPool],
                    'normal_head': [nn.AvgPool2d(7,1)]},
        'resnet50': {'conv_channels':2048,'head_id': -2,'clip_func':resnet_obj_clip,'adaptive_head': [DAI_AvgPool],
                    'normal_head': [nn.AvgPool2d(7,1)]},
        'densenet': {'conv_channels':1024,'head_id': -1,'clip_func':densenet_obj_clip,'adaptive_head': [nn.ReLU(inplace=True),DAI_AvgPool]
                    ,'normal_head': [nn.ReLU(inplace=True),nn.AvgPool2d(7,1)]}
        }

        # name = ''.join([x for x in model_name.lower() if x.isalpha()])
        name = model_name.lower()
        meta = models_meta[name]
        if pre_trained_back:
            model = pre_trained_back
        else:    
            modules = list(self.model.children())
            # l = meta['clip_func'](modules,meta['head_id'])
            l = modules[:meta['head_id']]
            if self.dream_model:
                # for layer in self.dream_model.children():
                #     if(type(layer).__name__) == 'Conv2d':
                #         layer.in_channels,layer.out_channels = meta['conv_channels'],meta['conv_channels']
                #     if(type(layer).__name__) == 'BatchNorm2d':
                #         layer.num_features = meta['conv_channels']
                # self.dream_model[0].in_channels = meta['conv_channels']
                # self.dream_model[-4].out_channels = meta['conv_channels']
                # self.dream_model[-3].num_features = meta['conv_channels']
                l+=self.dream_model

            # if type(head).__name__ != 'dict':
            model = nn.Sequential(*l)
        for layer in head.children():
            if(type(layer).__name__) == 'StdConv':
                conv_module = layer
                break
        # temp_conv = head.sconv0.conv
        conv_layer = conv_module.conv
        temp_args = [conv_layer.out_channels,conv_layer.kernel_size,conv_layer.stride,conv_layer.padding]
        temp_args.insert(0,meta['conv_channels'])
        conv_layer = nn.Conv2d(*temp_args)
        conv_module.conv = conv_layer
        # print(head)
        # model.add_module('adaptive_avg_pool',DAI_AvgPool)
        model.add_module('custom_head',head)
        # else:
        #     head['criterion'] = criterion
        #     self.num_outputs = head['num_outputs']
        #     fc = modules[-1]
        #     in_features =  fc.in_features
        #     fc = FC(
        #             num_inputs = in_features,
        #             num_outputs = head['num_outputs'],
        #             layers = head['layers'],
        #             model_type = head['model_type'],
        #             output_non_linearity = head['output_non_linearity'],
        #             dropout_p = dropout_p,
        #             criterion = head['criterion'],
        #             optimizer_name = None,
        #             device = device
        #             )
        #     if adaptive:
        #         l += meta['adaptive_head']
        #     else:
        #         l += meta['normal_head']
        #     model = nn.Sequential(*l)
        #     model.add_module('fc',fc)
        self.model = model
        self.head = head
        
        # if type(head).__name__ == 'dict':
        #     print('{}: setting head: inputs: {} hidden:{} outputs: {}'.format(model_name,
        #                                                                   in_features,
        #                                                                   head['layers'],
        #                                                                   head['num_outputs']))
        # else:
        print('{}: setting head: {}'.format(model_name,type(head).__name__))    

    def set_up_object_detection(self,anc_grids,anc_zooms,anc_ratios,num_classes,num_colr = 12,drop_out = 0.5):

        # print('Would you like to give your own values for anchor_grids, anchor_zooms,and anchor_ratios? The default values are: {}, {} and {}'
        # .format(anc_grids,anc_zooms,anc_ratios))
        # print('If so, you may call the function "set_up_object_detection" with your own paramteres.')

        cmap = get_cmap(num_colr)
        self.colr_list = [cmap(float(x)) for x in range(num_colr)]
        self.num_colr = num_colr
        self.create_anchors(anc_grids,anc_zooms,anc_ratios)
        self.custom_head = CustomSSD_MultiHead(len(anc_grids),self.k,num_classes,drop_out,-4.)
        self.loss_f = FocalLoss(num_classes,device=self.device)

    def create_anchors(self,anc_grids,anc_zooms,anc_ratios):
    
        anchor_scales = [(anz*i,anz*j) for anz in anc_zooms for (i,j) in anc_ratios]
        k = len(anchor_scales)
        anc_offsets = [1/(o*2) for o in anc_grids]
        anc_x = np.concatenate([np.repeat(np.linspace(ao, 1-ao, ag), ag)
                                for ao,ag in zip(anc_offsets,anc_grids)])
        anc_y = np.concatenate([np.tile(np.linspace(ao, 1-ao, ag), ag)
                                for ao,ag in zip(anc_offsets,anc_grids)])
        anc_ctrs = np.repeat(np.stack([anc_x,anc_y], axis=1), k, axis=0)
        anc_sizes  =   np.concatenate([np.array([[o/ag,p/ag] for i in range(ag*ag) for o,p in anchor_scales])
                    for ag in anc_grids])
        grid_sizes = torch.tensor(np.concatenate([np.array(
                                [ 1/ag for i in range(ag*ag) for o,p in anchor_scales])
                    for ag in anc_grids])).float().unsqueeze(1).to(self.device)
        anchors = torch.tensor(np.concatenate([anc_ctrs, anc_sizes], axis=1)).float().to(self.device)
        anchor_cnr = hw2corners(anchors[:,:2], anchors[:,2:])
        self.anchors,self.anchor_cnr,self.grid_sizes,self.k = anchors,anchor_cnr,grid_sizes,k

    # def draw_im(im, ann, cats):
    #     ax = img_grid(im, figsize=(16,8))
    #     for b,c in ann:
    #         b = bb_hw(b)
    #         draw_rect(ax, b)
    #         draw_text(ax, b[:2], cats[c], sz=16)

    # def draw_idx(i):
    #     im_a = trn_anno[i]
    # #     im = open_image(IMG_PATH/trn_fns[i])
    #     im = Image.open(IMG_PATH/trn_fns[i]).convert('RGB')
    #     draw_im(im, im_a)

    def intersect(self,box_a, box_b):

        max_xy = torch.min(box_a[:, None, 2:], box_b[None, :, 2:])
        min_xy = torch.max(box_a[:, None, :2], box_b[None, :, :2])
        inter = torch.clamp((max_xy - min_xy), min=0)
        return inter[:, :, 0] * inter[:, :, 1]

    def box_sz(self,b): return ((b[:, 2]-b[:, 0]) * (b[:, 3]-b[:, 1]))

    def jaccard(self,box_a, box_b):

        inter = self.intersect(box_a, box_b)
        union = self.box_sz(box_a).unsqueeze(1) + self.box_sz(box_b).unsqueeze(0) - inter
        return inter / union

    def get_y(self, bbox, clas):

        bbox = bbox.view(-1,4)/self.image_size[0]
        bb_keep = ((bbox[:,2]-bbox[:,0])>0).nonzero()[:,0]
        return bbox[bb_keep],clas[bb_keep]

    def actn_to_bb(self, actn):
        actn_bbs = torch.tanh(actn)
        # print(self.grid_sizes.size())
        # print(self.anchors[:,:2].size())
        actn_centers = (actn_bbs[:,:2]/2 * self.grid_sizes) + self.anchors[:,:2]
        actn_hw = (actn_bbs[:,2:]/2+1) * self.anchors[:,2:]
        return hw2corners(actn_centers, actn_hw)

    def map_to_ground_truth(self, overlaps, print_it=False):

        prior_overlap, prior_idx = overlaps.max(1)
    #     if print_it: print(prior_overlap)
        gt_overlap, gt_idx = overlaps.max(0)
        gt_overlap[prior_idx] = 1.99
        for i,o in enumerate(prior_idx): gt_idx[o] = i
        return gt_overlap,gt_idx

    def ssd_1_loss(self, b_c,b_bb,bbox,clas,print_it=False):

        anchor_cnr = hw2corners(self.anchors[:,:2], self.anchors[:,2:])
        bbox,clas = self.get_y(bbox,clas)
        a_ic = self.actn_to_bb(b_bb)
        overlaps = self.jaccard(bbox.data, anchor_cnr.data)
        gt_overlap,gt_idx = self.map_to_ground_truth(overlaps,print_it)
        gt_clas = clas[gt_idx]
        pos = gt_overlap > 0.4
        pos_idx = torch.nonzero(pos)[:,0]
        gt_clas[1-pos] = len(self.class_names)
        gt_bbox = bbox[gt_idx]
        loc_loss = ((a_ic[pos_idx] - gt_bbox[pos_idx]).abs()).mean()
        clas_loss  = self.loss_f(b_c, gt_clas)
        return loc_loss, clas_loss

    def ssd_loss(self, pred, targ, print_it=False):

        lcs,lls = 0.,0.
        for b_c,b_bb,bbox,clas in zip(*pred,*targ):
            loc_loss,clas_loss = self.ssd_1_loss(b_c,b_bb,bbox,clas)
            lls += loc_loss
            lcs += clas_loss
        if print_it: print(f'loc: {lls.data[0]}, clas: {lcs.data[0]}')
        return lls+lcs

    def set_loss(self,loss):
        self.loss_f = loss    

    # def dai_plot_results(self,thresh,loader,model):
    
    #     dai_x,dai_y = next(iter(loader))
    #     dai_x = dai_x.to(self.device)
    #     dai_y = [torch.tensor(l).to(self.device) for l in dai_y]
    #     dai_batch = model(dai_x)
    #     dai_b_clas,dai_b_bb = dai_batch
    #     dai_x = dai_x.cpu()
    #     dai_y = [torch.tensor(l).cpu() for l in dai_y]


    #     fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    #     for idx,ax in enumerate(axes.flat):
    #         ima = dai.denorm_img(dai_x[idx])
    #         bbox,clas = self.get_y(dai_y[0][idx], dai_y[1][idx])
    #         a_ic = self.actn_to_bb(dai_b_bb[idx])
    #         clas_pr, clas_ids = dai_b_clas[idx].max(1)
    #         clas_pr = clas_pr.sigmoid()
    #         self.show_objects(ax, ima, a_ic, clas_ids, clas_pr, clas_pr.max().data[0]*thresh)
    #     plt.tight_layout()  

    def batch_loss(self,model,loader,crit):
        
        data_batch = next(iter(loader))
        dai_x,dai_y = data_batch[0],data_batch[1]
        dai_x = dai_x.to(self.device)
        dai_y = [torch.tensor(l).to(self.device) if type(l).__name__ == 'Tensor' else l.to(device) for l in dai_y]
        dai_batch = model(dai_x)
        return crit(dai_batch,dai_y)

    def show_objects(self, ax, y, num_plate, ima, bbox, clas, prs=None, thresh=0.4,ocr_net=None,ocr_dp=None):

        return self.show_objects_(ax, y, num_plate, ima, ((bbox*self.image_size[0]).long()).numpy(),
            (clas).numpy(), (prs).numpy() if prs is not None else None, thresh,ocr_net=ocr_net,ocr_dp=ocr_dp)

    def show_objects_(self, ax, y, num_plate, im, bbox, clas=None, prs=None, thresh=0.3,ocr_net=None,ocr_dp=None):

        # ocr_net = data_processing.load_obj('/home/farhan/hamza/Object_Detection/best_lpr_chars_net.pkl')
        # ocr_dp = data_processing.load_obj('/home/farhan/hamza/lpr_chars/DP_lpr_chars.pkl')
        bb = [bb_hw(o) for o in bbox.reshape(-1,4)]
        # print(bb)
        if prs is None:  prs  = [None]*len(bb)
        if clas is None: clas = [None]*len(bb)
        ax = img_grid(im, ax=ax)
        for i,(b,c,pr) in enumerate(zip(bb, clas, prs)):
            if((b[2]>0) and (pr is None or pr > thresh)):
                draw_rect(ax, b, color=self.colr_list[i % self.num_colr])
                txt = f'{i}: '
                if c is not None: txt += ('bg' if c==len(self.class_names) else self.class_names[c])
                if pr is not None: txt += f' {pr:.2f}'
                draw_text(ax, b[:2], txt, color=self.colr_list[i % self.num_colr])
                resize_w = 512
                resize_h = 512    
                im_resized = cv2.resize(im,(resize_w,resize_h))
                im_r,im_c = im.shape[0],im.shape[1]
                row_scale = resize_h/im_r
                col_scale = resize_w/im_c
                b = hw_bb(b)
                b[0] = int(np.round(b[0]*col_scale))
                b[1] = int(np.round(b[1]*row_scale))
                b[2] = int(np.round(b[2]*col_scale))
                b[3] = int(np.round(b[3]*row_scale))
                b = bb_hw(b)
                margin = 12
                if b[1] >= 0 and b[0] >= 0:
                    try:
                        if b[1] == 0 or b[0] == 0:
                            im2 = im_resized[b[1]:b[1]+b[3],b[0]:b[0]+b[2]]
                        else:    
                            im2 = im_resized[b[1]-margin:b[1]+b[3]+margin,b[0]-margin:b[0]+b[2]+margin]
                        plt.imsave('carlp.png',im2)
                    except:
                        im2 = im_resized[b[1]:b[1]+b[3],b[0]:b[0]+b[2]]
                        plt.imsave('carlp.png',im2)
                    # print(im2.shape)
                    chars = get_lp_chars('carlp.png',size = 150,char_width = 9)['char_list']
                    plot_in_row(chars,figsize = (8,8))
                    all_found = len(chars) == len(num_plate)
                    for char_id,char in enumerate(chars):
                        # print(char.shape)
                        img = get_test_input(imgs = [char],size = (40,40),show=True)
                        # img_ = cv2.resize(char, (40,40))
                        class_conf = ocr_net.predict(img)[0].max(0)[0]
                        class_id = ocr_net.predict(img)[0].max(0)[1]
                        # if all_found:
                        #     char_name = num_plate[char_id]
                        #     file_name = num_plate+'_'+char_name+'.png'
                        #     name_count = 0
                        #     file_path = os.path.join('/home/farhan/hamza/lpr_chars/',file_name)
                        #     while os.path.exists(file_path):    
                        #         file_name = num_plate+'_'+char_name+'_'+str(name_count)+'.png'
                        #         file_path = os.path.join('/home/farhan/hamza/lpr_chars/',file_name)
                        #         name_count+=1        
                        #     plt.imsave(file_path,img_)
                        pred_name = ocr_dp.class_names[class_id]
                        # pred_name = chr(int(ocr_dp.class_names[class_id]))
                        print(pred_name,class_conf)
                        # print(ocr_dp.class_names[class_id], ' ', class_conf)    
        del ocr_net
        del ocr_dp

        return ax

    def show_nms(self,loader = None,num = 10,img_batch = None,score_thresh = 0.25,nms_overlap = 0.1,dp = None):

        if loader:
            data_batch = next(iter(loader))
            x,y = data_batch[0],data_batch[1]
            batch = self.predict(x)
            pred_clas,pred_bbox = batch
            x = x.cpu()

            for i in range(num):
                print(i)
                ima = dp.denorm_img(x[i])
                num_plate = dp.class_names[y[i]]
                box_coords = self.actn_to_bb(pred_bbox[i])
                conf_scores = pred_clas[i].sigmoid().t().data
                self.show_nms_(ima,y,num_plate,box_coords,conf_scores,score_thresh,nms_overlap)
        else:
            x = img_batch
            batch = self.predict(x)
            pred_clas,pred_bbox = batch
            x = x.cpu()
            for i in range(len(x)):
                print(i)
                ima = dp.denorm_img(x[i])
                box_coords = self.actn_to_bb(pred_bbox[i])
                conf_scores = pred_clas[i].sigmoid().t().data
                self.show_nms_(ima,box_coords,conf_scores,score_thresh)

    def show_nms_lpr(self,loader = None,score_thresh = 0.25,nms_overlap = 0.1,dp = None):

        ocr_net = data_processing.load_obj('/home/farhan/hamza/Object_Detection/best_lpr_chars_net.pkl')
        ocr_dp = data_processing.load_obj('/home/farhan/hamza/lpr_chars/DP_lpr_chars.pkl')
        for data_batch in loader:
            x,y = data_batch[0],data_batch[1]
            batch = self.predict(x)
            pred_clas,pred_bbox = batch
            x = x.cpu()

            for i in range(len(x)):
                # print(i)
                ima = dp.denorm_img(x[i])
                num_plate = dp.class_names[y[i]]
                box_coords = self.actn_to_bb(pred_bbox[i])
                conf_scores = pred_clas[i].sigmoid().t().data
                self.show_nms_lpr_(ima,y,num_plate,box_coords,conf_scores,score_thresh,nms_overlap,ocr_net,ocr_dp)

    def show_nms_lpr_(self,ima,y,num_plate,box_coords,conf_scores,score_thresh = 0.25,
                      nms_overlap = 0.1,ocr_net = None,ocr_dp = None):

        out1,out2,cc = [],[],[]
        for cl in range(0, len(conf_scores)-1):
            c_mask = conf_scores[cl] > score_thresh
            if c_mask.sum() == 0: continue
            scores = conf_scores[cl][c_mask]
            l_mask = c_mask.unsqueeze(1).expand_as(box_coords)
            boxes = box_coords[l_mask].view(-1, 4)
            ids, count = nms(boxes.data, scores, nms_overlap, 50)
            ids = ids[:count]
            out1.append(scores[ids])
            out2.append(boxes.data[ids])
            cc.append([cl]*count)
        # return(out1,out2)    
        if len(cc)> 0:    
            cc = torch.from_numpy(np.concatenate(cc))
            out1 = torch.cat(out1).cpu()
            out2 = torch.cat(out2).cpu()
            fig, ax = plt.subplots(figsize=(8,8))
            ax = self.show_objects(ax, y, num_plate, ima, out2, cc, out1, score_thresh,ocr_net=ocr_net,ocr_dp=ocr_dp)
            plt.show()
        # else:
        #     plt.imshow(ima)
        #     plt.show()

    def show_nms_(self,ima,y,num_plate,box_coords,conf_scores,score_thresh = 0.25,nms_overlap = 0.1):

        out1,out2,cc = [],[],[]
        for cl in range(0, len(conf_scores)-1):
            c_mask = conf_scores[cl] > score_thresh
            if c_mask.sum() == 0: continue
            scores = conf_scores[cl][c_mask]
            l_mask = c_mask.unsqueeze(1).expand_as(box_coords)
            boxes = box_coords[l_mask].view(-1, 4)
            ids, count = nms(boxes.data, scores, nms_overlap, 50)
            ids = ids[:count]
            out1.append(scores[ids])
            out2.append(boxes.data[ids])
            cc.append([cl]*count)
        # return(out1,out2)    
        if len(cc)> 0:    
            cc = torch.from_numpy(np.concatenate(cc))
            out1 = torch.cat(out1).cpu()
            out2 = torch.cat(out2).cpu()
            fig, ax = plt.subplots(figsize=(8,8))
            ax = self.show_objects(ax, y, num_plate, ima, out2, cc, out1, score_thresh)
            plt.show()
        else:
            plt.imshow(ima)
            plt.show()

class ResNetUnetObjectDetection(TransferNetworkImg):

    def __init__(self,
                 model_name='resnet34',
                 model_type='obj_detection',
                 lr=0.02,
                 one_cycle_factor = 0.5,
                 criterion= nn.NLLLoss(),
                 optimizer_name = 'Adam',
                 dropout_p=0.45,
                 pretrained=True,
                 device=None,
                 best_accuracy=0.,
                 best_validation_loss=None,
                 best_model_file ='best_resnet_unet_obj.pth',
                 chkpoint_file ='resnet_unet_obj_chkpoint_file.pth',
                 head = {'num_outputs':10,
                    'layers':[],
                    'model_type':'classifier'
                 },
                 add_extra = True,
                 class_names = None,
                 num_classes = None,
                 image_size = (224,224)):
        super().__init__(model_name = model_name,
                 model_type = model_type,
                 lr = lr,
                 one_cycle_factor = one_cycle_factor,
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
                 add_extra = add_extra,
                 set_params = False,
                 class_names = class_names,
                 num_classes = num_classes,
                 set_head = False)

        self.obj = True
        self.image_size = image_size
        temp_g = int(np.ceil(image_size[0]/4))
        grids = [temp_g]
        for _ in range(6):
            temp_g = int(np.ceil(temp_g/2))
            grids.append(temp_g)   
        # grid1 = int(np.ceil(image_size[0]/4))
        # grid2 = int(np.ceil(grid1/2))
        # grid3 = int(np.ceil(grid2/2))
        # grid4 = int(np.ceil(grid3/2))
        self.set_up_object_detection(anc_grids=grids, anc_zooms=[0.7,1.,1.3],anc_ratios=[(1.,1.),(1.,0.5),(0.5,1.)],
                                    num_classes=num_classes,drop_out=dropout_p)
        self.set_up_model(model_name=model_name,head = self.custom_head,unet_class=resnet_unet.Unet,device=device)
        super(ResNetUnetObjectDetection,self).set_model_params(
                                              criterion = self.ssd_loss,
                                              optimizer_name = optimizer_name,
                                              lr = lr,
                                              one_cycle_factor = one_cycle_factor,
                                              dropout_p = dropout_p,
                                              model_name = model_name,
                                              model_type = model_type,
                                              best_accuracy = best_accuracy,
                                              best_validation_loss = best_validation_loss,
                                              best_model_file = best_model_file,
                                              chkpoint_file = chkpoint_file,
                                              head = head,
                                              class_names = class_names,
                                              num_classes = num_classes
                                              )
        self.model = self.model.to(device)

    def set_up_model(self,
                        model_name = 'resnet34',
                        head = {'num_outputs':10,
                                'layers':[],
                                'class_names': None,
                                'model_type':'classifier'
                               },
                        unet_class = resnet_unet.Unet,
                        criterion = nn.NLLLoss(),  
                        adaptive = True,       
                        dropout_p = 0.45,
                        device = None):

        # models_meta = {
        # 'resnet': {'head_id': -2, 'adaptive_head': [DAI_AvgPool,Flatten()],'normal_head': [nn.AvgPool2d(7,1),Flatten()]},
        # 'densenet': {'head_id': -1,'adaptive_head': [nn.ReLU(inplace=True),DAI_AvgPool,Flatten()]
        #             ,'normal_head': [nn.ReLU(inplace=True),nn.AvgPool2d(7,1),Flatten()]}
        # }

        models_meta = {
        'resnet34': {'conv_channels':512,'head_id': -2, 'adaptive_head': [DAI_AvgPool],'normal_head': [nn.AvgPool2d(7,1)]},
        'resnet50': {'conv_channels':2048,'head_id': -2, 'adaptive_head': [DAI_AvgPool],'normal_head': [nn.AvgPool2d(7,1)]},
        'densenet': {'conv_channels':1024,'head_id': -1,'adaptive_head': [nn.ReLU(inplace=True),DAI_AvgPool]
                    ,'normal_head': [nn.ReLU(inplace=True),nn.AvgPool2d(7,1)]}
        }

        # name = ''.join([x for x in model_name.lower() if x.isalpha()])
        name = model_name.lower()
        meta = models_meta[name]
        modules = list(self.model.children())
        l = modules[:meta['head_id']]
        if self.dream_model:
            l+=self.dream_model

        model = nn.Sequential(*l)
        model = unet_class(model,meta['conv_channels'])
        self.unet_model = model

        if type(head).__name__ != 'dict':
            # model = nn.Sequential(*l)
            for layer in head.children():
                if(type(layer).__name__) == 'StdConv':
                    conv_module = layer
                    break
            # temp_conv = head.sconv0.conv
            conv_layer = conv_module.conv
            temp_args = [conv_layer.out_channels,conv_layer.kernel_size,conv_layer.stride,conv_layer.padding]
            # temp_args.insert(0,meta['conv_channels'])
            temp_args.insert(0,3)
            conv_layer = nn.Conv2d(*temp_args)
            conv_module.conv = conv_layer
            # print(head)
            # model.add_module('adaptive_avg_pool',DAI_AvgPool)
            model.add_module('custom_head',head)
        else:
            head['criterion'] = criterion
            self.num_outputs = head['num_outputs']
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
        
        if type(head).__name__ == 'dict':
            print('{}: setting head: inputs: {} hidden:{} outputs: {}'.format(model_name,
                                                                          in_features,
                                                                          head['layers'],
                                                                          head['num_outputs']))
        else:
            print('{}: setting head: {}'.format(model_name,type(head).__name__))

    def forward(self,x):
        x = self.unet_model(x)
        x = self.head(x)
        return x

    def set_up_object_detection(self,anc_grids,anc_zooms,anc_ratios,num_classes,num_colr = 12,drop_out = 0.5):

        # print('Would you like to give your own values for anchor_grids, anchor_zooms,and anchor_ratios? The default values are: {}, {} and {}'
        # .format(anc_grids,anc_zooms,anc_ratios))
        # print('If so, you may call the function "set_up_object_detection" with your own paramteres.')

        cmap = get_cmap(num_colr)
        self.colr_list = [cmap(float(x)) for x in range(num_colr)]
        self.num_colr = num_colr
        self.create_anchors(anc_grids,anc_zooms,anc_ratios)
        self.custom_head = UNet_MultiHead(self.k,num_classes,drop_out,-4.)
        self.loss_f = FocalLoss(num_classes,device=self.device)

    def create_anchors(self,anc_grids,anc_zooms,anc_ratios):
    
        anchor_scales = [(anz*i,anz*j) for anz in anc_zooms for (i,j) in anc_ratios]
        k = len(anchor_scales)
        anc_offsets = [1/(o*2) for o in anc_grids]
        anc_x = np.concatenate([np.repeat(np.linspace(ao, 1-ao, ag), ag)
                                for ao,ag in zip(anc_offsets,anc_grids)])
        anc_y = np.concatenate([np.tile(np.linspace(ao, 1-ao, ag), ag)
                                for ao,ag in zip(anc_offsets,anc_grids)])
        anc_ctrs = np.repeat(np.stack([anc_x,anc_y], axis=1), k, axis=0)
        anc_sizes  =   np.concatenate([np.array([[o/ag,p/ag] for i in range(ag*ag) for o,p in anchor_scales])
                    for ag in anc_grids])
        grid_sizes = torch.tensor(np.concatenate([np.array(
                                [ 1/ag for i in range(ag*ag) for o,p in anchor_scales])
                    for ag in anc_grids])).float().unsqueeze(1).to(self.device)
        anchors = torch.tensor(np.concatenate([anc_ctrs, anc_sizes], axis=1)).float().to(self.device)
        anchor_cnr = hw2corners(anchors[:,:2], anchors[:,2:])
        self.anchors,self.anchor_cnr,self.grid_sizes,self.k = anchors,anchor_cnr,grid_sizes,k

    # def draw_im(im, ann, cats):
    #     ax = img_grid(im, figsize=(16,8))
    #     for b,c in ann:
    #         b = bb_hw(b)
    #         draw_rect(ax, b)
    #         draw_text(ax, b[:2], cats[c], sz=16)

    # def draw_idx(i):
    #     im_a = trn_anno[i]
    # #     im = open_image(IMG_PATH/trn_fns[i])
    #     im = Image.open(IMG_PATH/trn_fns[i]).convert('RGB')
    #     draw_im(im, im_a)


    def show_objects_(self, ax, im, bbox, clas=None, prs=None, thresh=0.3):

        bb = [bb_hw(o) for o in bbox.reshape(-1,4)]
        # print(bb)
        if prs is None:  prs  = [None]*len(bb)
        if clas is None: clas = [None]*len(bb)
        ax = img_grid(im, ax=ax)
        for i,(b,c,pr) in enumerate(zip(bb, clas, prs)):
            if((b[2]>0) and (pr is None or pr > thresh)):
                print(b)
                draw_rect(ax, b, color=self.colr_list[i % self.num_colr])
                txt = f'{i}: '
                if c is not None: txt += ('bg' if c==len(self.class_names) else self.class_names[c])
                if pr is not None: txt += f' {pr:.2f}'
                draw_text(ax, b[:2], txt, color=self.colr_list[i % self.num_colr])
        return ax

    def intersect(self,box_a, box_b):

        max_xy = torch.min(box_a[:, None, 2:], box_b[None, :, 2:])
        min_xy = torch.max(box_a[:, None, :2], box_b[None, :, :2])
        inter = torch.clamp((max_xy - min_xy), min=0)
        return inter[:, :, 0] * inter[:, :, 1]

    def box_sz(self,b): return ((b[:, 2]-b[:, 0]) * (b[:, 3]-b[:, 1]))

    def jaccard(self,box_a, box_b):

        inter = self.intersect(box_a, box_b)
        union = self.box_sz(box_a).unsqueeze(1) + self.box_sz(box_b).unsqueeze(0) - inter
        return inter / union

    def get_y(self, bbox, clas):

        bbox = bbox.view(-1,4)/self.image_size[0]
        bb_keep = ((bbox[:,2]-bbox[:,0])>0).nonzero()[:,0]
        return bbox[bb_keep],clas[bb_keep]

    def actn_to_bb(self, actn):
        actn_bbs = torch.tanh(actn)
        # print(self.grid_sizes.size())
        # print(self.anchors[:,:2].size())
        actn_centers = (actn_bbs[:,:2]/2 * self.grid_sizes) + self.anchors[:,:2]
        actn_hw = (actn_bbs[:,2:]/2+1) * self.anchors[:,2:]
        return hw2corners(actn_centers, actn_hw)

    def map_to_ground_truth(self, overlaps, print_it=False):

        prior_overlap, prior_idx = overlaps.max(1)
    #     if print_it: print(prior_overlap)
        gt_overlap, gt_idx = overlaps.max(0)
        gt_overlap[prior_idx] = 1.99
        for i,o in enumerate(prior_idx): gt_idx[o] = i
        return gt_overlap,gt_idx

    def ssd_1_loss(self, b_c,b_bb,bbox,clas,print_it=False):

        anchor_cnr = hw2corners(self.anchors[:,:2], self.anchors[:,2:])
        bbox,clas = self.get_y(bbox,clas)
        a_ic = self.actn_to_bb(b_bb)
        overlaps = self.jaccard(bbox.data, anchor_cnr.data)
        gt_overlap,gt_idx = self.map_to_ground_truth(overlaps,print_it)
        gt_clas = clas[gt_idx]
        pos = gt_overlap > 0.4
        pos_idx = torch.nonzero(pos)[:,0]
        gt_clas[1-pos] = len(self.class_names)
        gt_bbox = bbox[gt_idx]
        loc_loss = ((a_ic[pos_idx] - gt_bbox[pos_idx]).abs()).mean()
        clas_loss  = self.loss_f(b_c, gt_clas)
        return loc_loss, clas_loss

    def ssd_loss(self, pred, targ, print_it=False):

        lcs,lls = 0.,0.
        for b_c,b_bb,bbox,clas in zip(*pred,*targ):
            loc_loss,clas_loss = self.ssd_1_loss(b_c,b_bb,bbox,clas)
            lls += loc_loss
            lcs += clas_loss
        if print_it: print(f'loc: {lls.data[0]}, clas: {lcs.data[0]}')
        return lls+lcs

    def set_loss(self,loss):
        self.loss_f = loss    

    def show_objects(self, ax, ima, bbox, clas, prs=None, thresh=0.4):

        return self.show_objects_(ax, ima, ((bbox*self.image_size[0]).long()).numpy(),
            (clas).numpy(), (prs).numpy() if prs is not None else None, thresh)

    # def dai_plot_results(self,thresh,loader,model):
    
    #     dai_x,dai_y = next(iter(loader))
    #     dai_x = dai_x.to(self.device)
    #     dai_y = [torch.tensor(l).to(self.device) for l in dai_y]
    #     dai_batch = model(dai_x)
    #     dai_b_clas,dai_b_bb = dai_batch
    #     dai_x = dai_x.cpu()
    #     dai_y = [torch.tensor(l).cpu() for l in dai_y]


    #     fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    #     for idx,ax in enumerate(axes.flat):
    #         ima = dai.denorm_img(dai_x[idx])
    #         bbox,clas = self.get_y(dai_y[0][idx], dai_y[1][idx])
    #         a_ic = self.actn_to_bb(dai_b_bb[idx])
    #         clas_pr, clas_ids = dai_b_clas[idx].max(1)
    #         clas_pr = clas_pr.sigmoid()
    #         self.show_objects(ax, ima, a_ic, clas_ids, clas_pr, clas_pr.max().data[0]*thresh)
    #     plt.tight_layout()  

    def batch_loss(self,model,loader,crit):
        
        data_batch = next(iter(loader))
        dai_x,dai_y = data_batch[0],data_batch[1]
        dai_x = dai_x.to(self.device)
        dai_y = [torch.tensor(l).to(self.device) if type(l).__name__ == 'Tensor' else l.to(device) for l in dai_y]
        dai_batch = model(dai_x)
        return crit(dai_batch,dai_y)

    def show_nms(self,loader = None,num = 10,img = None,score_thresh = 0.25,nms_overlap = 0.1,dp = None):

        if loader:    
            x = next(iter(loader))[0]
            batch = self.predict(x)
            pred_clas,pred_bbox = batch
            x = x.cpu()

            for i in range(num):
                print(i)
                ima = dp.denorm_img(x[i])
                box_coords = self.actn_to_bb(pred_bbox[i])
                conf_scores = pred_clas[i].sigmoid().t().data
                self.show_nms_(ima,box_coords,conf_scores,score_thresh)
        else:
            x = img
            batch = self.predict(x)
            pred_clas,pred_bbox = batch
            x = x.cpu()
            ima = x[0].numpy().transpose(1,2,0)
            box_coords = self.actn_to_bb(pred_bbox[0])
            conf_scores = pred_clas[0].sigmoid().t().data
            # return(box_coords,conf_scores,pred_bbox,pred_clas)
            self.show_nms_(ima,box_coords,conf_scores,score_thresh,nms_overlap)

    def show_nms_(self,ima,box_coords,conf_scores,score_thresh = 0.25,nms_overlap = 0.1):

        out1,out2,cc = [],[],[]
        for cl in range(0, len(conf_scores)-1):
            c_mask = conf_scores[cl] > score_thresh
            if c_mask.sum() == 0: continue
            scores = conf_scores[cl][c_mask]
            l_mask = c_mask.unsqueeze(1).expand_as(box_coords)
            boxes = box_coords[l_mask].view(-1, 4)
            ids, count = nms(boxes.data, scores, nms_overlap, 50)
            ids = ids[:count]
            out1.append(scores[ids])
            out2.append(boxes.data[ids])
            cc.append([cl]*count)
        # return(out1,out2)    
        if len(cc)> 0:    
            cc = torch.from_numpy(np.concatenate(cc))
            out1 = torch.cat(out1).cpu()
            out2 = torch.cat(out2).cpu()
            fig, ax = plt.subplots(figsize=(8,8))
            ax = self.show_objects(ax, ima, out2, cc, out1, score_thresh)
            plt.show()

class ResNetUnetObjectDetection2(TransferNetworkImg):

    def __init__(self,
                 model_name='resnet34',
                 model_type='obj_detection',
                 lr=0.02,
                 one_cycle_factor = 0.5,
                 criterion= nn.NLLLoss(),
                 optimizer_name = 'Adam',
                 dropout_p=0.45,
                 pretrained=True,
                 device=None,
                 best_accuracy=0.,
                 best_validation_loss=None,
                 best_model_file ='best_resnet_unet_obj.pth',
                 chkpoint_file ='resnet_unet_obj_chkpoint_file.pth',
                 head = {'num_outputs':10,
                    'layers':[],
                    'model_type':'classifier'
                 },
                 add_extra = True,
                 class_names = None,
                 num_classes = None,
                 image_size = (224,224)):
        super().__init__(model_name = model_name,
                 model_type = model_type,
                 lr = lr,
                 one_cycle_factor = one_cycle_factor,
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
                 add_extra = add_extra,
                 set_params = False,
                 class_names = class_names,
                 num_classes = num_classes,
                 set_head = False)

        self.obj = True
        self.image_size = image_size
        temp_g = int(np.ceil(image_size[0]/8))
        grids = [temp_g]
        for _ in range(5):
            temp_g = int(np.ceil(temp_g/2))
            grids.append(temp_g)   
        # grid1 = int(np.ceil(image_size[0]/4))
        # grid2 = int(np.ceil(grid1/2))
        # grid3 = int(np.ceil(grid2/2))
        # grid4 = int(np.ceil(grid3/2))
        self.set_up_object_detection(anc_grids=grids, anc_zooms=[0.7,1.,1.3],anc_ratios=[(1.,1.),(1.,0.5),(0.5,1.)],
                                    num_classes=num_classes,drop_out=dropout_p)
        self.set_up_model(model_name=model_name,head = self.custom_head,unet_class=resnet_unet_2.Unet,device=device)
        super(ResNetUnetObjectDetection2,self).set_model_params(
                                              criterion = self.ssd_loss,
                                              optimizer_name = optimizer_name,
                                              lr = lr,
                                              one_cycle_factor = one_cycle_factor,
                                              dropout_p = dropout_p,
                                              model_name = model_name,
                                              model_type = model_type,
                                              best_accuracy = best_accuracy,
                                              best_validation_loss = best_validation_loss,
                                              best_model_file = best_model_file,
                                              chkpoint_file = chkpoint_file,
                                              head = head,
                                              class_names = class_names,
                                              num_classes = num_classes
                                              )
        self.model = self.model.to(device)

    def set_up_model(self,
                        model_name = 'resnet34',
                        head = {'num_outputs':10,
                                'layers':[],
                                'class_names': None,
                                'model_type':'classifier'
                               },
                        unet_class = resnet_unet_2.Unet,
                        criterion = nn.NLLLoss(),  
                        adaptive = True,       
                        dropout_p = 0.45,
                        device = None):

        # models_meta = {
        # 'resnet': {'head_id': -2, 'adaptive_head': [DAI_AvgPool,Flatten()],'normal_head': [nn.AvgPool2d(7,1),Flatten()]},
        # 'densenet': {'head_id': -1,'adaptive_head': [nn.ReLU(inplace=True),DAI_AvgPool,Flatten()]
        #             ,'normal_head': [nn.ReLU(inplace=True),nn.AvgPool2d(7,1),Flatten()]}
        # }

        models_meta = {
        'resnet34': {'conv_channels':512,'head_id': -2, 'adaptive_head': [DAI_AvgPool],'normal_head': [nn.AvgPool2d(7,1)]},
        'resnet50': {'conv_channels':2048,'head_id': -2, 'adaptive_head': [DAI_AvgPool],'normal_head': [nn.AvgPool2d(7,1)]},
        'densenet': {'conv_channels':1024,'head_id': -1,'adaptive_head': [nn.ReLU(inplace=True),DAI_AvgPool]
                    ,'normal_head': [nn.ReLU(inplace=True),nn.AvgPool2d(7,1)]}
        }

        # name = ''.join([x for x in model_name.lower() if x.isalpha()])
        name = model_name.lower()
        meta = models_meta[name]
        modules = list(self.model.children())
        l = modules[:meta['head_id']]
        if self.dream_model:
            l+=self.dream_model

        model = nn.Sequential(*l)
        model = model.to(device)
        head = head.to(device)
        model = unet_class(model,meta['conv_channels'],head)
        self.model = model
        self.head = head
        
        print('{}: setting head: {}'.format(model_name,type(head).__name__))

    def forward(self,x):
        return self.model.forward(x)

    def set_up_object_detection(self,anc_grids,anc_zooms,anc_ratios,num_classes,num_colr = 12,drop_out = 0.5):

        # print('Would you like to give your own values for anchor_grids, anchor_zooms,and anchor_ratios? The default values are: {}, {} and {}'
        # .format(anc_grids,anc_zooms,anc_ratios))
        # print('If so, you may call the function "set_up_object_detection" with your own paramteres.')

        cmap = get_cmap(num_colr)
        self.colr_list = [cmap(float(x)) for x in range(num_colr)]
        self.num_colr = num_colr
        self.create_anchors(anc_grids,anc_zooms,anc_ratios)
        self.custom_head = UNet_MultiHead2(self.k,num_classes,drop_out,-4.)
        self.loss_f = FocalLoss(num_classes,device=self.device)

    def create_anchors(self,anc_grids,anc_zooms,anc_ratios):
    
        anchor_scales = [(anz*i,anz*j) for anz in anc_zooms for (i,j) in anc_ratios]
        k = len(anchor_scales)
        anc_offsets = [1/(o*2) for o in anc_grids]
        anc_x = np.concatenate([np.repeat(np.linspace(ao, 1-ao, ag), ag)
                                for ao,ag in zip(anc_offsets,anc_grids)])
        anc_y = np.concatenate([np.tile(np.linspace(ao, 1-ao, ag), ag)
                                for ao,ag in zip(anc_offsets,anc_grids)])
        anc_ctrs = np.repeat(np.stack([anc_x,anc_y], axis=1), k, axis=0)
        anc_sizes  =   np.concatenate([np.array([[o/ag,p/ag] for i in range(ag*ag) for o,p in anchor_scales])
                    for ag in anc_grids])
        grid_sizes = torch.tensor(np.concatenate([np.array(
                                [ 1/ag for i in range(ag*ag) for o,p in anchor_scales])
                    for ag in anc_grids])).float().unsqueeze(1).to(self.device)
        anchors = torch.tensor(np.concatenate([anc_ctrs, anc_sizes], axis=1)).float().to(self.device)
        anchor_cnr = hw2corners(anchors[:,:2], anchors[:,2:])
        self.anchors,self.anchor_cnr,self.grid_sizes,self.k = anchors,anchor_cnr,grid_sizes,k

    # def draw_im(im, ann, cats):
    #     ax = img_grid(im, figsize=(16,8))
    #     for b,c in ann:
    #         b = bb_hw(b)
    #         draw_rect(ax, b)
    #         draw_text(ax, b[:2], cats[c], sz=16)

    # def draw_idx(i):
    #     im_a = trn_anno[i]
    # #     im = open_image(IMG_PATH/trn_fns[i])
    #     im = Image.open(IMG_PATH/trn_fns[i]).convert('RGB')
    #     draw_im(im, im_a)


    def show_objects_(self, ax, im, bbox, clas=None, prs=None, thresh=0.3):

        bb = [bb_hw(o) for o in bbox.reshape(-1,4)]
        # print(bb)
        if prs is None:  prs  = [None]*len(bb)
        if clas is None: clas = [None]*len(bb)
        ax = img_grid(im, ax=ax)
        for i,(b,c,pr) in enumerate(zip(bb, clas, prs)):
            if((b[2]>0) and (pr is None or pr > thresh)):
                print(b)
                draw_rect(ax, b, color=self.colr_list[i % self.num_colr])
                txt = f'{i}: '
                if c is not None: txt += ('bg' if c==len(self.class_names) else self.class_names[c])
                if pr is not None: txt += f' {pr:.2f}'
                draw_text(ax, b[:2], txt, color=self.colr_list[i % self.num_colr])
        return ax

    def intersect(self,box_a, box_b):

        max_xy = torch.min(box_a[:, None, 2:], box_b[None, :, 2:])
        min_xy = torch.max(box_a[:, None, :2], box_b[None, :, :2])
        inter = torch.clamp((max_xy - min_xy), min=0)
        return inter[:, :, 0] * inter[:, :, 1]

    def box_sz(self,b): return ((b[:, 2]-b[:, 0]) * (b[:, 3]-b[:, 1]))

    def jaccard(self,box_a, box_b):

        inter = self.intersect(box_a, box_b)
        union = self.box_sz(box_a).unsqueeze(1) + self.box_sz(box_b).unsqueeze(0) - inter
        return inter / union

    def get_y(self, bbox, clas):

        bbox = bbox.view(-1,4)/self.image_size[0]
        bb_keep = ((bbox[:,2]-bbox[:,0])>0).nonzero()[:,0]
        return bbox[bb_keep],clas[bb_keep]

    def actn_to_bb(self, actn):
        actn_bbs = torch.tanh(actn)
        # print(self.grid_sizes.size())
        # print(self.anchors[:,:2].size())
        actn_centers = (actn_bbs[:,:2]/2 * self.grid_sizes) + self.anchors[:,:2]
        actn_hw = (actn_bbs[:,2:]/2+1) * self.anchors[:,2:]
        return hw2corners(actn_centers, actn_hw)

    def map_to_ground_truth(self, overlaps, print_it=False):

        prior_overlap, prior_idx = overlaps.max(1)
    #     if print_it: print(prior_overlap)
        gt_overlap, gt_idx = overlaps.max(0)
        gt_overlap[prior_idx] = 1.99
        for i,o in enumerate(prior_idx): gt_idx[o] = i
        return gt_overlap,gt_idx

    def ssd_1_loss(self, b_c,b_bb,bbox,clas,print_it=False):

        anchor_cnr = hw2corners(self.anchors[:,:2], self.anchors[:,2:])
        bbox,clas = self.get_y(bbox,clas)
        a_ic = self.actn_to_bb(b_bb)
        overlaps = self.jaccard(bbox.data, anchor_cnr.data)
        gt_overlap,gt_idx = self.map_to_ground_truth(overlaps,print_it)
        gt_clas = clas[gt_idx]
        pos = gt_overlap > 0.4
        pos_idx = torch.nonzero(pos)[:,0]
        gt_clas[1-pos] = len(self.class_names)
        gt_bbox = bbox[gt_idx]
        loc_loss = ((a_ic[pos_idx] - gt_bbox[pos_idx]).abs()).mean()
        clas_loss  = self.loss_f(b_c, gt_clas)
        return loc_loss, clas_loss

    def ssd_loss(self, pred, targ, print_it=False):

        lcs,lls = 0.,0.
        for b_c,b_bb,bbox,clas in zip(*pred,*targ):
            loc_loss,clas_loss = self.ssd_1_loss(b_c,b_bb,bbox,clas)
            lls += loc_loss
            lcs += clas_loss
        if print_it: print(f'loc: {lls.data[0]}, clas: {lcs.data[0]}')
        return lls+lcs

    def set_loss(self,loss):
        self.loss_f = loss    

    def show_objects(self, ax, ima, bbox, clas, prs=None, thresh=0.4):

        return self.show_objects_(ax, ima, ((bbox*self.image_size[0]).long()).numpy(),
            (clas).numpy(), (prs).numpy() if prs is not None else None, thresh)

    # def dai_plot_results(self,thresh,loader,model):
    
    #     dai_x,dai_y = next(iter(loader))
    #     dai_x = dai_x.to(self.device)
    #     dai_y = [torch.tensor(l).to(self.device) for l in dai_y]
    #     dai_batch = model(dai_x)
    #     dai_b_clas,dai_b_bb = dai_batch
    #     dai_x = dai_x.cpu()
    #     dai_y = [torch.tensor(l).cpu() for l in dai_y]


    #     fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    #     for idx,ax in enumerate(axes.flat):
    #         ima = dai.denorm_img(dai_x[idx])
    #         bbox,clas = self.get_y(dai_y[0][idx], dai_y[1][idx])
    #         a_ic = self.actn_to_bb(dai_b_bb[idx])
    #         clas_pr, clas_ids = dai_b_clas[idx].max(1)
    #         clas_pr = clas_pr.sigmoid()
    #         self.show_objects(ax, ima, a_ic, clas_ids, clas_pr, clas_pr.max().data[0]*thresh)
    #     plt.tight_layout()  

    def batch_loss(self,model,loader,crit):
        
        data_batch = next(iter(loader))
        dai_x,dai_y = data_batch[0],data_batch[1]
        dai_x = dai_x.to(self.device)
        dai_y = [torch.tensor(l).to(self.device) if type(l).__name__ == 'Tensor' else l.to(device) for l in dai_y]
        dai_batch = model(dai_x)
        return crit(dai_batch,dai_y)

    def show_nms(self,loader = None,num = 10,img = None,score_thresh = 0.25,nms_overlap = 0.1,dp = None):

        if loader:    
            x = next(iter(loader))[0]
            batch = self.predict(x)
            pred_clas,pred_bbox = batch
            x = x.cpu()

            for i in range(num):
                print(i)
                ima = dp.denorm_img(x[i])
                box_coords = self.actn_to_bb(pred_bbox[i])
                conf_scores = pred_clas[i].sigmoid().t().data
                self.show_nms_(ima,box_coords,conf_scores,score_thresh)
        else:
            x = img
            batch = self.predict(x)
            pred_clas,pred_bbox = batch
            x = x.cpu()
            ima = x[0].numpy().transpose(1,2,0)
            box_coords = self.actn_to_bb(pred_bbox[0])
            conf_scores = pred_clas[0].sigmoid().t().data
            # return(box_coords,conf_scores,pred_bbox,pred_clas)
            self.show_nms_(ima,box_coords,conf_scores,score_thresh,nms_overlap)

    def show_nms_(self,ima,box_coords,conf_scores,score_thresh = 0.25,nms_overlap = 0.1):

        out1,out2,cc = [],[],[]
        for cl in range(0, len(conf_scores)-1):
            c_mask = conf_scores[cl] > score_thresh
            if c_mask.sum() == 0: continue
            scores = conf_scores[cl][c_mask]
            l_mask = c_mask.unsqueeze(1).expand_as(box_coords)
            boxes = box_coords[l_mask].view(-1, 4)
            ids, count = nms(boxes.data, scores, nms_overlap, 50)
            ids = ids[:count]
            out1.append(scores[ids])
            out2.append(boxes.data[ids])
            cc.append([cl]*count)
        # return(out1,out2)    
        if len(cc)> 0:    
            cc = torch.from_numpy(np.concatenate(cc))
            out1 = torch.cat(out1).cpu()
            out2 = torch.cat(out2).cpu()
            fig, ax = plt.subplots(figsize=(8,8))
            ax = self.show_objects(ax, ima, out2, cc, out1, score_thresh)
            plt.show()

class UnetObjectDetection(Network):

    def __init__(self,
                 model_name='Unet_obj',
                 model_type='obj_detection',
                 lr=0.02,
                 one_cycle_factor = 0.5,
                 criterion= nn.NLLLoss(),
                 optimizer_name = 'Adam',
                 dropout_p=0.45,
                 pretrained=True,
                 device=None,
                 best_accuracy=0.,
                 best_validation_loss=None,
                 best_model_file ='best_unet_obj.pth',
                 chkpoint_file ='unet_obj_chkpoint_file.pth',
                 head = {'num_outputs':10,
                    'layers':[],
                    'model_type':'classifier'
                 },
                 class_names = None,
                 num_classes = None,
                 image_size = (224,224)):
        super().__init__(device=device)
        self.obj = True
        self.image_size = image_size
        temp_g = int(np.ceil(image_size[0]/4))
        grids = [temp_g]
        for _ in range(6):
            temp_g = int(np.ceil(temp_g/2))
            grids.append(temp_g)   
        # grid1 = int(np.ceil(image_size[0]/4))
        # grid2 = int(np.ceil(grid1/2))
        # grid3 = int(np.ceil(grid2/2))
        # grid4 = int(np.ceil(grid3/2))
        self.set_up_object_detection(anc_grids=grids, anc_zooms=[0.7,1.,1.3],anc_ratios=[(1.,1.),(1.,0.5),(0.5,1.)],
                            num_classes=num_classes,drop_out=dropout_p)
        self.set_up_model(head = self.custom_head,unet_class=unet_model.UNet,device=device,unet_out_channels=3)
        self.set_model_params(
                            criterion = self.ssd_loss,
                            optimizer_name = optimizer_name,
                            lr = lr,
                            one_cycle_factor = one_cycle_factor,
                            dropout_p = dropout_p,
                            model_name = model_name,
                            model_type = model_type,
                            best_accuracy = best_accuracy,
                            best_validation_loss = best_validation_loss,
                            best_model_file = best_model_file,
                            chkpoint_file = chkpoint_file,
                            head = head,
                            class_names = class_names,
                            num_classes = num_classes
                            )
        self.model = self.model.to(device)
        
    def set_model_params(self,criterion,
                         optimizer_name,
                         lr,
                         one_cycle_factor,
                         dropout_p,
                         model_name,
                         model_type,
                         best_accuracy,
                         best_validation_loss,
                         best_model_file,
                         chkpoint_file,
                         head,
                         class_names,
                         num_classes):

        super(UnetObjectDetection, self).set_model_params(
                                              criterion = criterion,
                                              optimizer_name = optimizer_name,
                                              lr = lr,
                                              one_cycle_factor = one_cycle_factor,
                                              dropout_p = dropout_p,
                                              model_name = model_name,
                                              model_type = model_type,
                                              best_accuracy = best_accuracy,
                                              best_validation_loss = best_validation_loss,
                                              best_model_file = best_model_file,
                                              chkpoint_file = chkpoint_file,
                                              class_names = class_names,
                                              num_classes = num_classes
                                              )

        # self.head = head
        if len(class_names) == 0:
            self.class_names = {k:str(v) for k,v in enumerate(list(range(head['num_outputs'])))}
        # if 'class_names' in head.keys():
        #     if head['class_names'] is not None:
        #         if len(head['class_names']) > 0:
        #             self.class_names = head['class_names']       
        # self.to(self.device)
        
    def get_model_params(self):
        params = super(UnetObjectDetection, self).get_model_params()
        params['device'] = self.device
        params['head'] = self.head
        return params

    def set_up_model(self,model_name = 'Unet_obj',
                        head = {'num_outputs':10,
                                'layers':[],
                                'class_names': None,
                                'model_type':'classifier'
                               },
                        unet_class = unet_model.UNet,
                        unet_out_channels = 3,
                        criterion = nn.NLLLoss(),
                        adaprive=True,
                        dropout_p = 0.45,
                        device = None):

        model = unet_class(3,unet_out_channels)
        self.unet_model = model

        if type(head).__name__ != 'dict':
            # model = nn.Sequential(*l)
            for layer in head.children():
                if(type(layer).__name__) == 'StdConv':
                    conv_module = layer
                    break
            # temp_conv = head.sconv0.conv
            conv_layer = conv_module.conv
            temp_args = [conv_layer.out_channels,conv_layer.kernel_size,conv_layer.stride,conv_layer.padding]
            # temp_args.insert(0,meta['conv_channels'])
            temp_args.insert(0,3)
            conv_layer = nn.Conv2d(*temp_args)
            conv_module.conv = conv_layer
            # print(head)
            # model.add_module('adaptive_avg_pool',DAI_AvgPool)
            model.add_module('custom_head',head)
        # else:
        #     head['criterion'] = criterion
        #     self.num_outputs = head['num_outputs']
        #     fc = modules[-1]
        #     in_features =  fc.in_features
        #     fc = FC(
        #             num_inputs = in_features,
        #             num_outputs = head['num_outputs'],
        #             layers = head['layers'],
        #             model_type = head['model_type'],
        #             output_non_linearity = head['output_non_linearity'],
        #             dropout_p = dropout_p,
        #             criterion = head['criterion'],
        #             optimizer_name = None,
        #             device = device
        #             )
        #     if adaptive:
        #         l += meta['adaptive_head']
        #     else:
        #         l += meta['normal_head']
        #     model = nn.Sequential(*l)
        #     model.add_module('fc',fc)
        self.model = model
        self.head = head
        
        if type(head).__name__ == 'dict':
            # print('{}: setting head: inputs: {} hidden:{} outputs: {}'.format(model_name,
            #                                                               in_features,
            #                                                               head['layers'],
            #                                                               head['num_outputs']))
            pass
        else:
            print('{}: setting head: {}'.format(model_name,type(head).__name__))

    def forward(self,x):
        x = self.unet_model(x)
        x = self.head(x)
        return x

    def set_up_object_detection(self,anc_grids,anc_zooms,anc_ratios,num_classes,num_colr = 12,drop_out = 0.5):

        # print('Would you like to give your own values for anchor_grids, anchor_zooms,and anchor_ratios? The default values are: {}, {} and {}'
        # .format(anc_grids,anc_zooms,anc_ratios))
        # print('If so, you may call the function "set_up_object_detection" with your own paramteres.')

        cmap = get_cmap(num_colr)
        self.colr_list = [cmap(float(x)) for x in range(num_colr)]
        self.num_colr = num_colr
        self.create_anchors(anc_grids,anc_zooms,anc_ratios)
        self.custom_head = UNet_MultiHead(self.k,num_classes,drop_out,-4.)
        self.loss_f = FocalLoss(num_classes,device=self.device)

    def create_anchors(self,anc_grids,anc_zooms,anc_ratios):
    
        anchor_scales = [(anz*i,anz*j) for anz in anc_zooms for (i,j) in anc_ratios]
        k = len(anchor_scales)
        anc_offsets = [1/(o*2) for o in anc_grids]
        anc_x = np.concatenate([np.repeat(np.linspace(ao, 1-ao, ag), ag)
                                for ao,ag in zip(anc_offsets,anc_grids)])
        anc_y = np.concatenate([np.tile(np.linspace(ao, 1-ao, ag), ag)
                                for ao,ag in zip(anc_offsets,anc_grids)])
        anc_ctrs = np.repeat(np.stack([anc_x,anc_y], axis=1), k, axis=0)
        anc_sizes  =   np.concatenate([np.array([[o/ag,p/ag] for i in range(ag*ag) for o,p in anchor_scales])
                    for ag in anc_grids])
        grid_sizes = torch.tensor(np.concatenate([np.array(
                                [ 1/ag for i in range(ag*ag) for o,p in anchor_scales])
                    for ag in anc_grids])).float().unsqueeze(1).to(self.device)
        anchors = torch.tensor(np.concatenate([anc_ctrs, anc_sizes], axis=1)).float().to(self.device)
        anchor_cnr = hw2corners(anchors[:,:2], anchors[:,2:])
        self.anchors,self.anchor_cnr,self.grid_sizes,self.k = anchors,anchor_cnr,grid_sizes,k

    # def draw_im(im, ann, cats):
    #     ax = img_grid(im, figsize=(16,8))
    #     for b,c in ann:
    #         b = bb_hw(b)
    #         draw_rect(ax, b)
    #         draw_text(ax, b[:2], cats[c], sz=16)

    # def draw_idx(i):
    #     im_a = trn_anno[i]
    # #     im = open_image(IMG_PATH/trn_fns[i])
    #     im = Image.open(IMG_PATH/trn_fns[i]).convert('RGB')
    #     draw_im(im, im_a)


    def show_objects_(self, ax, im, bbox, clas=None, prs=None, thresh=0.3):

        bb = [bb_hw(o) for o in bbox.reshape(-1,4)]
        # print(bb)
        if prs is None:  prs  = [None]*len(bb)
        if clas is None: clas = [None]*len(bb)
        ax = img_grid(im, ax=ax)
        for i,(b,c,pr) in enumerate(zip(bb, clas, prs)):
            if((b[2]>0) and (pr is None or pr > thresh)):
                print(b)
                draw_rect(ax, b, color=self.colr_list[i % self.num_colr])
                txt = f'{i}: '
                if c is not None: txt += ('bg' if c==len(self.class_names) else self.class_names[c])
                if pr is not None: txt += f' {pr:.2f}'
                draw_text(ax, b[:2], txt, color=self.colr_list[i % self.num_colr])
        return ax

    def intersect(self,box_a, box_b):

        max_xy = torch.min(box_a[:, None, 2:], box_b[None, :, 2:])
        min_xy = torch.max(box_a[:, None, :2], box_b[None, :, :2])
        inter = torch.clamp((max_xy - min_xy), min=0)
        return inter[:, :, 0] * inter[:, :, 1]

    def box_sz(self,b): return ((b[:, 2]-b[:, 0]) * (b[:, 3]-b[:, 1]))

    def jaccard(self,box_a, box_b):

        inter = self.intersect(box_a, box_b)
        union = self.box_sz(box_a).unsqueeze(1) + self.box_sz(box_b).unsqueeze(0) - inter
        return inter / union

    def get_y(self, bbox, clas):

        bbox = bbox.view(-1,4)/self.image_size[0]
        bb_keep = ((bbox[:,2]-bbox[:,0])>0).nonzero()[:,0]
        return bbox[bb_keep],clas[bb_keep]

    def actn_to_bb(self, actn):
        actn_bbs = torch.tanh(actn)
        # print(self.grid_sizes.size())
        # print(self.anchors[:,:2].size())
        actn_centers = (actn_bbs[:,:2]/2 * self.grid_sizes) + self.anchors[:,:2]
        actn_hw = (actn_bbs[:,2:]/2+1) * self.anchors[:,2:]
        return hw2corners(actn_centers, actn_hw)

    def map_to_ground_truth(self, overlaps, print_it=False):

        prior_overlap, prior_idx = overlaps.max(1)
    #     if print_it: print(prior_overlap)
        gt_overlap, gt_idx = overlaps.max(0)
        gt_overlap[prior_idx] = 1.99
        for i,o in enumerate(prior_idx): gt_idx[o] = i
        return gt_overlap,gt_idx

    def ssd_1_loss(self, b_c,b_bb,bbox,clas,print_it=False):

        anchor_cnr = hw2corners(self.anchors[:,:2], self.anchors[:,2:])
        bbox,clas = self.get_y(bbox,clas)
        a_ic = self.actn_to_bb(b_bb)
        overlaps = self.jaccard(bbox.data, anchor_cnr.data)
        gt_overlap,gt_idx = self.map_to_ground_truth(overlaps,print_it)
        gt_clas = clas[gt_idx]
        pos = gt_overlap > 0.4
        pos_idx = torch.nonzero(pos)[:,0]
        gt_clas[1-pos] = len(self.class_names)
        gt_bbox = bbox[gt_idx]
        loc_loss = ((a_ic[pos_idx] - gt_bbox[pos_idx]).abs()).mean()
        clas_loss  = self.loss_f(b_c, gt_clas)
        return loc_loss, clas_loss

    def ssd_loss(self, pred, targ, print_it=False):

        lcs,lls = 0.,0.
        for b_c,b_bb,bbox,clas in zip(*pred,*targ):
            loc_loss,clas_loss = self.ssd_1_loss(b_c,b_bb,bbox,clas)
            lls += loc_loss
            lcs += clas_loss
        if print_it: print(f'loc: {lls.data[0]}, clas: {lcs.data[0]}')
        return lls+lcs

    def set_loss(self,loss):
        self.loss_f = loss    

    def show_objects(self, ax, ima, bbox, clas, prs=None, thresh=0.4):

        return self.show_objects_(ax, ima, ((bbox*self.image_size[0]).long()).numpy(),
            (clas).numpy(), (prs).numpy() if prs is not None else None, thresh)

    # def dai_plot_results(self,thresh,loader,model):
    
    #     dai_x,dai_y = next(iter(loader))
    #     dai_x = dai_x.to(self.device)
    #     dai_y = [torch.tensor(l).to(self.device) for l in dai_y]
    #     dai_batch = model(dai_x)
    #     dai_b_clas,dai_b_bb = dai_batch
    #     dai_x = dai_x.cpu()
    #     dai_y = [torch.tensor(l).cpu() for l in dai_y]


    #     fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    #     for idx,ax in enumerate(axes.flat):
    #         ima = dai.denorm_img(dai_x[idx])
    #         bbox,clas = self.get_y(dai_y[0][idx], dai_y[1][idx])
    #         a_ic = self.actn_to_bb(dai_b_bb[idx])
    #         clas_pr, clas_ids = dai_b_clas[idx].max(1)
    #         clas_pr = clas_pr.sigmoid()
    #         self.show_objects(ax, ima, a_ic, clas_ids, clas_pr, clas_pr.max().data[0]*thresh)
    #     plt.tight_layout()  

    def batch_loss(self,model,loader,crit):
        
        data_batch = next(iter(loader))
        dai_x,dai_y = data_batch[0],data_batch[1]
        dai_x = dai_x.to(self.device)
        dai_y = [torch.tensor(l).to(self.device) if type(l).__name__ == 'Tensor' else l.to(device) for l in dai_y]
        dai_batch = model(dai_x)
        return crit(dai_batch,dai_y)

    def show_nms(self,loader = None,num = 10,img = None,score_thresh = 0.25,nms_overlap = 0.1,dp = None):

        if loader:    
            x = next(iter(loader))[0]
            batch = self.predict(x)
            pred_clas,pred_bbox = batch
            x = x.cpu()

            for i in range(num):
                print(i)
                ima = dp.denorm_img(x[i])
                box_coords = self.actn_to_bb(pred_bbox[i])
                conf_scores = pred_clas[i].sigmoid().t().data
                self.show_nms_(ima,box_coords,conf_scores,score_thresh)
        else:
            x = img
            batch = self.predict(x)
            pred_clas,pred_bbox = batch
            x = x.cpu()
            ima = x[0].numpy().transpose(1,2,0)
            box_coords = self.actn_to_bb(pred_bbox[0])
            conf_scores = pred_clas[0].sigmoid().t().data
            # return(box_coords,conf_scores,pred_bbox,pred_clas)
            self.show_nms_(ima,box_coords,conf_scores,score_thresh,nms_overlap)

    def show_nms_(self,ima,box_coords,conf_scores,score_thresh = 0.25,nms_overlap = 0.1):

        out1,out2,cc = [],[],[]
        for cl in range(0, len(conf_scores)-1):
            c_mask = conf_scores[cl] > score_thresh
            if c_mask.sum() == 0: continue
            scores = conf_scores[cl][c_mask]
            l_mask = c_mask.unsqueeze(1).expand_as(box_coords)
            boxes = box_coords[l_mask].view(-1, 4)
            ids, count = nms(boxes.data, scores, nms_overlap, 50)
            ids = ids[:count]
            out1.append(scores[ids])
            out2.append(boxes.data[ids])
            cc.append([cl]*count)
        # return(out1,out2)    
        if len(cc)> 0:    
            cc = torch.from_numpy(np.concatenate(cc))
            out1 = torch.cat(out1).cpu()
            out2 = torch.cat(out2).cpu()
            fig, ax = plt.subplots(figsize=(8,8))
            ax = self.show_objects(ax, ima, out2, cc, out1, score_thresh)
            plt.show()            

class DarknetObjectDetection(Network):
    def __init__(self,
                 model_name='Darknet',
                 model_type='obj_detection',
                 lr=0.02,
                 one_cycle_factor = 0.5,
                 criterion = nn.NLLLoss(),
                 optimizer_name = 'Adam',
                 dropout_p=0.45,
                 pretrained=True,
                 device=None,
                 best_accuracy=0.,
                 best_validation_loss=None,
                 best_model_file ='best_darknet.pth',
                 chkpoint_file ='darknet_chkpoint_file.pth',
                 head = {'num_outputs':10,
                    'layers':[],
                    'model_type':'classifier'
                 },
                 class_names = [],
                 num_classes = None,
                 set_params = True,
                 set_head = True,
                 image_size = (224,224),
                 weights_path = '/home/farhan/data/Object_Detection/yolov3.weights',
                 cfg_path = '/home/farhan/data/dreamai/cfg/DAI.cfg'
                 ):
        
        super().__init__(device=device)
        self.obj = True
        self.image_size = image_size
        self.set_up_object_detection(anc_grids=[13,26,52], anc_zooms=[0.7,1.,1.3], anc_ratios=[(1.,1.),(1.,0.5),(0.5,1.)], num_classes=num_classes)
        self.model = Darknet(cfg_path,k=self.k,num_classes=num_classes)
        if weights_path:
            self.model.load_weights(weights_path)
        print('Using Darknet for Object Detection.')    
        self.set_model_params(criterion = self.darknet_loss,
                            optimizer_name = optimizer_name,
                            lr = lr,
                            one_cycle_factor = one_cycle_factor,
                            dropout_p = dropout_p,
                            model_name = model_name,
                            model_type = model_type,
                            best_accuracy = best_accuracy,
                            best_validation_loss = best_validation_loss,
                            best_model_file = best_model_file,
                            chkpoint_file = chkpoint_file,
                            head = head,
                            class_names = class_names,
                            num_classes = num_classes
                            )

        self.model = self.model.to(device)   
        
    def set_model_params(self,criterion,
                         optimizer_name,
                         lr,
                         one_cycle_factor,
                         dropout_p,
                         model_name,
                         model_type,
                         best_accuracy,
                         best_validation_loss,
                         best_model_file,
                         chkpoint_file,
                         head,
                         class_names,
                         num_classes):
        
        print('Best accuracy = {:.3f}'.format(best_accuracy))
        
        super(DarknetObjectDetection, self).set_model_params(
                                              criterion = criterion,
                                              optimizer_name = optimizer_name,
                                              lr = lr,
                                              one_cycle_factor = one_cycle_factor,
                                              dropout_p = dropout_p,
                                              model_name = model_name,
                                              model_type = model_type,
                                              best_accuracy = best_accuracy,
                                              best_validation_loss = best_validation_loss,
                                              best_model_file = best_model_file,
                                              chkpoint_file = chkpoint_file,
                                              class_names = class_names,
                                              num_classes = num_classes
                                              )

        # self.head = head
        if len(class_names) == 0:
            self.class_names = {k:str(v) for k,v in enumerate(list(range(head['num_outputs'])))}
        # if 'class_names' in head.keys():
        #     if head['class_names'] is not None:
        #         if len(head['class_names']) > 0:
        #             self.class_names = head['class_names']       
        # self.to(self.device)

    def forward(self,x):
        return self.model.forward(x)
        
    def get_model_params(self):
        params = super(DarknetObjectDetection, self).get_model_params()
        params['device'] = self.device
        return params

    def set_up_object_detection(self,anc_grids,anc_zooms,anc_ratios,num_classes,num_colr = 12):

        # print('Would you like to give your own values for anchor_grids, anchor_zooms,and anchor_ratios? The default values are: {}, {} and {}'
        # .format(anc_grids,anc_zooms,anc_ratios))
        # print('If so, you may call the function "set_up_object_detection" with your own paramteres.')

        cmap = get_cmap(num_colr)
        self.colr_list = [cmap(float(x)) for x in range(num_colr)]
        self.num_colr = num_colr
        self.create_anchors(anc_grids,anc_zooms,anc_ratios)
        self.loss_f = FocalLoss(num_classes,device=self.device)

    def create_anchors(self,anc_grids,anc_zooms,anc_ratios):
    
        anchor_scales = [(anz*i,anz*j) for anz in anc_zooms for (i,j) in anc_ratios]
        k = len(anchor_scales)
        anc_offsets = [1/(o*2) for o in anc_grids]
        anc_x = np.concatenate([np.repeat(np.linspace(ao, 1-ao, ag), ag)
                                for ao,ag in zip(anc_offsets,anc_grids)])
        anc_y = np.concatenate([np.tile(np.linspace(ao, 1-ao, ag), ag)
                                for ao,ag in zip(anc_offsets,anc_grids)])
        anc_ctrs = np.repeat(np.stack([anc_x,anc_y], axis=1), k, axis=0)
        anc_sizes  =   np.concatenate([np.array([[o/ag,p/ag] for i in range(ag*ag) for o,p in anchor_scales])
                    for ag in anc_grids])
        grid_sizes = torch.tensor(np.concatenate([np.array(
                                [ 1/ag for i in range(ag*ag) for o,p in anchor_scales])
                    for ag in anc_grids])).float().unsqueeze(1).to(self.device)
        anchors = torch.tensor(np.concatenate([anc_ctrs, anc_sizes], axis=1)).float().to(self.device)
        anchor_cnr = hw2corners(anchors[:,:2], anchors[:,2:])
        self.anchors,self.anchor_cnr,self.grid_sizes,self.k = anchors,anchor_cnr,grid_sizes,k

    # def draw_im(im, ann, cats):
    #     ax = img_grid(im, figsize=(16,8))
    #     for b,c in ann:
    #         b = bb_hw(b)
    #         draw_rect(ax, b)
    #         draw_text(ax, b[:2], cats[c], sz=16)

    # def draw_idx(i):
    #     im_a = trn_anno[i]
    # #     im = open_image(IMG_PATH/trn_fns[i])
    #     im = Image.open(IMG_PATH/trn_fns[i]).convert('RGB')
    #     draw_im(im, im_a)


    def show_objects_(self, ax, im, bbox, clas=None, prs=None, thresh=0.3):

        bb = [bb_hw(o) for o in bbox.reshape(-1,4)]
        if prs is None:  prs  = [None]*len(bb)
        if clas is None: clas = [None]*len(bb)
        ax = img_grid(im, ax=ax)
        for i,(b,c,pr) in enumerate(zip(bb, clas, prs)):
            if((b[2]>0) and (pr is None or pr > thresh)):
                print(b)
                draw_rect(ax, b, color=self.colr_list[i % self.num_colr])
                txt = f'{i}: '
                if c is not None: txt += ('bg' if c==len(self.class_names) else self.class_names[c])
                if pr is not None: txt += f' {pr:.2f}'
                # print(txt,b)    
                draw_text(ax, b[:2], txt, color=self.colr_list[i % self.num_colr])
        return ax        

    def intersect(self,box_a, box_b):

        max_xy = torch.min(box_a[:, None, 2:], box_b[None, :, 2:])
        min_xy = torch.max(box_a[:, None, :2], box_b[None, :, :2])
        inter = torch.clamp((max_xy - min_xy), min=0)
        return inter[:, :, 0] * inter[:, :, 1]

    def box_sz(self,b): return ((b[:, 2]-b[:, 0]) * (b[:, 3]-b[:, 1]))

    def jaccard(self,box_a, box_b):

        inter = self.intersect(box_a, box_b)
        union = self.box_sz(box_a).unsqueeze(1) + self.box_sz(box_b).unsqueeze(0) - inter
        return inter / union

    def get_y(self, bbox, clas):

        bbox = bbox.view(-1,4)/self.image_size[0]
        bb_keep = ((bbox[:,2]-bbox[:,0])>0).nonzero()[:,0]
        return bbox[bb_keep],clas[bb_keep]

    def actn_to_bb(self, actn):
        actn_bbs = torch.tanh(actn)
        # print(self.grid_sizes.size())
        # print(self.anchors[:,:2].size())
        actn_centers = (actn_bbs[:,:2]/2 * self.grid_sizes) + self.anchors[:,:2]
        actn_hw = (actn_bbs[:,2:]/2+1) * self.anchors[:,2:]
        return hw2corners(actn_centers, actn_hw)

    def map_to_ground_truth(self, overlaps, print_it=False):

        prior_overlap, prior_idx = overlaps.max(1)
    #     if print_it: print(prior_overlap)
        gt_overlap, gt_idx = overlaps.max(0)
        gt_overlap[prior_idx] = 1.99
        for i,o in enumerate(prior_idx): gt_idx[o] = i
        return gt_overlap,gt_idx

    def darknet_1_loss(self, b_c,b_bb,bbox,clas,print_it=False):

        anchor_cnr = hw2corners(self.anchors[:,:2], self.anchors[:,2:])
        bbox,clas = self.get_y(bbox,clas)
        a_ic = self.actn_to_bb(b_bb)
        overlaps = self.jaccard(bbox.data, anchor_cnr.data)
        gt_overlap,gt_idx = self.map_to_ground_truth(overlaps,print_it)
        gt_clas = clas[gt_idx]
        pos = gt_overlap > 0.4
        pos_idx = torch.nonzero(pos)[:,0]
        gt_clas[1-pos] = len(self.class_names)
        gt_bbox = bbox[gt_idx]
        loc_loss = ((a_ic[pos_idx] - gt_bbox[pos_idx]).abs()).mean()
        clas_loss  = self.loss_f(b_c, gt_clas)
        return loc_loss, clas_loss

    def darknet_loss(self, pred, targ, print_it=False):

        lcs,lls = 0.,0.
        for b_c,b_bb,bbox,clas in zip(*pred,*targ):
            # print(b_bb.size())
            loc_loss,clas_loss = self.darknet_1_loss(b_c,b_bb,bbox,clas)
            lls += loc_loss
            lcs += clas_loss
        if print_it: print(f'loc: {lls.data[0]}, clas: {lcs.data[0]}')
        return lls+lcs

    def set_loss(self,loss):
        self.loss_f = loss    

    def show_objects(self, ax, ima, bbox, clas, prs=None, thresh=0.4):

        return self.show_objects_(ax, ima, ((bbox*self.image_size[0]).long()).numpy(),
            (clas).numpy(), (prs).numpy() if prs is not None else None, thresh)

    # def dai_plot_results(self,thresh,loader,model):
    
    #     dai_x,dai_y = next(iter(loader))
    #     dai_x = dai_x.to(self.device)
    #     dai_y = [torch.tensor(l).to(self.device) for l in dai_y]
    #     dai_batch = model(dai_x)
    #     dai_b_clas,dai_b_bb = dai_batch
    #     dai_x = dai_x.cpu()
    #     dai_y = [torch.tensor(l).cpu() for l in dai_y]


    #     fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    #     for idx,ax in enumerate(axes.flat):
    #         ima = dai.denorm_img(dai_x[idx])
    #         bbox,clas = self.get_y(dai_y[0][idx], dai_y[1][idx])
    #         a_ic = self.actn_to_bb(dai_b_bb[idx])
    #         clas_pr, clas_ids = dai_b_clas[idx].max(1)
    #         clas_pr = clas_pr.sigmoid()
    #         self.show_objects(ax, ima, a_ic, clas_ids, clas_pr, clas_pr.max().data[0]*thresh)
    #     plt.tight_layout()  

    def batch_loss(self,model,loader,crit):
        
        data_batch = next(iter(loader))
        dai_x,dai_y = data_batch[0],data_batch[1]
        dai_x = dai_x.to(self.device)
        dai_y = [torch.tensor(l).to(self.device) if type(l).__name__ == 'Tensor' else l.to(device) for l in dai_y]
        dai_batch = model(dai_x)
        return crit(dai_batch,dai_y)

    def show_nms(self,loader = None,num = 10,img = None,score_thresh = 0.25,nms_overlap = 0.1,dp = None):

        if loader:    
            x = next(iter(loader))[0]
            batch = self.predict(x)
            pred_clas,pred_bbox = batch
            x = x.cpu()

            for i in range(num):
                print(i)
                ima = dp.denorm_img(x[i])
                box_coords = self.actn_to_bb(pred_bbox[i])
                conf_scores = pred_clas[i].sigmoid().t().data
                self.show_nms_(ima,box_coords,conf_scores,score_thresh)
        else:
            x = img
            batch = self.predict(x)
            pred_clas,pred_bbox = batch
            x = x.cpu()
            ima = x[0].numpy().transpose(1,2,0)
            box_coords = self.actn_to_bb(pred_bbox[0])
            conf_scores = pred_clas[0].sigmoid().t().data
            # return(box_coords,conf_scores,pred_bbox,pred_clas)
            self.show_nms_(ima,box_coords,conf_scores,score_thresh,nms_overlap)

    def show_nms_(self,ima,box_coords,conf_scores,score_thresh = 0.25,nms_overlap = 0.1):

        out1,out2,cc = [],[],[]
        for cl in range(0, len(conf_scores)-1):
            c_mask = conf_scores[cl] > score_thresh
            if c_mask.sum() == 0: continue
            scores = conf_scores[cl][c_mask]
            l_mask = c_mask.unsqueeze(1).expand_as(box_coords)
            boxes = box_coords[l_mask].view(-1, 4)
            ids, count = nms(boxes.data, scores, nms_overlap, 50)
            ids = ids[:count]
            out1.append(scores[ids])
            out2.append(boxes.data[ids])
            cc.append([cl]*count)
        # return(out1,out2)    
        if len(cc)> 0:    
            cc = torch.from_numpy(np.concatenate(cc))
            out1 = torch.cat(out1).cpu()
            out2 = torch.cat(out2).cpu()
            fig, ax = plt.subplots(figsize=(8,8))
            ax = self.show_objects(ax, ima, out2, cc, out1, score_thresh)
            plt.show()

                
