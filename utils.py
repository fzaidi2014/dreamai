from collections import defaultdict
import math
import torch
from torch.utils.data.sampler import SubsetRandomSampler,SequentialSampler,BatchSampler
import numpy as np
from torch import nn
import cv2

class Printer(nn.Module):
    def forward(self,x):
        print(x.size())
        return x

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

DAI_AvgPool = nn.AdaptiveAvgPool2d(1)

def update_classwise_accuracies(preds,labels,class_correct,class_totals):
    correct = np.squeeze(preds.eq(labels.data.view_as(preds)))
    for i in range(labels.shape[0]):
        label = labels.data[i].item()
        class_correct[label] += correct[i].item()
        class_totals[label] += 1

def get_accuracies(class_names,class_correct,class_totals):
    accuracy = (100*np.sum(list(class_correct.values()))/np.sum(list(class_totals.values())))
    class_accuracies = [(class_names[i],100.0*(class_correct[i]/class_totals[i])) 
                        for i in class_names.keys() if class_totals[i] > 0]
    return accuracy,class_accuracies

def flatten_tensor(x):
    return x.view(x.shape[0],-1)

def split_image_data(train_data,test_data=None,batch_size=20,num_workers=0,
                     valid_size=0.2,sampler=SubsetRandomSampler):
    
    num_train = len(train_data)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    split = int(np.floor(valid_size * num_train))
    train_idx, valid_idx = indices[split:], indices[:split]

    train_sampler = sampler(train_idx)
    valid_sampler = sampler(valid_idx)

    if test_data is not None:
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size,
        num_workers=num_workers)
    else:
        train_idx, test_idx = train_idx[split:],train_idx[:split]
        train_sampler = sampler(train_idx)
        test_sampler = sampler(test_idx)
        
        test_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                                   sampler=test_sampler, num_workers=num_workers)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                               sampler=train_sampler, num_workers=num_workers)
    
    valid_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, 
                                               sampler=valid_sampler, num_workers=num_workers)
    
    return train_loader,valid_loader,test_loader

def split_data(train_data,test_data=None,batch_size=20,num_workers=0,
                     valid_size=0.2,sampler=SubsetRandomSampler):
    
    num_train = len(train_data)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    split = int(np.floor(valid_size * num_train))
    train_idx, valid_idx = indices[split:], indices[:split]

    train_sampler = sampler(train_idx)
    valid_sampler = sampler(valid_idx)

    if test_data is not None:
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size,
        num_workers=num_workers)
    else:
        train_idx, test_idx = train_idx[split:],train_idx[:split]
        train_sampler = sampler(train_idx)
        test_sampler = sampler(test_idx)
        
        test_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                                   sampler=test_sampler, num_workers=num_workers)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                               sampler=train_sampler, num_workers=num_workers)
    
    valid_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, 
                                               sampler=valid_sampler, num_workers=num_workers)
    
    return train_loader,valid_loader,test_loader

def calculate_img_stats(dataset):
    imgs_ = torch.stack([img for img,_ in dataset],dim=3)
    imgs_ = imgs_.view(3,-1)
    imgs_mean = imgs_.mean(dim=1)
    imgs_std = imgs_.std(dim=1)
    return imgs_mean,imgs_std

def create_csv_from_folder(folder_path,outfile,cols=['id','path']):
    
    f = glob.glob(folder_path+'/*.*')
    
    ids = []
    for elem in f:
        t = elem[elem.rfind('/')+1:]
        ids.append(t[:t.rfind('.')])
    data = {cols[0]:ids,cols[1]:f}    
    df = pd.DataFrame(data,columns=cols)
    df.to_csv(outfile,index=False)

def normalize_minmax(values):
    min_ = min(values)
    return (values-min_)/(max(values)- min_)

def denormalize_minmax(values,orig_values):
    min_ = min(orig_values)
    v = (values * (max(orig_values) - min_) + min_)
    return np.array([np.round(e) for e in v])

def rmse(inputs,targets):
    return torch.sqrt(torch.mean((inputs - targets) ** 2))

def compute_center_loss(features, centers, targets):
    features = features.view(features.size(0), -1)
    target_centers = centers[targets]
    criterion = torch.nn.MSELoss()
    center_loss = criterion(features, target_centers)
    return center_loss


def get_center_delta(features, centers, targets, alpha, device):
    # implementation equation (4) in the center-loss paper
    features = features.view(features.size(0), -1)
    targets, indices = torch.sort(targets)
    target_centers = centers[targets]
    features = features[indices]

    delta_centers = target_centers - features
    uni_targets, indices = torch.unique(
            targets.cpu(), sorted=True, return_inverse=True)

    uni_targets = uni_targets.to(device)
    indices = indices.to(device)

    delta_centers = torch.zeros(
        uni_targets.size(0), delta_centers.size(1)
    ).to(device).index_add_(0, indices, delta_centers)

    targets_repeat_num = uni_targets.size()[0]
    uni_targets_repeat_num = targets.size()[0]
    targets_repeat = targets.repeat(
            targets_repeat_num).view(targets_repeat_num, -1)
    uni_targets_repeat = uni_targets.unsqueeze(1).repeat(
            1, uni_targets_repeat_num)
    same_class_feature_count = torch.sum(
            targets_repeat == uni_targets_repeat, dim=1).float().unsqueeze(1)

    delta_centers = delta_centers / (same_class_feature_count + 1.0) * alpha
    result = torch.zeros_like(centers)
    result[uni_targets, :] = delta_centers
    return result

def get_test_input(path = "/home/farhan/Downloads/dog-cycle-car.png",size = (224,224)):
    img = cv2.imread(path)
    img = cv2.resize(img, size)          #Resize to the input dimension
    img_ =  img[:,:,::-1].transpose((2,0,1))  # BGR -> RGB | H X W C -> C X H X W
    img_ = img_[np.newaxis,:,:,:]/255.0       #Add a channel at 0 (for batch) | Normalise
    img_ = torch.from_numpy(img_).float()     #Convert to float
    # img_ = Variable(img_)                     # Convert to Variable
    return img_
