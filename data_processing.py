from dai_imports import*
from obj_utils import*
import utils

class my_image_csv_dataset(Dataset):
    
    def __init__(self, data_dir, data, transforms_ = None, obj = False,
                    minorities = None, diffs = None, bal_tfms = None):
        
        self.data_dir = data_dir
        self.data = data
        self.transforms_ = transforms_
        self.tfms = None
        self.obj = obj
        self.minorities = minorities
        self.diffs = diffs
        self.bal_tfms = bal_tfms
        assert transforms_ is not None, print('Please pass some transforms.')
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        img_path = os.path.join(self.data_dir,self.data.iloc[index, 0])
        img = Image.open(img_path)
        img = img.convert('RGB')
        y = self.data.iloc[index, 1]    
        if self.minorities and self.bal_tfms:
            if y in self.minorities:
                if hasattr(self.bal_tfms,'transforms'):
                    for tr in self.bal_tfms.transforms:
                        tr.p = self.diffs[y]
                    l = [self.bal_tfms]
                    l.extend(self.transforms_)
                    self.tfms = transforms.Compose(l)    
                else:            
                    for t in self.bal_tfms:
                        t.p = self.diffs[y]
                    self.transforms_[1:1] = self.bal_tfms    
                    self.tfms = transforms.Compose(self.transforms_)
                    # print(self.tfms)
            else:
                self.tfms = transforms.Compose(self.transforms_)
        else:    
            self.tfms = transforms.Compose(self.transforms_)    
        x = self.tfms(img)
        if self.obj:
            s = x.size()[1]
            if isinstance(s,tuple):
                s = s[0]
            row_scale = s/img.size[0]
            col_scale = s/img.size[1]
            y = rescale_bbox(y,row_scale,col_scale)
            y.squeeze_()
            y2 = self.data.iloc[index, 2]
            y = (y,y2)
        return (x,y)


class my_image_folder(DatasetFolder):
    
    def __init__(self, root, transform=None, target_transform=None,
                 loader=default_loader, minorities=None, diffs = None, bal_tfms=None, tta_tfms = None):
        
        super(my_image_folder, self).__init__(root, loader, IMG_EXTENSIONS,
                                          transform=transform,
                                          target_transform=target_transform)
        self.imgs = self.samples
        self.minorities = minorities
        self.diffs = diffs
        self.bal_tfms = bal_tfms
        self.tta_tfms = tta_tfms
        self.tfms = None

    def __getitem__(self,index):
        
        path, target = self.samples[index]        
        sample = self.loader(path)
        if self.transform:
            if self.minorities and self.bal_tfms:
                if target in self.minorities:
                    if hasattr(self.bal_tfms,'transforms'):
                        for tr in self.bal_tfms.transforms:
                            tr.p = self.diffs[target]
                        l = [self.bal_tfms]
                        l.extend(self.transform)
                        self.tfms = transforms.Compose(l)    
                    else:            
                        for t in self.bal_tfms:
                            t.p = self.diffs[target]
                        self.tfms = transforms.Compose(self.bal_tfms + self.transform )
                else:
                    self.tfms = transforms.Compose(self.transform)
            elif self.tta_tfms:
                self.tfms = self.tta_tfms
            else:    
                self.tfms = transforms.Compose(self.transform)
            sample = self.tfms(sample)
        if self.target_transform:
            target = self.target_transform(target)
        return sample, target

def extract_data(dt):

    x = []
    y = []
    for a,b in dt:
        x.append(a)
        y.append(b)
    return x,y

def listdir_fullpath(d):
        return [os.path.join(d, f) for f in os.listdir(d)]     

def get_minorities(df,thresh=0.8):

    c = df.iloc[:,1].value_counts()
    lc = list(c)
    max_count = lc[0]
    diffs = [1-(x/max_count) for x in lc]
    diffs = dict((k,v) for k,v in zip(c.keys(),diffs))
    minorities = [c.keys()[x] for x,y in enumerate(lc) if y < (thresh*max_count)]
    return minorities,diffs

def csv_from_path(path, img_dest):
    
    labels_paths = listdir_fullpath(path)
    tr_images = []
    tr_labels = []
    for l in labels_paths:
        if os.path.isdir(l):
            for i in listdir_fullpath(l):
                name = i.split('/')[-1]
                label = l.split('/')[-1]
                new_name = '/{}_'.format(label)+name
                new_name = '/'.join(i.split('/')[:-2])+new_name
                os.rename(i,new_name)
                tr_images.append(new_name)
                tr_labels.append(label)
            os.rmdir(l)    
    tr_img_label = {'Img':tr_images, 'Label': tr_labels}
    csv = pd.DataFrame(tr_img_label,columns=['Img','Label'])
    csv = csv.sample(frac=1).reset_index(drop=True)
    return csv    

def add_extension(a,e):
    a = [x+e for x in a]
    return a

def one_hot(targets, multi = False):
    if multi:
        binerizer = MultiLabelBinarizer()
        dai_1hot = binerizer.fit_transform(targets)
    else:
        binerizer = LabelBinarizer()
        dai_1hot = binerizer.fit_transform(targets)
    return dai_1hot,binerizer.classes_

def get_index(arr,a):
    for i in range(len(arr)):
        if sum(arr[i] == a) == len(a):
            return i
    return False

def rescale_bbox(bb,row_scale,col_scale):
    bb = bb.reshape((-1,4))
    for b in bb:
        r1,c1,r2,c2 = b
        b[0] = int(np.round(r1*col_scale))
        b[1] = int(np.round(c1*row_scale))
        b[2] = int(np.round(r2*col_scale))
        b[3] = int(np.round(c2*row_scale))

    # bb = torch.tensor([bb_hw(b) for b in bb.reshape(-1,4)])
    # for b in bb:
    #     r1,c1,r2,c2 = b
    #     b[0] = int(np.round(r1*row_scale))
    #     b[1] = int(np.round(c1*col_scale))
    #     b[2] = int(np.round(r2*row_scale))
    #     b[3] = int(np.round(c2*col_scale))
    #     if(sum(b)) == 1:
    #         b[0],b[1],b[2],b[3] = 0,0,0,0

    bb = bb.reshape((1,-1))        
    return bb

def get_img_stats(dataset,sz):
    size = len(dataset)//sz
    i = 0
    imgs = []
    for img,_ in dataset:
        # print(img.size())
        if i > size:
            break
        imgs.append(img)
        i+=1
    imgs_ = torch.stack(imgs,dim=3)
    imgs_ = imgs_.view(3,-1)
    imgs_mean = imgs_.mean(dim=1)
    imgs_std = imgs_.std(dim=1)
    return imgs_mean,imgs_std

def split_df(train_df,test_size = 0.15):
    try:    
        train_df,val_df = train_test_split(train_df,test_size = test_size,random_state = 2,stratify = train_df.iloc[:,1])
    except:
        train_df,val_df = train_test_split(train_df,test_size = test_size,random_state = 2)
    train_df = train_df.reset_index(drop = True)
    val_df =  val_df.reset_index(drop = True)
    return train_df,val_df    

def save_obj(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

class DataProcessor:
    
    def __init__(self, data_path = None, train_csv = None, val_csv = None, reg = False,
                    tr_name = 'train', val_name = 'val', test_name = 'test', extension = None, setup_data = True):
        
        print('+------------------------------------+')
        print('|              Dream AI              |')
        print('+------------------------------------+')
        print()
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        self.data_path,self.train_csv,self.val_csv,self.reg,self.tr_name,self.val_name,self.test_name,self.extension = (data_path,train_csv,
                                                                                            val_csv,reg,tr_name,val_name,test_name,extension)
        
        self.obj = False
        self.multi_label = False
        
        if setup_data:
            self.set_up_data()
                
    def set_up_data(self):

        data_path,train_csv,val_csv,tr_name,val_name,test_name = (self.data_path,self.train_csv,self.val_csv,self.tr_name,self.val_name,self.test_name)

        # check if paths given and also set paths
        
        if not data_path:
            data_path = os.getcwd() + '/'
        tr_path = os.path.join(data_path,tr_name)
        val_path = os.path.join(data_path,val_name)
        test_path = os.path.join(data_path,test_name)

        if os.path.exists(os.path.join(data_path,tr_name+'.csv')):
            train_csv = tr_name+'.csv'
        if os.path.exists(os.path.join(data_path,val_name+'.csv')):
            val_csv = val_name+'.csv'
        if os.path.exists(os.path.join(data_path,test_name+'.csv')):
            test_csv = test_name+'.csv'    

        # paths to csv

        if not train_csv:
            print('no')
            train_csv,val_csv = self.data_from_paths_to_csv(data_path,tr_path,val_path)

        train_csv_path = os.path.join(data_path,train_csv)
        train_df = pd.read_csv(train_csv_path)
        if 'Unnamed: 0' in train_df.columns:
            train_df = train_df.drop('Unnamed: 0', 1)
        if len(train_df.columns) > 2:
            self.obj = True    
        img_names = [str(x) for x in list(train_df.iloc[:,0])]
        if self.extension:
            img_names = add_extension(img_names,self.extension)
        if val_csv:
            val_csv_path = os.path.join(data_path,val_csv)
            val_df = pd.read_csv(val_csv_path)
            val_targets =  list(map(str,list(val_df.iloc[:,1])))    
        targets = list(map(str,list(train_df.iloc[:,1])))
        lengths = [len(t) for t in [s.split() for s in targets]]
        self.target_lengths = lengths
        split_targets = [t.split() for t in targets]
        if self.obj:
            print('\nObject Detection\n')

            # bounding boxes

            int_targets = [list(map(float,x)) for x in split_targets]
            zero_targets = np.zeros((len(targets),max(lengths)),dtype=int)
            for i,t in enumerate(zero_targets):
                t[len(t)-len(int_targets[i]):] = int_targets[i]
                zero_targets[i] = t
            train_df.iloc[:,1] = [torch.from_numpy(z).type(torch.FloatTensor) for z in zero_targets]

            # one-hot classes

            obj_targets = list(map(str,list(train_df.iloc[:,2])))
            obj_split_targets = [t.split() for t in obj_targets]
            try:
                obj_split_targets = [list(map(int,x)) for x in obj_split_targets]
            except:
                pass
            dai_onehot,onehot_classes = one_hot(obj_split_targets,True)
            train_df['one_hot'] = [torch.from_numpy(x).type(torch.FloatTensor) for x in dai_onehot]

            # class indexes

            c_names = list(onehot_classes)
            class_idx = [[c_names.index(i) for i in c] for c in obj_split_targets]
            zero_idx = np.zeros((len(targets),max(lengths)//4),dtype=int)
            # print(zero_idx.shape)
            for i,t in enumerate(zero_idx):
                # temp_l = len(class_idx[i])
                # if temp_l > 90:
                #     print(i,temp_l)
                t[len(t)-len(class_idx[i]):] = class_idx[i]
                zero_idx[i] = t
            train_df.iloc[:,2] = [torch.from_numpy(z).type(torch.LongTensor) for z in zero_idx]
            self.data_dir,self.num_classes,self.class_names = data_path,len(onehot_classes),onehot_classes
            # self.set_up_object_detection([4,2,1],[0.7, 1., 1.3],[(1.,1.), (1.,0.5), (0.5,1.)])

        elif self.reg:
            print('\nRegression\n')
            int_targets = [list(map(int,x)) for x in split_targets]
            zero_targets = np.zeros((len(targets),max(lengths)),dtype=int)
            for i,t in enumerate(zero_targets):
                t[len(t)-len(int_targets[i]):] = int_targets[i]
                zero_targets[i] = t
            train_df.iloc[:,1] = [torch.from_numpy(z).type(torch.FloatTensor) for z in zero_targets]
            self.data_dir,self.num_classes,self.class_names = data_path, max(lengths),np.unique(zero_targets,axis=1)
        elif lengths[1:] != lengths[:-1]:
            self.multi_label = True
            print('\nMulti-label Classification\n')
            try:
                split_targets = [list(map(int,x)) for x in split_targets]
            except:
                pass
            dai_onehot,onehot_classes = one_hot(split_targets,self.multi_label)
            train_df.iloc[:,1] = [torch.from_numpy(x).type(torch.FloatTensor) for x in dai_onehot]
            self.data_dir,self.num_classes,self.class_names = data_path,len(onehot_classes),onehot_classes
        else:
            print('\nSingle-label Classification\n')
            unique_targets = list(np.unique(targets))
            target_ids = [unique_targets.index(x) for x in targets]
            train_df.iloc[:,1] = target_ids
            if val_csv:
                target_ids = [unique_targets.index(x) for x in val_targets]
                val_df.iloc[:,1] = target_ids
            self.data_dir,self.num_classes,self.class_names = data_path,len(unique_targets),unique_targets

        # self.models_path = os.path.join(self.data_dir, 'models')
        # os.makedirs(self.models_path,exist_ok=True)

        if not val_csv:
            train_df,val_df = split_df(train_df)
        val_df,test_df = split_df(val_df)
        tr_images = [str(x) for x in list(train_df.iloc[:,0])]
        val_images = [str(x) for x in list(val_df.iloc[:,0])]
        test_images = [str(x) for x in list(test_df.iloc[:,0])]
        if self.extension:
            tr_images = add_extension(tr_images,self.extension)
            val_images = add_extension(val_images,self.extension)
            test_images = add_extension(test_images,self.extension)
        train_df.iloc[:,0] = tr_images
        val_df.iloc[:,0] = val_images
        test_df.iloc[:,0] = test_images
        train_df.to_csv(os.path.join(data_path,'train.csv'),index=False)
        val_df.to_csv(os.path.join(data_path,'val.csv'),index=False)
        test_df.to_csv(os.path.join(data_path,'test.csv'),index=False)
        self.minorities,self.class_diffs = None,None
        if (not self.obj) or (not self.multi_label):
            self.minorities,self.class_diffs = get_minorities(train_df)
        self.data_dfs = {self.tr_name:train_df, self.val_name:val_df, self.test_name:test_df}
        data_dict = {'data_dfs':self.data_dfs,'data_dir':self.data_dir,'num_classes':self.num_classes,'class_names':self.class_names,
                'minorities':self.minorities,'class_diffs':self.class_diffs,'obj':self.obj,'multi_label':self.multi_label}
        save_obj(data_dict,os.path.join(self.data_dir,'data_dict.pkl'))
        self.data_dict = data_dict
        return data_dict

    def data_from_paths_to_csv(self,data_path,tr_path,val_path):
            
        train_df = csv_from_path(tr_path,tr_path)
        train_df.to_csv(os.path.join(data_path,self.tr_name+'.csv'),index=False)
        ret = (self.tr_name+'.csv',None)
        val_exists = os.path.exists(val_path)
        if val_exists:
            val_df = csv_from_path(val_path,tr_path)
            val_df.to_csv(os.path.join(data_path,self.val_name+'.csv'),index=False)
            ret = (self.tr_name+'.csv',self.val_name+'.csv')
        return ret
        
    def get_data(self, data_dict = None, s = (224,224), dataset = my_image_csv_dataset, bs = 32, balance = False, tfms = None,
                                 bal_tfms = None, tta = False, num_workers = 4):
        
        self.image_size = s
        if not data_dict:
            data_dict = self.data_dict
        data_dfs,data_dir,minorities,class_diffs,obj,multi_label = (data_dict['data_dfs'],data_dict['data_dir'],data_dict['minorities'],
                                                        data_dict['class_diffs'],data_dict['obj'],data_dict['multi_label'])
        if obj or multi_label:
           balance = False                                                 
        if tta:
            tta_tfms = {self.tr_name: transforms.Compose( 
                [
#                 transforms.TenCrop(s),
                transforms.FiveCrop(s[0]),    
                transforms.Lambda(lambda crops:torch.stack([transforms.ToTensor()(crop) for crop in crops])),
                transforms.Lambda(lambda crops:torch.stack(
                [transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(crop) for crop in crops]))
                    
                ]),
                self.val_name:  transforms.Compose(
                [
#                 transforms.TenCrop(s),
                transforms.FiveCrop(s[0]),
                transforms.Lambda(lambda crops:torch.stack([transforms.ToTensor()(crop) for crop in crops])),
                transforms.Lambda(lambda crops:torch.stack(
                [transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(crop) for crop in crops]))
                ]),
                self.test_name:  transforms.Compose(
                [
#                 transforms.TenCrop(s),
                transforms.FiveCrop(s[0]),
                transforms.Lambda(lambda crops:torch.stack([transforms.ToTensor()(crop) for crop in crops])),
                transforms.Lambda(lambda crops:torch.stack(
                [transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(crop) for crop in crops]))
                ])}
#             tta_tfms = {self.tr_name: transforms.Compose([
#                 transforms.Resize(s),
#                 transforms.ToTensor(),
#                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#                 ]),
#                 self.val_name: transforms.Compose([
#                 transforms.Resize(s),    
#                 transforms.ToTensor(),
#                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#                 ]) }
            
        else:
            tta_tfms = None
        
        if not bal_tfms:
            bal_tfms = { self.tr_name: [transforms.RandomHorizontalFlip()],
                           
                         self.val_name: None,
                         self.test_name: None 
                       }
        else:
            bal_tfms = {self.tr_name: bal_tfms, self.val_name: None, self.test_name: None}
        if obj:
            resize_transform = transforms.Resize(s)
        else:
            resize_transform = transforms.RandomResizedCrop(s[0])    
        if not tfms:
            tfms = [
                resize_transform,
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]
        else:
            
            tfms_temp = [
                resize_transform,
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]
            tfms_temp[1:1] = tfms
            tfms = tfms_temp
            print(tfms)
        
        data_transforms = {
            self.tr_name: tfms,
            self.val_name: [
                # transforms.Resize(s[0]+50),
                # transforms.CenterCrop(s[0]),
                transforms.Resize(s),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ],
            self.test_name: [
                # transforms.Resize(s[0]+50),
                # transforms.CenterCrop(s[0]),
                transforms.Resize(s),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]
        }

        temp_tfms = [resize_transform, transforms.ToTensor()]
        temp_dataset = dataset(os.path.join(data_dir,self.tr_name),data_dfs[self.tr_name],temp_tfms)
        self.img_mean,self.img_std = get_img_stats(temp_dataset,60)
        data_transforms[self.tr_name][-1].mean,data_transforms[self.tr_name][-1].std = self.img_mean,self.img_std
        data_transforms[self.val_name][-1].mean,data_transforms[self.val_name][-1].std = self.img_mean,self.img_std
        data_transforms[self.test_name][-1].mean,data_transforms[self.test_name][-1].std = self.img_mean,self.img_std

        if balance:
            image_datasets = {x: dataset(os.path.join(data_dir,self.tr_name),data_dfs[x],
                                        data_transforms[x],obj,minorities,class_diffs,bal_tfms[x])
                        for x in [self.tr_name, self.val_name, self.test_name]}    
        else:
            image_datasets = {x: dataset(os.path.join(data_dir,self.tr_name),data_dfs[x],
                                                            data_transforms[x],obj)
                        for x in [self.tr_name, self.val_name, self.test_name]}
        
        dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=bs,
                                                     shuffle=True, num_workers=num_workers)
                      for x in [self.tr_name, self.val_name, self.test_name]}
        dataset_sizes = {x: len(image_datasets[x]) for x in [self.tr_name, self.val_name, self.test_name]}
        
        self.image_datasets,self.dataloaders,self.dataset_sizes = (image_datasets,dataloaders,
                                                                                    dataset_sizes)
        
        return image_datasets,dataloaders,dataset_sizes

    def imshow(self,inp, title=None):
        
        """Imshow for Tensor."""
        inp = self.denorm_img(inp)
        plt.imshow(inp)
        if title:
            plt.title(title)
        plt.pause(0.001)

    def denorm_img(self,inp):

        inp = inp.numpy().transpose((1, 2, 0))
        mean = self.img_mean.numpy()
        std = self.img_std.numpy()
        inp = std * inp + mean
        inp = np.clip(inp, 0, 1)
        return inp    
        
    def show_data(self,folder_name = 'train', size = (64,64), bs = 5):
        
        self.get_data(size,bs)
        inputs, classes = next(iter(self.dataloaders[folder_name]))
        out = torchvision.utils.make_grid(inputs)
        if self.reg:
            print(classes)
            self.imshow(out, title=[x for x in classes])    
        elif self.multi_label:
            self.imshow(out, title=[self.class_names[np.nonzero(x.type(torch.LongTensor))] for x in classes])    
        else:    
            self.imshow(out, title=[self.class_names[x] for x in classes])

    # def set_up_object_detection(self,anc_grids,anc_zooms,anc_ratios,num_colr = 12):

    #     # print('Would you like to give your own values for anchor_grids, anchor_zooms,and anchor_ratios? The default values are: {}, {} and {}'
    #     # .format(anc_grids,anc_zooms,anc_ratios))
    #     # print('If so, you may call the function "set_up_object_detection" with your own paramteres.')

    #     cmap = get_cmap(num_colr)
    #     self.colr_list = [cmap(float(x)) for x in range(num_colr)]
    #     self.num_colr = num_colr
    #     self.create_anchors(anc_grids,anc_zooms,anc_ratios)
    #     self.custom_head = SSD_MultiHead(self.k,self.num_classes,0.45,-4.)
    #     self.loss_f = FocalLoss(self.num_classes)

    # def create_anchors(self,anc_grids,anc_zooms,anc_ratios):
    
    #     anchor_scales = [(anz*i,anz*j) for anz in anc_zooms for (i,j) in anc_ratios]
    #     k = len(anchor_scales)
    #     anc_offsets = [1/(o*2) for o in anc_grids]
    #     anc_x = np.concatenate([np.repeat(np.linspace(ao, 1-ao, ag), ag)
    #                             for ao,ag in zip(anc_offsets,anc_grids)])
    #     anc_y = np.concatenate([np.tile(np.linspace(ao, 1-ao, ag), ag)
    #                             for ao,ag in zip(anc_offsets,anc_grids)])
    #     anc_ctrs = np.repeat(np.stack([anc_x,anc_y], axis=1), k, axis=0)
    #     anc_sizes  =   np.concatenate([np.array([[o/ag,p/ag] for i in range(ag*ag) for o,p in anchor_scales])
    #                 for ag in anc_grids])
    #     grid_sizes = torch.tensor(np.concatenate([np.array(
    #                             [ 1/ag for i in range(ag*ag) for o,p in anchor_scales])
    #                 for ag in anc_grids])).float().unsqueeze(1).to(self.device)
    #     anchors = torch.tensor(np.concatenate([anc_ctrs, anc_sizes], axis=1)).float().to(self.device)
    #     anchor_cnr = hw2corners(anchors[:,:2], anchors[:,2:])
    #     self.anchors,self.anchor_cnr,self.grid_sizes,self.k = anchors,anchor_cnr,grid_sizes,k        








