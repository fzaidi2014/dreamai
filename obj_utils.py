from dai_imports import*

def resnet_obj_clip(children, idx = -3):
    return children[:idx]

def densenet_obj_clip(children,idx = -2):
    return children[0][:idx]

class StdConv(nn.Module):
    def __init__(self, nin, nout, stride=2, drop=0.1):
        super().__init__()
        self.conv = nn.Conv2d(nin, nout, 3, stride=stride, padding=1)
        self.bn = nn.BatchNorm2d(nout)
        self.drop = nn.Dropout(drop)
        
    def forward(self, x): return self.drop(self.bn(F.relu(self.conv(x))))
        
def flatten_conv(x,k):
    # print(x.size())
    bs,nf,gx,gy = x.size()
    x = x.permute(0,2,3,1).contiguous()
    return x.view(bs,-1,nf//k)

class OutConv(nn.Module):
    def __init__(self, k, nin, num_classes, bias=-4.):
        super().__init__()
        self.k = k
        self.oconv1 = nn.Conv2d(nin, (num_classes+1)*k, 3, padding=1)
        self.oconv2 = nn.Conv2d(nin, 4*k, 3, padding=1)
        self.oconv1.bias.data.zero_().add_(bias)
        
    def forward(self, x):        
        return [flatten_conv(self.oconv1(x), self.k),
                flatten_conv(self.oconv2(x), self.k)]

def get_grids(image_size,initial_pow = 4,ceil = True):
    if ceil:
        g = int(np.ceil(image_size[0]/(2**initial_pow)))
    else:
        g = image_size[0]//(2**initial_pow)
    grids = []
    while g != 1:
        g = int(np.ceil(g/2))
        grids.append(g)
    return grids    

class CustomSSD_MultiHead(nn.Module):
    def __init__(self, num_grids, k, num_classes, drop, bias):
        super().__init__()
        self.k = k
        self.num_classes = num_classes
        self.bias = bias
        self.drop = nn.Dropout(drop)
        self.sconv0 = StdConv(512,256, stride=1, drop=drop)
        self.outconv0 = OutConv(k, 256, num_classes, bias)
        self.std_convs = nn.ModuleList()
        self.out_convs = nn.ModuleList()
        for _ in range(num_grids-1):
            self.std_convs.append(StdConv(256,256, drop=drop))
            self.out_convs.append(OutConv(k, 256, num_classes, bias))
    def forward(self, x):        
        class_list = []
        bbox_list = []
        # print(x.size())
        x = self.drop(F.relu(x))
        x = self.sconv0(x)
        class_out,bbox_out = self.outconv0(x)
        class_list.append(class_out)
        bbox_list.append(bbox_out)
        for std,out in zip(self.std_convs,self.out_convs):
            x = std(x)
            # print(x.size())
            class_out,bbox_out = out(x)
            class_list.append(class_out)
            bbox_list.append(bbox_out)
        return [torch.cat(class_list, dim=1),
                torch.cat(bbox_list, dim=1)]
    
class SSD_MultiHead(nn.Module):
    def __init__(self, k, num_classes, drop, bias):
        super().__init__()
        self.drop = nn.Dropout(drop)
        self.sconv0 = StdConv(512,256, stride=1, drop=drop)
        self.sconv1 = StdConv(256,256, drop=drop)
        self.sconv2 = StdConv(256,256, drop=drop)
        self.sconv3 = StdConv(256,256, drop=drop)
        self.out0 = OutConv(k, 256, num_classes, bias)
        self.out1 = OutConv(k, 256, num_classes, bias)
        self.out2 = OutConv(k, 256, num_classes, bias)
        self.out3 = OutConv(k, 256, num_classes, bias)

    def forward(self, x):        
        x = self.drop(F.relu(x))
        x = self.sconv0(x)
        x = self.sconv1(x)
        o1c,o1l = self.out1(x)
        # print(o1c.size())
        x = self.sconv2(x)
        o2c,o2l = self.out2(x)
        # print(o2c.size())
        x = self.sconv3(x)
        o3c,o3l = self.out3(x)
        # print(o3c.size())
        return [torch.cat([o1c,o2c,o3c], dim=1),
                torch.cat([o1l,o2l,o3l], dim=1)]

class UNet_MultiHead(nn.Module):
    def __init__(self, k, num_classes, drop, bias):
        super().__init__()
        self.drop = nn.Dropout(drop)
        self.sconv0 = StdConv(512,256, stride=1, drop=drop)
        self.sconv1 = StdConv(256,256, drop=drop)
        self.sconv2 = StdConv(256,256, drop=drop)
        self.sconv3 = StdConv(256,256, drop=drop)
        self.sconv4 = StdConv(256,256, drop=drop)
        self.sconv5 = StdConv(256,256, drop=drop)
        self.sconv6 = StdConv(256,256, drop=drop)
        self.sconv7 = StdConv(256,256, drop=drop)
        self.sconv8 = StdConv(256,256, drop=drop)
        self.out0 = OutConv(k, 256, num_classes, bias)
        self.out1 = OutConv(k, 256, num_classes, bias)
        self.out2 = OutConv(k, 256, num_classes, bias)
        self.out3 = OutConv(k, 256, num_classes, bias)
        self.out4 = OutConv(k, 256, num_classes, bias)
        self.out5 = OutConv(k, 256, num_classes, bias)
        self.out6 = OutConv(k, 256, num_classes, bias)

    def forward(self, x):        
        x = self.drop(F.relu(x))
        x = self.sconv0(x)
        x = self.sconv1(x)
        x = self.sconv2(x)
        o1c,o1l = self.out0(x)
        # print(o1c.size())
        x = self.sconv3(x)
        o2c,o2l = self.out1(x)
        # print(o2c.size())
        x = self.sconv4(x)
        o3c,o3l = self.out2(x)
        # print(o3c.size())
        x = self.sconv5(x)
        o4c,o4l = self.out3(x)
        # print(o4c.size())
        x = self.sconv6(x)
        o5c,o5l = self.out4(x)
        # print(o5c.size())
        x = self.sconv7(x)
        o6c,o6l = self.out5(x)
        # print(o6c.size())
        x = self.sconv8(x)
        o7c,o7l = self.out6(x)
        # print(o7c.size())
        return [torch.cat([o1c,o2c,o3c,o4c,o5c,o6c,o7c], dim=1),
                torch.cat([o1l,o2l,o3l,o4l,o5l,o6l,o7l], dim=1)]

class UNet_MultiHead2(nn.Module):
    def __init__(self, k, num_classes, drop, bias):
        super().__init__()
        self.drop = nn.Dropout(drop)
        self.sconv0 = StdConv(512,256, stride=1, drop=drop)
        self.sconv1 = StdConv(256,256, drop=drop)
        self.sconv2 = StdConv(256,256, drop=drop)
        self.sconv3 = StdConv(256,256, drop=drop)
        self.sconv4 = StdConv(256,256, drop=drop)
        self.sconv5 = StdConv(256,256, drop=drop)
        self.sconv6 = StdConv(256,256, drop=drop)
        self.out0 = OutConv(k, 256, num_classes, bias)
        self.out1 = OutConv(k, 256, num_classes, bias)
        self.out2 = OutConv(k, 256, num_classes, bias)
        self.out3 = OutConv(k, 256, num_classes, bias)
        self.out4 = OutConv(k, 256, num_classes, bias)
        self.out5 = OutConv(k, 256, num_classes, bias)

    def forward(self, x):        
        x = self.drop(F.relu(x))
        x = self.sconv0(x)
        x = self.sconv1(x)
        o1c,o1l = self.out0(x)
        x = self.sconv2(x)
        o2c,o2l = self.out1(x)
        x = self.sconv3(x)
        o3c,o3l = self.out2(x)
        x = self.sconv4(x)
        o4c,o4l = self.out3(x)
        x = self.sconv5(x)
        o5c,o5l = self.out4(x)
        x = self.sconv6(x)
        o6c,o6l = self.out5(x)
        return [torch.cat([o1c,o2c,o3c,o4c,o5c,o6c], dim=1),
                torch.cat([o1l,o2l,o3l,o4l,o5l,o6l], dim=1)]

def one_hot_embedding(labels, num_classes):
    return torch.eye(num_classes)[labels.data.cpu()]

class BCE_Loss(nn.Module):
    def __init__(self, num_classes, device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
        super().__init__()
        self.num_classes = num_classes
        self.device = device

    def forward(self, pred, targ):
        t = one_hot_embedding(targ, self.num_classes+1)
        # t = torch.tensor(t[:,:-1].contiguous()).to(self.device)
        t = t[:,:-1].contiguous().to(self.device)
        x = pred[:,:-1]
        # print('lala')
        w = self.get_weight(x.clone().detach(),t)
        # print(w)
        return F.binary_cross_entropy_with_logits(x, t, w, size_average=False)/self.num_classes
    
    def get_weight(self,x,t): return None

class FocalLoss(BCE_Loss):
    def get_weight(self,x,t):
        alpha,gamma = 0.25,1
        p = x.sigmoid()
        pt = p*t + (1-p)*(1-t)
        w = alpha*t + (1-alpha)*(1-t)
        return w * (1-pt)**(gamma)
        # print('pow')
        # return w * torch.pow(1-pt,gamma)

def get_cmap(N):
    color_norm  = mcolors.Normalize(vmin=0, vmax=N-1)
    return cmx.ScalarMappable(norm=color_norm, cmap='Set3').to_rgba

def draw_im(im,bbox,label = None):
    ax = img_grid(im, figsize=(16,16))
    draw_rect(ax,bbox)
    if label:
        draw_text(ax, 'bbox'[:2], label, sz=16)

def img_grid(im, figsize=None, ax=None):

    if not ax: fig,ax = plt.subplots(figsize=figsize)
    ax.imshow(im)
    ax.set_xticks(np.linspace(0, im.shape[0], 8))
    ax.set_yticks(np.linspace(0, im.shape[1], 8))
    ax.grid()
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    return ax

def draw_outline(o, lw):

    o.set_path_effects([patheffects.Stroke(
        linewidth=lw, foreground='black'), patheffects.Normal()])

def draw_rect(ax, b, color='white'):

    patch = ax.add_patch(patches.Rectangle(b[:2], *b[-2:], fill=False, edgecolor=color, lw=2))
    draw_outline(patch, 4)

def draw_text(ax, xy, txt, sz=14, color='white'):

    text = ax.text(*xy, txt,verticalalignment='top', color=color, fontsize=sz, weight='bold')
    draw_outline(text, 1)

def bb_hw(a): return np.array([a[1],a[0],a[3]-a[1]+1,a[2]-a[0]+1])
def hw_bb(bb): return [ bb[1], bb[0], bb[3] + bb[1] - 1, bb[2] + bb[0] - 1 ]

def hw2corners(ctr, hw): return torch.cat([ctr-hw/2, ctr+hw/2], dim=1)

def nms(boxes, scores, overlap=0.5, top_k=100):

    keep = scores.new(scores.size(0)).zero_().long()
    if boxes.numel() == 0: return keep
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    area = torch.mul(x2 - x1, y2 - y1)
    v, idx = scores.sort(0)  # sort in ascending order
    idx = idx[-top_k:]  # indices of the top-k largest vals
    xx1 = boxes.new()
    yy1 = boxes.new()
    xx2 = boxes.new()
    yy2 = boxes.new()
    w = boxes.new()
    h = boxes.new()

    count = 0
    while idx.numel() > 0:
        i = idx[-1]  # index of current largest val
        keep[count] = i
        count += 1
        if idx.size(0) == 1: break
        idx = idx[:-1]  # remove kept element from view
        # load bboxes of next highest vals
        torch.index_select(x1, 0, idx, out=xx1)
        torch.index_select(y1, 0, idx, out=yy1)
        torch.index_select(x2, 0, idx, out=xx2)
        torch.index_select(y2, 0, idx, out=yy2)
        # store element-wise max with next highest score
        xx1 = torch.clamp(xx1, min=x1[i])
        yy1 = torch.clamp(yy1, min=y1[i])
        xx2 = torch.clamp(xx2, max=x2[i])
        yy2 = torch.clamp(yy2, max=y2[i])
        w.resize_as_(xx2)
        h.resize_as_(yy2)
        w = xx2 - xx1
        h = yy2 - yy1
        # check sizes of xx1 and xx2.. after each iteration
        w = torch.clamp(w, min=0.0)
        h = torch.clamp(h, min=0.0)
        inter = w*h
        # IoU = i / (area(a) + area(b) - i)
        rem_areas = torch.index_select(area, 0, idx)  # load remaining areas)
        union = (rem_areas - inter) + area[i]
        IoU = inter/union  # store result in iou
        # keep only elements with an IoU <= overlap
        idx = idx[IoU.le(overlap)]
    return keep, count
    