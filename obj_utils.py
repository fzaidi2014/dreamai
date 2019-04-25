from dai_imports import*
import utils

def detectCharacterCandidates(region,size = None):
    
    LicensePlate = namedtuple("LicensePlateRegion", ["success", "plate", "thresh", "candidates", 'num_chars'])
    num_chars = 0
    plate = region
#     plt.imshow(imutils.resize(plate, width=size))
#     plt.show()
    V = cv2.split(cv2.cvtColor(plate, cv2.COLOR_BGR2HSV))[2]
    T = threshold_local(V, 29, offset=15, method="gaussian")
    thresh = (V > T).astype("uint8") * 255
    thresh = cv2.bitwise_not(thresh)
    if size:
        plate = imutils.resize(plate, width=size)
        thresh = imutils.resize(thresh, width=size)
    # utils.plt_show(plate)
    # utils.plt_show()
    labels = measure.label(thresh, neighbors=8, background=0)
    charCandidates = np.zeros(thresh.shape, dtype="uint8")
    for label in np.unique(labels):
        if label == 0:
            continue
        labelMask = np.zeros(thresh.shape, dtype="uint8")
        labelMask[labels == label] = 255
        cnts = cv2.findContours(labelMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if imutils.is_cv2() else cnts[1]
        if len(cnts) > 0:
            c = max(cnts, key=cv2.contourArea)
            (boxX, boxY, boxW, boxH) = cv2.boundingRect(c)
            aspectRatio = boxW / float(boxH)
            solidity = cv2.contourArea(c) / float(boxW * boxH)
            heightRatio = boxH / float(plate.shape[0])
            keepAspectRatio = aspectRatio < 1.0
            keepSolidity = solidity > 0.15
            keepHeight = heightRatio > 0.4 and heightRatio < 0.95
            if keepAspectRatio and keepSolidity and keepHeight:
                hull = cv2.convexHull(c)
                cv2.drawContours(charCandidates, [hull], -1, 255, -1)
                num_chars+=1
    charCandidates = segmentation.clear_border(charCandidates)
    return LicensePlate(success=True, plate=plate, thresh=thresh,
        candidates=charCandidates,num_chars = num_chars)

def get_lp_chars(path,size = None, char_width = 3):
#     size = 350
    img = cv2.imread(path)
    char_dict = {'char_list':[],'char_coords':[]}
    res = detectCharacterCandidates(img,size = size)
    thresh = np.dstack([res.thresh] * 3)
    # print(thresh.shape)
    cv2.imwrite('thresh.png',thresh)
    img = cv2.imread('thresh.png',0)
    cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU,img)
    image, contrs, hier = cv2.findContours(img, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    num_rects = 0
    digit_contours = []
    for c in contrs:
        x, y, w, h = cv2.boundingRect(c)
        if w >= char_width and h > w:
            num_rects+=1
            print(x,y,w,h)
            digit_contours.append(c)
    digit_contours = sorted(digit_contours,key = lambda x:cv2.boundingRect(x)[0])
    for c in digit_contours:
        x, y, w, h = cv2.boundingRect(c)
        char = img[y:y+h,x:x+w]
        char = cv2.cvtColor(char,cv2.COLOR_GRAY2RGB)
        row, col = char.shape[:2]
        # bottom = char[row-2:row, 0:col]
        bordersize = 10
        char = cv2.copyMakeBorder(char, top=bordersize, bottom=bordersize, left=bordersize, right=bordersize,
                                    borderType= cv2.BORDER_CONSTANT, value=[0,0,0] )
        char_dict['char_list'].append(char)
        char_dict['char_coords'].append((x,y,w,h))        
    return char_dict        
    #         cv2.imwrite('char.png',char)
    #         cv2.rectangle(img, (x, y), (x + w, y + h), 255, 1)

    # cv2.drawContours(img, contours, -1, (255, 255, 0), 1)
    # cv2.imwrite("output.png",img)
    # img = cv2.imread('output.png')
    # print(img.shape)
    # plt.imshow(img)
    # plt.show()


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
    
    dai_x,dai_y = next(iter(loader))
    dai_x = dai_x.to(self.device)
    dai_y = [torch.tensor(l).to(self.device) if type(l).__name__ == 'Tensor' else l.to(device) for l in dai_y]
    dai_batch = model(dai_x)
    return crit(dai_batch,dai_y)

def show_objects(self, ax, ima, bbox, clas, prs=None, thresh=0.4):

    return self.show_objects_(ax, ima, ((bbox*self.image_size[0]).long()).numpy(),
        (clas).numpy(), (prs).numpy() if prs is not None else None, thresh)

def show_objects_(self, ax, im, bbox, clas=None, prs=None, thresh=0.3):

    ocr_net = data_processing.load_obj('/home/farhan/hamza/Object_Detection/best_cr_resnet34_net.pkl')
    ocr_dp = data_processing.load_obj('/home/farhan/hamza/LPR_OCR/DP_lpr_ocr.pkl')

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
            # if pr > 0.75:
                # print(im.shape)
                # plt.imsave('good_res.png',im)
                # data_processing.save_obj(b,'good_res.pth')
                # plt.imshow(im[b[1]:b[1]+b[3]][b[0]:b[0]+b[2]])
                # plt.show()
            resize_w = 512
            resize_h = 512    
            im_resized = cv2.resize(im,(resize_w,resize_h))
            im_r,im_c = im.shape[0],im.shape[1]
            row_scale = resize_h/im_r
            col_scale = resize_w/im_c
            b[1] = int(np.round(b[1]*row_scale))
            b[3] = int(np.round(b[3]*row_scale))
            b[0] = int(np.round(b[0]*col_scale))
            b[2] = int(np.round(b[2]*col_scale))
            margin = 12
            try:
                im2 = im_resized[b[1]-margin:b[1]+b[3]+margin,b[0]-margin:b[0]+b[2]+margin]
                plt.imsave('carlp.png',im2)
            except:
                im2 = im_resized[b[1]-margin:b[1]+b[3],b[0]-margin:b[0]+b[2]]
                plt.imsave('carlp.png',im2)
            print(im2.shape)
            chars = get_lp_chars('carlp.png',size = 150,char_width = 7)['char_list']
            utils.plot_in_row(chars,figsize = (8,8))
            for i in chars:
                print(i.shape)
                img = utils.get_test_input(imgs = [i],size = (40,40))
                class_conf = ocr_net.predict(img)[0].max(0)[0]
                class_id = ocr_net.predict(img)[0].max(0)[1]
                # print(chr(int(ocr_dp.class_names[class_id])))
                print(ocr_dp.class_names[class_id], ' ', class_conf)
    del ocr_net
    del ocr_dp

    return ax

def show_nms(self,loader = None,num = 10,img_batch = None,score_thresh = 0.25,nms_overlap = 0.1,dp = None):

    if loader:    
        x,_ = next(iter(loader))
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