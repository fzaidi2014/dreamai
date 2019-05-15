from dai_imports import*

class SaveFeatures():
    features=None
    def __init__(self, m): self.hook = m.register_forward_hook(self.hook_fn)
    def hook_fn(self, module, input, output): self.features = output
    def remove(self): self.hook.remove()

class UnetBlock(nn.Module):
    def __init__(self, up_in, x_in, n_out):
        super().__init__()
        up_out = x_out = n_out//2
        self.x_conv  = nn.Conv2d(x_in,  x_out,  1)
        self.tr_conv = nn.ConvTranspose2d(up_in, up_out, 2, stride=2)
        self.bn = nn.BatchNorm2d(n_out)
        
    def forward(self, up_p, x_p):
        up_p = self.tr_conv(up_p)
        x_p = self.x_conv(x_p)
        # print(up_p.size(),x_p.size())
        cat_p = torch.cat([up_p,x_p], dim=1)
        return self.bn(F.relu(cat_p))

class Unet(nn.Module):
    def __init__(self, backbone, in_channels, head):
        super().__init__()
        self.backbone = backbone
        for layer in head.children():
            if(type(layer).__name__) == 'StdConv':
                conv_module = layer
                break
        conv_layer = conv_module.conv
        temp_args = [conv_layer.out_channels,conv_layer.kernel_size,conv_layer.stride,conv_layer.padding]
        temp_args.insert(0,3)
        conv_layer = nn.Conv2d(*temp_args)
        conv_module.conv = conv_layer
        self.head = head
        self.sfs = [SaveFeatures(backbone[i]) for i in [2,4,5,6]]
        self.up1 = UnetBlock(in_channels,256,256)
        self.up2 = UnetBlock(256,128,256)
        self.up3 = UnetBlock(256,64,256)
        self.up4 = UnetBlock(256,64,256)
        self.up5 = UnetBlock(256,3,16)
        self.up6 = nn.ConvTranspose2d(256, 3, 1)
        
    def forward(self,x):
        inp = x
        x = F.relu(self.backbone(x))
        # print(x.size())
        x = self.up1(x, self.sfs[3].features)
        # print(x.size())
        x = self.up2(x, self.sfs[2].features)
        # print(x.size())
        x = self.up3(x, self.sfs[1].features)
        # print(x.size())
        # x = self.up4(x, self.sfs[0].features)
        # print(x.size())
        # x = self.up5(x, inp)
        # print(x.size())
        x = self.up6(x)
        # print(x.size())
        x = self.head(x)
        return x#[:,0]
    
    def close(self):
        for sf in self.sfs: sf.remove()