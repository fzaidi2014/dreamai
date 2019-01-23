class StyleTransfer(Network):
    def __init__(self,content_img_path,layer_dict,content_layer_key,
                 style_img_path,style_weights,max_size=512,
                 norm_values=[(0.485, 0.456, 0.406),(0.229, 0.224, 0.225)],
                 model_name='vgg',content_weight=1,style_weight=1e6,
                 target_image_file='target_img.pkl'

                ):
        super().__init__()
        if model_name.lower() == 'vgg':
                self.model = models.vgg19(pretrained=True).features
        self.freeze()
        self.model_name = model_name
        self.model.to(self.device)
        self.content_img = load_image(content_img_path,norm_values=norm_values).to(self.device)
        # Resize style to match content, makes code easier
        self.style_img = load_image(style_img_path,shape=self.content_img.shape[-2:],norm_values=norm_values).to(self.device)
        self.layer_dict = layer_dict
        self.content_layer_key = content_layer_key
        self.target_image_file = target_image_file

        self.content_features = self._extract_features('content')
        self.style_features = self._extract_features('style')

        # calculate the gram matrices for each layer of our style representation
        self.style_grams = {}
        for layer in self.style_features:
            _,d,h,w = self.style_features[layer].shape           
            self.style_grams[layer] = gram_matrix(self.style_features[layer],d,h,w)

        self.target = self.content_img.clone().requires_grad_(True).to(self.device)
        
        self.style_weights = style_weights

        self.content_weight = content_weight
        self.style_weight = style_weight

    def _extract_features(self,image_type):                   
            
        features = {}
        if image_type == 'content':
            x = self.content_img
        elif image_type == 'style':
            x = self.style_img
        elif image_type == 'target':
            x = self.target  
        # model._modules is a dictionary holding each module in the model
        if self.model_name.lower() == 'vgg':
            for name, layer in self.model._modules.items():
                x = layer(x)
                if name in self.layer_dict:
                    features[self.layer_dict[name]] = x
                
        return features            
                
    def fit(self,show_every=400,optimizer='Adam',lr=0.003,steps=2000):
        import matplotlib.pyplot as plt
        # iteration hyperparameters
        set_optimizer(self,[self.target],optimizer=optimizer,lr=lr)

        steps = steps  # decide how many iterations to update your image (5000)

        for ii in range(1, steps+1):
            
            # get the features from your target image
            target_features = self._extract_features('target')
            
            # the content loss
            content_loss = torch.mean((target_features[self.content_layer_key] - self.content_features[self.content_layer_key])**2)
            
            style_loss = 0
            for layer in self.style_weights:
                target_feature = target_features[layer]
                _, d, h, w = target_feature.shape
                target_gram = gram_matrix(target_feature,d,h,w)
                # get the "style" style representation
                style_gram = self.style_grams[layer]
                # the style loss for one layer, weighted appropriately
                layer_style_loss = self.style_weights[layer] * torch.mean((target_gram - style_gram)**2)
                # add to the style loss
                style_loss += layer_style_loss / (d * h * w)
                
            # calculate the *total* loss
            total_loss = self.content_weight * content_loss + self.style_weight * style_loss
            
            # update your target image
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
            
            # display intermediate images and print the loss
            if  ii % show_every == 0:
                print('Total loss: ', total_loss.item())
                plt.imshow(im_convert(self.target))
                plt.show()
                in_key = input('press "c" to continue, any other key to exit: ')
                if in_key.lower() != 'c':
                    print('saving target image in file {}'.format(self.target_image_file))
                    torch.save(self.target,self.target_image_file)
                    break
                else:
                    fname = self.target_image_file
                    temp = fname.split('.')
                    fname = temp[0]+'_'+str(ii) + '.' + temp[1]
                    print('saving target image in file {}'.format(fname))
                    torch.save(self.target,fname)
