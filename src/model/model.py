# Model for the Self-Seupervised Classification model - PyTorch

import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class Model(nn.Module):
    def __init__(self, backbone_dim=2048, out_dim = 128, num_classes=10, backbone='resnet18',  pretrained=False,   num_layers_cls=2,  ):
        super(Model, self).__init__()
        self.backbone_dim = backbone_dim
        self.out_dim = out_dim
        self.backbone = backbone
        self.pretrained = pretrained
        self.num_classes = num_classes
        self.num_layers_cls = num_layers_cls
        
        self.backbone = self._get_model()
        self.backbone.fc = nn.Identity()

        # Classifier Head input_size, output_size, num_hidden_layers=2, hidden_size=256, activation='relu', batch_norm=False
        self.classifier_head = MLPhead(input_size=self.backbone_dim, output_size = self.out_dim, num_hidden_layers=self.num_layers_cls)

        # Classifier final layer    
        self.classifier_final = nn.utils.weight_norm(nn.Linear(self.out_dim, self.num_classes, bias=False))
        self.classifier_final.weight_g.data.fill_(1.0)
        #setattr(self.classifier_final, 'weight_g', nn.Parameter(self.classifier_final.weight_g))
        


    def _get_model(self):
        if self.backbone == 'resnet18':
            model = models.resnet18(pretrained=self.pretrained)
        elif self.backbone == 'resnet34':
            model = models.resnet34(pretrained=self.pretrained)
        elif self.backbone == 'resnet50':
            model = models.resnet50(pretrained=self.pretrained)
        elif self.backbone == 'resnet101':
            model = models.resnet101(pretrained=self.pretrained)
        elif self.backbone == 'resnet152':
            model = models.resnet152(pretrained=self.pretrained)
        else:
            raise ValueError("Invalid backbone: {}".format(self.backbone))
        return model

    #def forward(self, x):
    #    x = self.backbone(x)
    #    x = self.classifier(x)
    #    x = self.classifier_final(x)
    #    return x


    def forward(self, x, cls_num=None, return_embds=False):
        if isinstance(x, list):  # multiple views
            bs_size = x[0].shape[0]

            if return_embds:
                # run backbone forward pass separately on each resolution input.
                idx_crops = th.cumsum(th.unique_consecutive(th.Tensor([inp.shape[-1] for inp in x]), return_counts=True)[1], 0)
                start_idx = 0
                for end_idx in idx_crops:
                    _out = self.backbone(th.cat(x[start_idx: end_idx]))
                    if start_idx == 0:
                        output = _out
                    else:
                        output = th.cat((output, _out))
                    start_idx = end_idx

                # run classification head forward pass on concatenated features
                embds = self.classifier_head(output)
                # convert back to list of views
                embds = [embds[x: x + bs_size] for x in range(0, len(embds), bs_size)]
                return embds
            else:  # input is embds
                # concatenate features
                x = th.cat(x, 0)
                out = self.classifier_final(x)
                output = [out[x: x + bs_size] for x in range(0, len(out), bs_size)]

        else:  # single view
            x = self.backbone(x)
            x = self.mlp_head(x)

            if return_embds:
                return x
            else:
                # apply only cls_num
                output = self.classifier_final(x)

        return output

# Classifier head for the Self-Seupervised Classification model - PyTorch

class MLPhead(nn.Module):
    def __init__(self, input_size, output_size, num_hidden_layers=2, hidden_size=256, activation='relu', batch_norm=False):
        super().__init__()
        
        if num_hidden_layers == 0:
            self.mlp = nn.Linear(input_size, output_size)
        else:
            layers = [nn.Linear(input_size, hidden_size)]
            for i in range(num_hidden_layers-1):
                
                if batch_norm:
                    layers.append(nn.BatchNorm1d(hidden_size))
                
                if activation == 'relu':
                    layers.append(nn.ReLU())
                elif activation == 'leaky_relu':
                    layers.append(nn.LeakyReLU())
                
                layers.append(nn.Linear(hidden_size, hidden_size))

            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_size))
            
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'leaky_relu':
                layers.append(nn.LeakyReLU())

            layers.append(nn.Linear(hidden_size, output_size))
            self.mlp = nn.Sequential(*layers)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

            
    def forward(self, x):
        x = self.mlp(x)
        x = F.normalize(x, dim=1) # p=2, eps=1e-7
        return x

