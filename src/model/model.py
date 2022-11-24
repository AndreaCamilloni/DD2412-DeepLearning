# Model for the Self-Seupervised Classification model - PyTorch

import math 
import warnings

import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class Model(nn.Module):
    def __init__(self, backbone_dim=512, hidden_dim = 4096, dim = 128, num_classes=5, backbone='resnet18',  pretrained=False,   num_layers_cls=1, activation_cls='relu', use_bn = False):
        super(Model, self).__init__()
        #self.backbone_dim = backbone_dim
        self.hidden_dim = hidden_dim
        self.dim = dim 
        self.backbone = backbone
        self.pretrained = pretrained
        self.num_classes = num_classes
        self.num_layers_cls = num_layers_cls
        self.activation_cls = activation_cls
        self.use_bn = use_bn
        
        self.backbone = self._get_model()
        try:
            self.backbone_dim = self.backbone.fc.weight.shape[1]
        except:
            self.backbone_dim = backbone_dim

        self.backbone.fc = nn.Identity()


        # Classifier Head input_size, output_size, num_hidden_layers=2, hidden_size=256, activation='relu', batch_norm=False
        self.classifier_head = MLPhead(input_size=self.backbone_dim, output_size = self.dim, num_hidden_layers=self.num_layers_cls, hidden_size=self.hidden_dim, activation=self.activation_cls, batch_norm=self.use_bn)

        # Classifier final layer    
        #print("Classifier final layer - input: ", self.dim)
        #print("Classifier final layer - output: ", self.num_classes[0])
        classifier_final_layer = nn.utils.weight_norm(nn.Linear(dim, self.num_classes[0], bias=False))
        classifier_final_layer.weight_g.data.fill_(1.0)
        self.classifier_final = classifier_final_layer
        #setattr(self, 'classifier_final', classifier_final_layer)
        


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
            #print("bs_size: ", bs_size)
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
                #out = getattr(self, "classifier_final")(x)
                out = self.classifier_final(x)
                output = [out[x: x + bs_size] for x in range(0, len(out), bs_size)]

        else:  # single view
            x = self.backbone(x)
            x = self.classifier_head(x)

            if return_embds:
                return x
            else:
                # apply only cls_num
                #output = getattr(self, "classifier_final")(x)
                output = self.classifier_final(x)

        return output

# Classifier head for the Self-Seupervised Classification model - PyTorch

class MLPhead(nn.Module):
    def __init__(self, input_size, output_size, num_hidden_layers=2, hidden_size=256, activation='relu', batch_norm=False):
        super().__init__()
        
        # we skipped one case (no hidden layers)

        if num_hidden_layers == 0:
            self.mlp = nn.Identity()
        elif num_hidden_layers == 1:
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

#    @staticmethod
#    def _init_weights(m):
#        if type(m) == nn.Linear:
#            nn.init.xavier_uniform_(m.weight)
#            if m.bias is not None:
#                nn.init.zeros_(m.bias)
    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
            
    def forward(self, x):
        x = self.mlp(x)
        x = F.normalize(x, p=2, dim=1, eps=1e-7) # p=2, eps=1e-7
        return x




def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with th.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)
