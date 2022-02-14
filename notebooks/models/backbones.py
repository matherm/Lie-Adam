import torch
from torchvision import models as mods
import torchvision

def get_eff_net(layer=6):
    eff = mods.efficientnet_b4(pretrained=True)
    net = eff.features[:layer]
    return net

def get_wide_resnet(pooling=4):
    net = torchvision.models.wide_resnet50_2(pretrained=True)
    layers = [net.conv1,
              net.bn1,
              net.relu,
              net.maxpool]
    layers += [net.layer1, net.layer2, net.layer3, net.layer4][:pooling]
    net = torch.nn.Sequential(*layers)
    return net

def get_vgg(pooling=4):
    net = torch.hub.load('pytorch/vision:v0.10.0', 'vgg19', pretrained=True).features
    
    idx_pooling_layer = []
    for i, module in enumerate( net.modules() ):
        if not isinstance(module, nn.Sequential):
            if isinstance(module, nn.MaxPool2d):
                idx_pooling_layer.append(i)
    F = idx_pooling_layer[pooling]
    
    net = net[:self.F]
    
    # Sometimes inplace ReLUs cause problems with Laplace
    for module in net.modules():
        if not isinstance(module, nn.Sequential):
            if isinstance(module, nn.ReLU):
                module.inplace = False

    return net


