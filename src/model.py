import torch
import torch.nn as nn
from torch.hub import load
import math
import timm
import torch.nn.init as init

MODEL_LIST = {
    'dinov2_s':{
        'name':'dinov2_vits14',
        'embedding_size':384,
        'patch_size':14
    },
    'dinov2_b':{
        'name':'dinov2_vitb14',
        'embedding_size':768,
        'patch_size':14
    },
    'dinov2_l':{
        'name':'dinov2_vitl14',
        'embedding_size':1024,
        'patch_size':14
    },
    'dinov2_g':{
        'name':'dinov2_vitg14',
        'embedding_size':1536,
        'patch_size':14
    },
    'siglip_384':{
        'name': 'vit_so400m_patch14_siglip_384',
        'embedding_size':1152,
        'path_size': 14
    }
}

class linear_head(nn.Module):
    def __init__(self, embedding_size = 384, num_classes = 10, *args, **kwargs):
        super(linear_head, self).__init__()
        self.fc = nn.Linear(embedding_size, num_classes)

        init.xavier_uniform_(self.fc.weight)
        
        if self.fc.bias is not None:
            init.zeros_(self.fc.bias)

    def forward(self, x):
        return self.fc(x)
    
class multi_linear_head(nn.Module):
    def __init__(self, embedding_size = 384, num_classes = 10, hidden_dims=[512, 256], dropout=0.1):
        super(multi_linear_head, self).__init__()

        layers = []
        current_dim = embedding_size

        for hidden_dim in hidden_dims:
            linear_layer = nn.Linear(current_dim, hidden_dim)
            init.kaiming_normal_(linear_layer.weight, mode='fan_out', nonlinearity='relu')
            layers.append(linear_layer)
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            current_dim = hidden_dim

        final_layer = nn.Linear(current_dim, num_classes)
        init.kaiming_normal_(final_layer.weight, mode='fan_out', nonlinearity='relu')
        layers.append(final_layer)

        self.classifier_head = nn.Sequential(*layers)

    def forward(self, x):
        return self.classifier_head(x)

class Classifier(nn.Module):
    def __init__(self, num_classes, backbone = 'dinov2_s', head = 'single', hidden_dims = [512, 256], dropout=0.1, freeze_backbone=False, **kwargs):
        super(Classifier, self).__init__()
        self.heads = {
            'single':linear_head,
            'mlp': multi_linear_head
        }
        self.backbones = MODEL_LIST
        
        if backbone in ['dinov2_s', 'dinov2_b', 'dinov2_l', 'dinov2_g']:
            self.backbone = load('facebookresearch/dinov2', self.backbones[backbone]['name'])
        else:
            self.backbone = timm.create_model(self.backbones[backbone]['name'], pretrained=True, num_classes=0)

        self.head = self.heads[head](self.backbones[backbone]['embedding_size'], num_classes, hidden_dims, dropout=dropout)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return x