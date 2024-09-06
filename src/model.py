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

def bilinear_interpolate_pos_encoding(self, x, w, h):
    previous_dtype = x.dtype
    npatch = x.shape[1] - 1
    N = self.pos_embed.shape[1] - 1
    if npatch == N and w == h:
        return self.pos_embed.to(x.device)  # Move to same device as input
    pos_embed = self.pos_embed.float().to(x.device)  # Move to same device as input
    class_pos_embed = pos_embed[:, 0]
    patch_pos_embed = pos_embed[:, 1:]
    dim = x.shape[-1]
    w0 = w // self.patch_size
    h0 = h // self.patch_size
    M = int(math.sqrt(N))  # Recover the number of patches in each dimension
    assert N == M * M
    kwargs = {}
    if self.interpolate_offset:
        sx = float(w0 + self.interpolate_offset) / M
        sy = float(h0 + self.interpolate_offset) / M
        kwargs["scale_factor"] = (sx, sy)
    else:
        kwargs["size"] = (w0, h0)
    patch_pos_embed = nn.functional.interpolate(
        patch_pos_embed.reshape(1, M, M, dim).permute(0, 3, 1, 2),
        mode="bilinear",
        antialias=self.interpolate_antialias,
        **kwargs,
    )
    assert (w0, h0) == patch_pos_embed.shape[-2:]
    patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
    return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1).to(previous_dtype).to(x.device)  # x와 동일한 장치로 이동


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
             # Monkey patching the interpolate_pos_encoding method to allow for bilinear interpolation
            self.backbone.interpolate_pos_encoding = bilinear_interpolate_pos_encoding.__get__(self.backbone, type(self.backbone))
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