import torch
import torch.nn as nn
from torch.hub import load
import math

dino_backbones = {
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
}

def bilinear_interpolate_pos_encoding(self, x, w, h):
    previous_dtype = x.dtype
    npatch = x.shape[1] - 1
    N = self.pos_embed.shape[1] - 1
    if npatch == N and w == h:
        return self.pos_embed.to(x.device)  # x와 동일한 장치로 이동
    pos_embed = self.pos_embed.float().to(x.device)  # 위치 인코딩 텐서를 x와 동일한 장치로 이동
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
    def __init__(self, embedding_size = 384, num_classes = 10):
        super(linear_head, self).__init__()
        self.fc = nn.Linear(embedding_size, num_classes)

    def forward(self, x):
        return self.fc(x)
    

class Classifier(nn.Module):
    def __init__(self, num_classes, backbone = 'dinov2_s', head = 'linear', backbones = dino_backbones):
        super(Classifier, self).__init__()
        self.heads = {
            'linear':linear_head
        }
        self.backbones = dino_backbones
        self.backbone = load('facebookresearch/dinov2', self.backbones[backbone]['name'])

        # Monkey patching the interpolate_pos_encoding method to allow for bilinear interpolation
        self.backbone.interpolate_pos_encoding = bilinear_interpolate_pos_encoding.__get__(self.backbone, type(self.backbone))
        
        self.head = self.heads[head](self.backbones[backbone]['embedding_size'],num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return x
