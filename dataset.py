import os.path as osp
from torch.utils.data import Dataset
import torch
import json
import numpy as np
from torchvision import transforms


class ImageTransform(object):
    def __init__(self, resize, mean, std):
        self.data_transform = {
            'train': transforms.Compose([
                transforms.Resize(resize),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ]),
            'val': transforms.Compose([
                transforms.Resize(resize),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
        }

    def __call__(self, img, phase='train'):
        return self.data_transform[phase](img)

def get_rcs_class_probs(data_root, temperature):
    with open(osp.join(data_root, 'class_stats.json'), 'r') as of:
        class_stats = json.load(of)

    overall_class_stats = {
        k: v 
        for k, v in sorted(class_stats.items(), key=lambda item: item[1])
    }

    freq = torch.tensor(list(overall_class_stats.values()))
    freq = freq / torch.sum(freq)
    freq = 1 - freq
    freq = torch.softmax(freq / temperature, dim=-1)

    return list(overall_class_stats.keys()), freq.numpy()


class ReIDDataset(object):

    def __init__(self, cfg, dataset):
        self.cfg = cfg
        self.dataset = dataset

        rcs_cfg = cfg.get('rare_clas_sampling')
        self.rcs_enabled = rcs_cfg is not None

        if self.rcs_enabled:
            self.rcs_class_temp = rcs_cfg['class_temp']
            
            self.rcs_classes, self.rcs_classprob = get_rcs_class_probs(cfg['data_root'], self.rcs_class_temp)

            with open(osp.join(cfg['data_root'], 'class_stats.json'), 'r') as of:
                class_stats = json.load(of)

            self.samples_with_class = {
                int(k): []
                for k in self.rcs_classes
            }

            for i, (image, label) in enumerate(dataset):
                if label in self.samples_with_class:
                    self.samples_with_class[label].append(i)

            for c in self.rcs_classes:
                if len(self.samples_with_class[c]) == 0:
                    raise ValueError(f"No samples found for class {c}. Check the dataset.")

    
    def get_rare_class_sample(self):
        c = np.random.choice(self.rcs_classes, p=self.rcs_classprob)
        sample_idx = np.random.choice(self.samples_with_class[c])
        
        return self.dataset[sample_idx]
    

    def __getitem__(self, idx):
        if self.rcs_enabled:
            return self.get_rare_class_sample()
        else:
            return self.dataset[idx]
        

    def __len__(self):
        return len(self.dataset)
    

class CustomDatasetWrapper(Dataset):
    def __init__(self, custom_dataset, transform=None, phase='train'):
        self.custom_dataset = custom_dataset
        self.transform = transform
        self.phase = phase

    def __getitem__(self, idx):
        img, label = self.custom_dataset[idx]
        if self.transform:
            img = self.transform(img, self.phase)

        return img, label
    
    def __len__(self):
        return len(self.custom_dataset)