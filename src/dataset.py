import os.path as osp
from torch.utils.data import Dataset, Sampler
import torch
import json
import numpy as np
from torchvision import transforms
import glob
from PIL import Image
import itertools

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

def make_datapath_list(data_root):
    target_path = osp.join(data_root, '*/*')
    path_list = glob.glob(target_path)
    return path_list

def extract_label_from_path(file_path, data_root):
    # Assuming the directory structure is: data_root/class_name/image_file
    class_name = osp.basename(osp.dirname(file_path))
    return class_name

class ReIDDataset(Dataset):
    def __init__(self, cfg, transform=None, phase='train'):
        self.cfg = cfg
        self.transform = transform
        self.phase = phase

        # Generate list of file paths
        self.file_list = make_datapath_list(cfg['data_root'])

        # Extract unique class labels and assign numeric labels
        self.class_names = sorted({extract_label_from_path(f, cfg['data_root']) for f in self.file_list})
        self.class_to_idx = {class_name: idx for idx, class_name in enumerate(self.class_names)}

        # Assign labels based on file paths
        self.labels = [self.class_to_idx[extract_label_from_path(f, cfg['data_root'])] for f in self.file_list]

        # Rare Class Sampling (RCS) setup
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

            for i, label in enumerate(self.labels):
                if label in self.samples_with_class:
                    self.samples_with_class[label].append(i)

            for c in self.rcs_classes:
                if len(self.samples_with_class[c]) == 0:
                    raise ValueError(f"No samples found for class {c}. Check the dataset.")

    def get_rare_class_sample(self):
        c = np.random.choice(self.rcs_classes, p=self.rcs_classprob)
        sample_idx = np.random.choice(self.samples_with_class[c])
        img_path = self.file_list[sample_idx]
        label = self.labels[sample_idx]
        img = Image.open(img_path).convert('RGB')
        return img, label
    
    def __getitem__(self, idx):
        if self.rcs_enabled:
            img, label = self.get_rare_class_sample()
        else:
            img_path = self.file_list[idx]
            label = self.labels[idx]
            img = Image.open(img_path).convert('RGB')

        if self.transform:
            img = self.transform(img, self.phase)

        return img, label

    def __len__(self):
        return len(self.file_list)
    
class InfiniteSampler(Sampler):
    def __init__(self, dataset):
        self.dataset = dataset

    def __iter__(self):
        # itertools.cycle을 사용하여 데이터셋 인덱스를 무한 반복
        return iter(itertools.cycle(range(len(self.dataset))))

    def __len__(self):
        # len을 무한으로 설정할 수 없으므로, 매우 큰 값으로 설정
        return int(1e10)