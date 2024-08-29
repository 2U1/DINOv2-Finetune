import torch.distributed as dist
import os
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import itertools
import numpy as np
from src.model import Classifier
from src.dataset import ReIDDataset, ImageTransform, InfiniteSampler, EvalDataset
from src.config import Config
import json
from tqdm import tqdm
from torch.amp import autocast
import json

def text_to_number(text):
    units = {"k": 1000, "m": 1000000, "b": 1000000000}
    if text[-1].lower() in units:
        return int(float(text[:-1]) * units[text[-1].lower()])
    else:
        return int(text)
    
def save_final_model(model, optimizer, scheduler, save_dir, cfg, class_to_idx):
    final_dir = os.path.join(save_dir, "final_model")
    os.makedirs(final_dir, exist_ok=True)

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'class_to_idx': class_to_idx
    }

    final_model_path = os.path.join(final_dir, "final_model.pth")
    torch.save(checkpoint, final_model_path)

    # Config 파일 저장
    config_path = os.path.join(final_dir, "config.json")
    with open(config_path, 'w') as f:
        json.dump(cfg, f, indent=4)
    
def save_checkpoint(model, optimizer, scheduler, epoch, iteration, save_dir, cfg, class_to_idx, is_best=False):
    # Iteration별로 폴더 생성
    checkpoint_dir = os.path.join(save_dir, f"checkpoint-{iteration}")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # 모델, 옵티마이저, 스케줄러 상태 저장
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'epoch': epoch,
        'iteration': iteration,
        'class_to_idx': class_to_idx
    }

    # 체크포인트 저장 경로
    checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.pth")
    torch.save(checkpoint, checkpoint_path)

    # Config 파일 저장
    config_path = os.path.join(checkpoint_dir, "config.json")
    with open(config_path, 'w') as f:
        json.dump(cfg, f, indent=4)

    # 최상의 성능 모델 저장
    if is_best:
        best_checkpoint_dir = os.path.join(save_dir, f"best_checkpoint-{iteration}")
        os.makedirs(best_checkpoint_dir, exist_ok=True)
        best_checkpoint_path = os.path.join(best_checkpoint_dir, "best_checkpoint.pth")
        torch.save(checkpoint, best_checkpoint_path)

        # Config 파일 저장
        best_config_path = os.path.join(best_checkpoint_dir, "config.json")
        with open(best_config_path, 'w') as f:
            json.dump(cfg, f, indent=4)
    
def get_scheduler(optimizer, scheduler_type, num_iterations, warmup_ratio):
    if scheduler_type == 'linear':
        warmup_steps = int(num_iterations * warmup_ratio)
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1/num_iterations,
            total_iters=warmup_steps
        )
    elif scheduler_type == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=num_iterations // 3, gamma=0.1)
    elif scheduler_type == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_iterations)
    elif scheduler_type == 'exponential':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")
    return scheduler

def train_step(model, data, target, optimizer, criterion):
    optimizer.zero_grad()
    with autocast('cuda'):
        output = model(data)
        loss = criterion(output, target)
    loss.backward()
    optimizer.step()
    return loss.item()

def evaluate(model, dataloader, criterion, device):
    model.eval()  # 평가 모드로 설정
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in dataloader:
            data = data.to(device)
            target = target.to(device)

            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item() * data.size(0)
            _, predicted = torch.max(output, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    avg_loss = total_loss / total
    accuracy = 100 * correct / total

    return avg_loss, accuracy

def train(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_cfg = cfg['dataset']
    eval_interval = cfg['eval_step']
    num_iterations = text_to_number(cfg['iterations'])

    total_batch_size = cfg['batch_per_gpu'] * torch.cuda.device_count()

    transform = ImageTransform(cfg['resize'], cfg['mean'], cfg['std'])

    # Train dataset
    train_dataset = ReIDDataset(cfg=dataset_cfg['train'], transform=transform, phase='train')
    class_to_idx = train_dataset.class_to_idx  # 클래스 정보를 가져옴
    train_sampler = InfiniteSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, batch_size=total_batch_size, sampler=train_sampler, num_workers=8)

    # Eval dataset
    eval_dataset = EvalDataset(data_root=dataset_cfg['eval']['data_root'], transform=transform, phase='val', class_to_idx=class_to_idx) 
    eval_dataloader = DataLoader(eval_dataset, batch_size=total_batch_size, shuffle=False, num_workers=8)

    model = Classifier(num_classes=cfg['num_classes'], backbone=cfg['backbone'], head=cfg['head']).to(device)
    
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    # Set different learning rates for the backbone and head
    backbone_params = model.module.backbone.parameters() if isinstance(model, nn.DataParallel) else model.backbone.parameters()
    head_params = model.module.head.parameters() if isinstance(model, nn.DataParallel) else model.head.parameters()
    
    # Optimizer with different learning rates for backbone and classifier
    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': cfg['backbone_lr']},
        {'params': head_params, 'lr': cfg['head_lr'] }
    ], weight_decay=cfg['weight_decay'], betas=(cfg['beta1'], cfg['beta2']), eps=cfg['eps'])

    
    scheduler = get_scheduler(optimizer, cfg['scheduler_type'], num_iterations, cfg['warmup_ratio'])
    
    criterion = nn.CrossEntropyLoss()

    output_dir = cfg.get('output_dir', './outputs')
    os.makedirs(output_dir, exist_ok=True)

    best_val_accuracy = 0.0

    pbar = tqdm(range(num_iterations), desc=f"Training", dynamic_ncols=True)

    for iteration in pbar:
        model.train()
        data, target = next(iter(train_dataloader))
        data = data.to(device)
        target = target.to(device)
        loss = train_step(model, data, target, optimizer, criterion)
        
        # pbar.set_postfix({'loss': loss, 'learning_rate': scheduler.get_last_lr()[0]})
        tqdm.write(f"Iteration {iteration}: Loss = {loss:.5f}, Backbone LR = {scheduler.get_last_lr()[0]}, Head LR = {scheduler.get_last_lr()[1]}")

        if iteration % eval_interval == 0 :
            val_loss, val_accuracy = evaluate(model, eval_dataloader, criterion, device)
            tqdm.write(f"Evaluation at iteration {iteration}: Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")

            save_checkpoint(model, optimizer, scheduler, 0, iteration, output_dir, cfg, class_to_idx, is_best=(val_accuracy > best_val_accuracy))
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy

        scheduler.step()

    save_final_model(model, optimizer, scheduler, output_dir, cfg, class_to_idx)
    print(f"Final model saved to {os.path.join(output_dir, 'final_model')}")


if __name__ == '__main__':

    cfg = Config().get_cfg()

    train(cfg)