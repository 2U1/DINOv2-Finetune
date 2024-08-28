from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed import init_process_group, destroy_process_group
import torch.distributed as dist
import argparse
import os
import torch.multiprocessing as mp
import torch
from torch.utils.data import DataLoader, DistributedSampler
import torch.nn as nn
import itertools
import numpy as np
from torchvision.datasets import ImageFolder
from .model import Classifier
from .dataset import CustomDatasetWrapper, ReIDDataset, ImageTransform
from .config import Config
import json
from tqdm import tqdm

def text_to_number(text):
    units = {"k": 1000, "m": 1000000, "b": 1000000000}
    if text[-1].lower() in units:
        return int(float(text[:-1]) * units[text[-1].lower()])
    else:
        return int(text)
    
def save_final_model(model, optimizer, scheduler, save_dir, cfg):
    final_dir = os.path.join(save_dir, "final_model")
    os.makedirs(final_dir, exist_ok=True)

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }

    final_model_path = os.path.join(final_dir, "final_model.pth")
    torch.save(checkpoint, final_model_path)

    # Config 파일 저장
    config_path = os.path.join(final_dir, "config.json")
    with open(config_path, 'w') as f:
        json.dump(cfg, f, indent=4)
    
def save_checkpoint(model, optimizer, scheduler, epoch, iteration, save_dir, cfg, is_best=False):
    # Iteration별로 폴더 생성
    checkpoint_dir = os.path.join(save_dir, f"checkpoint-{iteration}")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # 모델, 옵티마이저, 스케줄러 상태 저장
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'epoch': epoch,
        'iteration': iteration
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
    output = model(data)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
    return loss.item()

def evaluate(model, dataloader, criterion, rank):
    model.eval()  # 평가 모드로 설정
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in dataloader:
            data = data.to(rank)
            target = target.to(rank)

            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item() * data.size(0)
            _, predicted = torch.max(output, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    avg_loss = total_loss / total
    accuracy = 100 * correct / total

    return avg_loss, accuracy

def train(rank, world_size, cfg):
    global_rank = rank
    
    init_process_group(
        backend='nccl',
        init_method="evn://",
        rank=global_rank,
        world_size=world_size,
    )

    dataset_cfg = cfg['dataset']
    eval_interval = cfg['eval_step']
    num_iterations = text_to_number(cfg['iterations'])

    transform = ImageTransform(cfg['resize'], cfg['mean'], cfg['std'])

    # Train dataset
    train_original_dataset = ImageFolder(dataset_cfg['train']['data_root'])
    train_dataset = ReIDDataset(cfg, train_original_dataset)
    wrapped_train_dataset = CustomDatasetWrapper(train_dataset, transform=transform, phase='train')
    train_sampler = DistributedSampler(wrapped_train_dataset, num_replicas=world_size, rank=global_rank, shuffle=True)
    train_dataloader = DataLoader(wrapped_train_dataset, batch_size=cfg['batch_size'], sampler=train_sampler)

    # Eval dataset
    eval_original_dataset = ImageFolder(dataset_cfg['eval']['data_root'])
    eval_dataset = ReIDDataset(cfg, eval_original_dataset)
    wrapped_eval_dataset = CustomDatasetWrapper(eval_dataset, transform=transform, phase='val')
    eval_sampler = DistributedSampler(wrapped_eval_dataset, num_replicas=world_size, rank=global_rank, shuffle=False)
    eval_dataloader = DataLoader(wrapped_eval_dataset, batch_size=cfg['batch_size'], sampler=eval_sampler)
    
    model = Classifier(num_classes=cfg['num_classes'], backbone=cfg['backbone'], head=cfg['head'])
    model = FSDP(model)
    
    optimizer = torch.optim.AdamW(model.parameters(), 
                                  lr=cfg['learning_rate'],
                                  weight_decay=cfg['weight_decay'],
                                  betas=(cfg['beta1'], cfg['beta2']),
                                  eps=cfg['eps']
                                  )
    
    scheduler = get_scheduler(optimizer, cfg['scheduler_type'], num_iterations, cfg['warmup_ratio'])
    
    criterion = nn.CrossEntropyLoss()

    inifinite_datalodaer = itertools.cycle(train_dataloader)

    output_dir = cfg.get('output_dir', './outputs')
    os.makedirs(output_dir, exist_ok=True)

    best_val_accuracy = 0.0

    pbar = tqdm(range(num_iterations), desc=f"Training", dynamic_ncols=True)

    for iteration in pbar:
        model.train()
        data, target = next(inifinite_datalodaer)
        data = data.to(rank)
        target = target.to(rank)
        loss = train_step(model, data, target, optimizer, criterion)
        
        pbar.set_postfix({'loss': loss, 'learning_rate': scheduler.get_last_lr()[0]})

        if iteration % eval_interval == 0 and rank == 0:
            val_loss, val_accuracy = evaluate(model, eval_dataloader, criterion, rank)
            print(f"Evaluation at iteration {iteration}: Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")

            save_checkpoint(model, optimizer, scheduler, 0, iteration, output_dir, cfg, is_best=(val_accuracy > best_val_accuracy))
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy

        scheduler.step()

    if rank == 0:  # 주 프로세스에서만 저장
        save_final_model(model, optimizer, scheduler, output_dir, cfg)
        print(f"Final model saved to {os.path.join(output_dir, 'final_model')}")


    destroy_process_group()

def main(cfg, args):
    world_size = args.world_size
    mp.spawn(train, args=(world_size, cfg), nprocs=world_size, join=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a classification model with FSDP')
    parser.add_argument('--world-size', type=int, default=torch.cuda.device_count(), help='Number of processes')
    parser.add_argument("--rank", type=int, default=0, help="Rank of the current process")
    parser.add_argument("--master_addr", type=str, default="127.0.0.1", help="Address of the master node")
    parser.add_argument("--master_port", type=str, default="29500", help="Port of the master node")
    args = parser.parse_args()


    cfg = Config().get_cfg()

    main(cfg, args)