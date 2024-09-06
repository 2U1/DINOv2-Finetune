import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.amp import autocast

import os
import json
from tqdm import tqdm
import glob

from src.model import Classifier
from src.dataset import ReIDDataset, ImageTransform, InfiniteSampler, EvalDataset
from src.config import Config
from src.loss import LossFactory


torch.random.manual_seed(0)
np.random.seed(0)

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

    config_path = os.path.join(final_dir, "config.json")
    with open(config_path, 'w') as f:
        json.dump(cfg, f, indent=4)
    
def save_checkpoint(model, optimizer, scheduler, epoch, iteration, save_dir, cfg, class_to_idx, is_best=False):
    max_checkpoints = cfg.get('max_checkpoints', 5)
    checkpoint_dir = os.path.join(save_dir, f"checkpoint-{iteration}")
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'epoch': epoch,
        'iteration': iteration,
        'class_to_idx': class_to_idx
    }

    checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.pth")
    torch.save(checkpoint, checkpoint_path)

    config_path = os.path.join(checkpoint_dir, "config.json")
    with open(config_path, 'w') as f:
        json.dump(cfg, f, indent=4)

    if is_best:
        best_checkpoint_dir = os.path.join(save_dir, f"best_checkpoint-{iteration}")
        os.makedirs(best_checkpoint_dir, exist_ok=True)
        best_checkpoint_path = os.path.join(best_checkpoint_dir, "best_checkpoint.pth")
        torch.save(checkpoint, best_checkpoint_path)

        best_config_path = os.path.join(best_checkpoint_dir, "config.json")
        with open(best_config_path, 'w') as f:
            json.dump(cfg, f, indent=4)

    manage_checkpoints(save_dir, max_checkpoints)


def manage_checkpoints(save_dir, max_checkpoints):
    checkpoint_dirs = sorted(glob.glob(os.path.join(save_dir, "checkpoint-*")), key=os.path.getmtime)

    if len(checkpoint_dirs) > max_checkpoints:
        num_to_remove = len(checkpoint_dirs) - max_checkpoints
        for i in range(num_to_remove):
            old_checkpoint_dir = checkpoint_dirs[i]
            print(f"Deleting old checkpoint: {old_checkpoint_dir}")
            os.system(f"rm -rf {old_checkpoint_dir}")
    

def get_scheduler(optimizer, scheduler_config, num_iterations):
    scheduler_type = scheduler_config['type']
    scheduler_params = scheduler_config.get('params', {})
    
    if scheduler_type == 'linear':
        warmup_ratio = scheduler_params.get('warmup_ratio', 0.03)
        warmup_steps = int(num_iterations * warmup_ratio)
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=scheduler_params.get('start_factor', 1/num_iterations),
            total_iters=warmup_steps
        )
    
    elif scheduler_type == 'step':
        step_size = scheduler_params.get('step_size', num_iterations // 3)
        gamma = scheduler_params.get('gamma', 0.1)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=step_size,
            gamma=gamma
        )
    
    elif scheduler_type == 'cosine':
        T_max = scheduler_params.get('T_max', num_iterations)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=T_max
        )
    
    elif scheduler_type == 'exponential':
        gamma = scheduler_params.get('gamma', 0.9)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=gamma
        )
    
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")
    
    return scheduler

def get_optimizer(optimizer_config, backbone_params=None, head_params=None):

    optimizer_type = optimizer_config['type']
    optimizer_params = optimizer_config['params']
    learning_rates = optimizer_config['learning_rate']

    optimizer_grouped_parameters = [
        {'params': head_params, 'lr': learning_rates['head_lr']},
    ]

    if backbone_params is not None:
        optimizer_grouped_parameters.append({'params': backbone_params, 'lr': learning_rates['backbone_lr']})

    if optimizer_type == 'AdamW':
        optimizer = optim.AdamW(optimizer_grouped_parameters, **optimizer_params)
    elif optimizer_type == 'SGD':
        optimizer = optim.SGD(optimizer_grouped_parameters, **optimizer_params)
    elif optimizer_type == 'Adam':
        optimizer = optim.Adam(optimizer_grouped_parameters, **optimizer_params)
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")
    
    return optimizer

def train_step(model, data, target, optimizer, criterion):
    optimizer.zero_grad()
    with autocast('cuda'):
        output = model(data)
        loss = criterion(output, target)
    loss.backward()
    optimizer.step()
    return loss.item()

def evaluate(model, dataloader, criterion, device):
    model.eval()
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

def make_supervised_dataset(cfg, epoch_based=False):
    dataset_cfg = cfg['dataset']
    transform = ImageTransform(cfg['resize'], cfg['mean'], cfg['std'])

    total_batch_size = cfg['batch_per_gpu'] * torch.cuda.device_count()
    
    if epoch_based:
        train_dataset = ReIDDataset(cfg=dataset_cfg['train'], transform=transform, phase='train')
        train_dataloader = DataLoader(train_dataset, batch_size=total_batch_size, shuffle=True, num_workers=8)

    else:
        train_dataset = ReIDDataset(cfg=dataset_cfg['train'], transform=transform, phase='train')
        train_sampler = InfiniteSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, batch_size=total_batch_size, sampler=train_sampler, num_workers=8)

    class_to_idx = train_dataset.get_class_to_idx()
    samples_per_class = train_dataset.get_samples_per_class()

    eval_dataset = EvalDataset(data_root=dataset_cfg['eval']['data_root'], transform=transform, phase='val', class_to_idx=class_to_idx) 
    eval_dataloader = DataLoader(eval_dataset, batch_size=total_batch_size, shuffle=False, num_workers=8)

    return dict(
        train_loader=train_dataloader,
        eval_loader=eval_dataloader,
        class_to_idx=class_to_idx,
        samples_per_class=samples_per_class
    )

def train(cfg):
    
    assert not ('num_train_epoch' in cfg and 'iterations' in cfg), \
        "Error: 'num_train_epoch' and 'iterations' cannot be used together. Choose one."
    
    epoch_based = False
    
    if 'num_train_epoch' in cfg:
        epoch_based = True
        if 'rare_class_sampling' in cfg['dataset']['train']:
            print("num_train_epoch detected. Removing 'rare_class_sampling' from config.")
            print("Epoch based is set. Eval step will be ignored and will be evaluated every epoch.")
            del cfg['dataset']['train']['rare_class_sampling']

        num_epochs = cfg['num_train_epoch']
        dataset_dict = make_supervised_dataset(cfg, epoch_based)
        num_iterations = len(dataset_dict['train_loader']) * num_epochs

    else:
        num_epochs = None
        num_iterations = text_to_number(cfg['iterations'])
        dataset_dict = make_supervised_dataset(cfg, epoch_based)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    eval_interval = cfg['eval_step']
    model_config = cfg['model']
    optimizer_config = cfg['optimizer']
    scheduler_config = cfg['scheduler']
    loss_config = cfg['loss']
    num_classes = cfg['model']['num_classes']

    model = Classifier(**model_config).to(device)

    train_dataloader = dataset_dict['train_loader']
    eval_dataloader = dataset_dict['eval_loader']
    class_to_idx = dataset_dict['class_to_idx']
    samples_per_class = dataset_dict['samples_per_class']
    
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    if model_config['freeze_backbone']:
        backbone_params = None
    else:
        backbone_params = model.module.backbone.parameters() if isinstance(model, nn.DataParallel) else model.backbone.parameters()
    
    head_params = model.module.head.parameters() if isinstance(model, nn.DataParallel) else model.head.parameters()
    
    optimizer = get_optimizer(optimizer_config, backbone_params, head_params)
    
    scheduler = get_scheduler(optimizer, scheduler_config, num_iterations)

    loss_factory = LossFactory(loss_config, samples_per_class, num_classes=num_classes)
    criterion = loss_factory.get_loss()

    output_dir = cfg.get('output_dir', './outputs')
    os.makedirs(output_dir, exist_ok=True)

    best_val_accuracy = 0.0

    if epoch_based:
        pbar = tqdm(total=num_iterations, desc=f"Training", dynamic_ncols=True)
        total_steps_per_epoch = len(train_dataloader)
        global_step = 0

        for epoch in range(num_epochs):
            model.train()
            for step, (data, target) in enumerate(train_dataloader):
                data = data.to(device)
                target = target.to(device)
                loss = train_step(model, data, target, optimizer, criterion)
                
                global_step += 1
                pbar.update(1)

                current_epoch_progress = epoch + (step + 1) / total_steps_per_epoch

                tqdm.write(f"Iteration {global_step}: Loss = {loss:.5f}, Backbone LR = {scheduler.get_last_lr()[0]}, Head LR = {scheduler.get_last_lr()[1]}, Epoch Progress = {current_epoch_progress:.2f}")

                scheduler.step()

            val_loss, val_accuracy = evaluate(model, eval_dataloader, criterion, device)
            tqdm.write(f"Evaluation at iteration {global_step}: Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")

            # Save the checkpoint if it's the best accuracy so far
            save_checkpoint(model, optimizer, scheduler, epoch, global_step, output_dir, cfg, class_to_idx, is_best=(val_accuracy > best_val_accuracy))
            
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
            
    else:
        pbar = tqdm(range(num_iterations), desc=f"Training", dynamic_ncols=True)
        for iteration in pbar:
            model.train()
            data, target = next(iter(train_dataloader))
            data = data.to(device)
            target = target.to(device)
            loss = train_step(model, data, target, optimizer, criterion)

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