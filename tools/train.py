import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.amp import autocast

import numpy as np
import os
from tqdm import tqdm
import argparse

from src.model import Classifier
from src.dataset import TrainDataset, ImageTransform, InfiniteSampler, EvalDataset
from configs import Config
from src.loss import LossFactory
from .utils import get_optimizer, get_scheduler, save_checkpoint, save_final_model, text_to_number


torch.random.manual_seed(0)
np.random.seed(0)

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
        train_dataset = TrainDataset(cfg=dataset_cfg['train'], transform=transform, phase='train')
        train_dataloader = DataLoader(train_dataset, batch_size=total_batch_size, shuffle=True, num_workers=8)

    else:
        train_dataset = TrainDataset(cfg=dataset_cfg['train'], transform=transform, phase='train')
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

def train(cfg, args):
    
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

    eval_interval = cfg.get('eval_step', 100)
    
    model_config = cfg['model']
    optimizer_config = cfg['optimizer']
    scheduler_config = cfg.get('scheduler', None)
    loss_config = cfg['loss']
    num_classes = cfg['model']['num_classes']

    do_eval = cfg.get('do_eval', False)

    model = Classifier(**model_config).to(device)

    train_dataloader = dataset_dict['train_loader']
    eval_dataloader = dataset_dict['eval_loader']
    class_to_idx = dataset_dict['class_to_idx']
    samples_per_class = dataset_dict['samples_per_class']

    available_gpus = min(cfg['num_gpu'], torch.cuda.device_count())

    if available_gpus > 1:
        device_ids = list(range(available_gpus))
        model = nn.DataParallel(model, device_ids=device_ids)
    
    if model_config['freeze_backbone']:
        backbone_params = None
    else:
        backbone_params = model.module.backbone.parameters() if isinstance(model, nn.DataParallel) else model.backbone.parameters()
    
    head_params = model.module.head.parameters() if isinstance(model, nn.DataParallel) else model.head.parameters()
    
    optimizer = get_optimizer(optimizer_config, backbone_params, head_params)

    if scheduler_config is not None:
        scheduler = get_scheduler(optimizer, scheduler_config, num_iterations)
    else:
        scheduler = None

    loss_factory = LossFactory(loss_config, samples_per_class, num_classes=num_classes)
    criterion = loss_factory.get_loss()

    output_dir = args.output_dir
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
                if scheduler is not None:
                    scheduler.step()
            
            if do_eval:
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

            if do_eval and iteration % eval_interval == 0 :
                val_loss, val_accuracy = evaluate(model, eval_dataloader, criterion, device)
                tqdm.write(f"Evaluation at iteration {iteration}: Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")

                save_checkpoint(model, optimizer, scheduler, 0, iteration, output_dir, cfg, class_to_idx, is_best=(val_accuracy > best_val_accuracy))
                if val_accuracy > best_val_accuracy:
                    best_val_accuracy = val_accuracy

            if scheduler is not None:
                scheduler.step()

    save_final_model(model, optimizer, scheduler, output_dir, cfg, class_to_idx)
    print(f"Final model saved to {os.path.join(output_dir, 'final_model')}")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, required=True, help="Path to config file")

    args = parser.parse_args()

    cfg = Config().get_cfg()

    train(cfg, args)