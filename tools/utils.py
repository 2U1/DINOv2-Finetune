import os
import torch
import json
import torch.optim as optim
import glob

def text_to_number(text):
    units = {"k": 1000, "m": 1000000, "b": 1000000000}
    if text[-1].lower() in units:
        return int(float(text[:-1]) * units[text[-1].lower()])
    else:
        return int(text)
    
def save_final_model(model, optimizer, scheduler, save_dir, cfg, class_to_idx):
    final_dir = os.path.join(save_dir, "final_model")
    os.makedirs(final_dir, exist_ok=True)

    training_args = {
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'class_to_idx': class_to_idx
    }

    final_model_path = os.path.join(final_dir, "dino_model.bin")
    torch.save(model.state_dict(), final_model_path)
    torch.save(training_args, os.path.join(final_dir, "training_args.bin"))

    with open(os.path.join(final_dir, "class_to_idx.json"), 'w') as f:
        json.dump(class_to_idx, f, indent=4)

    config_path = os.path.join(final_dir, "config.json")
    with open(config_path, 'w') as f:
        json.dump(cfg, f, indent=4)
    
def save_checkpoint(model, optimizer, scheduler, epoch, iteration, save_dir, cfg, class_to_idx, is_best=False):
    max_checkpoints = cfg.get('max_checkpoints', 5)
    checkpoint_dir = os.path.join(save_dir, f"checkpoint-{iteration}")
    os.makedirs(checkpoint_dir, exist_ok=True)

    training_args = {
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'epoch': epoch,
        'iteration': iteration
    }

    checkpoint_path = os.path.join(checkpoint_dir, "dino_model.bin")
    torch.save(model.state_dict(), checkpoint_path)
    torch.save(training_args, os.path.join(checkpoint_dir, "training_args.bin"))

    with open(os.path.join(checkpoint_dir, "class_to_idx.json"), 'w') as f:
        json.dump(class_to_idx, f, indent=4)

    config_path = os.path.join(checkpoint_dir, "config.json")
    with open(config_path, 'w') as f:
        json.dump(cfg, f, indent=4)

    if is_best:
        best_checkpoint_dir = os.path.join(save_dir, f"best_checkpoint-{iteration}")
        os.makedirs(best_checkpoint_dir, exist_ok=True)
        best_checkpoint_path = os.path.join(best_checkpoint_dir, "dino_model.bin")
        torch.save(model.state_dict(), best_checkpoint_path)
        torch.save(training_args, os.path.join(best_checkpoint_dir, "training_args.bin"))

        with open(os.path.join(best_checkpoint_dir, "class_to_idx.json"), 'w') as f:
            json.dump(class_to_idx, f, indent=4)

        best_config_path = os.path.join(best_checkpoint_dir, "config.json")
        with open(best_config_path, 'w') as f:
            json.dump(cfg, f, indent=4)

        manage_best_checkpoints(save_dir)

    manage_checkpoints(save_dir, max_checkpoints)


def manage_checkpoints(save_dir, max_checkpoints):
    checkpoint_dirs = sorted(glob.glob(os.path.join(save_dir, "checkpoint-*")), key=os.path.getmtime)

    if len(checkpoint_dirs) > max_checkpoints:
        num_to_remove = len(checkpoint_dirs) - max_checkpoints
        for i in range(num_to_remove):
            old_checkpoint_dir = checkpoint_dirs[i]
            print(f"Deleting old checkpoint: {old_checkpoint_dir}")
            os.system(f"rm -rf {old_checkpoint_dir}")

def manage_best_checkpoints(save_dir):
    # Get all directories with the "best_checkpoint-" prefix
    best_checkpoint_dirs = sorted(glob.glob(os.path.join(save_dir, "best_checkpoint-*")), key=os.path.getmtime)

    # Keep only the most recent best checkpoint, delete the rest
    if len(best_checkpoint_dirs) > 1:
        for old_best_dir in best_checkpoint_dirs[:-1]:  # Keep the last (most recent) one
            print(f"Deleting old best checkpoint: {old_best_dir}")
            os.system(f"rm -rf {old_best_dir}")
    

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