import os
import torch
from torch.utils.data import DataLoader
from src.model import Classifier
from src.dataset import ImageTransform, EvalDataset
from tqdm import tqdm
import json
import argparse

def evaluate(model, dataloader, device, k=5):
    model.eval()
    correct = 0
    total = 0
    top_k_correct = 0

    with torch.no_grad():
        for data, target in tqdm(dataloader, desc="Evaluating", leave=False):
            data = data.to(device)
            target = target.to(device)

            output = model(data)
            _, predicted = torch.max(output, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

            # Calculate top-k accuracy
            _, top_k_pred = output.topk(k, dim=1, largest=True, sorted=True)
            top_k_correct += sum([target[i] in top_k_pred[i] for i in range(target.size(0))])

    accuracy = 100 * correct / total
    top_k_accuracy = 100 * top_k_correct / total

    return accuracy, top_k_accuracy

def load_model_checkpoint(model_path):
    
    try:
        checkpoint_path = os.path.join(model_path, 'dino_model.bin')
        checkpoint = torch.load(checkpoint_path)
        checkpoint = remove_module_from_state_dict(checkpoint)

        return checkpoint

    except FileNotFoundError:
        print(f"Model checkpoint not found at {model_path}")
        exit()

def remove_module_from_state_dict(state_dict):
    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace('module.', '') if k.startswith('module.') else k
        new_state_dict[new_key] = v
    return new_state_dict

def main(args):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_path = args.model_path
    data_path = args.data_path
    batch_size = args.batch_size
    top_k = args.top_k

    cfg = json.load(open(os.path.join(model_path, 'config.json')))

    transform = ImageTransform(cfg['resize'], cfg['mean'], cfg['std'])

    # Remapping labels for evaluation dataset
    class_to_idx = json.load(open(os.path.join(model_path, 'class_to_idx.json')))

    model_cfg = cfg['model']
    
    model = Classifier(num_classes=model_cfg['num_classes'], backbone=model_cfg['backbone'], head=model_cfg['head']).to(device)
    state_dict = load_model_checkpoint(model_path)
    model.load_state_dict(state_dict)
    

    del state_dict

    if args.use_dp and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
        batch_size = batch_size * torch.cuda.device_count()

    # Eval dataset
    eval_dataset = EvalDataset(data_root=data_path, transform=transform, phase='val', class_to_idx=class_to_idx) 
    eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, num_workers=8)

    accuracy, top_k_accuracy = evaluate(model, eval_dataloader, device, k=top_k)
    
    print(f"Final Evaluation Accuracy: {accuracy:.2f}%")
    print(f"Final Top-{top_k} Accuracy: {top_k_accuracy:.2f}%")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate a trained model on a dataset")
    parser.add_argument('--model_path', type=str, required=True, help="Path to the model checkpoint")
    parser.add_argument('--data_path', type=str, required=True, help="Path to the evaluation dataset")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for evaluation")
    parser.add_argument('--top_k', type=int, default=5, help="Top-K accuracy to evaluate")
    parser.add_argument('--use_dp', action='store_true', help="Use DataParallel for multi-GPU evaluation")

    args = parser.parse_args()
    
    main(args)