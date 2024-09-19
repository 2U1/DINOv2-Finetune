import argparse
import json
import os
import os.path as osp

def calculate_class_stats(dataset_path):
    class_stats = {}
    for class_name in os.listdir(dataset_path):
        class_path = osp.join(dataset_path, class_name)
        if osp.isdir(class_path):
            image_files = [f for f in os.listdir(class_path) if osp.isfile(osp.join(class_path, f))]
            class_stats[class_name] = len(image_files)
    
    return class_stats

def parse_args():
    parser = argparse.ArgumentParser(description='Calculate image class stats for a classification dataset')
    parser.add_argument('dataset_path', help='Path to the image classification dataset')
    args = parser.parse_args()
    return args

def save_class_stats(out_dir, class_stats):
    os.makedirs(out_dir, exist_ok=True)
    with open(osp.join(out_dir, 'class_stats.json'), 'w') as of:
        json.dump(class_stats, of, indent=2)

def main():
    args = parse_args()
    dataset_path = args.dataset_path

    class_stats = calculate_class_stats(dataset_path)
    save_class_stats(dataset_path, class_stats)

if __name__ == '__main__':
    main()