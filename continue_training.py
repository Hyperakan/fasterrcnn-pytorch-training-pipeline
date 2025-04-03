"""
Continue training a Faster R-CNN model from a checkpoint.

USAGE:
python continue_training.py --weights path/to/checkpoint.pth --data data_configs/custom_data.yaml --model fasterrcnn_resnet50_fpn_v2 --epochs 15
"""
import torch
import argparse
import yaml
import os
import sys
from pathlib import Path

# Add the parent directory to the path so we can import from the project
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train import main as train_main
from utils.general import set_training_dir, SaveBestModel

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--weights', 
        required=True,
        help='path to the checkpoint to continue training from'
    )
    parser.add_argument(
        '--data', 
        required=True,
        help='path to the data config file'
    )
    parser.add_argument(
        '--model', 
        default='fasterrcnn_resnet50_fpn_v2',
        help='model name'
    )
    parser.add_argument(
        '--epochs', 
        default=15, 
        type=int,
        help='number of additional epochs to train for'
    )
    parser.add_argument(
        '--device', 
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='computation/training device'
    )
    parser.add_argument(
        '--batch', 
        default=4, 
        type=int,
        help='batch size for training'
    )
    parser.add_argument(
        '--lr', 
        default=0.001,
        type=float,
        help='learning rate for the optimizer'
    )
    parser.add_argument(
        '--imgsz', 
        default=640, 
        type=int,
        help='image size to feed to the network'
    )
    parser.add_argument(
        '--name', 
        default=None, 
        type=str,
        help='name for the output directory'
    )
    parser.add_argument(
        '--cosine-annealing', 
        dest='cosine_annealing', 
        action='store_true',
        help='use cosine annealing warm restarts'
    )
    return vars(parser.parse_args())

def main(args):
    # Load the checkpoint to get the current epoch
    checkpoint = torch.load(args['weights'], map_location=args['device'])
    current_epoch = checkpoint.get('epoch', 0)
    
    # Calculate the total number of epochs (current + additional)
    total_epochs = current_epoch + args['epochs']
    
    # Set up the training arguments
    train_args = {
        'model': args['model'],
        'data': args['data'],
        'device': args['device'],
        'epochs': total_epochs,  # Set to total epochs
        'workers': 4,
        'batch': args['batch'],
        'lr': args['lr'],
        'imgsz': args['imgsz'],
        'name': args['name'],
        'vis_transformed': False,
        'mosaic': 0.0,
        'use_train_aug': False,
        'cosine_annealing': args['cosine_annealing'],
        'weights': args['weights'],  # Path to the checkpoint
        'resume_training': True,  # Important: set to True to resume training
        'sync_bn': False,
        'distributed': False,
        'gpu': 0,
        'world_size': 1,
        'rank': -1,
        'dist_url': 'env://',
        'dist_backend': 'nccl'
    }
    
    # Create output directory
    if train_args['name'] is None:
        train_args['name'] = f"continued_training_{Path(args['weights']).stem}"
    
    # Run the training
    train_main(train_args)
    
    print(f"Training completed. Model trained for a total of {total_epochs} epochs.")
    print(f"Checkpoint saved in the outputs/training/{train_args['name']} directory.")

if __name__ == '__main__':
    args = parse_args()
    main(args) 