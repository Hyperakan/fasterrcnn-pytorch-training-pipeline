"""
Measure inference time of Faster R-CNN model.

USAGE:
python measure_inference_time.py --weights path/to/weights.pth --data data_configs/custom_data.yaml --model fasterrcnn_resnet50_fpn_v2
"""
import torch
import time
import argparse
import yaml
from pathlib import Path
import numpy as np
from models.create_fasterrcnn_model import return_fasterrcnn_resnet50_fpn_v2, return_fasterrcnn_resnet50_fpn
from datasets import create_valid_dataset, create_valid_loader
from torch_utils import utils

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--weights', 
        required=True,
        help='path to trained weights'
    )
    parser.add_argument(
        '--data', 
        required=True,
        help='path to data config file'
    )
    parser.add_argument(
        '--model', 
        default='fasterrcnn_resnet50_fpn_v2',
        help='model name'
    )
    parser.add_argument(
        '--batch', 
        default=1, 
        type=int,
        help='batch size for inference'
    )
    parser.add_argument(
        '--device', 
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='computation/training device'
    )
    parser.add_argument(
        '--imgsz', 
        default=640, 
        type=int,
        help='image size to feed to the network'
    )
    parser.add_argument(
        '--num-warmup', 
        default=10, 
        type=int,
        help='number of warmup iterations'
    )
    parser.add_argument(
        '--num-iter', 
        default=100, 
        type=int,
        help='number of iterations to measure'
    )
    return vars(parser.parse_args())

def measure_inference_time(model, data_loader, device, num_warmup=10, num_iter=100):
    model.eval()
    
    # Warmup
    print(f"Warming up for {num_warmup} iterations...")
    with torch.no_grad():
        for i, (images, _) in enumerate(data_loader):
            if i >= num_warmup:
                break
            images = list(img.to(device) for img in images)
            _ = model(images)
    
    # Measure inference time
    print(f"Measuring inference time for {num_iter} iterations...")
    times = []
    with torch.no_grad():
        for i, (images, _) in enumerate(data_loader):
            if i >= num_iter:
                break
            images = list(img.to(device) for img in images)
            
            # Measure time
            torch.cuda.synchronize() if device == 'cuda' else None
            start_time = time.time()
            
            _ = model(images)
            
            torch.cuda.synchronize() if device == 'cuda' else None
            end_time = time.time()
            
            times.append(end_time - start_time)
    
    # Calculate statistics
    times = np.array(times)
    mean_time = np.mean(times)
    std_time = np.std(times)
    fps = 1.0 / mean_time
    
    return {
        'mean_time': mean_time,
        'std_time': std_time,
        'fps': fps,
        'times': times
    }

def main(args):
    # Load data config
    with open(args['data']) as file:
        data_configs = yaml.safe_load(file)
    
    # Create model based on the model name
    if args['model'] == 'fasterrcnn_resnet50_fpn_v2':
        model = return_fasterrcnn_resnet50_fpn_v2(
            num_classes=data_configs['NC'],
            pretrained=False
        )
    elif args['model'] == 'fasterrcnn_resnet50_fpn':
        model = return_fasterrcnn_resnet50_fpn(
            num_classes=data_configs['NC'],
            pretrained=False
        )
    else:
        raise ValueError(f"Unsupported model: {args['model']}")
    
    # Load weights
    checkpoint = torch.load(args['weights'], map_location=args['device'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(args['device'])
    model.eval()
    
    # Create dataset and dataloader
    valid_dataset = create_valid_dataset(
        data_configs['VALID_DIR_IMAGES'],
        data_configs['VALID_DIR_LABELS'],
        args['imgsz'],
        data_configs['CLASSES']
    )
    valid_loader = create_valid_loader(
        valid_dataset,
        args['batch'],
        num_workers=4
    )
    
    # Measure inference time
    stats = measure_inference_time(
        model,
        valid_loader,
        args['device'],
        args['num_warmup'],
        args['num_iter']
    )
    
    # Print results
    print("\nInference Time Statistics:")
    print(f"Mean inference time: {stats['mean_time']*1000:.2f} ms")
    print(f"Std inference time: {stats['std_time']*1000:.2f} ms")
    print(f"FPS: {stats['fps']:.2f}")
    
    # Save results
    results = {
        'model': args['model'],
        'batch_size': args['batch'],
        'image_size': args['imgsz'],
        'device': args['device'],
        'mean_time_ms': float(stats['mean_time'] * 1000),
        'std_time_ms': float(stats['std_time'] * 1000),
        'fps': float(stats['fps'])
    }
    
    # Save to file
    output_dir = Path('inference_results')
    output_dir.mkdir(exist_ok=True)
    
    import json
    with open(output_dir / 'inference_time.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"\nResults saved to {output_dir / 'inference_time.json'}")

if __name__ == '__main__':
    args = parse_args()
    main(args) 