"""
Plot evaluation metrics (F1, Precision, Recall, PR curves) for trained Faster R-CNN model.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import yaml
import argparse
from pathlib import Path

from datasets import create_valid_dataset, create_valid_loader
from models.create_fasterrcnn_model import create_model
from utils.general import set_training_dir

def calculate_metrics(pred_boxes, pred_scores, pred_labels, true_boxes, true_labels, iou_threshold=0.5):
    """Calculate precision and recall for different confidence thresholds."""
    # Sort predictions by confidence score
    sorted_indices = torch.argsort(pred_scores, descending=True)
    pred_boxes = pred_boxes[sorted_indices]
    pred_scores = pred_scores[sorted_indices]
    pred_labels = pred_labels[sorted_indices]
    
    # Initialize metrics
    precisions = []
    recalls = []
    f1_scores = []
    thresholds = []
    
    # Calculate metrics for different confidence thresholds
    for threshold in np.arange(0, 1.1, 0.1):
        # Filter predictions by confidence threshold
        mask = pred_scores >= threshold
        filtered_boxes = pred_boxes[mask]
        filtered_labels = pred_labels[mask]
        
        # Calculate IoU between predictions and ground truth
        tp = 0
        fp = 0
        fn = len(true_boxes)
        
        if len(filtered_boxes) > 0:
            ious = box_iou(filtered_boxes, true_boxes)
            max_ious, matched_gt_idx = torch.max(ious, dim=1)
            
            # Count true positives and false positives
            for i, (iou, pred_label) in enumerate(zip(max_ious, filtered_labels)):
                if iou >= iou_threshold and pred_label == true_labels[matched_gt_idx[i]]:
                    tp += 1
                    fn -= 1
                else:
                    fp += 1
        
        # Calculate precision and recall
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        # Calculate F1 score
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
        thresholds.append(threshold)
    
    return np.array(precisions), np.array(recalls), np.array(f1_scores), np.array(thresholds)

def box_iou(boxes1, boxes2):
    """Calculate IoU between two sets of boxes."""
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    
    union = area1[:, None] + area2 - inter
    
    return inter / union

def plot_metrics(precisions, recalls, f1_scores, thresholds, out_dir):
    """Plot evaluation metrics."""
    # Create output directory if it doesn't exist
    metrics_dir = Path(out_dir) / 'metrics'
    metrics_dir.mkdir(exist_ok=True)
    
    # Plot precision, recall, and F1 scores vs threshold
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, precisions, 'b-', label='Precision')
    plt.plot(thresholds, recalls, 'r-', label='Recall')
    plt.plot(thresholds, f1_scores, 'g-', label='F1 Score')
    plt.xlabel('Confidence Threshold')
    plt.ylabel('Score')
    plt.title('Precision, Recall, and F1 Score vs Confidence Threshold')
    plt.legend()
    plt.grid(True)
    plt.savefig(metrics_dir / 'metrics_vs_threshold.png')
    plt.close()
    
    # Plot precision-recall curve
    plt.figure(figsize=(10, 6))
    plt.plot(recalls, precisions, 'b-')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.grid(True)
    plt.savefig(metrics_dir / 'precision_recall_curve.png')
    plt.close()

def main(args):
    # Load the data configurations
    with open(args.data) as f:
        data_configs = yaml.safe_load(f)
    
    # Create validation dataset and loader
    valid_dataset = create_valid_dataset(
        data_configs['VALID_DIR_IMAGES'],
        data_configs['VALID_DIR_LABELS'],
        args.imgsz,
        data_configs['CLASSES']
    )
    valid_loader = create_valid_loader(valid_dataset, 1, num_workers=4)
    
    # Load model
    checkpoint = torch.load(args.weights, map_location=args.device)
    num_classes = len(data_configs['CLASSES'])
    
    # Create model and load weights
    model = create_model[args.model](num_classes=num_classes, pretrained=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(args.device)
    model.eval()
    
    # Lists to store all predictions and ground truths
    all_pred_boxes = []
    all_pred_scores = []
    all_pred_labels = []
    all_true_boxes = []
    all_true_labels = []
    
    print('Evaluating model...')
    with torch.no_grad():
        for images, targets in tqdm(valid_loader):
            images = list(img.to(args.device) for img in images)
            
            # Get predictions
            outputs = model(images)
            
            # Store predictions and ground truths
            for output, target in zip(outputs, targets):
                pred_boxes = output['boxes']
                pred_scores = output['scores']
                pred_labels = output['labels']
                
                true_boxes = target['boxes'].to(args.device)
                true_labels = target['labels'].to(args.device)
                
                if len(pred_boxes) > 0 and len(true_boxes) > 0:
                    all_pred_boxes.append(pred_boxes)
                    all_pred_scores.append(pred_scores)
                    all_pred_labels.append(pred_labels)
                    all_true_boxes.append(true_boxes)
                    all_true_labels.append(true_labels)
    
    # Concatenate all predictions and ground truths
    all_pred_boxes = torch.cat(all_pred_boxes)
    all_pred_scores = torch.cat(all_pred_scores)
    all_pred_labels = torch.cat(all_pred_labels)
    all_true_boxes = torch.cat(all_true_boxes)
    all_true_labels = torch.cat(all_true_labels)
    
    # Calculate metrics
    precisions, recalls, f1_scores, thresholds = calculate_metrics(
        all_pred_boxes, all_pred_scores, all_pred_labels,
        all_true_boxes, all_true_labels
    )
    
    # Plot metrics
    out_dir = Path(args.weights).parent
    plot_metrics(precisions, recalls, f1_scores, thresholds, out_dir)
    print(f'Plots saved in {out_dir}/metrics/')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', required=True, help='path to model weights')
    parser.add_argument('--data', required=True, help='path to data config file')
    parser.add_argument('--model', default='fasterrcnn_resnet50_fpn_v2', help='model name')
    parser.add_argument('--imgsz', default=640, type=int, help='image size')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', help='device')
    args = parser.parse_args()
    
    main(args) 