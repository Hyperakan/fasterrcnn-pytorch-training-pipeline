import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
from pathlib import Path
import argparse

def plot_class_metrics(stats, classes, output_dir):
    """Plot per-class AP and AR metrics."""
    plt.figure(figsize=(12, 6))
    
    # Get per-class metrics
    ap_values = np.array(stats['map_per_class'])
    ar_values = np.array(stats['mar_100_per_class'])
    
    # Skip the background class (index 0)
    ap_values = ap_values[1:]
    ar_values = ar_values[1:]
    classes = classes[1:]  # Skip background class
    
    # Create bar positions
    x = np.arange(len(classes))
    width = 0.35
    
    # Plot bars
    plt.bar(x - width/2, ap_values, width, label='AP', color='skyblue')
    plt.bar(x + width/2, ar_values, width, label='AR', color='lightcoral')
    
    # Customize plot
    plt.xlabel('Classes')
    plt.ylabel('Score')
    plt.title('Per-class AP and AR Scores')
    plt.xticks(x, classes, rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    
    # Save plot
    plt.savefig(Path(output_dir) / 'class_metrics.png')
    plt.close()

def plot_size_metrics(stats, output_dir):
    """Plot mAP for different object sizes."""
    plt.figure(figsize=(10, 6))
    
    # Get size metrics
    sizes = ['small', 'medium', 'large']
    map_values = [stats[f'map_{size}'] for size in sizes]
    
    # Create bar plot
    plt.bar(sizes, map_values, color=['lightblue', 'lightgreen', 'lightcoral'])
    
    # Customize plot
    plt.xlabel('Object Size')
    plt.ylabel('mAP')
    plt.title('mAP for Different Object Sizes')
    
    # Add value labels on top of bars
    for i, v in enumerate(map_values):
        plt.text(i, v, f'{v:.3f}', ha='center', va='bottom')
    
    # Save plot
    plt.savefig(Path(output_dir) / 'size_metrics.png')
    plt.close()

def create_summary_table(stats, classes, output_dir):
    """Create a summary table of all metrics."""
    # Skip background class for per-class metrics
    class_metrics = {
        class_name: {
            'AP': float(ap),
            'AR': float(ar)
        }
        for class_name, ap, ar in zip(classes[1:], stats['map_per_class'][1:], stats['mar_100_per_class'][1:])
    }
    
    summary = {
        'Overall Metrics': {
            'mAP': float(stats['map']),
            'mAR': float(stats['mar_100']),
            'mAP (IoU=0.5)': float(stats['map_50']),
            'mAP (IoU=0.75)': float(stats['map_75'])
        },
        'Per-class Metrics': class_metrics
    }
    
    # Save summary to JSON
    with open(Path(output_dir) / 'metrics_summary.json', 'w') as f:
        json.dump(summary, f, indent=4)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stats', required=True, help='Path to the evaluation stats JSON file')
    parser.add_argument('--classes', required=True, help='Path to the classes list file')
    parser.add_argument('--output', default='visualization_output', help='Output directory for visualizations')
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load stats and classes
    with open(args.stats) as f:
        stats = json.load(f)
    with open(args.classes) as f:
        classes = json.load(f)
    
    # Create visualizations
    plot_class_metrics(stats, classes, output_dir)
    plot_size_metrics(stats, output_dir)
    create_summary_table(stats, classes, output_dir)
    
    print(f"Visualizations saved to {output_dir}")

if __name__ == '__main__':
    main() 