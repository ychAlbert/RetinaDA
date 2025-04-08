import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from data_loader import get_data_loaders
from models import UNet
from evaluation import SegmentationEvaluator

def parse_args():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(description='Test Retinal Vessel Segmentation Models')
    
    # Data parameters
    parser.add_argument('--data_path', type=str, default='./RentinaDA', help='Path to the dataset directory')
    parser.add_argument('--datasets', type=str, nargs='+', default=['DRIVE', 'STARE', 'CHASDB', 'HRF', 'LES-AV', 'RAVIR'], 
                        help='List of datasets to test on')
    parser.add_argument('--test_dataset', type=str, default=None, help='Specific dataset to test on (if None, test on all)')
    
    # Model parameters
    parser.add_argument('--model_path', type=str, required=True, help='Path to the saved model')
    parser.add_argument('--model_type', type=str, default='global', choices=['global', 'local'], 
                        help='Type of model to load (global or local)')
    parser.add_argument('--client_id', type=str, default=None, help='Client ID for local model (required if model_type is local)')
    
    # Testing parameters
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for testing')
    parser.add_argument('--threshold', type=float, default=0.5, help='Threshold for binary segmentation')
    
    # Output parameters
    parser.add_argument('--save_dir', type=str, default='./results/test_results', help='Directory to save results')
    parser.add_argument('--save_images', action='store_true', help='Save segmentation results as images')
    
    # Other parameters
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID to use')
    
    return parser.parse_args()

def main():
    # Parse arguments
    args = parse_args()
    
    # Set device
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Load datasets
    print("Loading datasets...")
    if args.test_dataset is not None:
        # Test on a specific dataset
        dataset_names = [args.test_dataset]
    else:
        # Test on all specified datasets
        dataset_names = args.datasets
    
    data_loaders = get_data_loaders(
        base_path=args.data_path,
        dataset_names=dataset_names,
        batch_size=args.batch_size,
        num_workers=4,
        augment=False  # No augmentation for testing
    )
    
    # Initialize model
    print("Initializing model...")
    model = UNet(
        n_channels=3,
        n_classes=1,
        bilinear=True,
        with_domain_features=True
    ).to(device)
    
    # Load model weights
    print(f"Loading model from {args.model_path}...")
    if args.model_type == 'global':
        model_file = os.path.join(args.model_path, 'global_model.pth')
    else:  # local model
        if args.client_id is None:
            raise ValueError("Client ID must be specified for local model")
        model_file = os.path.join(args.model_path, f'local_model_{args.client_id}.pth')
    
    model.load_state_dict(torch.load(model_file, map_location=device))
    model.eval()
    
    # Initialize evaluator
    evaluator = SegmentationEvaluator(device)
    
    # Evaluate model on datasets
    print("Evaluating model...")
    results = evaluator.evaluate_all_datasets(
        model=model,
        datasets=data_loaders,
        threshold=args.threshold
    )
    
    # Save results
    model_name = 'global' if args.model_type == 'global' else f'local_{args.client_id}'
    results_file = os.path.join(args.save_dir, f'{model_name}_results.txt')
    
    with open(results_file, 'w') as f:
        f.write(f"Results for {model_name} model:\n")
        for dataset_name, metrics in results.items():
            f.write(f"\n{dataset_name}:\n")
            for metric_name, value in metrics.items():
                f.write(f"{metric_name}: {value:.4f}\n")
    
    print(f"Results saved to {results_file}")
    
    # Plot metrics
    evaluator.plot_metrics(
        results=results,
        save_path=os.path.join(args.save_dir, f'{model_name}_metrics.png')
    )
    
    # Save segmentation results as images if requested
    if args.save_images:
        print("Saving segmentation results...")
        for dataset_name, loaders in data_loaders.items():
            save_dir = os.path.join(args.save_dir, f'{model_name}_{dataset_name}_segmentations')
            evaluator.save_segmentation_results(
                model=model,
                data_loader=loaders['test'],
                save_dir=save_dir,
                threshold=args.threshold
            )
    
    print("Testing completed.")

if __name__ == '__main__':
    main()