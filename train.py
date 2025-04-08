import os
import argparse
import torch
import numpy as np
import random
from datetime import datetime
import matplotlib.pyplot as plt

from data_loader import get_data_loaders
from models import UNet
from federated_domain_adaptation import FederatedDomainAdaptation
from evaluation import SegmentationEvaluator

def set_seed(seed):
    """
    Set random seed for reproducibility
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def parse_args():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(description='Federated Domain Adaptation for Retinal Vessel Segmentation')
    
    # Data parameters
    parser.add_argument('--data_path', type=str, default='./RentinaDA', help='Path to the dataset directory')
    parser.add_argument('--datasets', type=str, nargs='+', default=['DRIVE', 'STARE', 'CHASDB', 'HRF', 'LES-AV', 'RAVIR'], 
                        help='List of datasets to use')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training')
    parser.add_argument('--num_rounds', type=int, default=50, help='Number of federated learning rounds')
    parser.add_argument('--local_epochs', type=int, default=5, help='Number of local epochs per round')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--eval_every', type=int, default=5, help='Evaluate global model every N rounds')
    
    # Model parameters
    parser.add_argument('--lambda_adv', type=float, default=0.1, help='Weight for adversarial loss')
    parser.add_argument('--lambda_distill', type=float, default=0.5, help='Weight for distillation loss')
    parser.add_argument('--lambda_mmd', type=float, default=0.1, help='Weight for MMD loss')
    
    # Other parameters
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID to use')
    parser.add_argument('--save_dir', type=str, default='./results', help='Directory to save results')
    parser.add_argument('--exp_name', type=str, default=None, help='Experiment name')
    
    return parser.parse_args()

def main():
    # Parse arguments
    args = parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Set device
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create experiment directory
    if args.exp_name is None:
        args.exp_name = f"fedda_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    save_dir = os.path.join(args.save_dir, args.exp_name)
    os.makedirs(save_dir, exist_ok=True)
    
    # Load datasets
    print("Loading datasets...")
    data_loaders = get_data_loaders(
        base_path=args.data_path,
        dataset_names=args.datasets,
        batch_size=args.batch_size,
        num_workers=4,
        augment=True
    )
    
    # Create federated learning configuration
    fed_args = {
        'lr': args.lr,
        'num_rounds': args.num_rounds,
        'local_epochs': args.local_epochs,
        'eval_every': args.eval_every,
        'lambda_adv': args.lambda_adv,
        'lambda_distill': args.lambda_distill,
        'lambda_mmd': args.lambda_mmd
    }
    
    # Initialize federated learning system
    print("Initializing federated learning system...")
    fed_system = FederatedDomainAdaptation(
        datasets=data_loaders,
        device=device,
        args=fed_args
    )
    
    # Train federated learning system
    print("Starting federated learning training...")
    metrics = fed_system.train(args.num_rounds)
    
    # Save models
    print("Saving models...")
    fed_system.save_models(save_dir)
    
    # Evaluate final models
    print("\nEvaluating final models...")
    evaluator = SegmentationEvaluator(device)
    
    # Evaluate global model
    print("\nEvaluating global model:")
    global_results = evaluator.evaluate_all_datasets(
        model=fed_system.global_model,
        datasets=data_loaders
    )
    
    # Save global model results
    evaluator.plot_metrics(
        results=global_results,
        save_path=os.path.join(save_dir, 'global_model_metrics.png')
    )
    
    # Evaluate local models
    local_results = {}
    for client_id in data_loaders.keys():
        print(f"\nEvaluating local model for {client_id}:")
        local_model = fed_system.local_models[client_id]
        
        # Evaluate on all datasets to measure generalization
        client_results = evaluator.evaluate_all_datasets(
            model=local_model,
            datasets=data_loaders
        )
        
        local_results[client_id] = client_results
        
        # Save segmentation results for visual inspection
        evaluator.save_segmentation_results(
            model=local_model,
            data_loader=data_loaders[client_id]['test'],
            save_dir=os.path.join(save_dir, f'segmentation_results_{client_id}')
        )
    
    # Plot training curves
    plt.figure(figsize=(10, 5))
    plt.plot(metrics['val_loss'], label='Validation Loss')
    plt.xlabel('Evaluation Round')
    plt.ylabel('Loss')
    plt.title('Validation Loss During Training')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'training_loss.png'))
    
    # Plot Dice scores over rounds
    plt.figure(figsize=(12, 6))
    for client_id in data_loaders.keys():
        dice_scores = [metrics['test_metrics'][round]['dice_scores'][client_id] 
                      for round in metrics['test_metrics'].keys()]
        plt.plot(list(metrics['test_metrics'].keys()), dice_scores, label=client_id)
    
    plt.xlabel('Round')
    plt.ylabel('Dice Score')
    plt.title('Dice Scores During Training')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'dice_scores.png'))
    
    print(f"\nTraining completed. Results saved to {save_dir}")

if __name__ == '__main__':
    main()