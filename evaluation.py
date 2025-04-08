import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, jaccard_score
import matplotlib.pyplot as plt
import os
from PIL import Image
import cv2

class SegmentationEvaluator:
    """
    Evaluator for retinal vessel segmentation models
    """
    def __init__(self, device):
        """
        Initialize the evaluator
        
        Args:
            device (torch.device): Device to run evaluation on
        """
        self.device = device
    
    def evaluate(self, model, data_loader, threshold=0.5):
        """
        Evaluate a model on a dataset
        
        Args:
            model (nn.Module): Model to evaluate
            data_loader (DataLoader): Data loader for evaluation
            threshold (float): Threshold for binary segmentation
        
        Returns:
            dict: Dictionary of evaluation metrics
        """
        model.eval()
        
        dice_scores = []
        iou_scores = []
        accuracy_scores = []
        precision_scores = []
        recall_scores = []
        specificity_scores = []
        f1_scores = []
        
        with torch.no_grad():
            for batch in data_loader:
                images = batch['image'].to(self.device)
                masks = batch['mask'].to(self.device)
                
                # Forward pass
                outputs, _, _ = model(images)
                
                # Apply sigmoid and threshold
                probs = torch.sigmoid(outputs)
                preds = (probs > threshold).float()
                
                # Calculate metrics for each image in batch
                for i in range(images.size(0)):
                    pred = preds[i, 0].cpu().numpy()
                    mask = masks[i, 0].cpu().numpy()
                    
                    # Calculate Dice coefficient
                    dice = self._dice_coefficient(pred, mask)
                    dice_scores.append(dice)
                    
                    # Calculate IoU (Jaccard index)
                    iou = self._iou(pred, mask)
                    iou_scores.append(iou)
                    
                    # Flatten arrays for sklearn metrics
                    pred_flat = pred.flatten()
                    mask_flat = mask.flatten()
                    
                    # Calculate accuracy
                    acc = accuracy_score(mask_flat, pred_flat)
                    accuracy_scores.append(acc)
                    
                    # Calculate precision
                    prec = precision_score(mask_flat, pred_flat, zero_division=1)
                    precision_scores.append(prec)
                    
                    # Calculate recall (sensitivity)
                    rec = recall_score(mask_flat, pred_flat, zero_division=1)
                    recall_scores.append(rec)
                    
                    # Calculate specificity
                    tn = np.sum((pred_flat == 0) & (mask_flat == 0))
                    fp = np.sum((pred_flat == 1) & (mask_flat == 0))
                    spec = tn / (tn + fp + 1e-8)
                    specificity_scores.append(spec)
                    
                    # Calculate F1 score
                    f1 = f1_score(mask_flat, pred_flat, zero_division=1)
                    f1_scores.append(f1)
        
        # Calculate mean metrics
        metrics = {
            'dice': np.mean(dice_scores),
            'iou': np.mean(iou_scores),
            'accuracy': np.mean(accuracy_scores),
            'precision': np.mean(precision_scores),
            'recall': np.mean(recall_scores),
            'specificity': np.mean(specificity_scores),
            'f1': np.mean(f1_scores)
        }
        
        return metrics
    
    def _dice_coefficient(self, pred, target):
        """
        Calculate Dice coefficient
        
        Args:
            pred (np.ndarray): Predicted binary mask
            target (np.ndarray): Ground truth binary mask
        
        Returns:
            float: Dice coefficient
        """
        smooth = 1e-8
        intersection = np.sum(pred * target)
        return (2. * intersection + smooth) / (np.sum(pred) + np.sum(target) + smooth)
    
    def _iou(self, pred, target):
        """
        Calculate IoU (Jaccard index)
        
        Args:
            pred (np.ndarray): Predicted binary mask
            target (np.ndarray): Ground truth binary mask
        
        Returns:
            float: IoU score
        """
        smooth = 1e-8
        intersection = np.sum(pred * target)
        union = np.sum(pred) + np.sum(target) - intersection
        return (intersection + smooth) / (union + smooth)
    
    def evaluate_all_datasets(self, model, datasets, threshold=0.5):
        """
        Evaluate a model on multiple datasets
        
        Args:
            model (nn.Module): Model to evaluate
            datasets (dict): Dictionary of datasets
            threshold (float): Threshold for binary segmentation
        
        Returns:
            dict: Dictionary of evaluation metrics for each dataset
        """
        results = {}
        
        for dataset_name, loaders in datasets.items():
            test_loader = loaders['test']
            metrics = self.evaluate(model, test_loader, threshold)
            results[dataset_name] = metrics
            
            print(f"\nResults for {dataset_name}:")
            for metric_name, value in metrics.items():
                print(f"{metric_name}: {value:.4f}")
        
        # Calculate average metrics across all datasets
        avg_metrics = {}
        for metric in results[list(results.keys())[0]].keys():
            avg_metrics[metric] = np.mean([results[dataset][metric] for dataset in results.keys()])
        
        results['average'] = avg_metrics
        
        print(f"\nAverage results across all datasets:")
        for metric_name, value in avg_metrics.items():
            print(f"{metric_name}: {value:.4f}")
        
        return results
    
    def save_segmentation_results(self, model, data_loader, save_dir, threshold=0.5):
        """
        Save segmentation results as images
        
        Args:
            model (nn.Module): Model to evaluate
            data_loader (DataLoader): Data loader for evaluation
            save_dir (str): Directory to save results
            threshold (float): Threshold for binary segmentation
        """
        os.makedirs(save_dir, exist_ok=True)
        model.eval()
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                images = batch['image'].to(self.device)
                masks = batch['mask'].to(self.device)
                image_paths = batch['image_path']
                
                # Forward pass
                outputs, _, _ = model(images)
                
                # Apply sigmoid and threshold
                probs = torch.sigmoid(outputs)
                preds = (probs > threshold).float()
                
                # Save results for each image in batch
                for i in range(images.size(0)):
                    # Get image name from path
                    image_name = os.path.basename(image_paths[i]).split('.')[0]
                    
                    # Convert tensors to numpy arrays
                    image = images[i].permute(1, 2, 0).cpu().numpy() * 255
                    image = image.astype(np.uint8)
                    
                    mask = masks[i, 0].cpu().numpy() * 255
                    mask = mask.astype(np.uint8)
                    
                    pred = preds[i, 0].cpu().numpy() * 255
                    pred = pred.astype(np.uint8)
                    
                    # Create overlay of prediction on image
                    overlay = image.copy()
                    overlay[pred > 0] = [255, 0, 0]  # Red color for predicted vessels
                    
                    # Create comparison image (original, ground truth, prediction, overlay)
                    h, w = image.shape[:2]
                    comparison = np.zeros((h, w * 4, 3), dtype=np.uint8)
                    
                    # Original image
                    comparison[:, :w] = image
                    
                    # Ground truth mask (convert to RGB)
                    mask_rgb = np.zeros((h, w, 3), dtype=np.uint8)
                    mask_rgb[:, :, 0] = mask
                    mask_rgb[:, :, 1] = mask
                    mask_rgb[:, :, 2] = mask
                    comparison[:, w:w*2] = mask_rgb
                    
                    # Prediction mask (convert to RGB)
                    pred_rgb = np.zeros((h, w, 3), dtype=np.uint8)
                    pred_rgb[:, :, 0] = pred
                    pred_rgb[:, :, 1] = pred
                    pred_rgb[:, :, 2] = pred
                    comparison[:, w*2:w*3] = pred_rgb
                    
                    # Overlay
                    comparison[:, w*3:w*4] = overlay
                    
                    # Save comparison image
                    save_path = os.path.join(save_dir, f"{image_name}_comparison.png")
                    cv2.imwrite(save_path, cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR))
    
    def plot_metrics(self, results, save_path=None):
        """
        Plot evaluation metrics for multiple datasets
        
        Args:
            results (dict): Dictionary of evaluation metrics for each dataset
            save_path (str, optional): Path to save the plot
        """
        # Exclude 'average' from plotting individual datasets
        datasets = [dataset for dataset in results.keys() if dataset != 'average']
        metrics = list(results[datasets[0]].keys())
        
        # Create figure with subplots for each metric
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.flatten()
        
        # Plot each metric
        for i, metric in enumerate(metrics):
            if i < len(axes):
                values = [results[dataset][metric] for dataset in datasets]
                axes[i].bar(datasets, values)
                axes[i].set_title(metric.capitalize())
                axes[i].set_ylim(0, 1)
                axes[i].set_xticklabels(datasets, rotation=45, ha='right')
                
                # Add average line
                if 'average' in results:
                    avg_value = results['average'][metric]
                    axes[i].axhline(y=avg_value, color='r', linestyle='--', label=f'Avg: {avg_value:.3f}')
                    axes[i].legend()
        
        # Remove any unused subplots
        for i in range(len(metrics), len(axes)):
            fig.delaxes(axes[i])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"Metrics plot saved to {save_path}")
        
        plt.show()