import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import seaborn as sns
import pandas as pd
from PIL import Image
import cv2

from data_loader import get_data_loaders
from models import UNet

class FeatureVisualizer:
    """
    Utility class for visualizing features and results from federated domain adaptation
    """
    def __init__(self, device):
        """
        Initialize the visualizer
        
        Args:
            device (torch.device): Device to run visualization on
        """
        self.device = device
    
    def extract_features(self, model, data_loaders, num_samples=100):
        """
        Extract bottleneck features from model for visualization
        
        Args:
            model (nn.Module): Model to extract features from
            data_loaders (dict): Dictionary of data loaders
            num_samples (int): Maximum number of samples per dataset
        
        Returns:
            tuple: (features, labels, domain_labels)
        """
        model.eval()
        features_list = []
        labels_list = []
        domain_labels_list = []
        dataset_names = []
        
        with torch.no_grad():
            for dataset_name, loaders in data_loaders.items():
                loader = loaders['test']
                samples_count = 0
                
                for batch in loader:
                    if samples_count >= num_samples:
                        break
                    
                    images = batch['image'].to(self.device)
                    masks = batch['mask'].to(self.device)
                    domain = batch['domain']
                    
                    # Forward pass to get features
                    _, _, bottleneck_features = model(images)
                    
                    # Reshape features for visualization
                    batch_features = bottleneck_features.view(bottleneck_features.size(0), -1).cpu().numpy()
                    
                    # Get binary labels (vessel or background) from masks
                    # We'll use the average mask value as a simple label
                    batch_labels = masks.mean(dim=[1, 2, 3]).cpu().numpy()
                    
                    # Get domain labels
                    batch_domains = torch.argmax(domain, dim=1).cpu().numpy()
                    
                    # Add to lists
                    features_list.append(batch_features)
                    labels_list.append(batch_labels)
                    domain_labels_list.append(batch_domains)
                    dataset_names.extend([dataset_name] * len(batch_labels))
                    
                    samples_count += len(batch_labels)
        
        # Concatenate all features and labels
        features = np.vstack(features_list)
        labels = np.concatenate(labels_list)
        domain_labels = np.concatenate(domain_labels_list)
        
        return features, labels, domain_labels, dataset_names
    
    def visualize_tsne(self, features, domain_labels, dataset_names, save_path=None, title="t-SNE Visualization of Features"):
        """
        Visualize features using t-SNE
        
        Args:
            features (np.ndarray): Feature vectors
            domain_labels (np.ndarray): Domain labels
            dataset_names (list): List of dataset names
            save_path (str, optional): Path to save the visualization
            title (str): Title for the plot
        """
        # Apply t-SNE for dimensionality reduction
        print("Computing t-SNE embedding...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
        features_tsne = tsne.fit_transform(features)
        
        # Create a DataFrame for easier plotting
        df = pd.DataFrame({
            'x': features_tsne[:, 0],
            'y': features_tsne[:, 1],
            'domain': domain_labels,
            'dataset': dataset_names
        })
        
        # Create a mapping of domain indices to dataset names
        domain_mapping = {0: 'DRIVE', 1: 'STARE', 2: 'CHASDB', 3: 'HRF', 4: 'LES-AV', 5: 'RAVIR'}
        df['domain_name'] = df['domain'].map(lambda x: domain_mapping.get(x, f'Unknown-{x}'))
        
        # Plot using seaborn
        plt.figure(figsize=(12, 10))
        sns.scatterplot(x='x', y='y', hue='domain_name', data=df, palette='tab10', s=100, alpha=0.7)
        
        plt.title(title, fontsize=16)
        plt.xlabel('t-SNE dimension 1', fontsize=12)
        plt.ylabel('t-SNE dimension 2', fontsize=12)
        plt.legend(title='Dataset', fontsize=10)
        
        if save_path:
            plt.savefig(save_path)
            print(f"t-SNE visualization saved to {save_path}")
        
        plt.show()
    
    def visualize_pca(self, features, domain_labels, dataset_names, save_path=None, title="PCA Visualization of Features"):
        """
        Visualize features using PCA
        
        Args:
            features (np.ndarray): Feature vectors
            domain_labels (np.ndarray): Domain labels
            dataset_names (list): List of dataset names
            save_path (str, optional): Path to save the visualization
            title (str): Title for the plot
        """
        # Apply PCA for dimensionality reduction
        print("Computing PCA embedding...")
        pca = PCA(n_components=2, random_state=42)
        features_pca = pca.fit_transform(features)
        
        # Create a DataFrame for easier plotting
        df = pd.DataFrame({
            'x': features_pca[:, 0],
            'y': features_pca[:, 1],
            'domain': domain_labels,
            'dataset': dataset_names
        })
        
        # Create a mapping of domain indices to dataset names
        domain_mapping = {0: 'DRIVE', 1: 'STARE', 2: 'CHASDB', 3: 'HRF', 4: 'LES-AV', 5: 'RAVIR'}
        df['domain_name'] = df['domain'].map(lambda x: domain_mapping.get(x, f'Unknown-{x}'))
        
        # Plot using seaborn
        plt.figure(figsize=(12, 10))
        sns.scatterplot(x='x', y='y', hue='domain_name', data=df, palette='tab10', s=100, alpha=0.7)
        
        # Add variance explained
        explained_var = pca.explained_variance_ratio_
        plt.title(f"{title}\nExplained variance: {explained_var[0]:.2f}, {explained_var[1]:.2f}", fontsize=16)
        plt.xlabel(f'PC1 ({explained_var[0]:.2%} variance)', fontsize=12)
        plt.ylabel(f'PC2 ({explained_var[1]:.2%} variance)', fontsize=12)
        plt.legend(title='Dataset', fontsize=10)
        
        if save_path:
            plt.savefig(save_path)
            print(f"PCA visualization saved to {save_path}")
        
        plt.show()
    
    def compare_models(self, models_dict, data_loader, save_dir, threshold=0.5):
        """
        Compare segmentation results from multiple models
        
        Args:
            models_dict (dict): Dictionary of models to compare
            data_loader (DataLoader): Data loader for evaluation
            save_dir (str): Directory to save comparison results
            threshold (float): Threshold for binary segmentation
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # Set all models to evaluation mode
        for model_name, model in models_dict.items():
            model.eval()
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                images = batch['image'].to(self.device)
                masks = batch['mask'].to(self.device)
                image_paths = batch['image_path']
                
                # Process each image in the batch
                for i in range(images.size(0)):
                    # Get image name from path
                    image_name = os.path.basename(image_paths[i]).split('.')[0]
                    
                    # Convert image and mask to numpy arrays
                    image = images[i].permute(1, 2, 0).cpu().numpy() * 255
                    image = image.astype(np.uint8)
                    
                    mask = masks[i, 0].cpu().numpy() * 255
                    mask = mask.astype(np.uint8)
                    
                    # Get predictions from each model
                    predictions = {}
                    for model_name, model in models_dict.items():
                        pred_mask, _, _ = model(images[i:i+1])
                        pred_prob = torch.sigmoid(pred_mask)
                        pred_binary = (pred_prob > threshold).float()
                        pred = pred_binary[0, 0].cpu().numpy() * 255
                        pred = pred.astype(np.uint8)
                        predictions[model_name] = pred
                    
                    # Create comparison image
                    h, w = image.shape[:2]
                    num_models = len(models_dict) + 2  # +2 for original image and ground truth
                    comparison = np.zeros((h, w * num_models, 3), dtype=np.uint8)
                    
                    # Add original image
                    comparison[:, :w] = image
                    
                    # Add ground truth mask
                    mask_rgb = np.zeros((h, w, 3), dtype=np.uint8)
                    mask_rgb[:, :, 1] = mask  # Green channel for ground truth
                    comparison[:, w:w*2] = mask_rgb
                    
                    # Add predictions from each model
                    col_idx = 2
                    for model_name, pred in predictions.items():
                        pred_rgb = np.zeros((h, w, 3), dtype=np.uint8)
                        pred_rgb[:, :, 0] = pred  # Red channel for predictions
                        comparison[:, w*col_idx:w*(col_idx+1)] = pred_rgb
                        col_idx += 1
                    
                    # Save comparison image
                    save_path = os.path.join(save_dir, f"{image_name}_model_comparison.png")
                    cv2.imwrite(save_path, cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR))
                    
                    # Create a version with labels
                    labeled_comparison = np.zeros((h + 30, w * num_models, 3), dtype=np.uint8)
                    labeled_comparison[30:, :] = comparison
                    
                    # Add labels
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(labeled_comparison, "Original", (w//2 - 40, 20), font, 0.6, (255, 255, 255), 2)
                    cv2.putText(labeled_comparison, "Ground Truth", (w + w//2 - 60, 20), font, 0.6, (255, 255, 255), 2)
                    
                    col_idx = 2
                    for model_name in models_dict.keys():
                        cv2.putText(labeled_comparison, model_name, (w*col_idx + w//2 - 60, 20), font, 0.6, (255, 255, 255), 2)
                        col_idx += 1
                    
                    # Save labeled comparison image
                    save_path = os.path.join(save_dir, f"{image_name}_labeled_comparison.png")
                    cv2.imwrite(save_path, cv2.cvtColor(labeled_comparison, cv2.COLOR_RGB2BGR))
                    
                # Only process a few batches to avoid generating too many images
                if batch_idx >= 5:
                    break
    
    def plot_performance_comparison(self, results_dict, metric='dice', save_path=None):
        """
        Plot performance comparison between different models
        
        Args:
            results_dict (dict): Dictionary of results for each model
            metric (str): Metric to compare (dice, iou, accuracy, etc.)
            save_path (str, optional): Path to save the plot
        """
        # Extract datasets and models
        datasets = list(next(iter(results_dict.values())).keys())
        if 'average' in datasets:
            datasets.remove('average')  # Remove average from individual comparisons
        
        models = list(results_dict.keys())
        
        # Create data for plotting
        data = []
        for model_name, model_results in results_dict.items():
            for dataset_name in datasets:
                data.append({
                    'Model': model_name,
                    'Dataset': dataset_name,
                    metric.capitalize(): model_results[dataset_name][metric]
                })
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Plot
        plt.figure(figsize=(12, 8))
        ax = sns.barplot(x='Dataset', y=metric.capitalize(), hue='Model', data=df)
        
        plt.title(f'{metric.capitalize()} Score Comparison', fontsize=16)
        plt.xlabel('Dataset', fontsize=14)
        plt.ylabel(f'{metric.capitalize()} Score', fontsize=14)
        plt.ylim(0, 1)
        plt.legend(title='Model')
        
        # Add value labels on bars
        for container in ax.containers:
            ax.bar_label(container, fmt='%.3f', fontsize=8)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"Performance comparison plot saved to {save_path}")
        
        plt.show()
        
        # Also plot average performance across datasets
        avg_data = []
        for model_name, model_results in results_dict.items():
            if 'average' in model_results:
                avg_data.append({
                    'Model': model_name,
                    metric.capitalize(): model_results['average'][metric]
                })
        
        if avg_data:
            avg_df = pd.DataFrame(avg_data)
            
            plt.figure(figsize=(10, 6))
            ax = sns.barplot(x='Model', y=metric.capitalize(), data=avg_df)
            
            plt.title(f'Average {metric.capitalize()} Score Across All Datasets', fontsize=16)
            plt.xlabel('Model', fontsize=14)
            plt.ylabel(f'{metric.capitalize()} Score', fontsize=14)
            plt.ylim(0, 1)
            
            # Add value labels on bars
            for container in ax.containers:
                ax.bar_label(container, fmt='%.3f')
            
            plt.tight_layout()
            
            if save_path:
                avg_save_path = save_path.replace('.png', '_average.png')
                plt.savefig(avg_save_path)
                print(f"Average performance plot saved to {avg_save_path}")
            
            plt.show()