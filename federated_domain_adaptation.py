import os
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from collections import OrderedDict
import random
from models import UNet, DomainDiscriminator, MMD_loss

class FederatedDomainAdaptation:
    """
    A novel federated learning framework with domain adaptation for retinal vessel segmentation
    
    This class implements a federated learning system where multiple clients (datasets)
    collaborate to train a global model while adapting to domain shifts between datasets.
    
    Key features:
    1. Personalized local models with domain-specific adaptations
    2. Multi-source domain adaptation using adversarial training
    3. Knowledge distillation from global to local models
    4. Mutual information minimization for domain-invariant features
    5. Federated model aggregation with domain importance weighting
    """
    def __init__(self, datasets, device, args):
        """
        Initialize the federated learning system
        
        Args:
            datasets (dict): Dictionary of datasets for each client
            device (torch.device): Device to run the models on
            args (dict): Configuration parameters
        """
        self.datasets = datasets
        self.device = device
        self.args = args
        
        # Initialize global model
        self.global_model = UNet(
            n_channels=3,
            n_classes=1,
            bilinear=True,
            with_domain_features=True
        ).to(device)
        
        # Initialize local models for each client
        self.local_models = {}
        for client_id in datasets.keys():
            self.local_models[client_id] = UNet(
                n_channels=3,
                n_classes=1,
                bilinear=True,
                with_domain_features=True
            ).to(device)
            # Initialize local model with global model weights
            self.local_models[client_id].load_state_dict(self.global_model.state_dict())
        
        # Initialize domain discriminator
        self.domain_discriminator = DomainDiscriminator().to(device)
        
        # Initialize MMD loss for domain adaptation
        self.mmd_loss = MMD_loss().to(device)
        
        # Set up optimizers
        self.global_optimizer = optim.Adam(self.global_model.parameters(), lr=args['lr'])
        
        self.local_optimizers = {}
        self.discriminator_optimizers = {}
        for client_id in datasets.keys():
            self.local_optimizers[client_id] = optim.Adam(
                self.local_models[client_id].parameters(), 
                lr=args['lr']
            )
            self.discriminator_optimizers[client_id] = optim.Adam(
                self.domain_discriminator.parameters(),
                lr=args['lr'] * 0.1  # Lower learning rate for discriminator
            )
        
        # Set up loss functions
        self.segmentation_loss = nn.BCEWithLogitsLoss()
        self.domain_loss = nn.CrossEntropyLoss()
        self.distillation_loss = nn.KLDivLoss(reduction='batchmean')
        
        # Track metrics
        self.metrics = {
            'train_loss': [],
            'val_loss': [],
            'test_metrics': {}
        }
    
    def train(self, num_rounds):
        """
        Train the federated learning system for a specified number of rounds
        
        Args:
            num_rounds (int): Number of federated learning rounds
        
        Returns:
            dict: Training metrics
        """
        for round_idx in range(num_rounds):
            print(f"\nFederated Learning Round {round_idx+1}/{num_rounds}")
            
            # Local training on each client
            local_weights = {}
            local_losses = []
            
            for client_id in self.datasets.keys():
                print(f"Training on client {client_id}")
                
                # Train local model
                local_loss = self.train_client(
                    client_id=client_id,
                    global_round=round_idx
                )
                local_losses.append(local_loss)
                
                # Get model weights
                local_weights[client_id] = self.local_models[client_id].state_dict()
            
            # Update global model (federated aggregation)
            self.aggregate_models(local_weights, round_idx)
            
            # Evaluate global model
            if (round_idx + 1) % self.args['eval_every'] == 0:
                self.evaluate_global_model(round_idx)
            
            # Update local models with global model
            for client_id in self.datasets.keys():
                # Personalized model update: blend global model with local model
                self.personalized_model_update(client_id, round_idx)
        
        return self.metrics
    
    def train_client(self, client_id, global_round):
        """
        Train a client's local model
        
        Args:
            client_id (str): Client identifier
            global_round (int): Current global round
        
        Returns:
            float: Average training loss
        """
        # Set models to training mode
        local_model = self.local_models[client_id]
        local_model.train()
        self.domain_discriminator.train()
        
        # Get data loader for this client
        train_loader = self.datasets[client_id]['train']
        
        # Set up optimizers
        optimizer = self.local_optimizers[client_id]
        discriminator_optimizer = self.discriminator_optimizers[client_id]
        
        epoch_loss = []
        for epoch in range(self.args['local_epochs']):
            batch_loss = []
            
            for batch_idx, data in enumerate(train_loader):
                images = data['image'].to(self.device)
                masks = data['mask'].to(self.device)
                domain_labels = data['domain'].to(self.device)
                
                # Clear gradients
                optimizer.zero_grad()
                discriminator_optimizer.zero_grad()
                
                # Forward pass
                # Alpha controls the strength of gradient reversal (increases over training)
                alpha = min(1.0, (global_round * self.args['local_epochs'] + epoch) / (self.args['num_rounds'] * self.args['local_epochs'] / 2))
                
                # Get predictions from local model
                pred_masks, domain_preds, features = local_model(images, alpha)
                
                # Segmentation loss
                seg_loss = self.segmentation_loss(pred_masks, masks)
                
                # Domain adversarial loss
                if domain_preds is not None:
                    domain_adv_loss = self.domain_loss(domain_preds, torch.argmax(domain_labels, dim=1))
                else:
                    domain_adv_loss = torch.tensor(0.0).to(self.device)
                
                # Knowledge distillation from global model (if not first round)
                if global_round > 0:
                    # Get predictions from global model
                    with torch.no_grad():
                        self.global_model.eval()
                        global_pred_masks, _, global_features = self.global_model(images)
                    
                    # Feature-level distillation loss
                    distill_loss = F.mse_loss(features, global_features.detach())
                    
                    # MMD loss between local and global features
                    mmd = self.mmd_loss(features, global_features.detach())
                else:
                    distill_loss = torch.tensor(0.0).to(self.device)
                    mmd = torch.tensor(0.0).to(self.device)
                
                # Total loss
                loss = seg_loss + \
                       self.args['lambda_adv'] * domain_adv_loss + \
                       self.args['lambda_distill'] * distill_loss + \
                       self.args['lambda_mmd'] * mmd
                
                # Backward pass and optimize
                loss.backward()
                optimizer.step()
                
                # Train domain discriminator separately
                if global_round > 0:  # Start domain adaptation after first round
                    discriminator_optimizer.zero_grad()
                    
                    # Extract features without gradient reversal
                    with torch.no_grad():
                        _, _, features = local_model(images)
                    
                    # Domain classification
                    domain_preds = self.domain_discriminator(features.view(features.size(0), -1).detach())
                    d_loss = self.domain_loss(domain_preds, torch.argmax(domain_labels, dim=1))
                    
                    d_loss.backward()
                    discriminator_optimizer.step()
                
                batch_loss.append(loss.item())
            
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            print(f"Client {client_id} - Local Epoch {epoch+1}/{self.args['local_epochs']} - Loss: {epoch_loss[-1]:.4f}")
        
        return sum(epoch_loss) / len(epoch_loss)
    
    def aggregate_models(self, local_weights, global_round):
        """
        Aggregate local models into the global model using weighted averaging
        
        Args:
            local_weights (dict): Dictionary of local model weights
            global_round (int): Current global round
        """
        # Calculate weights for aggregation based on dataset sizes
        total_samples = sum([len(self.datasets[client_id]['train'].dataset) for client_id in self.datasets.keys()])
        client_weights = {client_id: len(self.datasets[client_id]['train'].dataset) / total_samples 
                         for client_id in self.datasets.keys()}
        
        # Initialize global model dict with zeros
        global_dict = OrderedDict()
        for k in local_weights[list(local_weights.keys())[0]].keys():
            global_dict[k] = torch.zeros_like(local_weights[list(local_weights.keys())[0]][k])
        
        # Weighted averaging of model parameters
        for client_id in local_weights.keys():
            weight = client_weights[client_id]
            for k in global_dict.keys():
                global_dict[k] += local_weights[client_id][k] * weight
        
        # Update global model
        self.global_model.load_state_dict(global_dict)
    
    def personalized_model_update(self, client_id, global_round):
        """
        Update local model with a personalized blend of global and local parameters
        
        Args:
            client_id (str): Client identifier
            global_round (int): Current global round
        """
        # Adaptive mixing parameter (increases with rounds to favor global knowledge)
        beta = min(0.8, 0.1 + global_round * 0.1)  # Max 0.8 to preserve some local knowledge
        
        # Get global and local model states
        global_dict = self.global_model.state_dict()
        local_dict = self.local_models[client_id].state_dict()
        
        # Create mixed model
        mixed_dict = OrderedDict()
        for k in global_dict.keys():
            mixed_dict[k] = beta * global_dict[k] + (1 - beta) * local_dict[k]
        
        # Update local model
        self.local_models[client_id].load_state_dict(mixed_dict)
    
    def evaluate_global_model(self, global_round):
        """
        Evaluate the global model on all clients' test data
        
        Args:
            global_round (int): Current global round
        """
        self.global_model.eval()
        
        val_loss = 0
        dice_scores = {}
        
        with torch.no_grad():
            for client_id in self.datasets.keys():
                test_loader = self.datasets[client_id]['test']
                client_loss = 0
                client_dice = 0
                
                for batch_idx, data in enumerate(test_loader):
                    images = data['image'].to(self.device)
                    masks = data['mask'].to(self.device)
                    
                    # Forward pass
                    pred_masks, _, _ = self.global_model(images)
                    
                    # Calculate loss
                    loss = self.segmentation_loss(pred_masks, masks)
                    client_loss += loss.item()
                    
                    # Calculate Dice score
                    pred_binary = (torch.sigmoid(pred_masks) > 0.5).float()
                    dice = (2 * (pred_binary * masks).sum()) / ((pred_binary + masks).sum() + 1e-8)
                    client_dice += dice.item()
                
                # Average metrics
                client_loss /= len(test_loader)
                client_dice /= len(test_loader)
                
                val_loss += client_loss
                dice_scores[client_id] = client_dice
                
                print(f"Client {client_id} - Test Loss: {client_loss:.4f}, Dice Score: {client_dice:.4f}")
        
        # Average validation loss across all clients
        val_loss /= len(self.datasets)
        
        # Store metrics
        self.metrics['val_loss'].append(val_loss)
        self.metrics['test_metrics'][global_round] = {
            'dice_scores': dice_scores,
            'avg_dice': sum(dice_scores.values()) / len(dice_scores)
        }
        
        print(f"Global Round {global_round+1} - Avg Test Loss: {val_loss:.4f}, Avg Dice: {self.metrics['test_metrics'][global_round]['avg_dice']:.4f}")
    
    def save_models(self, save_path):
        """
        Save global and local models
        
        Args:
            save_path (str): Directory to save models
        """
        os.makedirs(save_path, exist_ok=True)
        
        # Save global model
        torch.save(self.global_model.state_dict(), os.path.join(save_path, 'global_model.pth'))
        
        # Save local models
        for client_id in self.local_models.keys():
            torch.save(
                self.local_models[client_id].state_dict(),
                os.path.join(save_path, f'local_model_{client_id}.pth')
            )
    
    def load_models(self, load_path):
        """
        Load global and local models
        
        Args:
            load_path (str): Directory to load models from
        """
        # Load global model
        self.global_model.load_state_dict(torch.load(os.path.join(load_path, 'global_model.pth')))
        
        # Load local models
        for client_id in self.local_models.keys():
            model_path = os.path.join(load_path, f'local_model_{client_id}.pth')
            if os.path.exists(model_path):
                self.local_models[client_id].load_state_dict(torch.load(model_path))
            else:
                print(f"Local model for client {client_id} not found, initializing with global model")
                self.local_models[client_id].load_state_dict(self.global_model.state_dict())