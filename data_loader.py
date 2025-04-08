import os
import numpy as np
import cv2
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import random

class RetinalDataset(Dataset):
    """
    Dataset class for retinal vessel segmentation datasets
    """
    def __init__(self, dataset_path, dataset_name, is_train=True, transform=None, augment=False):
        """
        Args:
            dataset_path (str): Path to the dataset directory
            dataset_name (str): Name of the dataset (DRIVE, STARE, CHASEDB, HRF, LES-AV, RAVIR)
            is_train (bool): Whether to use training set or test set
            transform (callable, optional): Optional transform to be applied on a sample
            augment (bool): Whether to use data augmentation
        """
        self.dataset_path = dataset_path
        self.dataset_name = dataset_name
        self.is_train = is_train
        self.transform = transform
        self.augment = augment
        
        # Define paths based on dataset name
        if dataset_name == 'DRIVE':
            self.image_dir = os.path.join(dataset_path, 'DRIVE', 'images')
            self.mask_dir = os.path.join(dataset_path, 'DRIVE', '1st_manual')
        elif dataset_name == 'STARE':
            self.image_dir = os.path.join(dataset_path, 'STARE', 'images')
            self.mask_dir = os.path.join(dataset_path, 'STARE', '1st_manual')
        elif dataset_name == 'CHASDB':
            self.image_dir = os.path.join(dataset_path, 'CHASDB', 'images')
            self.mask_dir = os.path.join(dataset_path, 'CHASDB', '1st_manual')
        elif dataset_name == 'HRF':
            self.image_dir = os.path.join(dataset_path, 'HRF', 'images')
            self.mask_dir = os.path.join(dataset_path, 'HRF', 'manual1')
        elif dataset_name == 'LES-AV':
            self.image_dir = os.path.join(dataset_path, 'LES-AV', 'images')
            self.mask_dir = os.path.join(dataset_path, 'LES-AV', 'vessel-segmentations')
        elif dataset_name == 'RAVIR':
            self.image_dir = os.path.join(dataset_path, 'RAVIR', 'training_images')
            self.mask_dir = os.path.join(dataset_path, 'RAVIR', 'training_masks')
        else:
            raise ValueError(f"Dataset {dataset_name} not supported")
        
        # Get all image files
        self.image_files = sorted([f for f in os.listdir(self.image_dir) if f.endswith('.png')])
        
        # Split into train and test sets (80/20 split if not predefined)
        if is_train:
            self.image_files = self.image_files[:int(len(self.image_files) * 0.8)]
        else:
            self.image_files = self.image_files[int(len(self.image_files) * 0.8):]
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        
        # For mask, we need to find the corresponding mask file
        # The naming convention might differ between datasets
        mask_name = img_name
        mask_path = os.path.join(self.mask_dir, mask_name)
        
        # If mask file doesn't exist with the same name, try to find a match
        if not os.path.exists(mask_path):
            base_name = os.path.splitext(img_name)[0]
            potential_masks = [f for f in os.listdir(self.mask_dir) if base_name in f]
            if potential_masks:
                mask_name = potential_masks[0]
                mask_path = os.path.join(self.mask_dir, mask_name)
            else:
                raise FileNotFoundError(f"No matching mask found for {img_name}")
        
        # Load image and mask
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # Ensure mask is binary (0 or 1)
        _, mask = cv2.threshold(mask, 127, 1, cv2.THRESH_BINARY)
        
        # Apply data augmentation if enabled
        if self.augment:
            image, mask = self._augment(image, mask)
        
        # Convert to tensors
        image = torch.from_numpy(image.transpose((2, 0, 1))).float() / 255.0
        mask = torch.from_numpy(mask).float().unsqueeze(0)
        
        # Apply additional transforms if provided
        if self.transform:
            image = self.transform(image)
        
        # Create domain label (one-hot encoding)
        domain_idx = {'DRIVE': 0, 'STARE': 1, 'CHASDB': 2, 'HRF': 3, 'LES-AV': 4, 'RAVIR': 5}[self.dataset_name]
        domain_label = torch.zeros(6)
        domain_label[domain_idx] = 1
        
        return {
            'image': image,
            'mask': mask,
            'domain': domain_label,
            'dataset_name': self.dataset_name,
            'image_path': img_path
        }
    
    def _augment(self, image, mask):
        """
        Apply data augmentation to image and mask
        """
        # Random horizontal flip
        if random.random() > 0.5:
            image = cv2.flip(image, 1)
            mask = cv2.flip(mask, 1)
        
        # Random vertical flip
        if random.random() > 0.5:
            image = cv2.flip(image, 0)
            mask = cv2.flip(mask, 0)
        
        # Random rotation
        angle = random.randint(-20, 20)
        h, w = image.shape[:2]
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
        image = cv2.warpAffine(image, M, (w, h))
        mask = cv2.warpAffine(mask, M, (w, h))
        
        # Random brightness and contrast adjustment
        alpha = 1.0 + random.uniform(-0.2, 0.2)  # contrast
        beta = random.uniform(-20, 20)  # brightness
        image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
        
        return image, mask

def get_data_loaders(base_path, dataset_names, batch_size=8, num_workers=4, augment=True):
    """
    Create data loaders for multiple datasets
    
    Args:
        base_path (str): Base path to the datasets
        dataset_names (list): List of dataset names to include
        batch_size (int): Batch size for data loaders
        num_workers (int): Number of workers for data loading
        augment (bool): Whether to use data augmentation
    
    Returns:
        dict: Dictionary containing train and test data loaders for each dataset
    """
    data_loaders = {}
    
    for dataset_name in dataset_names:
        # Create training dataset
        train_dataset = RetinalDataset(
            dataset_path=base_path,
            dataset_name=dataset_name,
            is_train=True,
            augment=augment
        )
        
        # Create test dataset
        test_dataset = RetinalDataset(
            dataset_path=base_path,
            dataset_name=dataset_name,
            is_train=False,
            augment=False
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        
        data_loaders[dataset_name] = {
            'train': train_loader,
            'test': test_loader
        }
    
    return data_loaders