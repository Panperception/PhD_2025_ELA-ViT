"""
Data loading utilities for ELA-ViT experiments
Supports datasets used in Section 4 of the paper
"""

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


def get_cifar10_loaders(
    batch_size: int = 8,
    num_workers: int = 0,
    train_subset_size: int = None
):
    """
    Get CIFAR-10 data loaders.
    
    Args:
        batch_size: Batch size for training and validation
        num_workers: Number of data loading workers
        train_subset_size: If specified, use only this many training samples
        
    Returns:
        train_loader, val_loader
    """
    # Standard ViT preprocessing (Section 4)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    
    # Load datasets
    train_dataset = datasets.CIFAR10(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )
    
    val_dataset = datasets.CIFAR10(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )
    
    # Create subset if specified
    if train_subset_size is not None:
        train_dataset = Subset(train_dataset, list(range(train_subset_size)))
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    return train_loader, val_loader


def get_cifar100_loaders(
    batch_size: int = 8,
    num_workers: int = 0,
    train_subset_size: int = None
):
    """
    Get CIFAR-100 data loaders.
    
    Args:
        batch_size: Batch size for training and validation
        num_workers: Number of data loading workers
        train_subset_size: If specified, use only this many training samples
        
    Returns:
        train_loader, val_loader
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    
    train_dataset = datasets.CIFAR100(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )
    
    val_dataset = datasets.CIFAR100(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )
    
    if train_subset_size is not None:
        train_dataset = Subset(train_dataset, list(range(train_subset_size)))
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    return train_loader, val_loader
