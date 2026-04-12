# """
# UTRP-based Pre-AF Classification using CNNs.
#
# This package provides a complete pipeline for:
#   1. Generating UTRP datasets from PAF ECG data
#   2. Training CNN classifiers (custom or ResNet-18)
#   3. Patient-stratified cross-validation
#   4. Comprehensive evaluation and visualization
#
# Quick Start:
#     # Generate dataset
#     python src/CNN/dataset_generator.py
#
#     # Train single-scale ResNet-18
#     python src/CNN/run_classification.py --model single --arch resnet
#
#     # Inspect dataset
#     python src/CNN/inspect_dataset.py
#
# Modules:
#     - base_cnn: Abstract base class with training/evaluation methods
#     - models: CNN architectures (SingleScaleCNN, MultiScaleCNN, ResNet18Transfer)
#     - dataset: PyTorch Dataset and data loading utilities
#     - cross_validation: Patient-stratified K-fold splitter
#     - training: Cross-validation runner with hyperparameter tuning
#     - dataset_generator: UTRP dataset generation from PAF data
#     - run_classification: Main entry point for experiments
#     - inspect_dataset: Dataset inspection and visualization
# """
#
# from src.CNN.models import SingleScaleCNN, MultiScaleCNN, ResNet18Transfer
# from src.CNN.dataset import UTRPDataset, compute_class_weights, create_dataloaders
# from src.CNN.training import CrossValidationRunner, set_seed
#
# __all__ = [
#     'SingleScaleCNN',
#     'MultiScaleCNN',
#     'ResNet18Transfer',
#     'UTRPDataset',
#     'compute_class_weights',
#     'create_dataloaders',
#     'CrossValidationRunner',
#     'set_seed'
# ]