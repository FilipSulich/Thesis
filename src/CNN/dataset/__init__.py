"""
Dataset module - UTRP dataset loader and H5 extraction utilities
"""

import sys
from multiprocessing import get_context

# Set multiprocessing start method for macOS compatibility
# This prevents "RuntimeError: spawn not allowed" errors when using multiprocessing
if sys.platform == 'darwin':  # macOS
    try:
        if 'fork' in get_context().get_start_method():
            import multiprocessing
            multiprocessing.set_start_method('fork', force=True)
    except RuntimeError:
        pass  # Already set

from .dataset import UTRPDataset
from .dataset_extractor import H5ImageExtractor, FolderBasedUTRPDataset, extract_h5_to_folders

__all__ = [
    'UTRPDataset',
    'H5ImageExtractor',
    'FolderBasedUTRPDataset',
    'extract_h5_to_folders',
]

