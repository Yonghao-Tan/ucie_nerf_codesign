"""
IBRNet Pruning Utilities Package
包含source view pruning和相关可视化功能
"""

from .source_view_pruning import (
    apply_source_view_pruning_optimized, 
    apply_source_view_pruning_original,
    apply_source_view_pruning_sparse,
    apply_source_view_pruning_sparse_vectorized,
    apply_source_view_pruning
)
from .visualization import visualize_depth_samples, visualize_mask_assignment
from .benchmark import benchmark_pruning_methods

__all__ = [
    'apply_source_view_pruning_optimized',
    'apply_source_view_pruning_original', 
    'apply_source_view_pruning_sparse',
    'apply_source_view_pruning_sparse_vectorized',
    'apply_source_view_pruning',
    'visualize_depth_samples',
    'visualize_mask_assignment',
    'benchmark_pruning_methods'
]
