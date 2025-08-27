"""
IBRNet Pruning Utilities Package
包含source view pruning和相关可视化功能
"""

from .source_view_pruning import (
    apply_source_view_pruning_optimized, 
    apply_source_view_pruning_optimized_aggregated,
    apply_source_view_pruning_sparse_vectorized,
    apply_source_view_pruning_sparse_vectorized_aggregated,
    apply_source_view_pruning_2x2_windows_variance_based,
    apply_source_view_pruning_2x2_windows_threshold_based,
    apply_source_view_pruning
)

__all__ = [
    'apply_source_view_pruning_optimized',
    'apply_source_view_pruning_optimized_aggregated',
    'apply_source_view_pruning_sparse_vectorized',
    'apply_source_view_pruning_sparse_vectorized_aggregated',
    'apply_source_view_pruning_2x2_windows_variance_based',
    'apply_source_view_pruning_2x2_windows_threshold_based',
    'apply_source_view_pruning',
]
