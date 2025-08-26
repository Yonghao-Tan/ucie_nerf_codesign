import matplotlib.pyplot as plt
import numpy as np
import os

def create_grouped_bar_chart(save_path='psnr_comparison.png', dpi=300):
    """
    Creates a grouped bar chart with 12 bars (4 groups of 3 bars each)
    for comparing baseline, full sr, and patched sr across 4 datasets
    
    Parameters:
    save_path (str): Path to save the figure
    dpi (int): Resolution for saved image
    """
    
    # Sample data - you can replace these values with your actual data
    datasets = ['fern', 'trex', 'horns', 'fortress']
    
    # PSNR values for each method across the 4 datasets
    baseline_values = [28.5, 30.2, 26.8, 32.1]
    full_sr_values = [31.2, 33.5, 29.4, 35.8]
    patched_sr_values = [30.8, 32.9, 28.9, 34.5]
    
    # Set up the bar positions
    x = np.arange(len(datasets))  # positions for groups
    width = 0.25  # width of each bar
    
    # Create the figure and axis
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Create the bars
    bars1 = ax.bar(x - width, baseline_values, width, label='baseline', 
                   color='#1f77b4', alpha=0.8)
    bars2 = ax.bar(x, full_sr_values, width, label='full sr', 
                   color='#ff7f0e', alpha=0.8)
    bars3 = ax.bar(x + width, patched_sr_values, width, label='patched sr', 
                   color='#2ca02c', alpha=0.8)
    
    # Add value labels on top of each bar
    def add_value_labels(bars, values):
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                   f'{value:.1f}',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    add_value_labels(bars1, baseline_values)
    add_value_labels(bars2, full_sr_values)
    add_value_labels(bars3, patched_sr_values)
    
    # Customize the chart
    ax.set_xlabel('datasets', fontsize=14, fontweight='bold')
    ax.set_ylabel('PSNR', fontsize=14, fontweight='bold')
    ax.set_title('PSNR Comparison Across Different Methods and Datasets', 
                fontsize=16, fontweight='bold', pad=20)
    
    # Set x-axis labels and positions
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontsize=12)
    
    # Add legend
    ax.legend(fontsize=12, loc='upper left')
    
    # Add grid for better readability
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    # Adjust y-axis limits to accommodate labels
    y_max = max(max(baseline_values), max(full_sr_values), max(patched_sr_values))
    ax.set_ylim(0, y_max * 1.15)
    
    # Improve layout
    plt.tight_layout()
    
    # Save the figure instead of showing it
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    print(f"Chart saved as: {save_path}")
    
    # Close the figure to free memory
    plt.close(fig)
    
    return save_path

def create_custom_chart(datasets, baseline_values, full_sr_values, patched_sr_values, 
                       save_path='custom_psnr_comparison.png', dpi=300):
    """
    Function to create chart with custom data and save it
    
    Parameters:
    datasets (list): List of dataset names
    baseline_values (list): PSNR values for baseline method
    full_sr_values (list): PSNR values for full sr method
    patched_sr_values (list): PSNR values for patched sr method
    save_path (str): Path to save the figure
    dpi (int): Resolution for saved image
    """
    
    # Validate input
    if not (len(datasets) == len(baseline_values) == len(full_sr_values) == len(patched_sr_values)):
        raise ValueError("All input lists must have the same length")
    
    x = np.arange(len(datasets))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    bars1 = ax.bar(x - width, baseline_values, width, label='baseline', 
                   color='#1f77b4', alpha=0.8)
    bars2 = ax.bar(x, full_sr_values, width, label='full sr', 
                   color='#ff7f0e', alpha=0.8)
    bars3 = ax.bar(x + width, patched_sr_values, width, label='patched sr', 
                   color='#2ca02c', alpha=0.8)
    
    def add_value_labels(bars, values):
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                   f'{value:.2f}',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    add_value_labels(bars1, baseline_values)
    add_value_labels(bars2, full_sr_values)
    add_value_labels(bars3, patched_sr_values)
    
    ax.set_xlabel('datasets', fontsize=14, fontweight='bold')
    ax.set_ylabel('PSNR', fontsize=14, fontweight='bold')
    ax.set_title('PSNR Comparison Across Different Methods and Datasets', 
                fontsize=16, fontweight='bold', pad=20)
    
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontsize=12)
    ax.legend(fontsize=12, loc='upper left')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    y_max = max(max(baseline_values), max(full_sr_values), max(patched_sr_values))
    # ax.set_ylim(0, y_max * 1.15)
    ax.set_ylim(24., 30.)
    
    plt.tight_layout()
    
    # Save the figure instead of showing it
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    print(f"Custom chart saved as: {save_path}")
    
    # Close the figure to free memory
    plt.close(fig)
    
    return save_path

if __name__ == "__main__":
    # Create the chart with default sample data and save it
    print("Creating and saving grouped bar chart...")
    saved_path = create_grouped_bar_chart()
    
    # Example of how to use with custom data and save:
    print("\nExample with custom data:")
    custom_datasets = ['Flower', 'Fortress', 'Horns', 'Room']
    custom_baseline = [26.63, 29.78, 26.16, 29.61]
    custom_full_sr = [26.62, 29.31, 25.54, 28.59]
    custom_patched_sr = [26.58, 29.25, 25.48, 28.52]
    
    # Save in PNG format
    create_custom_chart(custom_datasets, custom_baseline, custom_full_sr, custom_patched_sr, 
                       'figure4_pre.png')
    
    # Save in multiple formats
    # print("\nSaving in multiple formats:")
    # save_multiple_formats(custom_datasets, custom_baseline, custom_full_sr, custom_patched_sr, 'multi_format_chart')