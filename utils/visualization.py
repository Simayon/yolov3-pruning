import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path

def plot_pruning_results(results_df, save_dir):
    save_dir = Path(save_dir)
    
    # Set the style
    plt.style.use('seaborn')
    sns.set_palette("husl")

    # Create figure with multiple subplots
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: Metrics vs Pruning Ratio
    ax1 = axes[0]
    ax1.plot(results_df['pruning_ratio'], results_df['mAP'], marker='o', label='mAP')
    ax1.plot(results_df['pruning_ratio'], results_df['precision'], marker='s', label='Precision')
    ax1.plot(results_df['pruning_ratio'], results_df['recall'], marker='^', label='Recall')
    ax1.set_xlabel('Pruning Ratio')
    ax1.set_ylabel('Score')
    ax1.set_title('Model Performance vs Pruning Ratio')
    ax1.legend()
    ax1.grid(True)

    # Plot 2: Model Size vs Pruning Ratio
    ax2 = axes[1]
    ax2.plot(results_df['pruning_ratio'], results_df['model_size_mb'], 
             marker='o', color='green', linewidth=2)
    ax2.set_xlabel('Pruning Ratio')
    ax2.set_ylabel('Model Size (MB)')
    ax2.set_title('Model Size vs Pruning Ratio')
    ax2.grid(True)

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(save_dir / 'pruning_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_comparison_table(results_df, save_dir):
    save_dir = Path(save_dir)
    
    # Create a styled HTML table
    styled_df = results_df.style\
        .format({
            'pruning_ratio': '{:.2f}',
            'mAP': '{:.3f}',
            'precision': '{:.3f}',
            'recall': '{:.3f}',
            'model_size_mb': '{:.1f}'
        })\
        .background_gradient(cmap='YlOrRd', subset=['mAP', 'precision', 'recall'])\
        .background_gradient(cmap='YlOrRd_r', subset=['model_size_mb'])
    
    # Save to HTML
    styled_df.to_html(save_dir / 'comparison_table.html')

def generate_report(results_df, save_dir):
    """Generate a comprehensive report of the pruning results"""
    save_dir = Path(save_dir)
    
    with open(save_dir / 'pruning_report.md', 'w') as f:
        f.write("# YOLOv3 Model Pruning Report\n\n")
        
        # Summary Statistics
        f.write("## Summary Statistics\n\n")
        f.write(f"- Initial model size: {results_df['model_size_mb'].iloc[0]:.1f} MB\n")
        f.write(f"- Final model size: {results_df['model_size_mb'].iloc[-1]:.1f} MB\n")
        f.write(f"- Size reduction: {((results_df['model_size_mb'].iloc[0] - results_df['model_size_mb'].iloc[-1]) / results_df['model_size_mb'].iloc[0] * 100):.1f}%\n\n")
        
        # Performance Impact
        f.write("## Performance Impact\n\n")
        f.write(f"- Initial mAP: {results_df['mAP'].iloc[0]:.3f}\n")
        f.write(f"- Final mAP: {results_df['mAP'].iloc[-1]:.3f}\n")
        f.write(f"- mAP change: {((results_df['mAP'].iloc[-1] - results_df['mAP'].iloc[0]) / results_df['mAP'].iloc[0] * 100):.1f}%\n\n")
        
        # Detailed Results Table
        f.write("## Detailed Results\n\n")
        f.write(results_df.to_markdown(index=False, floatfmt='.3f'))
