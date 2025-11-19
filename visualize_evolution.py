"""
Visualization Module for Evolution Progress
Creates plots showing fitness evolution over generations
"""

import matplotlib.pyplot as plt
import matplotlib
from typing import List, Dict, Any
import os

# Use non-interactive backend if display not available
try:
    matplotlib.use('Agg')
except:
    pass


def plot_evolution_history(history: List[Dict[str, Any]], output_file: str = "evolution_plot.png"):
    """
    Plot evolution history showing fitness over generations
    
    Args:
        history: List of generation statistics dictionaries
        output_file: Path to save the plot
    """
    if not history:
        print("No history data to plot")
        return
    
    generations = [h['generation'] for h in history]
    best_fitness = [h['best_fitness'] for h in history]
    avg_fitness = [h['avg_fitness'] for h in history]
    worst_fitness = [h['worst_fitness'] for h in history]
    std_fitness = [h.get('std_fitness', 0) for h in history]
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Evolution Progress - Kissing Number Problem (Dimension 2)', 
                 fontsize=16, fontweight='bold')
    
    # Plot 1: Fitness over generations (best, avg, worst)
    ax1 = axes[0, 0]
    ax1.plot(generations, best_fitness, 'g-', linewidth=2, label='Best Fitness', marker='o', markersize=4)
    ax1.plot(generations, avg_fitness, 'b-', linewidth=2, label='Average Fitness', marker='s', markersize=4)
    ax1.plot(generations, worst_fitness, 'r-', linewidth=1, label='Worst Fitness', marker='^', markersize=3, alpha=0.7)
    ax1.set_xlabel('Generation')
    ax1.set_ylabel('Fitness')
    ax1.set_title('Fitness Evolution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1.1])
    
    # Plot 2: Fitness improvement (best fitness)
    ax2 = axes[0, 1]
    ax2.plot(generations, best_fitness, 'g-', linewidth=3, marker='o', markersize=5)
    ax2.fill_between(generations, best_fitness, alpha=0.3, color='green')
    ax2.set_xlabel('Generation')
    ax2.set_ylabel('Best Fitness')
    ax2.set_title('Best Fitness Over Time')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1.1])
    
    # Add annotation for maximum
    max_gen = generations[best_fitness.index(max(best_fitness))]
    max_fit = max(best_fitness)
    ax2.annotate(f'Max: {max_fit:.3f}\nGen: {max_gen}', 
                xy=(max_gen, max_fit), 
                xytext=(max_gen + len(generations)*0.1, max_fit - 0.2),
                arrowprops=dict(arrowstyle='->', color='red', lw=2),
                fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))
    
    # Plot 3: Population diversity (standard deviation)
    ax3 = axes[1, 0]
    ax3.plot(generations, std_fitness, 'purple', linewidth=2, marker='d', markersize=4)
    ax3.fill_between(generations, std_fitness, alpha=0.3, color='purple')
    ax3.set_xlabel('Generation')
    ax3.set_ylabel('Standard Deviation')
    ax3.set_title('Population Diversity (Fitness Std Dev)')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Average fitness improvement
    ax4 = axes[1, 1]
    ax4.plot(generations, avg_fitness, 'b-', linewidth=2, marker='s', markersize=4)
    ax4.fill_between(generations, avg_fitness, alpha=0.3, color='blue')
    ax4.set_xlabel('Generation')
    ax4.set_ylabel('Average Fitness')
    ax4.set_title('Average Fitness Over Time')
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim([0, 1.1])
    
    # Add improvement annotation
    if len(avg_fitness) > 1:
        improvement = avg_fitness[-1] - avg_fitness[0]
        ax4.annotate(f'Improvement: +{improvement:.3f}', 
                    xy=(generations[-1], avg_fitness[-1]), 
                    xytext=(generations[-1] - len(generations)*0.2, avg_fitness[-1] + 0.1),
                    arrowprops=dict(arrowstyle='->', color='green', lw=2),
                    fontsize=10,
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Evolution plot saved to: {output_file}")
    plt.close()


def plot_fitness_distribution(history: List[Dict[str, Any]], output_file: str = "fitness_distribution.png"):
    """
    Plot fitness distribution over generations
    
    Args:
        history: List of generation statistics dictionaries
        output_file: Path to save the plot
    """
    if not history:
        return
    
    generations = [h['generation'] for h in history]
    best_fitness = [h['best_fitness'] for h in history]
    avg_fitness = [h['avg_fitness'] for h in history]
    worst_fitness = [h['worst_fitness'] for h in history]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Create shaded area between best and worst
    ax.fill_between(generations, worst_fitness, best_fitness, alpha=0.2, color='gray', label='Fitness Range')
    ax.plot(generations, best_fitness, 'g-', linewidth=2, label='Best', marker='o', markersize=5)
    ax.plot(generations, avg_fitness, 'b-', linewidth=2, label='Average', marker='s', markersize=4)
    ax.plot(generations, worst_fitness, 'r-', linewidth=1, label='Worst', marker='^', markersize=3, alpha=0.7)
    
    ax.set_xlabel('Generation', fontsize=12)
    ax.set_ylabel('Fitness', fontsize=12)
    ax.set_title('Fitness Distribution Over Generations', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.1])
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Fitness distribution plot saved to: {output_file}")
    plt.close()


def plot_convergence_analysis(history: List[Dict[str, Any]], output_file: str = "convergence_analysis.png"):
    """
    Plot convergence analysis showing when fitness stabilizes
    
    Args:
        history: List of generation statistics dictionaries
        output_file: Path to save the plot
    """
    if not history:
        return
    
    generations = [h['generation'] for h in history]
    best_fitness = [h['best_fitness'] for h in history]
    avg_fitness = [h['avg_fitness'] for h in history]
    
    # Calculate improvement rate
    improvements = []
    for i in range(1, len(best_fitness)):
        improvement = best_fitness[i] - best_fitness[i-1]
        improvements.append(improvement)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Convergence Analysis', fontsize=14, fontweight='bold')
    
    # Plot 1: Fitness improvement rate
    ax1 = axes[0]
    if improvements:
        ax1.plot(generations[1:], improvements, 'g-', linewidth=2, marker='o', markersize=4)
        ax1.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        ax1.fill_between(generations[1:], improvements, 0, alpha=0.3, 
                         where=[x >= 0 for x in improvements], color='green', label='Improvement')
        ax1.fill_between(generations[1:], improvements, 0, alpha=0.3, 
                         where=[x < 0 for x in improvements], color='red', label='Decline')
    ax1.set_xlabel('Generation')
    ax1.set_ylabel('Fitness Improvement')
    ax1.set_title('Fitness Improvement Rate')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Gap between best and average (convergence indicator)
    ax2 = axes[1]
    gaps = [best_fitness[i] - avg_fitness[i] for i in range(len(generations))]
    ax2.plot(generations, gaps, 'purple', linewidth=2, marker='s', markersize=4)
    ax2.fill_between(generations, gaps, alpha=0.3, color='purple')
    ax2.set_xlabel('Generation')
    ax2.set_ylabel('Best - Average Fitness')
    ax2.set_title('Population Convergence (Lower = More Converged)')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Convergence analysis plot saved to: {output_file}")
    plt.close()


def create_comprehensive_visualization(history: List[Dict[str, Any]], output_dir: str = "."):
    """
    Create all visualizations
    
    Args:
        history: List of generation statistics dictionaries
        output_dir: Directory to save plots
    """
    if not history:
        print("No history data available for visualization")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("Generating visualizations...")
    plot_evolution_history(history, os.path.join(output_dir, "evolution_plot.png"))
    plot_fitness_distribution(history, os.path.join(output_dir, "fitness_distribution.png"))
    plot_convergence_analysis(history, os.path.join(output_dir, "convergence_analysis.png"))
    
    print(f"\nAll visualizations saved to: {output_dir}/")
    print("  - evolution_plot.png: Main evolution progress (4 subplots)")
    print("  - fitness_distribution.png: Fitness range over generations")
    print("  - convergence_analysis.png: Convergence and improvement analysis")


if __name__ == "__main__":
    # Example usage
    print("Visualization module loaded. Import and use create_comprehensive_visualization()")

