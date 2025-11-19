"""
Animate Sphere Evolution
Creates an animation showing sphere positions on a plane at each iteration
"""

import json
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
import matplotlib
from typing import List, Dict, Any
import numpy as np
from sphere_solver import extract_sphere_info_from_genome, compute_kissing_number_spheres

# Use non-interactive backend if display not available
try:
    matplotlib.use('Agg')
except:
    pass


def load_sphere_data(history_file: str = "kissing_number_results.json") -> List[Dict[str, Any]]:
    """
    Load sphere data from evolution history
    
    Args:
        history_file: Path to JSON file with evolution results
        
    Returns:
        List of dictionaries with sphere information for each generation
    """
    try:
        with open(history_file, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File {history_file} not found.")
        return []
    
    history = data.get('history', [])
    top_individuals = data.get('top_individuals', [])
    best_individual = data.get('best_individual', {})
    
    # Create a mapping of generation to best genome
    gen_to_genome = {}
    if best_individual.get('genome'):
        gen_to_genome[best_individual.get('generation', -1)] = best_individual['genome']
    
    for ind in top_individuals:
        gen = ind.get('generation', -1)
        if gen not in gen_to_genome:
            gen_to_genome[gen] = ind.get('genome', '')
    
    # Extract sphere information for each generation
    sphere_data = []
    for gen_data in history:
        generation = gen_data.get('generation', -1)
        best_fitness = gen_data.get('best_fitness', 0.0)
        
        # Get sphere information if available
        sphere_count = gen_data.get('sphere_count')
        sphere_centers = gen_data.get('sphere_centers', [])
        central_center = gen_data.get('central_center', [0.0, 0.0])
        dimension = gen_data.get('dimension', 2)
        
        # If sphere info not in history, try to compute from genome
        if sphere_count is None and generation in gen_to_genome:
            genome = gen_to_genome[generation]
            sphere_info = extract_sphere_info_from_genome(
                genome,
                context={'problem': 'kissing_number_dimension_2', 'expected_answer': 6}
            )
            sphere_count = sphere_info.get('sphere_count')
            sphere_centers = sphere_info.get('centers', [])
            central_center = sphere_info.get('central_center', [0.0, 0.0])
            dimension = sphere_info.get('dimension', 2)
        
        # If still no sphere info, use default
        if sphere_count is None:
            sphere_info = compute_kissing_number_spheres(dimension=2, num_spheres=6)
            sphere_count = sphere_info.get('sphere_count', 6)
            sphere_centers = sphere_info.get('centers', [])
            central_center = sphere_info.get('central_center', [0.0, 0.0])
            dimension = sphere_info.get('dimension', 2)
        
        sphere_data.append({
            'generation': generation,
            'best_fitness': best_fitness,
            'sphere_count': sphere_count,
            'sphere_centers': sphere_centers,
            'central_center': central_center,
            'dimension': dimension
        })
    
    return sphere_data


def create_sphere_animation(
    history_file: str = "kissing_number_results.json",
    output_file: str = "sphere_evolution_animation.gif",
    fps: int = 2,
    radius: float = 1.0
):
    """
    Create an animation showing sphere positions at each generation
    
    Args:
        history_file: Path to JSON file with evolution results
        output_file: Path to save the animation (GIF or MP4)
        fps: Frames per second for the animation
        radius: Radius of the circles/spheres
    """
    sphere_data = load_sphere_data(history_file)
    
    if not sphere_data:
        print("No sphere data available for animation.")
        return
    
    # Set up the figure and axis
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_aspect('equal')
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_title('Sphere Evolution - Generation 0', fontsize=14, fontweight='bold')
    
    # Text for generation and fitness info
    info_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, 
                        fontsize=11, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Store circles for updating
    central_circle = None
    sphere_circles = []
    
    def animate(frame):
        """Animation function called for each frame"""
        nonlocal central_circle, sphere_circles
        
        # Clear previous circles
        if central_circle:
            central_circle.remove()
        for circle in sphere_circles:
            circle.remove()
        sphere_circles.clear()
        
        # Get data for current frame
        if frame >= len(sphere_data):
            frame = len(sphere_data) - 1
        
        data = sphere_data[frame]
        generation = data['generation']
        best_fitness = data['best_fitness']
        sphere_centers = data['sphere_centers']
        central_center = data['central_center']
        sphere_count = data['sphere_count']
        
        # Draw central circle
        central_circle = Circle(
            central_center, radius, 
            color='red', fill=True, alpha=0.6, 
            label='Central Circle'
        )
        ax.add_patch(central_circle)
        
        # Draw surrounding circles
        colors = plt.cm.viridis(np.linspace(0, 1, max(len(sphere_centers), 1)))
        for i, center in enumerate(sphere_centers):
            circle = Circle(
                center, radius,
                facecolor=colors[i % len(colors)], 
                fill=True, alpha=0.5,
                edgecolor='black', linewidth=1.5
            )
            ax.add_patch(circle)
            sphere_circles.append(circle)
        
        # Update title and info
        ax.set_title(f'Sphere Evolution - Generation {generation}', 
                    fontsize=14, fontweight='bold')
        info_text.set_text(
            f'Generation: {generation}\n'
            f'Best Fitness: {best_fitness:.4f}\n'
            f'Number of Spheres: {sphere_count}\n'
            f'Dimension: {data["dimension"]}D'
        )
        
        return [central_circle] + sphere_circles + [info_text]
    
    # Create animation
    num_frames = len(sphere_data)
    anim = animation.FuncAnimation(
        fig, animate, frames=num_frames,
        interval=1000/fps, blit=False, repeat=True
    )
    
    # Add legend
    ax.legend([Circle((0, 0), 0.1, color='red', alpha=0.6),
               Circle((0, 0), 0.1, color=plt.cm.viridis(0.5), alpha=0.5)],
              ['Central Circle', 'Surrounding Spheres'],
              loc='upper right')
    
    # Save animation
    print(f"Creating animation with {num_frames} frames...")
    try:
        # Try to save as GIF
        anim.save(output_file, writer='pillow', fps=fps)
        print(f"Animation saved to: {output_file}")
    except Exception as e:
        print(f"Error saving animation: {e}")
        print("Trying alternative format...")
        try:
            # Try MP4 if GIF fails
            if output_file.endswith('.gif'):
                output_file = output_file.replace('.gif', '.mp4')
            anim.save(output_file, writer='ffmpeg', fps=fps)
            print(f"Animation saved to: {output_file}")
        except Exception as e2:
            print(f"Error saving animation: {e2}")
            print("Saving as individual frames instead...")
            # Save as individual frames
            import os
            frames_dir = "animation_frames"
            os.makedirs(frames_dir, exist_ok=True)
            for i in range(num_frames):
                animate(i)
                plt.savefig(f"{frames_dir}/frame_{i:03d}.png", dpi=150, bbox_inches='tight')
            print(f"Frames saved to: {frames_dir}/")
    
    plt.close()


def create_static_comparison(
    history_file: str = "kissing_number_results.json",
    output_file: str = "sphere_comparison.png",
    generations_to_show: List[int] = None,
    radius: float = 1.0
):
    """
    Create a static comparison showing sphere positions at selected generations
    
    Args:
        history_file: Path to JSON file with evolution results
        output_file: Path to save the comparison image
        generations_to_show: List of generation numbers to show (None = show all)
        radius: Radius of the circles/spheres
    """
    sphere_data = load_sphere_data(history_file)
    
    if not sphere_data:
        print("No sphere data available for comparison.")
        return
    
    # Select generations to show
    if generations_to_show is None:
        # Show first, middle, and last generations
        if len(sphere_data) <= 3:
            generations_to_show = [d['generation'] for d in sphere_data]
        else:
            generations_to_show = [
                sphere_data[0]['generation'],
                sphere_data[len(sphere_data)//2]['generation'],
                sphere_data[-1]['generation']
            ]
    
    # Filter data
    selected_data = [d for d in sphere_data if d['generation'] in generations_to_show]
    
    num_plots = len(selected_data)
    cols = min(3, num_plots)
    rows = (num_plots + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 5*rows))
    if num_plots == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    fig.suptitle('Sphere Configuration Comparison Across Generations', 
                 fontsize=16, fontweight='bold')
    
    for idx, data in enumerate(selected_data):
        ax = axes[idx]
        ax.set_aspect('equal')
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
        ax.grid(True, alpha=0.3)
        
        generation = data['generation']
        best_fitness = data['best_fitness']
        sphere_centers = data['sphere_centers']
        central_center = data['central_center']
        sphere_count = data['sphere_count']
        
        # Draw central circle
        central_circle = Circle(
            central_center, radius,
            color='red', fill=True, alpha=0.6
        )
        ax.add_patch(central_circle)
        
        # Draw surrounding circles
        colors = plt.cm.viridis(np.linspace(0, 1, max(len(sphere_centers), 1)))
        for i, center in enumerate(sphere_centers):
            circle = Circle(
                center, radius,
                facecolor=colors[i % len(colors)],
                fill=True, alpha=0.5,
                edgecolor='black', linewidth=1.5
            )
            ax.add_patch(circle)
        
        ax.set_title(f'Generation {generation}\nFitness: {best_fitness:.4f}, Spheres: {sphere_count}',
                    fontsize=12, fontweight='bold')
        ax.set_xlabel('X', fontsize=10)
        ax.set_ylabel('Y', fontsize=10)
    
    # Hide unused subplots
    for idx in range(num_plots, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Comparison plot saved to: {output_file}")
    plt.close()


def main():
    """Main function"""
    import sys
    
    history_file = sys.argv[1] if len(sys.argv) > 1 else "kissing_number_results.json"
    
    print("=" * 80)
    print("CREATING SPHERE EVOLUTION ANIMATION")
    print("=" * 80)
    
    # Create animation
    create_sphere_animation(
        history_file=history_file,
        output_file="sphere_evolution_animation.gif",
        fps=2
    )
    
    # Create static comparison
    print("\n" + "=" * 80)
    print("CREATING STATIC COMPARISON")
    print("=" * 80)
    create_static_comparison(
        history_file=history_file,
        output_file="sphere_comparison.png"
    )
    
    print("\n" + "=" * 80)
    print("ANIMATION COMPLETE")
    print("=" * 80)
    print("Files created:")
    print("  - sphere_evolution_animation.gif: Animated visualization")
    print("  - sphere_comparison.png: Static comparison of key generations")


if __name__ == "__main__":
    main()

