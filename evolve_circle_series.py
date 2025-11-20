"""
Evolve Circle Series Animation
Creates four series of evolving circle center positions (n=3, 4, 5, 6)
Each series has 100 frames evolving towards optimal arrangement
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation, PillowWriter
import math
from typing import List, Tuple


def generate_initial_positions(n: int, radius: float = 1.0, noise: float = 0.5) -> np.ndarray:
    """
    Generate initial (suboptimal) positions for n circles around a central circle.
    
    Args:
        n: Number of circles
        radius: Radius of each circle
        noise: Amount of noise/randomness in initial positions
    
    Returns:
        Array of shape (n, 2) with circle centers
    """
    centers = []
    distance = 2 * radius  # Distance from center to circle center
    
    for i in range(n):
        # Start with optimal angle
        optimal_angle = 2 * math.pi * i / n
        
        # Add noise to angle
        angle = optimal_angle + np.random.uniform(-noise, noise)
        
        # Add some noise to distance
        actual_distance = distance * (1 + np.random.uniform(-0.1, 0.1))
        
        x = actual_distance * math.cos(angle)
        y = actual_distance * math.sin(angle)
        centers.append([x, y])
    
    return np.array(centers)


def optimize_towards_optimal(current_centers: np.ndarray, n: int, 
                             radius: float = 1.0, step_size: float = 0.05) -> np.ndarray:
    """
    Move current centers one step towards optimal arrangement.
    
    Args:
        current_centers: Current circle centers (n, 2)
        n: Number of circles
        radius: Radius of each circle
        step_size: How much to move towards optimal per step
    
    Returns:
        Updated centers (n, 2)
    """
    optimal_centers = []
    distance = 2 * radius
    
    for i in range(n):
        optimal_angle = 2 * math.pi * i / n
        optimal_x = distance * math.cos(optimal_angle)
        optimal_y = distance * math.sin(optimal_angle)
        optimal_centers.append([optimal_x, optimal_y])
    
    optimal_centers = np.array(optimal_centers)
    
    # Move towards optimal
    direction = optimal_centers - current_centers
    new_centers = current_centers + step_size * direction
    
    return new_centers


def generate_series(n: int, num_frames: int = 100, radius: float = 1.0) -> List[np.ndarray]:
    """
    Generate a series of evolving circle center positions.
    
    Args:
        n: Number of circles
        num_frames: Number of frames in the series
        radius: Radius of each circle
    
    Returns:
        List of arrays, each containing n circle centers
    """
    series = []
    
    # Start with initial (suboptimal) positions
    current_centers = generate_initial_positions(n, radius, noise=0.8)
    
    # Gradually reduce step size for smoother convergence
    for frame in range(num_frames):
        series.append(current_centers.copy())
        
        # Adaptive step size - smaller steps as we get closer to optimal
        progress = frame / num_frames
        step_size = 0.1 * (1 - progress * 0.7)  # Start at 0.1, end at 0.03
        
        # Optimize towards optimal
        current_centers = optimize_towards_optimal(
            current_centers, n, radius, step_size=step_size
        )
    
    return series


def create_animation(series_dict: dict, output_file: str = "circle_evolution.gif", 
                   fps: int = 10, radius: float = 1.0):
    """
    Create animation showing all four series evolving.
    
    Args:
        series_dict: Dictionary with keys 3, 4, 5, 6, each containing a list of center arrays
        output_file: Path to save the GIF
        fps: Frames per second
        radius: Radius of circles
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 14))
    axes = axes.flatten()
    
    # Set up each subplot
    for ax in axes:
        ax.set_aspect('equal')
        ax.set_xlim(-4, 4)
        ax.set_ylim(-4, 4)
        ax.axis('off')
        ax.set_facecolor('white')
    
    # Store patches for each subplot
    central_circles = [None] * 4
    sphere_circles = [[] for _ in range(4)]
    text_elements = [[] for _ in range(4)]
    
    # Get number of frames (should be 100 for each series)
    num_frames = len(series_dict[3])
    
    # Colors for each series
    series_colors = {
        3: 'orange',
        4: 'blue',
        5: 'purple',
        6: 'green'
    }
    
    def animate(frame):
        """Animation function"""
        nonlocal central_circles, sphere_circles, text_elements
        
        # Clear previous elements
        for i in range(4):
            if central_circles[i]:
                central_circles[i].remove()
            for circle in sphere_circles[i]:
                circle.remove()
            for text in text_elements[i]:
                text.remove()
            sphere_circles[i].clear()
            text_elements[i].clear()
        
        # Draw each series
        for idx, n in enumerate([3, 4, 5, 6]):
            ax = axes[idx]
            
            # Get centers for this frame
            centers = series_dict[n][frame]
            
            # Draw central circle
            central_circles[idx] = patches.Circle(
                [0, 0], radius,
                facecolor='red', fill=True, alpha=0.7,
                linewidth=2, edgecolor='darkred'
            )
            ax.add_patch(central_circles[idx])
            
            # Draw surrounding circles
            color = series_colors[n]
            cmap = plt.colormaps['viridis']
            colors = [cmap(i / max(n, 1)) for i in range(n)]
            
            for i, center in enumerate(centers):
                circle = patches.Circle(
                    center, radius,
                    facecolor=colors[i], fill=True,
                    alpha=0.6, edgecolor='black', linewidth=1.5
                )
                ax.add_patch(circle)
                sphere_circles[idx].append(circle)
            
            # Add title and frame info
            progress = (frame + 1) / num_frames
            title = ax.text(
                0, 0.95, f'n = {n} circles',
                fontsize=18, fontweight='bold', ha='center',
                transform=ax.transAxes, color=color
            )
            text_elements[idx].append(title)
            
            frame_text = ax.text(
                0, 0.05, f'Frame {frame+1}/{num_frames}',
                fontsize=12, ha='center',
                transform=ax.transAxes,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8)
            )
            text_elements[idx].append(frame_text)
            
            # Add progress indicator
            progress_text = ax.text(
                0, -0.05, f'Progress: {progress*100:.0f}%',
                fontsize=10, ha='center',
                transform=ax.transAxes, style='italic'
            )
            text_elements[idx].append(progress_text)
        
        # Return all elements
        all_elements = []
        for i in range(4):
            if central_circles[i]:
                all_elements.append(central_circles[i])
            all_elements.extend(sphere_circles[i])
            all_elements.extend(text_elements[i])
        return all_elements
    
    # Create animation
    print(f"Creating animation with {num_frames} frames...")
    anim = FuncAnimation(
        fig, animate, frames=num_frames,
        interval=1000/fps, blit=False, repeat=True
    )
    
    # Save animation
    try:
        writer = PillowWriter(fps=fps)
        anim.save(output_file, writer=writer)
        print(f"✓ Animation saved to: {output_file}")
    except Exception as e:
        print(f"Error saving animation: {e}")
        try:
            anim.save(output_file.replace('.gif', '.mp4'), writer='ffmpeg', fps=fps)
            print(f"✓ Animation saved as MP4")
        except Exception as e2:
            print(f"Error: {e2}")
    
    plt.close()


def main():
    """Main function"""
    print("=" * 80)
    print("EVOLVING CIRCLE SERIES ANIMATION")
    print("=" * 80)
    print("\nGenerating four series:")
    print("  - Series 1: n = 3 circles (100 frames)")
    print("  - Series 2: n = 4 circles (100 frames)")
    print("  - Series 3: n = 5 circles (100 frames)")
    print("  - Series 4: n = 6 circles (100 frames)")
    print("\nEach series evolves towards optimal arrangement...\n")
    
    # Generate all series
    series_dict = {}
    for n in [3, 4, 5, 6]:
        print(f"Generating series for n = {n}...")
        series_dict[n] = generate_series(n, num_frames=100, radius=1.0)
        print(f"  ✓ Generated {len(series_dict[n])} frames")
    
    print("\n" + "=" * 80)
    print("CREATING ANIMATION")
    print("=" * 80)
    
    # Create animation
    create_animation(
        series_dict,
        output_file="circle_evolution.gif",
        fps=10
    )
    
    print("\n" + "=" * 80)
    print("COMPLETE")
    print("=" * 80)
    print("\nAnimation saved to: circle_evolution.gif")
    print("\nSeries structure:")
    for n in [3, 4, 5, 6]:
        print(f"  Series {n}: {len(series_dict[n])} lists of {n} circle centers")
        print(f"    First frame centers shape: {series_dict[n][0].shape}")
        print(f"    Last frame centers shape: {series_dict[n][-1].shape}")


if __name__ == "__main__":
    main()

