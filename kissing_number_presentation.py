"""
Kissing Number Problem - Presentation Animation
Creates an educational animation explaining the kissing number problem
in 2D, showing the progression from 4, 5, to 6 spheres.
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation, PillowWriter
import math


def compute_sphere_positions(num_spheres: int, radius: float = 1.0) -> tuple:
    """
    Compute positions for spheres arranged around a central sphere.
    
    Args:
        num_spheres: Number of surrounding spheres
        radius: Radius of each sphere (unit circles)
    
    Returns:
        Tuple of (central_center, sphere_centers)
    """
    central_center = np.array([0.0, 0.0])
    sphere_centers = []
    
    # Distance from center to sphere center = 2 * radius (touching)
    distance = 2 * radius
    
    for i in range(num_spheres):
        angle = 2 * math.pi * i / num_spheres
        x = distance * math.cos(angle)
        y = distance * math.sin(angle)
        sphere_centers.append(np.array([x, y]))
    
    return central_center, np.array(sphere_centers)


def check_overlap(centers: np.ndarray, radius: float = 1.0) -> bool:
    """
    Check if any surrounding spheres overlap with each other.
    
    Args:
        centers: Array of sphere centers
        radius: Radius of each sphere
    
    Returns:
        True if there's overlap, False otherwise
    """
    n = len(centers)
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(centers[i] - centers[j])
            if dist < 2 * radius - 1e-6:  # Small tolerance for floating point
                return True
    return False


def create_presentation_animation(
    output_file: str = "kissing_number_presentation.gif",
    fps: int = 1,
    radius: float = 1.0,
    pause_frames: int = 30
):
    """
    Create a presentation animation explaining the kissing number problem.
    
    Args:
        output_file: Path to save the animation
        fps: Frames per second
        pause_frames: Number of frames to pause on each configuration
    """
    # Set up figure with larger size for presentation
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_aspect('equal')
    ax.set_xlim(-4.5, 4.5)
    ax.set_ylim(-4.5, 4.5)
    ax.axis('off')  # Remove axes for cleaner presentation
    
    # Store patches
    central_circle = None
    sphere_circles = []
    text_elements = []
    line_elements = []
    
    # Animation frames structure
    # Each configuration gets: intro frame + optimization frames + final frame
    configurations = [
        {
            'num_spheres': 4,
            'title': '4 Spheres',
            'description': 'Can we fit 4 circles around a central circle?',
            'result': 'Yes! But is this the maximum?',
            'color': 'orange'
        },
        {
            'num_spheres': 5,
            'title': '5 Spheres',
            'description': 'Can we fit 5 circles around a central circle?',
            'result': 'Yes! But can we fit more?',
            'color': 'blue'
        },
        {
            'num_spheres': 6,
            'title': '6 Spheres',
            'description': 'Can we fit 6 circles around a central circle?',
            'result': 'Yes! This is the kissing number for 2D!',
            'color': 'green'
        }
    ]
    
    # Calculate total frames
    frames_per_config = pause_frames + 10  # intro + pause + transition
    total_frames = len(configurations) * frames_per_config + 20  # + intro/outro
    
    def animate(frame):
        """Animation function"""
        nonlocal central_circle, sphere_circles, text_elements
        
        # Clear previous elements
        nonlocal line_elements
        if central_circle:
            central_circle.remove()
        for circle in sphere_circles:
            circle.remove()
        for text in text_elements:
            text.remove()
        for line in line_elements:
            line.remove()
        sphere_circles.clear()
        text_elements.clear()
        line_elements.clear()
        
        # Introduction frame
        if frame < 10:
            title_text = ax.text(0, 0.3, 'The Kissing Number Problem', 
                               fontsize=32, fontweight='bold', ha='center',
                               transform=ax.transAxes)
            subtitle_text = ax.text(0, 0.15, 
                                   'How many unit circles can touch a central unit circle?',
                                   fontsize=20, ha='center', transform=ax.transAxes)
            problem_text = ax.text(0, -0.1,
                                  'In 2D, the answer is 6',
                                  fontsize=18, ha='center', transform=ax.transAxes,
                                  style='italic')
            text_elements = [title_text, subtitle_text, problem_text]
            return text_elements
        
        # Adjust frame for configurations
        frame -= 10
        config_idx = frame // frames_per_config
        
        if config_idx >= len(configurations):
            # Outro frame
            title_text = ax.text(0, 0.2, 'Kissing Number in 2D = 6', 
                               fontsize=36, fontweight='bold', ha='center',
                               transform=ax.transAxes, color='green')
            subtitle_text = ax.text(0, 0.05,
                                   'This is the maximum number of non-overlapping',
                                   fontsize=18, ha='center', transform=ax.transAxes)
            problem_text = ax.text(0, -0.05,
                                  'unit circles that can touch a central unit circle',
                                  fontsize=18, ha='center', transform=ax.transAxes)
            text_elements = [title_text, subtitle_text, problem_text]
            return text_elements
        
        config = configurations[config_idx]
        num_spheres = config['num_spheres']
        local_frame = frame % frames_per_config
        
        # Compute sphere positions
        central_center, sphere_centers = compute_sphere_positions(num_spheres, radius)
        
        # Draw central circle
        central_circle = patches.Circle(
            central_center, radius,
            facecolor='red', fill=True, alpha=0.7,
            linewidth=2, edgecolor='darkred'
        )
        ax.add_patch(central_circle)
        
        # Draw surrounding circles with animation effect
        config_color = config['color']
        colors = matplotlib.colormaps['viridis'](np.linspace(0.3, 0.9, num_spheres))
        
        # Animate appearance
        alpha_factor = min(1.0, local_frame / 5.0) if local_frame < 5 else 1.0
        
        for i, center in enumerate(sphere_centers):
            circle = patches.Circle(
                center, radius,
                facecolor=colors[i], fill=True,
                alpha=0.6 * alpha_factor,
                edgecolor='black', linewidth=2
            )
            ax.add_patch(circle)
            sphere_circles.append(circle)
        
        # Add title
        title_text = ax.text(0, 0.95, config['title'], 
                           fontsize=28, fontweight='bold', ha='center',
                           transform=ax.transAxes, color=config_color)
        text_elements.append(title_text)
        
        # Add description
        if local_frame < pause_frames // 2:
            desc_text = ax.text(0, 0.88, config['description'],
                              fontsize=18, ha='center', transform=ax.transAxes)
            text_elements.append(desc_text)
        else:
            result_text = ax.text(0, 0.88, config['result'],
                                fontsize=18, ha='center', transform=ax.transAxes,
                                color=config_color, fontweight='bold')
            text_elements.append(result_text)
        
        # Add mathematical info
        overlap = check_overlap(sphere_centers, radius)
        info_y = 0.05
        info_text = ax.text(0, info_y,
                          f'Spheres: {num_spheres} | Overlap: {"Yes" if overlap else "No"}',
                          fontsize=14, ha='center', transform=ax.transAxes,
                          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        text_elements.append(info_text)
        
        # Add angle information for 6 spheres
        if num_spheres == 6 and local_frame > pause_frames // 2:
            angle_text = ax.text(0, -0.1,
                               'Each circle is 60° apart - perfect hexagonal arrangement!',
                               fontsize=14, ha='center', transform=ax.transAxes,
                               style='italic', color='green')
            text_elements.append(angle_text)
        
        # Draw lines from center to sphere centers for clarity
        if local_frame > 3:
            for center in sphere_centers:
                line = ax.plot([0, center[0]], [0, center[1]], 
                             'k--', alpha=0.3, linewidth=1)[0]
                line_elements.append(line)
        
        return [central_circle] + sphere_circles + text_elements + line_elements
    
    # Create animation
    print(f"Creating presentation animation with {total_frames} frames...")
    anim = FuncAnimation(
        fig, animate, frames=total_frames,
        interval=1000/fps, blit=False, repeat=True
    )
    
    # Save animation
    try:
        writer = PillowWriter(fps=fps)
        anim.save(output_file, writer=writer)
        print(f"✓ Animation saved to: {output_file}")
    except Exception as e:
        print(f"Error saving animation: {e}")
        # Try alternative
        try:
            anim.save(output_file.replace('.gif', '.mp4'), writer='ffmpeg', fps=fps)
            print(f"✓ Animation saved as MP4")
        except Exception as e2:
            print(f"Error: {e2}")
    
    plt.close()


def create_detailed_animation(
    output_file: str = "kissing_number_detailed.gif",
    fps: int = 2,
    radius: float = 1.0
):
    """
    Create a more detailed animation showing the optimization process.
    Shows gradual transition from 4 -> 5 -> 6 spheres.
    """
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_aspect('equal')
    ax.set_xlim(-4.5, 4.5)
    ax.set_ylim(-4.5, 4.5)
    ax.axis('off')
    
    central_circle = None
    sphere_circles = []
    text_elements = []
    line_elements = []
    
    # Frame structure: show gradual optimization
    total_frames = 200
    
    def animate(frame):
        nonlocal central_circle, sphere_circles, text_elements, line_elements
        
        # Clear
        if central_circle:
            central_circle.remove()
        for circle in sphere_circles:
            circle.remove()
        for text in text_elements:
            text.remove()
        for line in line_elements:
            line.remove()
        sphere_circles.clear()
        text_elements.clear()
        line_elements.clear()
        
        # Determine current number of spheres based on frame
        if frame < 30:
            # Introduction
            num_spheres = 0
            title_text = ax.text(0, 0.3, 'Kissing Number Problem - 2D', 
                               fontsize=32, fontweight='bold', ha='center',
                               transform=ax.transAxes)
            subtitle_text = ax.text(0, 0.15,
                                   'Optimization Process: Finding the Maximum',
                                   fontsize=20, ha='center', transform=ax.transAxes)
            text_elements = [title_text, subtitle_text]
        elif frame < 80:
            # Show 4 spheres
            num_spheres = 4
            progress = (frame - 30) / 50.0
        elif frame < 130:
            # Transition 4 -> 5
            num_spheres = 4
            if (frame - 80) % 10 < 5:
                num_spheres = 5
            progress = 1.0
        elif frame < 180:
            # Show 5 spheres
            num_spheres = 5
            progress = min(1.0, (frame - 130) / 50.0)
        elif frame < 200:
            # Transition 5 -> 6
            num_spheres = 5
            if (frame - 180) % 10 < 5:
                num_spheres = 6
            progress = 1.0
        else:
            # Final: 6 spheres
            num_spheres = 6
            progress = 1.0
        
        if num_spheres == 0:
            return text_elements
        
        # Compute positions
        central_center, sphere_centers = compute_sphere_positions(num_spheres, radius)
        
        # Draw central circle
        central_circle = patches.Circle(
            central_center, radius,
            facecolor='red', fill=True, alpha=0.7,
            linewidth=2.5, edgecolor='darkred'
        )
        ax.add_patch(central_circle)
        
        # Draw surrounding circles
        colors = matplotlib.colormaps['viridis'](np.linspace(0.2, 0.9, max(num_spheres, 1)))
        
        for i, center in enumerate(sphere_centers):
            alpha = 0.6 * progress
            circle = patches.Circle(
                center, radius,
                facecolor=colors[i % len(colors)], fill=True,
                alpha=alpha, edgecolor='black', linewidth=2
            )
            ax.add_patch(circle)
            sphere_circles.append(circle)
        
        # Title
        title_text = ax.text(0, 0.95, f'Step: {num_spheres} Spheres', 
                           fontsize=24, fontweight='bold', ha='center',
                           transform=ax.transAxes)
        text_elements.append(title_text)
        
        # Status
        overlap = check_overlap(sphere_centers, radius)
        status_text = ax.text(0, 0.05,
                            f'Spheres: {num_spheres} | Valid: {"✓" if not overlap else "✗"}',
                            fontsize=16, ha='center', transform=ax.transAxes,
                            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        text_elements.append(status_text)
        
        # Progress indicator
        if num_spheres < 6:
            progress_text = ax.text(0, -0.1,
                                   f'Optimizing... trying to fit more spheres',
                                   fontsize=14, ha='center', transform=ax.transAxes,
                                   style='italic')
            text_elements.append(progress_text)
        else:
            final_text = ax.text(0, -0.1,
                               '✓ Maximum reached! Kissing number = 6',
                               fontsize=16, ha='center', transform=ax.transAxes,
                               color='green', fontweight='bold')
            text_elements.append(final_text)
        
        return [central_circle] + sphere_circles + text_elements + line_elements
    
    print(f"Creating detailed animation with {total_frames} frames...")
    anim = FuncAnimation(
        fig, animate, frames=total_frames,
        interval=1000/fps, blit=False, repeat=True
    )
    
    try:
        writer = PillowWriter(fps=fps)
        anim.save(output_file, writer=writer)
        print(f"✓ Detailed animation saved to: {output_file}")
    except Exception as e:
        print(f"Error: {e}")
    
    plt.close()


def main():
    """Main function"""
    print("=" * 80)
    print("KISSING NUMBER PROBLEM - PRESENTATION ANIMATION")
    print("=" * 80)
    print("\nCreating two animations:")
    print("1. Presentation animation (slow, educational)")
    print("2. Detailed optimization animation\n")
    
    # Create presentation animation (slower, more educational)
    create_presentation_animation(
        output_file="kissing_number_presentation.gif",
        fps=1,  # Slow for presentation
        pause_frames=40
    )
    
    # Create detailed optimization animation
    create_detailed_animation(
        output_file="kissing_number_detailed.gif",
        fps=2
    )
    
    print("\n" + "=" * 80)
    print("ANIMATIONS COMPLETE")
    print("=" * 80)
    print("\nFiles created:")
    print("  - kissing_number_presentation.gif: Educational presentation")
    print("  - kissing_number_detailed.gif: Detailed optimization process")


if __name__ == "__main__":
    main()

