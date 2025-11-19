"""
Sphere Position Solver for Kissing Number Problem
Computes the positions of spheres/circles for the kissing number problem
"""

import numpy as np
from typing import List, Tuple, Dict, Any
import math


def compute_kissing_number_spheres(dimension: int = 2, num_spheres: int = None) -> Dict[str, Any]:
    """
    Compute sphere positions for kissing number problem
    
    Args:
        dimension: Dimension of the problem (2 for circles, 3 for spheres)
        num_spheres: Number of spheres to arrange (None to find maximum)
        
    Returns:
        Dictionary with sphere_count, centers, and other info
    """
    if dimension == 2:
        # For 2D, arrange circles around a central circle
        # The kissing number is 6 for dimension 2
        if num_spheres is None:
            num_spheres = 6
        
        # Central circle at origin
        central_center = [0.0, 0.0]
        radius = 1.0  # Unit circles
        
        # Arrange circles around central circle
        # Each circle touches the central circle, so distance from origin is 2*radius = 2
        centers = []
        for i in range(num_spheres):
            angle = 2 * math.pi * i / num_spheres
            x = 2 * radius * math.cos(angle)
            y = 2 * radius * math.sin(angle)
            centers.append([x, y])
        
        return {
            'sphere_count': num_spheres,
            'centers': centers,
            'central_center': central_center,
            'radius': radius,
            'dimension': dimension
        }
    
    elif dimension == 3:
        # For 3D, arrange spheres around a central sphere
        # The kissing number is 12 for dimension 3
        if num_spheres is None:
            num_spheres = 12
        
        # Central sphere at origin
        central_center = [0.0, 0.0, 0.0]
        radius = 1.0
        
        # For 3D, we can use icosahedral arrangement
        # This is more complex - using a simplified approach
        centers = []
        
        # First layer: 6 spheres in a plane (like 2D)
        for i in range(6):
            angle = 2 * math.pi * i / 6
            x = 2 * radius * math.cos(angle)
            y = 2 * radius * math.sin(angle)
            z = 0.0
            centers.append([x, y, z])
        
        # Second layer: 3 spheres above
        for i in range(3):
            angle = 2 * math.pi * i / 3
            x = radius * math.cos(angle)
            y = radius * math.sin(angle)
            z = math.sqrt(3) * radius
            centers.append([x, y, z])
        
        # Third layer: 3 spheres below
        for i in range(3):
            angle = 2 * math.pi * i / 3
            x = radius * math.cos(angle)
            y = radius * math.sin(angle)
            z = -math.sqrt(3) * radius
            centers.append([x, y, z])
        
        # Limit to requested number
        centers = centers[:num_spheres]
        
        return {
            'sphere_count': len(centers),
            'centers': centers,
            'central_center': central_center,
            'radius': radius,
            'dimension': dimension
        }
    
    else:
        raise ValueError(f"Dimension {dimension} not yet supported")


def extract_sphere_info_from_genome(genome: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Extract sphere information from a genome/prompt
    
    This attempts to parse the genome to determine:
    - Number of spheres mentioned
    - Dimension specified
    
    Args:
        genome: The evolved prompt/genome
        context: Optional context
        
    Returns:
        Dictionary with sphere_count and dimension
    """
    import re
    
    genome_lower = genome.lower()
    
    # Try to extract number of spheres
    sphere_count = None
    
    # Look for explicit numbers
    numbers = re.findall(r'\b(\d+)\b', genome)
    if numbers:
        # Check if any number is mentioned in context of spheres/circles
        for num_str in numbers:
            num = int(num_str)
            # Check if number appears near sphere/circle keywords
            num_pos = genome.lower().find(num_str)
            nearby_text = genome_lower[max(0, num_pos-20):min(len(genome_lower), num_pos+20)]
            if any(keyword in nearby_text for keyword in ['sphere', 'circle', 'kissing', 'number']):
                sphere_count = num
                break
    
    # Default to 6 for dimension 2 if not found
    if sphere_count is None:
        sphere_count = 6
    
    # Determine dimension
    dimension = 2  # Default
    if '3d' in genome_lower or 'three dimension' in genome_lower or '3-dimensional' in genome_lower:
        dimension = 3
    elif '2d' in genome_lower or 'two dimension' in genome_lower or '2-dimensional' in genome_lower:
        dimension = 2
    
    # Compute sphere positions
    sphere_info = compute_kissing_number_spheres(dimension=dimension, num_spheres=sphere_count)
    
    return sphere_info


def get_best_sphere_configuration(population: List, context: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Get the best sphere configuration from a population
    
    Args:
        population: List of individuals
        context: Optional context
        
    Returns:
        Dictionary with sphere information for best individual
    """
    if not population:
        return None
    
    # Get best individual (highest fitness)
    best = max(population, key=lambda x: x.fitness)
    
    # Extract sphere info from best genome
    sphere_info = extract_sphere_info_from_genome(best.genome, context)
    
    return {
        'best_fitness': best.fitness,
        'best_genome': best.genome,
        'generation': best.generation,
        **sphere_info
    }

