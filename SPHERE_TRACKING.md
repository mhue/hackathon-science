# Sphere Tracking in Evolution

This document explains how to track the number of spheres and their centers at each iteration of the evolution.

## Overview

The evolution system has been enhanced to track sphere information (number of spheres and their centers) at each generation. This information is computed from the best individual's genome at each iteration.

## Files Added

1. **`sphere_solver.py`** - Contains functions to compute sphere positions for the kissing number problem
2. **`analyze_spheres.py`** - Script to analyze and display sphere information from evolution history
3. **`animate_spheres.py`** - Creates animated visualization showing sphere positions at each iteration

## Usage

### Running Evolution with Sphere Tracking

The evolution loop automatically tracks sphere information when you run:

```bash
python3 kissing_number_evolve.py
```

Sphere information is stored in the evolution history for each generation.

### Analyzing Sphere Information

To view sphere information from a completed evolution run:

```bash
python3 analyze_spheres.py [results_file]
```

If no file is specified, it defaults to `kissing_number_results.json`.

### Creating Sphere Animation

To create an animated visualization showing sphere positions at each iteration:

```bash
python3 animate_spheres.py [results_file]
```

This creates two files:
- **`sphere_evolution_animation.gif`** - Animated GIF showing sphere evolution
- **`sphere_comparison.png`** - Static comparison of key generations

The animation shows:
- Red circle: Central sphere/circle
- Colored circles: Surrounding spheres arranged around the central one
- Generation number and fitness displayed at each frame

### Example Output

```
Generation 4:
  Best Fitness: 0.9600
  Number of Spheres: 6
  Dimension: 2D
  Central Center: [0.0, 0.0]
  Sphere Centers (6):
    Sphere 1: [2.0, 0.0]
    Sphere 2: [1.0, 1.732]
    Sphere 3: [-1.0, 1.732]
    Sphere 4: [-2.0, 0.0]
    Sphere 5: [-1.0, -1.732]
    Sphere 6: [1.0, -1.732]
```

## How It Works

1. **During Evolution**: At each generation, the evolution loop:
   - Evaluates all individuals
   - Identifies the best individual
   - Extracts sphere information from the best genome
   - Stores sphere count, centers, and dimension in the history

2. **Sphere Position Computation**: The `sphere_solver.py` module:
   - Parses the genome to extract the number of spheres mentioned
   - Determines the dimension (2D or 3D)
   - Computes optimal sphere positions using geometric calculations
   - For 2D: Arranges circles in a hexagonal pattern (kissing number = 6)
   - For 3D: Arranges spheres using icosahedral-like patterns (kissing number = 12)

3. **Analysis**: The `analyze_spheres.py` script:
   - Reads evolution history from JSON file
   - Displays sphere information for each generation
   - Computes sphere positions on-the-fly if not stored in history
   - Provides summary statistics

## Data Structure

Sphere information is stored in the evolution history with the following structure:

```python
{
    'generation': 4,
    'best_fitness': 0.96,
    'sphere_count': 6,
    'sphere_centers': [
        [2.0, 0.0],
        [1.0, 1.732],
        [-1.0, 1.732],
        [-2.0, 0.0],
        [-1.0, -1.732],
        [1.0, -1.732]
    ],
    'central_center': [0.0, 0.0],
    'dimension': 2
}
```

## Functions

### `compute_kissing_number_spheres(dimension, num_spheres)`

Computes sphere positions for a given dimension and number of spheres.

**Parameters:**
- `dimension`: 2 for circles, 3 for spheres
- `num_spheres`: Number of spheres to arrange (None for maximum)

**Returns:** Dictionary with sphere_count, centers, central_center, radius, dimension

### `extract_sphere_info_from_genome(genome, context)`

Extracts sphere information from a genome/prompt string.

**Parameters:**
- `genome`: The evolved prompt/genome string
- `context`: Optional context dictionary

**Returns:** Dictionary with sphere information

### `get_best_sphere_configuration(population, context)`

Gets sphere configuration for the best individual in a population.

**Parameters:**
- `population`: List of Individual objects
- `context`: Optional context dictionary

**Returns:** Dictionary with sphere information for best individual

## Notes

- Sphere positions are computed based on the best individual's genome at each generation
- The system attempts to parse numbers and keywords from genomes to determine sphere count
- For 2D, the default kissing number is 6 (hexagonal arrangement)
- For 3D, the default kissing number is 12 (icosahedral arrangement)
- If sphere information is missing from history, the analysis script computes it on-the-fly

