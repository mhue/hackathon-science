# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Overview

**Alpha Evolve** is an evolutionary algorithm system for optimizing prompts using genetic algorithms. The repository includes a specialized implementation for solving the kissing number problem (determining maximum sphere packing configurations in 2D/3D).

## Common Commands

### Running Evolution

```bash
# Main kissing number evolution (dimension 2)
python3 kissing_number_evolve.py

# Run base Alpha Evolve example
python3 alpha_evolve.py
```

### Analysis and Visualization

```bash
# Analyze sphere information from evolution results
python3 analyze_spheres.py [results_file]
python3 analyze_spheres.py kissing_number_results.json  # default

# Create animated visualization of sphere evolution
python3 animate_spheres.py [results_file]

# Visualize evolution progress (called automatically by evolution scripts)
python3 visualize_evolution.py
```

### Dependencies

```bash
# Core functionality: No dependencies needed (Python 3.7+ stdlib only)

# For visualization features (optional):
pip install matplotlib>=3.3.0 pillow>=8.0.0

# For sphere solver numerical operations:
pip install numpy>=1.20.0
```

## Architecture Overview

### Core Genetic Algorithm Pipeline

The system follows a modular GA architecture:

```
Evaluator → Individual → Population → EvolutionLoop → AlphaEvolve
```

**Flow:**
1. **Evaluator** (`evaluator.py`) - Computes fitness scores for prompts/genomes
2. **Individual** - Dataclass representing a genome with fitness and generation
3. **EvolutionLoop** (`evolution_loop.py`) - Implements GA operators (selection, crossover, mutation)
4. **AlphaEvolve** (`alpha_evolve.py`) - High-level API integrating all components

### Specialized Extension Pattern

Domain-specific problems extend `EvolutionLoop`:

```python
class KissingNumberEvolutionLoop(EvolutionLoop):
    # Override mutation operations with domain-specific words
    def _generate_word(self) -> str:
        # Returns math/geometry specific terms
    
    # Override crossover to reduce repetition
    def _crossover(self, parent1, parent2) -> str:
        # Custom crossover logic
```

### Sphere Tracking Integration

Evolution history automatically captures geometric information:

1. `EvolutionLoop.evolve()` calls `get_best_sphere_configuration()` each generation
2. `sphere_solver.py` extracts sphere count and dimension from best genome
3. `compute_kissing_number_spheres()` calculates optimal sphere positions
4. Results stored in history with keys: `sphere_count`, `sphere_centers`, `central_center`, `dimension`

### Visualization Pipeline

```
evolution history → visualize_evolution.py → [PNG plots]
                 → animate_spheres.py → [GIF animation]
```

**Generated outputs:**
- `evolution_plot.png` - 4-panel dashboard (fitness, diversity, convergence)
- `fitness_distribution.png` - Fitness range over generations
- `convergence_analysis.png` - Improvement rate and population convergence
- `sphere_evolution_animation.gif` - Animated sphere positions
- `sphere_comparison.png` - Static comparison of key generations

## Key Customization Points

### Custom Evaluator

Create domain-specific fitness functions:

```python
def custom_evaluator(genome: str, context: Dict[str, Any]) -> float:
    # Your evaluation logic - return 0.0 to 1.0
    return fitness_score

evaluator = CustomEvaluator(custom_evaluator)
alpha = AlphaEvolve(evaluator=evaluator)
```

**Pattern used in kissing number:**
- Award points for domain keywords (0.3 for "kissing number")
- Award points for dimension specificity (0.2 for "2D")
- Award points for mathematical reasoning (0.2 max)
- Award points for correct answer (0.15 for "6")
- Penalties for vague terms (-0.05 per term)

### Evolution Parameters

```python
alpha = AlphaEvolve(
    population_size=40,      # Individuals per generation
    mutation_rate=0.15,      # Probability of mutation (0-1)
    crossover_rate=0.7,      # Probability of crossover (0-1)
    elite_size=5,            # Top individuals preserved
    max_generations=30       # Maximum iterations
)
```

**Tuning guidance:**
- Higher `mutation_rate` increases exploration but slows convergence
- Higher `elite_size` preserves good solutions but reduces diversity
- Population too small: premature convergence; too large: slow evolution

### Specialized Mutation Operations

Extend `EvolutionLoop._mutate()` for domain-specific mutations:

```python
def _mutate(self, genome: str) -> str:
    mutation_type = random.choice(['swap', 'insert', 'delete', 'replace', 'expand'])
    
    # 'expand' adds domain-specific phrases
    if mutation_type == 'expand':
        expansions = ['using geometric reasoning', 'in two-dimensional space']
        genome += ' ' + random.choice(expansions)
```

### Template System

```python
# Register a template
alpha.register_template(
    name='my_template',
    template='Solve {{task}} using {{method}}',
    template_type=TemplateType.TASK,
    variables=['task', 'method']
)

# Use template in evolution
alpha.evolve_prompts(
    template_name='my_template',
    template_variables={'task': 'kissing number', 'method': 'geometry'}
)
```

## Important Implementation Details

### Genetic Algorithm Specifics

- **Selection:** Tournament selection with tournament size of 3
- **Crossover:** Single-point crossover at word boundaries
- **Mutation types:** swap, insert, delete, replace (word-level operations)
- **Elitism:** Top `elite_size` individuals always preserved
- **Convergence:** Detects when fitness improvement plateaus

### Default Evaluation Criteria

Base `Evaluator` uses weighted criteria (total = 1.0):
- Length: 0.1 (optimal 200-500 words)
- Clarity: 0.3 (sentence length, clarity indicators)
- Specificity: 0.3 (numbers, specific terms, examples)
- Completeness: 0.3 (required elements present)

**Override with `CustomEvaluator` for domain-specific evaluation.**

### Kissing Number Solutions

Known answers for reference:
- **Dimension 2:** 6 circles (hexagonal arrangement)
- **Dimension 3:** 12 spheres (icosahedral-like arrangement)

`sphere_solver.py` computes positions at distance 2*radius from center:
- 2D: `angle = 2π * i / num_spheres`
- 3D: Layered approach (6 in plane, 3 above, 3 below)

### Results Persistence

Evolution results saved to JSON:

```python
alpha.save_results('results.json', best_individual)
```

**Structure:**
```json
{
  "best_individual": {
    "genome": "...",
    "fitness": 0.96,
    "generation": 15
  },
  "evolution_history": [
    {
      "generation": 0,
      "best_fitness": 0.65,
      "avg_fitness": 0.52,
      "worst_fitness": 0.31,
      "std_fitness": 0.12,
      "sphere_count": 6,
      "sphere_centers": [[2.0, 0.0], ...],
      "dimension": 2
    }
  ],
  "top_prompts": [...]
}
```

### History Tracking

`EvolutionLoop.history` stores per-generation statistics:
- `generation`, `best_fitness`, `avg_fitness`, `worst_fitness`, `std_fitness`
- `sphere_count`, `sphere_centers`, `central_center`, `dimension` (if sphere solver available)

Access with: `alpha.get_evolution_history()`

## Working with Sphere Analysis

The sphere solver attempts to extract geometric information from evolved genomes:

1. **Number extraction:** Regex finds numbers near sphere/circle keywords
2. **Dimension detection:** Looks for "2D", "3D", "dimension X" in genome
3. **Position computation:** Uses trigonometry to place spheres at optimal distances
4. **Defaults:** If unclear, assumes 6 spheres in dimension 2

To add sphere tracking to custom evolution:

```python
from sphere_solver import get_best_sphere_configuration

# In your evolution loop, after evaluation:
sphere_info = get_best_sphere_configuration(population, context)
history_entry['sphere_count'] = sphere_info['sphere_count']
history_entry['sphere_centers'] = sphere_info['centers']
```

## Extending Visualization

Create custom plots by importing visualization functions:

```python
from visualize_evolution import create_comprehensive_visualization

history = alpha.get_evolution_history()
create_comprehensive_visualization(history, output_dir="./plots")
```

**Customization:** Modify plot functions in `visualize_evolution.py` to add domain-specific metrics or change aesthetics. All plots use matplotlib with 'Agg' backend for non-interactive environments.
