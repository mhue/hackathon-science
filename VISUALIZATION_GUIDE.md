# Evolution Visualization Guide

This guide explains the visualization plots generated during the kissing number evolution.

## Generated Plots

After running `kissing_number_evolve.py`, three visualization files are created:

### 1. `evolution_plot.png` (Main Dashboard)
A comprehensive 4-panel dashboard showing:

- **Top Left: Fitness Evolution**
  - Best fitness (green line with circles)
  - Average fitness (blue line with squares)
  - Worst fitness (red line with triangles)
  - Shows how the entire population improves over generations

- **Top Right: Best Fitness Over Time**
  - Focused view of the best individual's fitness
  - Annotated with the maximum fitness value and generation
  - Green shaded area showing the improvement

- **Bottom Left: Population Diversity**
  - Standard deviation of fitness scores
  - Higher values = more diverse population
  - Lower values = population converging (similar fitness)

- **Bottom Right: Average Fitness Over Time**
  - Shows overall population improvement
  - Annotated with total improvement from first to last generation

### 2. `fitness_distribution.png`
Shows the fitness range across generations:

- **Shaded gray area**: Range between worst and best fitness
- **Green line**: Best fitness
- **Blue line**: Average fitness  
- **Red line**: Worst fitness
- Useful for seeing how the fitness distribution narrows as evolution progresses

### 3. `convergence_analysis.png`
Two-panel analysis of convergence:

- **Left Panel: Fitness Improvement Rate**
  - Shows how much fitness improves each generation
  - Green areas = improvements
  - Red areas = declines
  - Helps identify when evolution plateaus

- **Right Panel: Population Convergence**
  - Gap between best and average fitness
  - Lower values = population is converging (similar individuals)
  - Higher values = diverse population with varying fitness

## Interpreting the Plots

### Good Evolution Signs:
- ✅ Steady upward trend in best fitness
- ✅ Average fitness increasing over time
- ✅ Population diversity decreasing (converging toward good solutions)
- ✅ Best fitness reaching near 1.0 (maximum)

### Potential Issues:
- ⚠️ Flat line = evolution stalled (may need more generations or higher mutation rate)
- ⚠️ High variance = population too diverse (may need more selection pressure)
- ⚠️ Declining fitness = evolution going wrong (check mutation/crossover rates)

## Example Interpretation

For the kissing number problem:
- **Fitness 1.0** = Perfect solution (mentions answer "6", all keywords present)
- **Fitness 0.7-0.9** = Good solution (most criteria met)
- **Fitness < 0.5** = Poor solution (missing key elements)

The evolution should show:
1. Initial fitness around 0.5-0.7 (seed prompts are decent)
2. Rapid improvement in early generations
3. Gradual convergence toward fitness 1.0
4. Population becoming more similar (lower diversity) as good solutions dominate

## Usage

Visualizations are automatically generated when you run:
```bash
python3 kissing_number_evolve.py
```

The plots are saved in the current directory and can be viewed with any image viewer or included in reports.

