# Kissing Number Evolution - Dimension 2

This module focuses on evolving an agent/prompt that can find the kissing number for dimension 2.

## Problem Description

The **kissing number problem** asks: What is the maximum number of non-overlapping unit circles (or spheres in higher dimensions) that can be arranged so that they all touch a central unit circle?

For **dimension 2**, the answer is **6**. You can arrange 6 unit circles around a central unit circle in a hexagonal pattern.

## Files

- `kissing_number_evolve.py` - Main evolution script
- `kissing_number_evolution.py` - Specialized evolution loop with domain-specific mutations

## Usage

Run the evolution:

```bash
python kissing_number_evolve.py
```

This will:
1. Initialize a population with seed prompts about the kissing number problem
2. Evolve the prompts using a genetic algorithm
3. Evaluate fitness based on:
   - Mentioning the kissing number problem
   - Specifying dimension 2
   - Including geometric/mathematical reasoning
   - Leading toward the answer (6)
4. Display the best evolved agents
5. Save results to `kissing_number_results.json`

## Evaluation Criteria

The fitness function rewards prompts that:
- ✓ Mention "kissing number" or related terms (0.3 points)
- ✓ Specify dimension 2 or 2D (0.2 points)
- ✓ Include geometric/mathematical keywords (0.2 points)
- ✓ Reference the answer 6 or hexagonal arrangements (0.15 points)
- ✓ Include problem-solving approaches (0.15 points)
- ✓ Have appropriate length and clarity (0.05 points)

## Expected Output

The evolution should produce prompts/agents that:
- Clearly state the kissing number problem for dimension 2
- Include geometric reasoning about circle arrangements
- Potentially mention hexagonal patterns or the answer 6
- Provide a clear approach to solving the problem

## Example Seed Prompts

The evolution starts with prompts like:
- "Find the kissing number for dimension 2. How many unit circles can touch a central unit circle?"
- "Determine the maximum number of non-overlapping circles that can be arranged around a central circle in 2D."
- "Solve the kissing number problem for dimension 2 using geometric reasoning."

## Customization

You can modify:
- `kissing_number_evaluator()` - Adjust fitness evaluation criteria
- `get_seed_prompts()` - Change initial population
- Evolution parameters in `main()` - Population size, mutation rate, etc.

