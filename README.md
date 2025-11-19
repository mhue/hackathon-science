# Alpha Evolve

An evolutionary algorithm system for evolving and optimizing prompts using genetic algorithms. Alpha Evolve combines evaluator code, evolution loops, and prompt templates to automatically improve prompt quality.

## Features

- **Evaluator Module**: Comprehensive fitness evaluation with multiple criteria (length, clarity, specificity, completeness)
- **Evolution Loop**: Full genetic algorithm implementation with selection, crossover, and mutation
- **Prompt Templates**: Template system for structured prompt generation and evolution
- **Integrated System**: Complete Alpha Evolve system combining all components

## Components

### 1. Evaluator (`evaluator.py`)
Evaluates fitness of prompts using multiple criteria:
- Length optimization
- Clarity scoring
- Specificity measurement
- Completeness assessment

### 2. Evolution Loop (`evolution_loop.py`)
Genetic algorithm implementation:
- Population initialization
- Fitness evaluation
- Selection (tournament selection)
- Crossover (single-point)
- Mutation (swap, insert, delete, replace)
- Elitism
- Convergence detection

### 3. Prompt Template (`prompt_template.py`)
Template management system:
- Template registration and management
- Variable extraction and filling
- Template evolution operations
- Predefined templates for common use cases

### 4. Alpha Evolve (`alpha_evolve.py`)
Main integration module combining all components:
- Unified API for prompt evolution
- Template-based prompt generation
- Result saving/loading
- Custom evaluator support

## Installation

```bash
# Clone or download the repository
cd hackathon-science

# No external dependencies required!
# Uses only Python standard library (Python 3.7+)
```

## Quick Start

### Basic Usage

```python
from alpha_evolve import AlphaEvolve

# Initialize Alpha Evolve
alpha = AlphaEvolve(
    population_size=30,
    mutation_rate=0.15,
    crossover_rate=0.7,
    elite_size=3,
    max_generations=20
)

# Evolve prompts from seed
seed_prompts = [
    "Write a Python function to calculate fibonacci numbers",
    "Create a function that computes fibonacci sequence"
]

best = alpha.evolve_prompts(seed_prompts=seed_prompts, verbose=True)

print(f"Best prompt: {best.genome}")
print(f"Fitness: {best.fitness}")
```

### Using Templates

```python
# Evolve using a template
best = alpha.evolve_prompts(
    template_name='code_generation',
    template_variables={
        'language': 'Python',
        'task': 'sorts a list',
        'requirements': 'be efficient',
        'output_format': 'the sorted list'
    }
)
```

### Custom Evaluator

```python
def my_evaluator(prompt: str, context: dict) -> float:
    # Your custom evaluation logic
    score = 0.0
    if 'Python' in prompt:
        score += 0.3
    if 'function' in prompt:
        score += 0.3
    if len(prompt) > 50:
        score += 0.4
    return min(score, 1.0)

alpha.set_custom_evaluator(my_evaluator)
```

### Evaluate Single Prompt

```python
evaluation = alpha.evaluate_prompt("Your prompt here")
print(evaluation)
# Output: {
#     'fitness': 0.85,
#     'length_score': 0.9,
#     'clarity_score': 0.8,
#     ...
# }
```

## Examples

Run the example script:

```bash
python alpha_evolve.py
```

This demonstrates:
1. Evolving prompts from seed
2. Using templates
3. Evaluating prompts
4. Getting top results

## API Reference

### AlphaEvolve Class

#### Methods

- `evolve_prompts(seed_prompts, template_name, template_variables, context, verbose)` - Evolve prompts
- `evaluate_prompt(prompt, context)` - Evaluate a single prompt
- `register_template(name, template, template_type, description, variables)` - Register new template
- `get_best_prompts(n)` - Get top N evolved prompts
- `get_evolution_history()` - Get evolution statistics history
- `set_custom_evaluator(evaluation_function)` - Set custom evaluation function
- `save_results(filepath, best_individual)` - Save results to JSON
- `load_results(filepath)` - Load results from JSON

### Evaluator Class

- `evaluate(individual, context)` - Evaluate fitness
- `batch_evaluate(individuals, context)` - Evaluate multiple prompts
- `get_evaluation_details(individual, context)` - Get detailed breakdown

### EvolutionLoop Class

- `initialize_population(seed_prompts)` - Initialize population
- `evolve(context, verbose)` - Run evolution loop
- `get_best_individuals(n)` - Get top individuals
- `get_history()` - Get evolution history

### TemplateManager Class

- `register_template(name, template, template_type, description, variables)` - Register template
- `get_template(name)` - Get template by name
- `evolve_template(name, operations)` - Evolve a template

## Configuration

### Evolution Parameters

- `population_size`: Number of individuals per generation (default: 50)
- `mutation_rate`: Probability of mutation (default: 0.1)
- `crossover_rate`: Probability of crossover (default: 0.7)
- `elite_size`: Number of top individuals to preserve (default: 5)
- `max_generations`: Maximum generations to run (default: 100)

### Evaluation Criteria

Default criteria weights:
- Length: 0.1
- Clarity: 0.3
- Specificity: 0.3
- Completeness: 0.3

Customize by creating an Evaluator with custom criteria:

```python
from evaluator import Evaluator

custom_evaluator = Evaluator({
    'length': 0.2,
    'clarity': 0.4,
    'specificity': 0.2,
    'completeness': 0.2
})
```

## Predefined Templates

- `code_generation`: For code generation tasks
- `problem_solving`: For analytical problem-solving
- `creative_writing`: For creative writing tasks
- `data_analysis`: For data analysis tasks

## License

MIT License - feel free to use and modify!

## Contributing

Contributions welcome! Areas for improvement:
- Additional mutation operators
- More sophisticated crossover strategies
- Advanced evaluation metrics
- Visualization tools
- Parallel evolution support

