"""
Kissing Number Evolution - Dimension 2
Evolves an agent/prompt that can find the kissing number for dimension 2
The kissing number is the maximum number of non-overlapping unit circles 
that can be arranged so they all touch a central unit circle.
For dimension 2, the answer is 6.
"""

from typing import Dict, Any
from evaluator import CustomEvaluator
from alpha_evolve import AlphaEvolve
from kissing_number_evolution import KissingNumberEvolutionLoop
from visualize_evolution import create_comprehensive_visualization


def kissing_number_evaluator(genome: str, context: Dict[str, Any] = None) -> float:
    """
    Evaluate fitness of a prompt/agent for finding kissing number in dimension 2.
    
    Higher fitness for prompts that:
    - Mention kissing number problem
    - Specify dimension 2
    - Include geometric/mathematical reasoning
    - Reference circles, spheres, or packing
    - Could lead to finding the answer 6
    """
    genome_lower = genome.lower()
    fitness = 0.0
    
    # Base score for mentioning kissing number (0.3)
    kissing_keywords = ['kissing number', 'kissing', 'contact number', 'sphere packing']
    if any(keyword in genome_lower for keyword in kissing_keywords):
        fitness += 0.3
    
    # Dimension 2 specificity (0.2)
    dimension_keywords = ['dimension 2', '2d', 'two dimension', '2-dimensional', 'plane', 'circle']
    if any(keyword in genome_lower for keyword in dimension_keywords):
        fitness += 0.2
    
    # Mathematical/geometric reasoning (0.2)
    math_keywords = ['geometric', 'geometry', 'arrange', 'pack', 'touch', 'tangent', 
                     'radius', 'distance', 'angle', 'hexagonal', 'regular']
    math_count = sum(1 for keyword in math_keywords if keyword in genome_lower)
    fitness += min(math_count * 0.05, 0.2)
    
    # Specific answer or reasoning toward 6 (0.15)
    if '6' in genome or 'six' in genome_lower:
        fitness += 0.15
    elif 'hexagon' in genome_lower or 'hexagonal' in genome_lower:
        fitness += 0.1
    
    # Problem-solving approach (0.15)
    approach_keywords = ['find', 'determine', 'calculate', 'solve', 'compute', 
                        'algorithm', 'method', 'approach', 'strategy']
    approach_count = sum(1 for keyword in approach_keywords if keyword in genome_lower)
    fitness += min(approach_count * 0.03, 0.15)
    
    # Clarity and completeness bonus
    if len(genome.split()) >= 10 and len(genome.split()) <= 100:
        fitness += 0.05
    
    # Penalty for being too vague or off-topic
    vague_keywords = ['random', 'arbitrary', 'guess', 'maybe', 'possibly']
    vague_count = sum(1 for keyword in vague_keywords if keyword in genome_lower)
    fitness -= min(vague_count * 0.05, 0.1)
    
    # Ensure fitness is between 0 and 1
    return max(0.0, min(1.0, fitness))


def get_seed_prompts() -> list:
    """Get seed prompts for kissing number problem"""
    return [
        "Find the kissing number for dimension 2. How many unit circles can touch a central unit circle?",
        "Determine the maximum number of non-overlapping circles that can be arranged around a central circle in 2D.",
        "What is the kissing number in two dimensions? Calculate how many unit circles can all touch one central circle.",
        "Solve the kissing number problem for dimension 2 using geometric reasoning.",
        "Find the maximum number of circles that can be packed around a central circle in the plane.",
        "Calculate the kissing number for 2D space. Consider the geometric arrangement of circles.",
        "Determine the kissing number in dimension 2 by analyzing circle packing and tangency conditions.",
        "What is the maximum number of unit circles that can simultaneously touch a central unit circle in 2D?",
        "Find the kissing number for dimension 2 using mathematical and geometric analysis.",
        "Solve for the kissing number in two-dimensional space. Consider hexagonal arrangements."
    ]


def main():
    """Main evolution loop for kissing number problem"""
    print("=" * 80)
    print("KISSING NUMBER EVOLUTION - DIMENSION 2")
    print("=" * 80)
    print("\nGoal: Evolve an agent that finds the kissing number for dimension 2")
    print("Expected answer: 6 (six unit circles can touch a central unit circle)")
    print("\n" + "=" * 80 + "\n")
    
    # Create custom evaluator
    evaluator = CustomEvaluator(kissing_number_evaluator)
    
    # Initialize Alpha Evolve with custom evaluator
    alpha = AlphaEvolve(
        evaluator=evaluator,
        population_size=40,
        mutation_rate=0.15,
        crossover_rate=0.7,
        elite_size=5,
        max_generations=30
    )
    
    # Replace evolution loop with specialized one
    alpha.evolution_loop = KissingNumberEvolutionLoop(
        evaluator=evaluator,
        population_size=40,
        mutation_rate=0.15,
        crossover_rate=0.7,
        elite_size=5,
        max_generations=30
    )
    
    # Get seed prompts
    seed_prompts = get_seed_prompts()
    
    print(f"Starting evolution with {len(seed_prompts)} seed prompts...")
    print(f"Population size: {alpha.evolution_loop.population_size}")
    print(f"Max generations: {alpha.evolution_loop.max_generations}\n")
    
    # Run evolution
    best = alpha.evolve_prompts(
        seed_prompts=seed_prompts,
        context={'problem': 'kissing_number_dimension_2', 'expected_answer': 6},
        verbose=True
    )
    
    # Display results
    print("\n" + "=" * 80)
    print("EVOLUTION COMPLETE")
    print("=" * 80)
    print("\nBest Individual:")
    print(f"  Fitness: {best.fitness:.4f}")
    print(f"  Generation: {best.generation}")
    print("  Prompt/Agent:")
    print(f"  {best.genome}")
    
    # Show top 5
    print("\n" + "-" * 80)
    print("TOP 5 EVOLVED AGENTS:")
    print("-" * 80)
    top_agents = alpha.get_best_prompts(5)
    for i, agent in enumerate(top_agents, 1):
        print(f"\n{i}. Fitness: {agent.fitness:.4f} (Generation {agent.generation})")
        print(f"   {agent.genome}")
    
    # Evaluation details for best
    print("\n" + "-" * 80)
    print("DETAILED EVALUATION OF BEST AGENT:")
    print("-" * 80)
    evaluation = alpha.evaluate_prompt(best.genome)
    for key, value in evaluation.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    # Check if answer is mentioned
    print("\n" + "-" * 80)
    print("ANSWER CHECK:")
    print("-" * 80)
    if '6' in best.genome or 'six' in best.genome.lower():
        print("✓ Answer (6) is mentioned in the evolved agent!")
    else:
        print("⚠ Answer (6) not explicitly mentioned, but agent may lead to it.")
    
    # Save results
    output_file = "kissing_number_results.json"
    alpha.save_results(output_file, best)
    print(f"\nResults saved to: {output_file}")
    
    # Create visualizations
    print("\n" + "=" * 80)
    print("CREATING VISUALIZATIONS")
    print("=" * 80)
    history = alpha.get_evolution_history()
    if history:
        create_comprehensive_visualization(history, output_dir=".")
    else:
        print("No evolution history available for visualization")
    
    return best, alpha


if __name__ == "__main__":
    best_agent, alpha_system = main()

