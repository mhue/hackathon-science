"""
Alpha Evolve - Main Integration Module
Combines Evaluator, Evolution Loop, and Prompt Template systems
"""

from typing import List, Dict, Any, Optional, Callable
import json
from evaluator import Evaluator, CustomEvaluator
from evolution_loop import EvolutionLoop, Individual
from prompt_template import TemplateManager, PromptTemplate, TemplateType, DEFAULT_TEMPLATES


class AlphaEvolve:
    """
    Main Alpha Evolve system that integrates all components
    """
    
    def __init__(
        self,
        evaluator: Evaluator = None,
        population_size: int = 50,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.7,
        elite_size: int = 5,
        max_generations: int = 100
    ):
        """
        Initialize Alpha Evolve system
        
        Args:
            evaluator: Custom evaluator (uses default if None)
            population_size: Size of population per generation
            mutation_rate: Probability of mutation
            crossover_rate: Probability of crossover
            elite_size: Number of elite individuals to preserve
            max_generations: Maximum number of generations
        """
        self.evaluator = evaluator or Evaluator()
        self.template_manager = TemplateManager()
        
        # Register default templates
        for name, template in DEFAULT_TEMPLATES.items():
            self.template_manager.register_template(
                name=name,
                template=template.template,
                template_type=template.template_type,
                description=template.description,
                variables=template.variables
            )
        
        self.evolution_loop = EvolutionLoop(
            evaluator=self.evaluator,
            population_size=population_size,
            mutation_rate=mutation_rate,
            crossover_rate=crossover_rate,
            elite_size=elite_size,
            max_generations=max_generations
        )
    
    def evolve_prompts(
        self,
        seed_prompts: List[str] = None,
        template_name: str = None,
        template_variables: Dict[str, Any] = None,
        context: Dict[str, Any] = None,
        verbose: bool = True
    ) -> Individual:
        """
        Evolve prompts using genetic algorithm
        
        Args:
            seed_prompts: Optional list of seed prompts to start evolution
            template_name: Optional template name to generate seed prompts
            template_variables: Variables for template filling
            context: Optional context for evaluation
            verbose: Whether to print progress
            
        Returns:
            Best evolved individual
        """
        # Generate seed prompts if not provided
        if not seed_prompts:
            if template_name:
                template = self.template_manager.get_template(template_name)
                if template:
                    if template_variables:
                        seed_prompts = [template.fill(**template_variables)]
                    else:
                        seed_prompts = [template.template]
                else:
                    raise ValueError(f"Template '{template_name}' not found")
            else:
                # Use default template
                default_template = list(DEFAULT_TEMPLATES.values())[0]
                seed_prompts = [default_template.template]
        
        # Initialize population
        self.evolution_loop.initialize_population(seed_prompts)
        
        # Run evolution
        best = self.evolution_loop.evolve(context=context, verbose=verbose)
        
        return best
    
    def evolve_template(
        self,
        template_name: str,
        operations: List[str] = None,
        context: Dict[str, Any] = None
    ) -> str:
        """
        Evolve a specific template
        
        Args:
            template_name: Name of template to evolve
            operations: List of evolution operations
            context: Optional context
            
        Returns:
            Evolved template string
        """
        return self.template_manager.evolve_template(template_name, operations)
    
    def register_template(
        self,
        name: str,
        template: str,
        template_type: TemplateType = TemplateType.INSTRUCTION,
        description: str = "",
        variables: List[str] = None
    ):
        """
        Register a new template
        
        Args:
            name: Unique name for template
            template: Template string
            template_type: Type of template
            description: Description
            variables: Optional list of variables
        """
        self.template_manager.register_template(
            name=name,
            template=template,
            template_type=template_type,
            description=description,
            variables=variables
        )
    
    def evaluate_prompt(self, prompt: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Evaluate a single prompt
        
        Args:
            prompt: Prompt to evaluate
            context: Optional context
            
        Returns:
            Evaluation details dictionary
        """
        return self.evaluator.get_evaluation_details(prompt, context)
    
    def get_best_prompts(self, n: int = 10) -> List[Individual]:
        """
        Get top N evolved prompts
        
        Args:
            n: Number of prompts to return
            
        Returns:
            List of top individuals
        """
        return self.evolution_loop.get_best_individuals(n)
    
    def get_evolution_history(self) -> List[Dict[str, Any]]:
        """Get evolution history"""
        return self.evolution_loop.get_history()
    
    def set_custom_evaluator(self, evaluation_function: Callable[[str, Dict], float]):
        """
        Set custom evaluation function
        
        Args:
            evaluation_function: Function that takes (prompt, context) and returns fitness
        """
        self.evaluator = CustomEvaluator(evaluation_function)
        self.evolution_loop.evaluator = self.evaluator
    
    def save_results(self, filepath: str, best_individual: Individual = None):
        """
        Save evolution results to file
        
        Args:
            filepath: Path to save file
            best_individual: Optional best individual to save
        """
        results = {
            'best_individual': {
                'genome': best_individual.genome if best_individual else None,
                'fitness': best_individual.fitness if best_individual else None,
                'generation': best_individual.generation if best_individual else None
            },
            'top_individuals': [
                {
                    'genome': ind.genome,
                    'fitness': ind.fitness,
                    'generation': ind.generation
                }
                for ind in self.get_best_prompts(10)
            ],
            'history': self.get_evolution_history()
        }
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
    
    def load_results(self, filepath: str) -> Dict[str, Any]:
        """
        Load evolution results from file
        
        Args:
            filepath: Path to load file from
            
        Returns:
            Loaded results dictionary
        """
        with open(filepath, 'r') as f:
            return json.load(f)


def main():
    """Example usage of Alpha Evolve"""
    
    # Initialize Alpha Evolve
    alpha = AlphaEvolve(
        population_size=30,
        mutation_rate=0.15,
        crossover_rate=0.7,
        elite_size=3,
        max_generations=20
    )
    
    # Example 1: Evolve prompts from seed
    print("=" * 60)
    print("Example 1: Evolving prompts from seed")
    print("=" * 60)
    
    seed_prompts = [
        "Write a Python function to calculate fibonacci numbers",
        "Create a function that computes fibonacci sequence",
        "Implement fibonacci number calculation in Python"
    ]
    
    best = alpha.evolve_prompts(
        seed_prompts=seed_prompts,
        verbose=True
    )
    
    print(f"\nBest evolved prompt:")
    print(f"Fitness: {best.fitness:.3f}")
    print(f"Generation: {best.generation}")
    print(f"Prompt: {best.genome}")
    
    # Example 2: Evolve using template
    print("\n" + "=" * 60)
    print("Example 2: Evolving using template")
    print("=" * 60)
    
    best2 = alpha.evolve_prompts(
        template_name='code_generation',
        template_variables={
            'language': 'Python',
            'task': 'sorts a list',
            'requirements': 'be efficient and handle edge cases',
            'output_format': 'the sorted list'
        },
        verbose=True
    )
    
    print(f"\nBest evolved prompt:")
    print(f"Fitness: {best2.fitness:.3f}")
    print(f"Prompt: {best2.genome}")
    
    # Example 3: Evaluate a prompt
    print("\n" + "=" * 60)
    print("Example 3: Evaluating a prompt")
    print("=" * 60)
    
    test_prompt = "Write a comprehensive Python function that efficiently sorts a list of integers using quicksort algorithm, handles edge cases like empty lists and single elements, and returns the sorted list with detailed comments explaining each step."
    
    evaluation = alpha.evaluate_prompt(test_prompt)
    print(f"\nEvaluation results:")
    for key, value in evaluation.items():
        print(f"  {key}: {value}")
    
    # Example 4: Get top prompts
    print("\n" + "=" * 60)
    print("Example 4: Top 5 evolved prompts")
    print("=" * 60)
    
    top_prompts = alpha.get_best_prompts(5)
    for i, ind in enumerate(top_prompts, 1):
        print(f"\n{i}. Fitness: {ind.fitness:.3f}")
        print(f"   Prompt: {ind.genome[:100]}...")


if __name__ == "__main__":
    main()

