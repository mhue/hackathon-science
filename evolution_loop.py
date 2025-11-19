"""
Evolution Loop Module for Alpha Evolve
Implements the genetic algorithm evolution loop
"""

from typing import List, Dict, Any, Tuple, Callable
import random
import copy
from dataclasses import dataclass
from evaluator import Evaluator
try:
    from sphere_solver import get_best_sphere_configuration
except ImportError:
    # Fallback if sphere_solver not available
    def get_best_sphere_configuration(population, context=None):
        return None


@dataclass
class Individual:
    """Represents an individual in the population"""
    genome: str
    fitness: float = 0.0
    generation: int = 0
    
    def __repr__(self):
        return f"Individual(fitness={self.fitness:.3f}, gen={self.generation})"


class EvolutionLoop:
    """Main evolution loop for genetic algorithm"""
    
    def __init__(
        self,
        evaluator: Evaluator,
        population_size: int = 50,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.7,
        elite_size: int = 5,
        max_generations: int = 100
    ):
        """
        Initialize evolution loop
        
        Args:
            evaluator: Evaluator instance for fitness calculation
            population_size: Size of population per generation
            mutation_rate: Probability of mutation (0-1)
            crossover_rate: Probability of crossover (0-1)
            elite_size: Number of top individuals to preserve
            max_generations: Maximum number of generations
        """
        self.evaluator = evaluator
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_size = elite_size
        self.max_generations = max_generations
        
        self.population: List[Individual] = []
        self.generation = 0
        self.history: List[Dict[str, Any]] = []
    
    def initialize_population(self, seed_prompts: List[str] = None):
        """
        Initialize population with seed prompts or random generation
        
        Args:
            seed_prompts: Optional list of seed prompts to start with
        """
        self.population = []
        
        if seed_prompts:
            # Use seed prompts and generate variations
            for prompt in seed_prompts[:self.population_size]:
                individual = Individual(
                    genome=prompt,
                    generation=0
                )
                self.population.append(individual)
            
            # Fill remaining with mutations/variations
            while len(self.population) < self.population_size:
                base = random.choice(seed_prompts)
                mutated = self._mutate(base)
                individual = Individual(
                    genome=mutated,
                    generation=0
                )
                self.population.append(individual)
        else:
            # Generate random initial population
            for _ in range(self.population_size):
                individual = Individual(
                    genome=self._generate_random(),
                    generation=0
                )
                self.population.append(individual)
    
    def evolve(self, context: Dict[str, Any] = None, verbose: bool = True) -> Individual:
        """
        Run evolution loop
        
        Args:
            context: Optional context for evaluation
            verbose: Whether to print progress
            
        Returns:
            Best individual found
        """
        if not self.population:
            raise ValueError("Population not initialized. Call initialize_population() first.")
        
        best_ever = None
        
        for generation in range(self.max_generations):
            self.generation = generation
            
            # Evaluate fitness
            self._evaluate_population(context)
            
            # Sort by fitness
            self.population.sort(key=lambda x: x.fitness, reverse=True)
            
            # Track best
            current_best = self.population[0]
            if best_ever is None or current_best.fitness > best_ever.fitness:
                best_ever = copy.deepcopy(current_best)
            
            # Record history
            stats = self._get_generation_stats()
            
            # Add sphere information for best individual
            sphere_info = get_best_sphere_configuration(self.population, context)
            if sphere_info:
                stats['sphere_count'] = sphere_info.get('sphere_count', 0)
                stats['sphere_centers'] = sphere_info.get('centers', [])
                stats['central_center'] = sphere_info.get('central_center', [])
                stats['dimension'] = sphere_info.get('dimension', 2)
            
            self.history.append(stats)
            
            if verbose:
                print(f"Generation {generation}: "
                      f"Best={current_best.fitness:.3f}, "
                      f"Avg={stats['avg_fitness']:.3f}, "
                      f"Best Ever={best_ever.fitness:.3f}")
            
            # Check convergence
            if self._check_convergence():
                if verbose:
                    print(f"Converged at generation {generation}")
                break
            
            # Create next generation
            self._create_next_generation()
        
        return best_ever
    
    def _evaluate_population(self, context: Dict[str, Any] = None):
        """Evaluate fitness of all individuals"""
        for individual in self.population:
            if individual.fitness == 0.0 or self.generation > 0:
                individual.fitness = self.evaluator.evaluate(individual.genome, context)
                individual.generation = self.generation
    
    def _create_next_generation(self):
        """Create next generation through selection, crossover, and mutation"""
        new_population = []
        
        # Elitism: keep top individuals
        elite = self.population[:self.elite_size]
        for ind in elite:
            new_population.append(copy.deepcopy(ind))
        
        # Generate rest through selection, crossover, and mutation
        while len(new_population) < self.population_size:
            # Selection
            parent1 = self._tournament_selection()
            parent2 = self._tournament_selection()
            
            # Crossover
            if random.random() < self.crossover_rate:
                child_genome = self._crossover(parent1.genome, parent2.genome)
            else:
                child_genome = parent1.genome
            
            # Mutation
            if random.random() < self.mutation_rate:
                child_genome = self._mutate(child_genome)
            
            child = Individual(
                genome=child_genome,
                generation=self.generation + 1
            )
            new_population.append(child)
        
        self.population = new_population
    
    def _tournament_selection(self, tournament_size: int = 3) -> Individual:
        """Tournament selection"""
        tournament = random.sample(self.population, min(tournament_size, len(self.population)))
        return max(tournament, key=lambda x: x.fitness)
    
    def _crossover(self, parent1: str, parent2: str) -> str:
        """Single-point crossover"""
        words1 = parent1.split()
        words2 = parent2.split()
        
        if len(words1) < 2 or len(words2) < 2:
            return parent1
        
        point1 = random.randint(1, len(words1) - 1)
        point2 = random.randint(1, len(words2) - 1)
        
        child = ' '.join(words1[:point1] + words2[point2:])
        return child
    
    def _mutate(self, genome: str) -> str:
        """Mutate genome through various operations"""
        words = genome.split()
        if len(words) < 2:
            return genome
        
        mutation_type = random.choice(['swap', 'insert', 'delete', 'replace'])
        
        if mutation_type == 'swap' and len(words) >= 2:
            i, j = random.sample(range(len(words)), 2)
            words[i], words[j] = words[j], words[i]
        
        elif mutation_type == 'insert':
            insert_pos = random.randint(0, len(words))
            new_word = self._generate_word()
            words.insert(insert_pos, new_word)
        
        elif mutation_type == 'delete' and len(words) > 1:
            delete_pos = random.randint(0, len(words) - 1)
            words.pop(delete_pos)
        
        elif mutation_type == 'replace':
            replace_pos = random.randint(0, len(words) - 1)
            words[replace_pos] = self._generate_word()
        
        return ' '.join(words)
    
    def _generate_random(self) -> str:
        """Generate random initial genome"""
        # Simple random generation - can be customized
        words = [
            'the', 'quick', 'brown', 'fox', 'jumps', 'over',
            'lazy', 'dog', 'and', 'runs', 'fast'
        ]
        length = random.randint(5, 15)
        return ' '.join(random.choices(words, k=length))
    
    def _generate_word(self) -> str:
        """Generate a random word for mutations"""
        words = [
            'the', 'a', 'an', 'and', 'or', 'but', 'with', 'for',
            'example', 'instance', 'case', 'method', 'approach',
            'result', 'outcome', 'goal', 'objective'
        ]
        return random.choice(words)
    
    def _check_convergence(self, threshold: float = 0.001) -> bool:
        """Check if population has converged"""
        if len(self.population) < 2:
            return False
        
        fitnesses = [ind.fitness for ind in self.population]
        avg_fitness = sum(fitnesses) / len(fitnesses)
        max_fitness = max(fitnesses)
        
        # Converged if variance is low and near max
        variance = sum((f - avg_fitness) ** 2 for f in fitnesses) / len(fitnesses)
        return variance < threshold and (max_fitness - avg_fitness) < threshold
    
    def _get_generation_stats(self) -> Dict[str, Any]:
        """Get statistics for current generation"""
        fitnesses = [ind.fitness for ind in self.population]
        return {
            'generation': self.generation,
            'best_fitness': max(fitnesses),
            'worst_fitness': min(fitnesses),
            'avg_fitness': sum(fitnesses) / len(fitnesses),
            'std_fitness': (sum((f - sum(fitnesses)/len(fitnesses))**2 for f in fitnesses) / len(fitnesses)) ** 0.5
        }
    
    def get_best_individuals(self, n: int = 10) -> List[Individual]:
        """Get top N individuals"""
        sorted_pop = sorted(self.population, key=lambda x: x.fitness, reverse=True)
        return sorted_pop[:n]
    
    def get_history(self) -> List[Dict[str, Any]]:
        """Get evolution history"""
        return self.history

