"""
Specialized Evolution Loop for Kissing Number Problem
Extends the base evolution loop with domain-specific mutations
"""

from typing import List
import random
from evolution_loop import EvolutionLoop


class KissingNumberEvolutionLoop(EvolutionLoop):
    """Evolution loop specialized for kissing number problem"""
    
    def _crossover(self, parent1: str, parent2: str) -> str:
        """Improved crossover that reduces repetition"""
        words1 = parent1.split()
        words2 = parent2.split()
        
        if len(words1) < 2 or len(words2) < 2:
            return parent1
        
        # Use shorter parent as base to avoid excessive length
        if len(words1) > len(words2):
            base_words, other_words = words2, words1
        else:
            base_words, other_words = words1, words2
        
        # Take a meaningful segment from other parent (avoid single words)
        segment_start = random.randint(0, max(0, len(other_words) - 3))
        segment_length = random.randint(2, min(5, len(other_words) - segment_start))
        segment = other_words[segment_start:segment_start + segment_length]
        
        # Insert segment at random position in base
        insert_pos = random.randint(0, len(base_words))
        result = base_words[:insert_pos] + segment + base_words[insert_pos:]
        
        # Remove duplicate consecutive words
        cleaned = []
        for word in result:
            if not cleaned or word != cleaned[-1]:
                cleaned.append(word)
        
        return ' '.join(cleaned)
    
    def _generate_word(self) -> str:
        """Generate domain-specific words for mutations"""
        math_words = [
            'circle', 'sphere', 'dimension', 'geometric', 'arrangement',
            'packing', 'tangent', 'touch', 'contact', 'radius', 'distance',
            'angle', 'hexagonal', 'regular', 'maximum', 'minimum', 'optimal',
            'calculate', 'determine', 'find', 'solve', 'compute', 'analyze',
            'unit', 'non-overlapping', 'plane', '2D', 'two-dimensional',
            'kissing', 'number', 'six', '6', 'arrange', 'pack', 'method',
            'approach', 'strategy', 'algorithm', 'reasoning', 'proof'
        ]
        return random.choice(math_words)
    
    def _mutate(self, genome: str) -> str:
        """Enhanced mutation with domain-specific operations"""
        words = genome.split()
        if len(words) < 2:
            return genome
        
        mutation_type = random.choice(['swap', 'insert', 'delete', 'replace', 'expand'])
        
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
        
        elif mutation_type == 'expand':
            # Add domain-specific phrases
            expansions = [
                'using geometric reasoning',
                'in two-dimensional space',
                'with unit circles',
                'considering tangency conditions',
                'through mathematical analysis',
                'by examining circle arrangements'
            ]
            expansion = random.choice(expansions)
            words.extend(expansion.split())
        
        return ' '.join(words)

