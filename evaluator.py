"""
Evaluator Module for Alpha Evolve
Evaluates the fitness of evolved prompts/code
"""

from typing import Dict, List, Any, Callable
import json
import re


class Evaluator:
    """Evaluates fitness of evolved individuals"""
    
    def __init__(self, evaluation_criteria: Dict[str, float] = None):
        """
        Initialize evaluator with evaluation criteria
        
        Args:
            evaluation_criteria: Dictionary mapping criterion names to weights
        """
        self.criteria = evaluation_criteria or {
            'length': 0.1,
            'clarity': 0.3,
            'specificity': 0.3,
            'completeness': 0.3
        }
    
    def evaluate(self, individual: str, context: Dict[str, Any] = None) -> float:
        """
        Evaluate an individual and return fitness score
        
        Args:
            individual: The prompt/code to evaluate
            context: Optional context for evaluation
            
        Returns:
            Fitness score between 0 and 1
        """
        scores = {}
        
        # Length score (prefer moderate length)
        length_score = self._evaluate_length(individual)
        scores['length'] = length_score
        
        # Clarity score (based on readability metrics)
        clarity_score = self._evaluate_clarity(individual)
        scores['clarity'] = clarity_score
        
        # Specificity score (presence of specific terms/details)
        specificity_score = self._evaluate_specificity(individual)
        scores['specificity'] = specificity_score
        
        # Completeness score (covers all necessary aspects)
        completeness_score = self._evaluate_completeness(individual, context)
        scores['completeness'] = completeness_score
        
        # Calculate weighted fitness
        fitness = sum(
            scores.get(criterion, 0) * weight
            for criterion, weight in self.criteria.items()
        )
        
        return min(max(fitness, 0.0), 1.0)
    
    def _evaluate_length(self, individual: str) -> float:
        """Evaluate length score (optimal around 200-500 words)"""
        word_count = len(individual.split())
        
        if word_count < 50:
            return 0.3
        elif word_count < 200:
            return 0.7
        elif word_count < 500:
            return 1.0
        elif word_count < 1000:
            return 0.8
        else:
            return 0.5
    
    def _evaluate_clarity(self, individual: str) -> float:
        """Evaluate clarity based on readability metrics"""
        sentences = re.split(r'[.!?]+', individual)
        avg_sentence_length = sum(len(s.split()) for s in sentences) / max(len(sentences), 1)
        
        # Prefer sentences between 10-25 words
        if 10 <= avg_sentence_length <= 25:
            clarity = 1.0
        elif 5 <= avg_sentence_length < 10 or 25 < avg_sentence_length <= 35:
            clarity = 0.7
        else:
            clarity = 0.4
        
        # Check for common clarity indicators
        clarity_indicators = ['because', 'therefore', 'specifically', 'for example', 'such as']
        indicator_count = sum(1 for indicator in clarity_indicators if indicator in individual.lower())
        clarity += min(indicator_count * 0.1, 0.2)
        
        return min(clarity, 1.0)
    
    def _evaluate_specificity(self, individual: str) -> float:
        """Evaluate specificity (presence of specific details)"""
        # Check for numbers, dates, specific terms
        has_numbers = bool(re.search(r'\d+', individual))
        has_specific_terms = bool(re.search(r'\b(?:exactly|precisely|specifically|concretely)\b', individual.lower()))
        has_examples = bool(re.search(r'\b(?:example|instance|case|scenario)\b', individual.lower()))
        
        specificity = 0.0
        if has_numbers:
            specificity += 0.3
        if has_specific_terms:
            specificity += 0.3
        if has_examples:
            specificity += 0.4
        
        return min(specificity, 1.0)
    
    def _evaluate_completeness(self, individual: str, context: Dict[str, Any] = None) -> float:
        """Evaluate completeness based on required elements"""
        # Check for common completeness indicators
        completeness_indicators = [
            'what', 'how', 'why', 'when', 'where',
            'step', 'process', 'method', 'approach',
            'goal', 'objective', 'result', 'outcome'
        ]
        
        found_indicators = sum(
            1 for indicator in completeness_indicators
            if indicator in individual.lower()
        )
        
        completeness = min(found_indicators / len(completeness_indicators), 1.0)
        
        # If context provided, check for required elements
        if context and 'required_elements' in context:
            required = context['required_elements']
            found_required = sum(
                1 for element in required
                if element.lower() in individual.lower()
            )
            completeness = (completeness + found_required / len(required)) / 2
        
        return completeness
    
    def batch_evaluate(self, individuals: List[str], context: Dict[str, Any] = None) -> List[float]:
        """
        Evaluate multiple individuals
        
        Args:
            individuals: List of prompts/code to evaluate
            context: Optional context for evaluation
            
        Returns:
            List of fitness scores
        """
        return [self.evaluate(ind, context) for ind in individuals]
    
    def get_evaluation_details(self, individual: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Get detailed evaluation breakdown
        
        Args:
            individual: The prompt/code to evaluate
            context: Optional context for evaluation
            
        Returns:
            Dictionary with detailed scores and metrics
        """
        return {
            'fitness': self.evaluate(individual, context),
            'length_score': self._evaluate_length(individual),
            'clarity_score': self._evaluate_clarity(individual),
            'specificity_score': self._evaluate_specificity(individual),
            'completeness_score': self._evaluate_completeness(individual, context),
            'word_count': len(individual.split()),
            'character_count': len(individual)
        }


class CustomEvaluator(Evaluator):
    """Custom evaluator that allows user-defined evaluation functions"""
    
    def __init__(self, evaluation_function: Callable[[str, Dict], float], **kwargs):
        """
        Initialize with custom evaluation function
        
        Args:
            evaluation_function: Function that takes (individual, context) and returns fitness score
        """
        super().__init__(**kwargs)
        self.custom_eval = evaluation_function
    
    def evaluate(self, individual: str, context: Dict[str, Any] = None) -> float:
        """Use custom evaluation function"""
        return self.custom_eval(individual, context or {})

