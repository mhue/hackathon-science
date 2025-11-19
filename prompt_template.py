"""
Prompt Template Module for Alpha Evolve
Manages prompt templates and their evolution
"""

from typing import Dict, List, Any, Optional
import re
from dataclasses import dataclass
from enum import Enum


class TemplateType(Enum):
    """Types of prompt templates"""
    INSTRUCTION = "instruction"
    QUESTION = "question"
    TASK = "task"
    CREATIVE = "creative"
    ANALYTICAL = "analytical"


@dataclass
class PromptTemplate:
    """Represents a prompt template"""
    template: str
    template_type: TemplateType
    variables: List[str]
    description: str = ""
    
    def __post_init__(self):
        """Extract variables from template"""
        if not self.variables:
            self.variables = self._extract_variables()
    
    def _extract_variables(self) -> List[str]:
        """Extract variable placeholders from template"""
        pattern = r'\{\{(\w+)\}\}'
        return re.findall(pattern, self.template)
    
    def fill(self, **kwargs) -> str:
        """
        Fill template with values
        
        Args:
            **kwargs: Values for template variables
            
        Returns:
            Filled prompt string
        """
        result = self.template
        for var in self.variables:
            value = kwargs.get(var, f"[{var}]")
            result = result.replace(f"{{{{{var}}}}}", str(value))
        return result
    
    def evolve(self, evolution_operations: List[str] = None) -> str:
        """
        Evolve template by applying operations
        
        Args:
            evolution_operations: List of operations to apply
            
        Returns:
            Evolved template string
        """
        evolved = self.template
        
        operations = evolution_operations or ['expand', 'clarify', 'add_examples']
        
        for op in operations:
            if op == 'expand':
                evolved = self._expand_template(evolved)
            elif op == 'clarify':
                evolved = self._clarify_template(evolved)
            elif op == 'add_examples':
                evolved = self._add_examples(evolved)
            elif op == 'simplify':
                evolved = self._simplify_template(evolved)
            elif op == 'add_context':
                evolved = self._add_context(evolved)
        
        return evolved
    
    def _expand_template(self, template: str) -> str:
        """Expand template with more details"""
        # Add expansion markers
        expansions = [
            " Please provide detailed information.",
            " Include specific examples and use cases.",
            " Consider edge cases and potential challenges."
        ]
        return template + " " + " ".join(expansions[:1])
    
    def _clarify_template(self, template: str) -> str:
        """Clarify template language"""
        clarifications = {
            r'\bdo\b': 'perform',
            r'\bmake\b': 'create',
            r'\bget\b': 'obtain',
            r'\bshow\b': 'demonstrate'
        }
        result = template
        for pattern, replacement in clarifications.items():
            result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
        return result
    
    def _add_examples(self, template: str) -> str:
        """Add example section to template"""
        if 'example' not in template.lower():
            return template + " For example, consider the following scenario: [scenario]."
        return template
    
    def _simplify_template(self, template: str) -> str:
        """Simplify template language"""
        # Remove redundant phrases
        simplifications = [
            (r'\bplease note that\b', ''),
            (r'\bit is important to\b', ''),
            (r'\bkindly\b', '')
        ]
        result = template
        for pattern, replacement in simplifications:
            result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
        return result.strip()
    
    def _add_context(self, template: str) -> str:
        """Add context section"""
        if 'context' not in template.lower():
            return f"Context: [context]. {template}"
        return template


class TemplateManager:
    """Manages collection of prompt templates"""
    
    def __init__(self):
        """Initialize template manager"""
        self.templates: Dict[str, PromptTemplate] = {}
        self.template_registry: Dict[TemplateType, List[str]] = {
            template_type: [] for template_type in TemplateType
        }
    
    def register_template(
        self,
        name: str,
        template: str,
        template_type: TemplateType,
        description: str = "",
        variables: List[str] = None
    ):
        """
        Register a new template
        
        Args:
            name: Unique name for template
            template: Template string with {{variable}} placeholders
            template_type: Type of template
            description: Description of template
            variables: Optional list of variables (auto-detected if None)
        """
        prompt_template = PromptTemplate(
            template=template,
            template_type=template_type,
            variables=variables or [],
            description=description
        )
        
        self.templates[name] = prompt_template
        self.template_registry[template_type].append(name)
    
    def get_template(self, name: str) -> Optional[PromptTemplate]:
        """Get template by name"""
        return self.templates.get(name)
    
    def get_templates_by_type(self, template_type: TemplateType) -> List[PromptTemplate]:
        """Get all templates of a specific type"""
        names = self.template_registry.get(template_type, [])
        return [self.templates[name] for name in names if name in self.templates]
    
    def evolve_template(self, name: str, operations: List[str] = None) -> str:
        """
        Evolve a template
        
        Args:
            name: Name of template to evolve
            operations: List of evolution operations
            
        Returns:
            Evolved template string
        """
        template = self.templates.get(name)
        if not template:
            raise ValueError(f"Template '{name}' not found")
        
        return template.evolve(operations)
    
    def create_from_seed(self, seed: str, template_type: TemplateType = TemplateType.INSTRUCTION) -> PromptTemplate:
        """
        Create template from seed text
        
        Args:
            seed: Seed text to convert to template
            template_type: Type of template
            
        Returns:
            New PromptTemplate instance
        """
        # Convert seed to template format
        template_str = seed
        
        return PromptTemplate(
            template=template_str,
            template_type=template_type,
            variables=[],
            description=f"Generated from seed: {seed[:50]}..."
        )
    
    def list_templates(self) -> List[str]:
        """List all registered template names"""
        return list(self.templates.keys())


# Predefined templates
DEFAULT_TEMPLATES = {
    'code_generation': PromptTemplate(
        template="Write a {{language}} function that {{task}}. The function should {{requirements}}. Return {{output_format}}.",
        template_type=TemplateType.TASK,
        variables=['language', 'task', 'requirements', 'output_format'],
        description="Template for code generation tasks"
    ),
    'problem_solving': PromptTemplate(
        template="Solve the following problem: {{problem}}. Show your reasoning step by step. Consider {{constraints}}.",
        template_type=TemplateType.ANALYTICAL,
        variables=['problem', 'constraints'],
        description="Template for problem-solving tasks"
    ),
    'creative_writing': PromptTemplate(
        template="Write a {{genre}} story about {{topic}}. Include {{elements}}. The story should be {{style}}.",
        template_type=TemplateType.CREATIVE,
        variables=['genre', 'topic', 'elements', 'style'],
        description="Template for creative writing tasks"
    ),
    'data_analysis': PromptTemplate(
        template="Analyze the following data: {{data}}. Identify {{patterns}}. Provide insights about {{focus_areas}}.",
        template_type=TemplateType.ANALYTICAL,
        variables=['data', 'patterns', 'focus_areas'],
        description="Template for data analysis tasks"
    )
}

