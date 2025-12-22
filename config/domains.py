#!/usr/bin/env python3
"""
Unified MoE Domain Configuration
Centralized domain management for expert routing
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import os

@dataclass
class DomainConfig:
    """Configuration for a specific domain"""
    name: str
    data_path: str
    train_file: str
    validation_file: str
    test_file: str
    instruction_template: str
    
    def get_file_path(self, split: str) -> str:
        """Get file path for specific split"""
        if split == 'train':
            return os.path.join(self.data_path, self.train_file)
        elif split == 'validation':
            return os.path.join(self.data_path, self.validation_file)
        elif split == 'test':
            return os.path.join(self.data_path, self.test_file)
        else:
            raise ValueError(f"Unknown split: {split}")
    
    def file_exists(self, split: str) -> bool:
        """Check if file exists for specific split"""
        return os.path.exists(self.get_file_path(split))

class DomainManager:
    """Centralized domain management"""
    
    def __init__(self):
        self.domains = self._initialize_domains()
    
    def _initialize_domains(self) -> Dict[str, DomainConfig]:
        """Initialize all domain configurations"""
        # Get max_length from actual config file
        try:
            from config.moe import get_model_config
            model_config = get_model_config()
            default_max_length = model_config.max_length
        except Exception:
            # Fallback to default value if config loading fails
            default_max_length = 1800  # Default from config.yaml
        return {
            "medical": DomainConfig(
                name="medical",
                data_path="data/processed/medical",
                train_file="medical_train.json",
                validation_file="medical_test.json",
                test_file="medical_test.json",
                instruction_template='''Answer the following multiple-choice question.
                    Choose only one option: A, B, C, D.
                    Your response MUST follow this format:

                    Answer: [letter]
                    Explanation: [your reasoning here]

                    Example:
                    Answer: C
                    Explanation: Because ...

                    Now, here is the question:\n\n{question}''',
                # max_length retained for backward compatibility in config
            ),

            "law": DomainConfig(
                name="law", 
                data_path="data/processed/law",
                train_file="law_train.json",
                validation_file="law_test.json",
                test_file="law_test.json", 
                instruction_template='''Answer the following multiple-choice question.
                    Choose only one option: A, B, C, D, or E.
                    Your response MUST follow this format:

                    Answer: [letter]
                    Explanation: [your reasoning here]

                    Example:
                    Answer: C
                    Explanation: Because ...

                    Now, here is the question:\n\n{question}''',
                # max_length retained for backward compatibility in config
            ),
            "math": DomainConfig(
                name="math",
                data_path="data/processed/math", 
                train_file="math_train.json",
                validation_file="math_test.json",
                test_file="math_test.json",
                instruction_template='''Answer the following multiple-choice question.
                    Choose only one option: A, B, C, D, or E.
                    Your response MUST follow this format:

                    Answer: [letter]
                    Explanation: [your reasoning here]

                    Example:
                    Answer: C
                    Explanation: Because ...

                    Now, here is the question:\n\n{question}''',
                # max_length retained for backward compatibility in config
            ),
            "code": DomainConfig(
                name="code",
                data_path="data/processed/code",
                train_file="code_train.json", 
                validation_file="code_test.json",
                test_file="code_test.json",
                instruction_template='''Answer the following multiple-choice question.
                    Choose only one option: A, B, C, or D.
                    Your response MUST follow this format:

                    Answer: [letter]
                    Explanation: [your reasoning here]

                    Example:
                    Answer: C
                    Explanation: Because ...

                    Now, here is the question:\n\n{question}''',
                # max_length retained for backward compatibility in config
            ),
            "MMLU": DomainConfig(
                name="MMLU",
                data_path="data/processed/MMLU",
                train_file="MMLU_train.json",
                validation_file="MMLU_test.json",
                test_file="MMLU_test.json",
                instruction_template='''Answer the following multiple-choice question.
                    Choose only one option: A, B, C, or D.
                    Your response MUST follow this format:

                    Answer: [letter]
                    Explanation: [your reasoning here]

                    Example:
                    Answer: C
                    Explanation: Because ...

                    Now, here is the question:\n\n{question}''',
                # max_length retained for backward compatibility in config
            ),
            "general": DomainConfig(
                name="general",
                data_path="data/processed/total",
                train_file="total_train.json",
                validation_file="total_test.json", 
                test_file="total_test.json",
                instruction_template='''Answer the following multiple-choice question.
                    Choose only one option: A, B, C, D, or E.
                    Your response MUST follow this format:

                    Answer: [letter]
                    Explanation: [your reasoning here]

                    Example:
                    Answer: C
                    Explanation: Because ...

                    Now, here is the question:\n\n{question}''',
                # max_length retained for backward compatibility in config
            )
                  
        }
    
    def get_domain(self, domain_name: str) -> DomainConfig:
        """Get domain configuration by name"""
        if domain_name not in self.domains:
            raise ValueError(f"Unknown domain: {domain_name}. Available: {list(self.domains.keys())}")
        return self.domains[domain_name]
    
    def get_available_domains(self) -> List[str]:
        """Get list of available domains"""
        return list(self.domains.keys())
    
    def check_data_availability(self, domain_name: str = None) -> Dict[str, bool]:
        """Check data availability for domain(s)"""
        if domain_name:
            domain = self.get_domain(domain_name)
            return {
                domain_name: all([
                    domain.file_exists('train'),
                    domain.file_exists('test')
                ])
            }
        else:
            availability = {}
            for name, domain in self.domains.items():
                availability[name] = all([
                    domain.file_exists('train'),
                    domain.file_exists('test')
                ])
            return availability
    
    def format_prompt(self, domain_name: str, **kwargs) -> str:
        """Format prompt for a specific domain"""
        domain = self.get_domain(domain_name)
        question = kwargs.get('question', '')
        return domain.instruction_template.format(question=question)
    
# Global domain manager instance
domain_manager = DomainManager()

def get_all_domains() -> List[str]:
    """Get list of all domain names"""
    return list(domain_manager.domains.keys())

def get_domain_config(domain_name: str) -> DomainConfig:
    """Get domain configuration by name"""
    return domain_manager.get_domain(domain_name)
