from datasets import load_dataset
import pandas as pd
from icm_core import Example
import logging

logger = logging.getLogger(__name__)

def prepare_country_data(country_name, n_samples=120):
    """
    Downloads and formats GlobalOpinionQA data for a specific country.
    """
    try:
        # Load the dataset from HuggingFace
        ds = load_dataset("anthropic/llm_global_opinions", split="train")
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return []
    
    examples = []
    
    # Map input country names to dataset keys
    # The dataset uses 2-letter ISO codes (US, IN, CN, etc.)
    code_map = {
        'United States': 'US', 'US': 'US', 
        'China': 'CN', 'CN': 'CN',
        'India': 'IN', 'IN': 'IN', 
        'Germany': 'DE', 'DE': 'DE', 
        'Nigeria': 'NG', 'NG': 'NG',
        'Japan': 'JP', 'JP': 'JP'
    }
    target_code = code_map.get(country_name, country_name)
    
    for item in ds:
        options = item['selections']['options']
        
        # Filter for binary-style Agree/Disagree questions
        # We only want questions where the options include both "Agree" and "Disagree"
        if not (any("Agree" in opt for opt in options) and any("Disagree" in opt for opt in options)):
            continue
            
        # Find the index of the "Agree" and "Disagree" options
        agree_idx = next((i for i, opt in enumerate(options) if "Agree" in opt), -1)
        disagree_idx = next((i for i, opt in enumerate(options) if "Disagree" in opt), -1)
        
        if agree_idx == -1 or disagree_idx == -1: continue
        
        try:
            # item['scores'] corresponds to the options list
            agree_scores = item['scores'][agree_idx]
            disagree_scores = item['scores'][disagree_idx]
            
            # Skip if the target country doesn't have data for this specific question
            if target_code not in agree_scores or target_code not in disagree_scores: 
                continue
                
            s_agree = agree_scores[target_code]
            s_disagree = disagree_scores[target_code]
            
            # Create Binary Label (1 = Agree, 0 = Disagree)
            # We determine the "Gold Label" by checking which score is higher for that country
            label = 1 if s_agree > s_disagree else 0
            
            examples.append(Example(
                id=str(len(examples)),
                question=item['question'],
                choice="Agree",
                label=label
            ))
            
            if len(examples) >= n_samples: break
            
        except Exception: 
            continue
            
    logger.info(f"Loaded {len(examples)} examples for {country_name}")
    return examples