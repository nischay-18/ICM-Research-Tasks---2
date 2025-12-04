import logging
from datasets import load_dataset
from typing import List
import json                                                                     
from dataclasses import dataclass, field                                        

@dataclass
class Example:                                                                  
    question: str
    choice: str
    label: int = 0
    predicted_label: int = 0
    id: str = ""
    
    def to_text(self, label_val: int) -> str:
        lbl_str = "Yes" if label_val == 1 else "No"
        return f"Question: {self.question}\nDoes the persona agree?: {lbl_str}"
                                                                                ### <--- END DATACLASS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def prepare_country_data(country_name, n_samples=120) -> List[Example]:
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
        # Ensure item is a dictionary before access (Fix for data items loaded as strings)
        if isinstance(item, str):                                               ### <--- ADD THIS CHECK
            try:
                item = json.loads(item)
            except json.JSONDecodeError:
                logger.warning("Skipping item: Failed to decode JSON string.")
                continue
        # 🌟 FIX: Check for the options list in its expected locations 🌟
        # The key for options is either 'options' or nested under 'selections'
        options = None
        if 'options' in item and isinstance(item['options'], list):
            options = item['options']
        elif 'selections' in item and isinstance(item['selections'], dict) and 'options' in item['selections'] and isinstance(item['selections']['options'], list):
            options = item['selections']['options']
            
        if options is None:
            logger.warning("Skipping item: Missing options list in expected locations.")
            continue                                                            ### <--- END FIX BLOCK

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
            if 'scores' not in item or not isinstance(item['scores'], dict):     ### <--- ADD CHECK FOR 'scores' KEY
                logger.warning("Skipping item: Missing or malformed 'scores'.")
                continue

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
            
        except Exception as e:                                                  
            logger.debug(f"Skipping item due to score/index mismatch or generic error: {e}") 
            continue
            
    logger.info(f"Loaded {len(examples)} examples for {country_name}")
    return examples