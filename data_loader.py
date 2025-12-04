import logging
from datasets import load_dataset
from typing import List
import json
from dataclasses import dataclass, field 

# --- DATACLASS DEFINITION (Required for Example objects) ---
# NOTE: This Example class definition is required if it's not imported from icm_core
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
# -----------------------------------------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def prepare_country_data(country_name, n_samples=120) -> List[Example]:
    """
    Downloads and formats GlobalOpinionQA data for a specific country.
    
    This function includes robust logic to handle the inconsistent location of 
    options and scores within the loaded dataset items.
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
        # 1. Ensure item is a dictionary (handles items loaded as JSON strings)
        if isinstance(item, str):
            try:
                item = json.loads(item)
            except json.JSONDecodeError:
                logger.warning("Skipping item: Failed to decode JSON string.")
                continue

        # 2. ROBUSTLY GET OPTIONS (The list of answer choices)
        options = item.get('options') 
        
        # Fallback check for nested options structure
        if not isinstance(options, list) and 'selections' in item and isinstance(item['selections'], dict) and 'options' in item['selections'] and isinstance(item['selections']['options'], list):
            options = item['selections']['options']
            
        if options is None or not options:
            logger.warning("Skipping item: Missing options list in 'options' or nested under 'selections'.")
            continue
            
        # 3. ROBUSTLY GET SCORES (The country opinion data)
        all_scores = None
        
        # Priority 1: Check where you observed the score data (item['selections'])
        potential_scores = item.get('selections')
        if isinstance(potential_scores, dict) and potential_scores:
            all_scores = potential_scores
        
        # Priority 2: Check the alternate key used by the dataset (item['scores'])
        if all_scores is None:
            potential_scores = item.get('scores')
            if isinstance(potential_scores, dict) and potential_scores:
                all_scores = potential_scores

        if all_scores is None:
            logger.warning("Skipping item: Missing or malformed score data in 'selections' or 'scores' key.")
            continue
        
        # 4. Filter for binary-style Agree/Disagree questions
        if not (any("Agree" in opt for opt in options) and any("Disagree" in opt for opt in options)):
            continue
            
        # Find the index of the "Agree" and "Disagree" options
        agree_idx = next((i for i, opt in enumerate(options) if "Agree" in opt), -1)
        disagree_idx = next((i for i, opt in enumerate(options) if "Disagree" in opt), -1)
        
        if agree_idx == -1 or disagree_idx == -1: continue
        
        try:
            
            agree_score_list = {}
            disagree_score_list = {}
            
            # Extract scores for the target options across all countries present in this item
            for country_code, scores_list in all_scores.items():
                if len(scores_list) > agree_idx:
                    agree_score_list[country_code] = scores_list[agree_idx]
                if len(scores_list) > disagree_idx:
                    disagree_score_list[country_code] = scores_list[disagree_idx]

            # Check if the target country has data for BOTH options
            if target_code not in agree_score_list or target_code not in disagree_score_list: 
                continue
                
            s_agree = agree_score_list[target_code]
            s_disagree = disagree_score_list[target_code]
            
            # Create Binary Label (1 = Agree, 0 = Disagree)
            label = 1 if s_agree > s_disagree else 0
            
            examples.append(Example(
                id=str(len(examples)),
                question=item['question'],
                choice="Agree",
                label=label
            ))
            
            if len(examples) >= n_samples: break
            
        except Exception as e: 
            # Catch exceptions from dictionary key lookups or indexing errors
            logger.debug(f"Skipping item due to score/index mismatch or generic error: {e}") 
            continue
            
    logger.info(f"Loaded {len(examples)} examples for {country_name}")
    return examples