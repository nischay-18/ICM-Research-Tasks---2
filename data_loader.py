# data_loader.py
import logging
import re
from datasets import load_dataset
from typing import List, Dict, Optional
import json
from dataclasses import dataclass

# --- DATACLASS DEFINITION ---
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


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cache the dataset to avoid reloading for each country
_dataset_cache = None


def get_dataset():
    """Load and cache the dataset."""
    global _dataset_cache
    if _dataset_cache is None:
        logger.info("Loading dataset from HuggingFace (this may take a moment)...")
        _dataset_cache = load_dataset("anthropic/llm_global_opinions", split="train")
        logger.info(f"Dataset loaded: {len(_dataset_cache)} total items")
    return _dataset_cache


def parse_defaultdict_string(s) -> Optional[Dict]:
    """Parse a string representation of defaultdict into a regular dictionary."""
    if isinstance(s, dict):
        return s if s else None
    
    if not isinstance(s, str):
        return None
    
    match = re.search(r"defaultdict\s*\([^,]+,\s*(\{.*\})\s*\)$", s.strip(), re.DOTALL)
    
    if match:
        dict_str = match.group(1)
        try:
            result = eval(dict_str, {"__builtins__": {}}, {})
            if isinstance(result, dict) and result:
                return result
        except Exception as e:
            logger.debug(f"Failed to eval dict string: {e}")
            return None
    
    try:
        result = json.loads(s)
        if isinstance(result, dict) and result:
            return result
    except (json.JSONDecodeError, TypeError):
        pass
    
    try:
        result = eval(s, {"__builtins__": {}}, {})
        if isinstance(result, dict) and result:
            return result
    except Exception:
        pass
    
    return None


def parse_options_string(s) -> Optional[List[str]]:
    """Parse options which might be a string representation of a list."""
    if isinstance(s, list):
        return s if s else None
    
    if not isinstance(s, str):
        return None
    
    try:
        result = eval(s, {"__builtins__": {}}, {})
        if isinstance(result, list) and result:
            return result
    except Exception:
        pass
    
    try:
        result = json.loads(s)
        if isinstance(result, list) and result:
            return result
    except (json.JSONDecodeError, TypeError):
        pass
    
    return None


def get_available_countries() -> Dict[str, int]:
    """
    Scan the dataset and return a dict of {country_name: num_binary_questions}.
    """
    ds = get_dataset()
    country_counts = {}
    
    for item in ds:
        options = parse_options_string(item.get('options'))
        if not options:
            continue
            
        has_agree = any('agree' in str(o).lower() and 'disagree' not in str(o).lower() for o in options)
        has_disagree = any('disagree' in str(o).lower() for o in options)
        
        if not (has_agree and has_disagree):
            continue
        
        all_scores = parse_defaultdict_string(item.get('selections'))
        if not all_scores:
            continue
            
        for country in all_scores.keys():
            country_counts[country] = country_counts.get(country, 0) + 1
    
    return dict(sorted(country_counts.items(), key=lambda x: -x[1]))


def prepare_country_data(country_name: str, n_samples: int = None) -> List[Example]:
    """
    Downloads and formats GlobalOpinionQA data for a specific country.
    
    Args:
        country_name: Name of the country - use EXACT name from dataset
        n_samples: Max number of samples. If None, returns ALL available data.
    
    Returns:
        List of Example objects with binary labels (1=Agree, 0=Disagree)
    """
    try:
        ds = get_dataset()
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return []
    
    examples = []
    
    # Use exact dataset keys - map common aliases
    code_map = {
        'US': 'United States',
        'USA': 'United States', 
        'United States': 'United States',
        'UK': 'Britain',
        'United Kingdom': 'Britain',
        'Britain': 'Britain',
        'Germany': 'Germany',
        'France': 'France',
        'Japan': 'Japan',
        'India': 'India',
        'China': 'China',
        'Australia': 'Australia',
        'Canada': 'Canada',
        'Brazil': 'Brazil',
        'Mexico': 'Mexico',
        'Spain': 'Spain',
        'Italy': 'Italy',
        'Poland': 'Poland',
        'Netherlands': 'Netherlands',
        'Sweden': 'Sweden',
        'South Korea': 'South Korea',
        'Russia': 'Russia',
        'Turkey': 'Turkey',
        'Argentina': 'Argentina',
        'Nigeria': 'Nigeria',
        'South Africa': 'South Africa',
    }
    
    target_code = code_map.get(country_name, country_name)
    
    stats = {
        'total': 0,
        'no_options': 0,
        'no_scores': 0,
        'no_agree_disagree': 0,
        'no_country': 0,
        'success': 0
    }
    
    seen_countries = set()
    
    for idx, item in enumerate(ds):
        stats['total'] += 1
        
        if idx == 0:
            logger.info(f"Dataset columns: {list(item.keys())}")
        
        question = item.get('question', '')
        if not question:
            continue
        
        options = parse_options_string(item.get('options'))
        if not options:
            stats['no_options'] += 1
            continue
        
        all_scores = parse_defaultdict_string(item.get('selections'))
        if not all_scores:
            stats['no_scores'] += 1
            continue
        
        if idx < 200:
            seen_countries.update(all_scores.keys())
        
        agree_idx = -1
        disagree_idx = -1
        
        for i, opt in enumerate(options):
            opt_str = str(opt).lower()
            if 'agree' in opt_str and 'disagree' not in opt_str and agree_idx == -1:
                agree_idx = i
            elif 'disagree' in opt_str and disagree_idx == -1:
                disagree_idx = i
        
        if agree_idx == -1 or disagree_idx == -1:
            stats['no_agree_disagree'] += 1
            continue
        
        if target_code not in all_scores:
            stats['no_country'] += 1
            continue
        
        try:
            country_scores = all_scores[target_code]
            
            max_idx = max(agree_idx, disagree_idx)
            if len(country_scores) <= max_idx:
                continue
            
            s_agree = float(country_scores[agree_idx])
            s_disagree = float(country_scores[disagree_idx])
            
            label = 1 if s_agree > s_disagree else 0
            
            examples.append(Example(
                id=str(len(examples)),
                question=question,
                choice="Agree",
                label=label
            ))
            stats['success'] += 1
            
            if n_samples is not None and len(examples) >= n_samples:
                break
                
        except (IndexError, KeyError, TypeError, ValueError) as e:
            logger.debug(f"Score extraction error: {e}")
            continue
    
    logger.info(f"=== Data Loading Stats for '{country_name}' (target: '{target_code}') ===")
    logger.info(f"Total items: {stats['total']}, No agree/disagree: {stats['no_agree_disagree']}, No country: {stats['no_country']}")
    logger.info(f"Successfully loaded: {stats['success']} examples")
    
    if stats['success'] == 0 and seen_countries:
        matching = [c for c in seen_countries if country_name.lower() in c.lower()]
        if matching:
            logger.warning(f"Did you mean one of these? {matching}")
        logger.info(f"Available countries: {sorted(list(seen_countries))[:20]}")
    
    return examples


if __name__ == "__main__":
    print("="*60)
    print("Scanning Dataset for Available Countries")
    print("="*60)
    
    country_counts = get_available_countries()
    
    print("\nCountries with binary (Agree/Disagree) questions:")
    print("-" * 50)
    for country, count in list(country_counts.items()):
        print(f"  {country:40} : {count:4} questions")
    
    print(f"\nTotal countries: {len(country_counts)}")