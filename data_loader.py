# data_loader.py
import logging
import re
from datasets import load_dataset
from typing import List, Dict, Optional, Tuple
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
    """
    Parse a string representation of defaultdict into a regular dictionary.
    
    The dataset stores selections as strings like:
    "defaultdict(<class 'list'>, {'Sweden': [0.06, 0.4, 0.38, 0.13, 0.03]})"
    
    Returns a regular dict: {'Sweden': [0.06, 0.4, 0.38, 0.13, 0.03]}
    """
    # If already a dict, return it
    if isinstance(s, dict):
        return s if s else None
    
    if not isinstance(s, str):
        return None
    
    # Pattern to extract the dictionary part from defaultdict string
    # Handles: defaultdict(<class 'list'>, {...})
    match = re.search(r"defaultdict\s*\([^,]+,\s*(\{.*\})\s*\)$", s.strip(), re.DOTALL)
    
    if match:
        dict_str = match.group(1)
        try:
            # Parse the dictionary string
            # Using eval with empty builtins for safety
            result = eval(dict_str, {"__builtins__": {}}, {})
            if isinstance(result, dict) and result:
                return result
        except Exception as e:
            logger.debug(f"Failed to eval dict string: {e}")
            return None
    
    # Try direct JSON parse as fallback
    try:
        result = json.loads(s)
        if isinstance(result, dict) and result:
            return result
    except (json.JSONDecodeError, TypeError):
        pass
    
    # Try direct eval for plain dict strings
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
    Useful for understanding what data is available.
    """
    ds = get_dataset()
    country_counts = {}
    
    for item in ds:
        options = parse_options_string(item.get('options'))
        if not options:
            continue
            
        # Check if binary agree/disagree
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
    Filters for Agree/Disagree questions and creates binary labels.
    
    Args:
        country_name: Name of the country (e.g., "US", "Germany", "Britain")
        n_samples: Max number of samples to return. If None, returns ALL available data.
    
    Returns:
        List of Example objects with binary labels (1=Agree, 0=Disagree)
    """
    try:
        ds = get_dataset()
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return []
    
    examples = []
    
    # Map country names to dataset keys (based on what appears in the dataset)
    # The dataset uses full country names, not ISO codes
    code_map = {
        'United States': 'US', 'US': 'US', 'USA': 'US',
        'China': 'China', 'CN': 'China',
        'India': 'India', 'IN': 'India',
        'Germany': 'Germany', 'DE': 'Germany',
        'Nigeria': 'Nigeria', 'NG': 'Nigeria',
        'Japan': 'Japan', 'JP': 'Japan',
        'Britain': 'Britain', 'UK': 'Britain', 'United Kingdom': 'Britain',
        'France': 'France', 'FR': 'France',
        'Australia': 'Australia', 'AU': 'Australia',
        'Sweden': 'Sweden', 'SE': 'Sweden',
        'Hungary': 'Hungary', 'HU': 'Hungary',
        'Tunisia': 'Tunisia', 'TN': 'Tunisia',
        'Argentina': 'Argentina', 'AR': 'Argentina',
        'Bangladesh': 'Bangladesh', 'BD': 'Bangladesh',
        'Belgium': 'Belgium', 'BE': 'Belgium',
        'Brazil': 'Brazil', 'BR': 'Brazil',
        'Canada': 'Canada', 'CA': 'Canada',
        'Egypt': 'Egypt', 'EG': 'Egypt',
        'Indonesia': 'Indonesia', 'ID': 'Indonesia',
        'Italy': 'Italy', 'IT': 'Italy',
        'Kenya': 'Kenya', 'KE': 'Kenya',
        'Mexico': 'Mexico', 'MX': 'Mexico',
        'Netherlands': 'Netherlands', 'NL': 'Netherlands',
        'Pakistan': 'Pakistan', 'PK': 'Pakistan',
        'Philippines': 'Philippines', 'PH': 'Philippines',
        'Poland': 'Poland', 'PL': 'Poland',
        'Russia': 'Russia', 'RU': 'Russia',
        'South Africa': 'South Africa', 'ZA': 'South Africa',
        'South Korea': 'South Korea', 'KR': 'South Korea',
        'Spain': 'Spain', 'ES': 'Spain',
        'Turkey': 'Turkey', 'TR': 'Turkey',
        'Vietnam': 'Vietnam', 'VN': 'Vietnam',
    }
    target_code = code_map.get(country_name, country_name)
    
    # Statistics for debugging
    stats = {
        'total': 0,
        'no_options': 0,
        'no_scores': 0,
        'no_agree_disagree': 0,
        'no_country': 0,
        'success': 0
    }
    
    # Track which country codes we actually see
    seen_countries = set()
    
    for idx, item in enumerate(ds):
        stats['total'] += 1
        
        # Debug first item
        if idx == 0:
            logger.info(f"Dataset columns: {list(item.keys())}")
            logger.info(f"'selections' type: {type(item.get('selections'))}")
            sel_str = str(item.get('selections', ''))[:300]
            logger.info(f"'selections' sample: {sel_str}")
        
        # Get question
        question = item.get('question', '')
        if not question:
            continue
        
        # Parse options
        options = parse_options_string(item.get('options'))
        if not options:
            stats['no_options'] += 1
            continue
        
        # Parse selections (the score data)
        all_scores = parse_defaultdict_string(item.get('selections'))
        if not all_scores:
            stats['no_scores'] += 1
            continue
        
        # Track seen countries (for debugging)
        if idx < 100:
            seen_countries.update(all_scores.keys())
        
        # Find Agree/Disagree indices
        agree_idx = -1
        disagree_idx = -1
        
        for i, opt in enumerate(options):
            opt_str = str(opt).lower()
            # Look for variations: "Agree", "Strongly agree", etc.
            if 'agree' in opt_str and 'disagree' not in opt_str and agree_idx == -1:
                agree_idx = i
            elif 'disagree' in opt_str and disagree_idx == -1:
                disagree_idx = i
        
        if agree_idx == -1 or disagree_idx == -1:
            stats['no_agree_disagree'] += 1
            continue
        
        # Check if target country exists in this item's data
        if target_code not in all_scores:
            stats['no_country'] += 1
            continue
        
        try:
            country_scores = all_scores[target_code]
            
            # Validate indices
            max_idx = max(agree_idx, disagree_idx)
            if len(country_scores) <= max_idx:
                continue
            
            s_agree = float(country_scores[agree_idx])
            s_disagree = float(country_scores[disagree_idx])
            
            # Binary label: 1 if agree > disagree, else 0
            label = 1 if s_agree > s_disagree else 0
            
            examples.append(Example(
                id=str(len(examples)),
                question=question,
                choice="Agree",
                label=label
            ))
            stats['success'] += 1
            
            # Only limit if n_samples is specified
            if n_samples is not None and len(examples) >= n_samples:
                break
                
        except (IndexError, KeyError, TypeError, ValueError) as e:
            logger.debug(f"Score extraction error: {e}")
            continue
    
    # Log statistics
    logger.info(f"=== Data Loading Stats for '{country_name}' (target: '{target_code}') ===")
    logger.info(f"Total items scanned: {stats['total']}")
    logger.info(f"Skipped - no options: {stats['no_options']}")
    logger.info(f"Skipped - no scores: {stats['no_scores']}")
    logger.info(f"Skipped - no agree/disagree: {stats['no_agree_disagree']}")
    logger.info(f"Skipped - no country data: {stats['no_country']}")
    logger.info(f"Successfully loaded: {stats['success']} examples")
    
    if n_samples is None:
        logger.info(f"Mode: ALL available data (no limit)")
    else:
        logger.info(f"Mode: Limited to {n_samples} samples")
    
    if seen_countries:
        logger.info(f"Sample of country codes in dataset: {sorted(list(seen_countries))[:20]}")
    
    return examples


if __name__ == "__main__":
    # Test the data loader
    print("="*60)
    print("Testing Data Loader")
    print("="*60)
    
    # Test parsing functions
    print("\n--- Testing Parsing Functions ---")
    test_str = "defaultdict(<class 'list'>, {'US': [0.1, 0.2, 0.3], 'Germany': [0.4, 0.5, 0.6]})"
    parsed = parse_defaultdict_string(test_str)
    print(f"Parse test result: {parsed}")
    assert parsed == {'US': [0.1, 0.2, 0.3], 'Germany': [0.4, 0.5, 0.6]}
    print("✓ Parse test passed!")
    
    # Show available countries
    print("\n--- Scanning Dataset for Available Countries ---")
    print("(This will take a moment...)")
    country_counts = get_available_countries()
    
    print("\nCountries with binary (Agree/Disagree) questions:")
    print("-" * 40)
    for country, count in list(country_counts.items())[:20]:
        print(f"  {country:25} : {count:4} questions")
    
    if len(country_counts) > 20:
        print(f"  ... and {len(country_counts) - 20} more countries")
    
    # Test loading data for US
    print("\n--- Testing Data Loading for US ---")
    
    # Test with limit
    print("\nWith n_samples=10:")
    examples_limited = prepare_country_data("US", n_samples=10)
    print(f"Loaded: {len(examples_limited)} examples")
    
    # Test without limit (all data)
    print("\nWith n_samples=None (ALL data):")
    examples_all = prepare_country_data("US", n_samples=None)
    print(f"Loaded: {len(examples_all)} examples (THIS IS ALL AVAILABLE DATA)")
    
    if examples_all:
        print(f"\nSample question: {examples_all[0].question[:100]}...")
        print(f"Sample label: {examples_all[0].label} ({'Agree' if examples_all[0].label == 1 else 'Disagree'})")
        
        # Label distribution
        agree_count = sum(1 for ex in examples_all if ex.label == 1)
        print(f"\nLabel distribution: {agree_count} agree ({100*agree_count/len(examples_all):.1f}%), "
              f"{len(examples_all) - agree_count} disagree ({100*(len(examples_all)-agree_count)/len(examples_all):.1f}%)")