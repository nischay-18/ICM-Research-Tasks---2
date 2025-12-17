# data_loader.py
"""
Data loader for GlobalOpinionQA dataset.
Extracts binary opinion questions and converts them to ICM-compatible format.
"""

import logging
import re
from datasets import load_dataset
from typing import List, Dict, Optional, Tuple
import json
from dataclasses import dataclass
from collections import defaultdict

# =============================================================================
# CONFIGURATION
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class Example:
    """Single question-answer pair for ICM training."""
    question: str
    choice: str  # Always "Agree" for binary classification
    label: int  # 1 = majority agrees, 0 = majority disagrees
    predicted_label: int = 0
    id: str = ""
    
    def to_text(self, label_val: int) -> str:
        """Convert to text format for ICM prompting."""
        lbl_str = "Yes" if label_val == 1 else "No"
        return f"Question: {self.question}\nDoes the persona agree?: {lbl_str}"


# =============================================================================
# DATASET CACHING
# =============================================================================

_dataset_cache = None

def get_dataset():
    """Load and cache the GlobalOpinionQA dataset."""
    global _dataset_cache
    if _dataset_cache is None:
        logger.info("Loading GlobalOpinionQA dataset from HuggingFace...")
        logger.info("(This may take 30-60 seconds on first run)")
        _dataset_cache = load_dataset("anthropic/llm_global_opinions", split="train")
        logger.info(f"✓ Dataset loaded: {len(_dataset_cache):,} total items")
    return _dataset_cache


# =============================================================================
# PARSING UTILITIES
# =============================================================================

def parse_defaultdict_string(s) -> Optional[Dict]:
    """
    Parse string representation of defaultdict/dict into regular dictionary.
    
    The dataset stores country scores as strings like:
    "defaultdict(<class 'list'>, {'Kenya': [0.45, 0.55], 'US': [0.60, 0.40]})"
    
    This function extracts the actual dictionary.
    """
    # Already a dict? Return it
    if isinstance(s, dict):
        return s if s else None
    
    if not isinstance(s, str):
        return None
    
    # Try to extract from defaultdict string
    match = re.search(r"defaultdict\s*\([^,]+,\s*(\{.*\})\s*\)$", s.strip(), re.DOTALL)
    if match:
        dict_str = match.group(1)
        try:
            result = eval(dict_str, {"__builtins__": {}}, {})
            if isinstance(result, dict) and result:
                return result
        except Exception as e:
            logger.debug(f"Failed to eval defaultdict string: {e}")
            return None
    
    # Try JSON parsing
    try:
        result = json.loads(s)
        if isinstance(result, dict) and result:
            return result
    except (json.JSONDecodeError, TypeError):
        pass
    
    # Try direct eval
    try:
        result = eval(s, {"__builtins__": {}}, {})
        if isinstance(result, dict) and result:
            return result
    except Exception:
        pass
    
    return None


def parse_options_string(s) -> Optional[List[str]]:
    """
    Parse options which might be a string representation of a list.
    
    Example input: "['Strongly Agree', 'Agree', 'Disagree', 'Strongly Disagree']"
    Output: ['Strongly Agree', 'Agree', 'Disagree', 'Strongly Disagree']
    """
    if isinstance(s, list):
        return s if s else None
    
    if not isinstance(s, str):
        return None
    
    # Try eval
    try:
        result = eval(s, {"__builtins__": {}}, {})
        if isinstance(result, list) and result:
            return result
    except Exception:
        pass
    
    # Try JSON
    try:
        result = json.loads(s)
        if isinstance(result, list) and result:
            return result
    except (json.JSONDecodeError, TypeError):
        pass
    
    return None


# =============================================================================
# BINARY DETECTION LOGIC
# =============================================================================

def find_binary_indices(options: List[str]) -> Tuple[List[int], List[int]]:
    """
    Find indices of positive and negative options for binary classification.
    
    This handles multiple formats:
    - Agree/Disagree
    - Yes/No
    - Support/Oppose
    - Favor/Against
    - Approve/Disapprove
    
    Returns:
        (positive_indices, negative_indices)
        Example: ([0, 1], [2, 3]) for ['Strongly Agree', 'Agree', 'Disagree', 'Strongly Disagree']
    """
    options_lower = [str(opt).lower().strip() for opt in options]
    
    positive_indices = []
    negative_indices = []
    
    # Pattern 1: Agree/Disagree
    for i, opt in enumerate(options_lower):
        if 'agree' in opt and 'disagree' not in opt:
            positive_indices.append(i)
        elif 'disagree' in opt:
            negative_indices.append(i)
    
    if positive_indices and negative_indices:
        return (positive_indices, negative_indices)
    
    # Pattern 2: Yes/No
    positive_indices = []
    negative_indices = []
    for i, opt in enumerate(options_lower):
        if opt == 'yes' or opt.startswith('yes,') or opt.startswith('yes '):
            positive_indices.append(i)
        elif opt == 'no' or opt.startswith('no,') or opt.startswith('no '):
            negative_indices.append(i)
    
    if positive_indices and negative_indices:
        return (positive_indices, negative_indices)
    
    # Pattern 3: Support/Oppose
    positive_indices = []
    negative_indices = []
    for i, opt in enumerate(options_lower):
        if 'support' in opt and 'oppose' not in opt:
            positive_indices.append(i)
        elif 'oppose' in opt:
            negative_indices.append(i)
    
    if positive_indices and negative_indices:
        return (positive_indices, negative_indices)
    
    # Pattern 4: Favor/Against
    positive_indices = []
    negative_indices = []
    for i, opt in enumerate(options_lower):
        if 'favor' in opt and 'against' not in opt:
            positive_indices.append(i)
        elif 'against' in opt:
            negative_indices.append(i)
    
    if positive_indices and negative_indices:
        return (positive_indices, negative_indices)
    
    # Pattern 5: Approve/Disapprove
    positive_indices = []
    negative_indices = []
    for i, opt in enumerate(options_lower):
        if 'approve' in opt and 'disapprove' not in opt:
            positive_indices.append(i)
        elif 'disapprove' in opt:
            negative_indices.append(i)
    
    if positive_indices and negative_indices:
        return (positive_indices, negative_indices)
    
    # No binary pattern found
    return ([], [])


# =============================================================================
# COUNTRY UTILITIES
# =============================================================================

def get_available_countries() -> Dict[str, int]:
    """
    Scan dataset and return dictionary of {country_name: num_binary_questions}.
    """
    ds = get_dataset()
    country_counts = defaultdict(int)
    
    logger.info("Scanning dataset for available countries...")
    
    for item in ds:
        options = parse_options_string(item.get('options'))
        if not options:
            continue
        
        # Check if question is binary
        positive_idx, negative_idx = find_binary_indices(options)
        if not positive_idx or not negative_idx:
            continue
        
        # Get country scores
        all_scores = parse_defaultdict_string(item.get('selections'))
        if not all_scores:
            continue
        
        # Count questions per country
        for country in all_scores.keys():
            country_counts[country] += 1
    
    # Sort by count (descending)
    return dict(sorted(country_counts.items(), key=lambda x: -x[1]))


# Country name mapping (handles common aliases)
COUNTRY_ALIASES = {
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
    'Kenya': 'Kenya',
    'Ethiopia': 'Ethiopia',
    'Zimbabwe': 'Zimbabwe',
    'Pakistan': 'Pakistan',
    'Lebanon': 'Lebanon',
}


# =============================================================================
# MAIN DATA PREPARATION FUNCTION
# =============================================================================

def prepare_country_data(country_name: str, n_samples: int = None) -> List[Example]:
    """
    Download and format GlobalOpinionQA data for a specific country.
    
    Args:
        country_name: Name of country (e.g., 'Kenya', 'US', 'Germany')
        n_samples: Max number of samples. If None, returns ALL available data.
    
    Returns:
        List of Example objects with binary labels (1=Positive, 0=Negative)
    
    Example:
        >>> examples = prepare_country_data('Kenya', n_samples=50)
        >>> print(f"Loaded {len(examples)} questions for Kenya")
    """
    # Load dataset
    try:
        ds = get_dataset()
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return []
    
    # Map country name to dataset key
    target_country = COUNTRY_ALIASES.get(country_name, country_name)
    
    # Statistics tracking
    stats = {
        'total_items': 0,
        'no_options': 0,
        'no_scores': 0,
        'not_binary': 0,
        'no_country_data': 0,
        'success': 0
    }
    
    examples = []
    seen_countries = set()
    
    logger.info(f"Processing data for: {country_name} (dataset key: {target_country})")
    
    # Process each item in dataset
    for idx, item in enumerate(ds):
        stats['total_items'] += 1
        
        # Extract question
        question = item.get('question', '')
        if not question:
            continue
        
        # Parse options
        options = parse_options_string(item.get('options'))
        if not options:
            stats['no_options'] += 1
            continue
        
        # Check if question is binary
        positive_indices, negative_indices = find_binary_indices(options)
        if not positive_indices or not negative_indices:
            stats['not_binary'] += 1
            continue
        
        # Parse country scores
        all_scores = parse_defaultdict_string(item.get('selections'))
        if not all_scores:
            stats['no_scores'] += 1
            continue
        
        # Track countries we've seen (for debugging)
        if idx < 100:
            seen_countries.update(all_scores.keys())
        
        # Check if this country has data
        if target_country not in all_scores:
            stats['no_country_data'] += 1
            continue
        
        # Extract scores for this country
        try:
            country_scores = all_scores[target_country]
            
            # Sum positive and negative scores
            positive_score = sum(float(country_scores[i]) for i in positive_indices)
            negative_score = sum(float(country_scores[i]) for i in negative_indices)
            
            # Assign binary label based on majority
            label = 1 if positive_score > negative_score else 0
            
            # Create Example object
            examples.append(Example(
                id=str(len(examples)),
                question=question,
                choice="Agree",  # Standard choice text for binary classification
                label=label
            ))
            
            stats['success'] += 1
            
            # Stop if we've reached desired sample size
            if n_samples is not None and len(examples) >= n_samples:
                break
        
        except (IndexError, KeyError, TypeError, ValueError) as e:
            logger.debug(f"Score extraction error for item {idx}: {e}")
            continue
    
    # Print summary
    logger.info("=" * 70)
    logger.info(f"Data Loading Summary for '{country_name}'")
    logger.info("=" * 70)
    logger.info(f"Total items processed: {stats['total_items']:,}")
    logger.info(f"Successfully loaded: {stats['success']} examples")
    logger.info(f"Skipped - no options: {stats['no_options']}")
    logger.info(f"Skipped - not binary: {stats['not_binary']}")
    logger.info(f"Skipped - no country data: {stats['no_country_data']}")
    
    if stats['success'] == 0 and seen_countries:
        # Suggest alternatives if country not found
        matching = [c for c in seen_countries if country_name.lower() in c.lower()]
        if matching:
            logger.warning(f"Did you mean one of these? {matching}")
        else:
            logger.info(f"Available countries: {sorted(list(seen_countries))[:20]}")
    
    return examples


# =============================================================================
# DATASET SPLITTING UTILITIES
# =============================================================================

def split_train_test(examples: List[Example], test_ratio: float = 0.3) -> Tuple[List[Example], List[Example]]:
    """
    Split examples into train and test sets.
    
    Args:
        examples: List of Example objects
        test_ratio: Fraction to use for testing (default: 0.3)
    
    Returns:
        (train_examples, test_examples)
    """
    import random
    random.seed(42)  # For reproducibility
    
    shuffled = examples.copy()
    random.shuffle(shuffled)
    
    split_idx = int(len(shuffled) * (1 - test_ratio))
    train = shuffled[:split_idx]
    test = shuffled[split_idx:]
    
    logger.info(f"Split: {len(train)} train, {len(test)} test")
    return train, test


# =============================================================================
# MAIN (FOR TESTING)
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("GlobalOpinionQA Data Loader - Test Run")
    print("=" * 70)
    
    # Test 1: Show available countries
    print("\n[TEST 1] Scanning for available countries...")
    countries = get_available_countries()
    
    print(f"\nFound {len(countries)} countries with binary questions:")
    print("-" * 70)
    for country, count in list(countries.items())[:15]:
        print(f"  {country:30} : {count:4} binary questions")
    
    # Test 2: Load data for a specific country
    print("\n[TEST 2] Loading data for Kenya...")
    kenya_data = prepare_country_data('Kenya', n_samples=None)
    
    if kenya_data:
        print(f"\n✓ Successfully loaded {len(kenya_data)} examples")
        print("\nExample:")
        ex = kenya_data[0]
        print(f"  Question: {ex.question}")
        print(f"  Label: {ex.label} ({'Majority agrees' if ex.label == 1 else 'Majority disagrees'})")
        
        # Test 3: Train/test split
        print("\n[TEST 3] Splitting into train/test...")
        train, test = split_train_test(kenya_data, test_ratio=0.3)
        print(f"  Train set: {len(train)} examples")
        print(f"  Test set: {len(test)} examples")
    else:
        print("\n✗ Failed to load data")
    
    print("\n" + "=" * 70)
    print("Testing complete!")