import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass, field
import asyncio
from tqdm import tqdm
import logging
import aiohttp
import copy
import math

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ICMConfig:
    """Configuration matching the paper's algorithm."""
    # Simulated Annealing parameters (from reference implementation)
    initial_temperature: float = 3.0      # Starting temperature
    final_temperature: float = 0.001      # Ending temperature  
    cooling_rate: float = 0.98            # Temperature decay per iteration
    
    # Objective function
    alpha: float = 100.0                  # Weight for P(D) vs I(D)
    
    # Search parameters
    max_iterations: int = 500             # Maximum SA iterations
    initial_examples: int = 20            # Start with K random labels
    
    # Context and batching
    n_shots_context: int = 8              # Context window size
    batch_size: int = 10                  # Batch size for API calls
    
    # Consistency penalty parameters
    class_balance_threshold: float = 0.8  # Penalize if one class > 80%
    consistency_weight: float = 1.0       # Weight for I(D) penalty
    
    # Restarts (optional, but less critical with proper SA)
    n_restarts: int = 1


@dataclass
class Example:
    question: str
    choice: str
    label: Optional[int] = None       # Gold label (ground truth)
    predicted_label: int = 0          # ICM-assigned label
    id: str = ""
    logprob_cache: Dict[int, float] = field(default_factory=dict)  # Cache logprobs
    
    def to_text(self, label_val: int) -> str:
        lbl_str = "Yes" if label_val == 1 else "No"
        return f"Statement: {self.question}\nAgree: {lbl_str}"


class VLLMClient:
    """Client for vLLM OpenAI-compatible API."""
    
    def __init__(self, api_key: str, base_url: str, model_name: str):
        self.base_url = base_url.rstrip('/')
        self.url = f"{self.base_url}/completions"
        self.headers = {"Content-Type": "application/json"}
        if api_key and api_key.strip():
            self.headers["Authorization"] = f"Bearer {api_key}"
        self.model_name = model_name
        logger.info(f"VLLMClient initialized: {self.base_url}")
    
    async def get_label_logprobs(
        self, 
        session: aiohttp.ClientSession, 
        prompt: str
    ) -> Tuple[float, float]:
        """Gets log probabilities for Yes/No tokens."""
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "max_tokens": 1,
            "temperature": 0.0,
            "logprobs": 10,
        }
        
        try:
            async with session.post(
                self.url, headers=self.headers, json=payload,
                timeout=aiohttp.ClientTimeout(total=60)
            ) as response:
                if response.status != 200:
                    return -100.0, -100.0
                
                result = await response.json()
                choices = result.get('choices', [])
                if not choices:
                    return -100.0, -100.0
                
                logprobs_data = choices[0].get('logprobs', {})
                top_logprobs_list = logprobs_data.get('top_logprobs', [])
                
                if not top_logprobs_list:
                    return -100.0, -100.0
                
                top_dict = top_logprobs_list[0]
                yes_score, no_score = -100.0, -100.0
                
                for token, logprob in top_dict.items():
                    t = token.lower().strip()
                    # Handle various tokenizations (yes, Yes, ĠYes, ' yes', y, Y)
                    if 'yes' in t or t == 'y':
                        yes_score = max(yes_score, logprob)
                    if 'no' in t or (t == 'n' and 'no' not in t):
                        no_score = max(no_score, logprob)
                
                return yes_score, no_score
                
        except Exception as e:
            logger.debug(f"Logprob error: {e}")
            return -100.0, -100.0

    async def get_completion(
        self, session: aiohttp.ClientSession, prompt: str
    ) -> str:
        """Gets text completion from the model."""
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "temperature": 0.0,
            "max_tokens": 5,
            "stop": ["\n", ".", ",", "\n\n"]
        }
        
        try:
            async with session.post(
                self.url, headers=self.headers, json=payload,
                timeout=aiohttp.ClientTimeout(total=60)
            ) as response:
                if response.status != 200:
                    return ""
                result = await response.json()
                choices = result.get('choices', [])
                return choices[0].get('text', "").strip() if choices else ""
        except Exception:
            return ""


class ICM:
    """
    Internal Coherence Maximization with proper Simulated Annealing.
    
    Objective: U(D) = α·P_θ(D) - I(D)
    
    Where:
    - P_θ(D) = Σᵢ log P(yᵢ | xᵢ, D\{i}) is mutual predictability
    - I(D) is the inconsistency/degeneracy penalty
    - α balances the two terms
    """
    
    def __init__(self, api_key: str, base_url: str, model_name: str, config: ICMConfig):
        self.config = config
        self.client = VLLMClient(api_key, base_url, model_name)
        self._context_cache = {}  # Cache for stable context selection

    def create_icm_prompt(self, target_ex: Example, context_examples: List[Example]) -> str:
        """Create ICM prompt for label prediction."""
        prompt = "Based on a specific persona's opinions on various statements:\n\n"
        
        for ex in context_examples:
            answer = "Yes" if ex.predicted_label == 1 else "No"
            prompt += f"Statement: {ex.question}\nPersona agrees: {answer}\n\n"
        
        prompt += f"Statement: {target_ex.question}\nPersona agrees:"
        return prompt
    
    def apply_llama3_chat_template(self, system_msg: str, user_msg: str) -> str:
        return (
            f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            f"{system_msg}<|eot_id|>"
            f"<|start_header_id|>user<|end_header_id|>\n\n"
            f"{user_msg}<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n"
        )

    def calculate_inconsistency_penalty(self, examples: List[Example]) -> float:
        """
        Calculate I(D) - the logical inconsistency penalty.
        
        Penalizes:
        1. Class imbalance (mode collapse to all-Yes or all-No)
        2. Can be extended with domain-specific logical constraints
        """
        if not examples:
            return 0.0
        
        # Count class distribution
        n_yes = sum(1 for ex in examples if ex.predicted_label == 1)
        n_total = len(examples)
        
        if n_total == 0:
            return 0.0
        
        yes_ratio = n_yes / n_total
        
        # Penalty for class imbalance (mode collapse)
        # Penalize if distribution is too skewed (> threshold)
        penalty = 0.0
        
        if yes_ratio > self.config.class_balance_threshold:
            # Too many Yes labels
            penalty = (yes_ratio - self.config.class_balance_threshold) * 10.0
        elif yes_ratio < (1 - self.config.class_balance_threshold):
            # Too many No labels
            penalty = ((1 - self.config.class_balance_threshold) - yes_ratio) * 10.0
        
        # Additional entropy-based penalty (encourage diversity)
        if yes_ratio > 0 and yes_ratio < 1:
            entropy = -yes_ratio * math.log(yes_ratio) - (1-yes_ratio) * math.log(1-yes_ratio)
            max_entropy = math.log(2)  # Maximum entropy for binary
            # Penalize low entropy (low diversity)
            penalty += (1 - entropy / max_entropy) * 2.0
        else:
            # Degenerate case: all same label
            penalty += 5.0  # Strong penalty for complete collapse
        
        return penalty * self.config.consistency_weight

    async def calculate_mutual_predictability(
        self, 
        examples: List[Example], 
        session: aiohttp.ClientSession,
        sample_size: int = None
    ) -> float:
        """
        Calculate P_θ(D) = Σᵢ log P(yᵢ | xᵢ, D\{i})
        
        For efficiency, we sample a subset of examples.
        """
        if not examples:
            return 0.0
        
        # Sample for efficiency
        if sample_size is None:
            sample_size = min(30, len(examples))
        
        indices = np.random.choice(len(examples), size=sample_size, replace=False)
        
        total_logprob = 0.0
        valid_count = 0
        
        for idx in indices:
            target = examples[idx]
            
            # Get context (all others, limited to n_shots)
            context = [ex for i, ex in enumerate(examples) if i != idx]
            context = context[:self.config.n_shots_context]
            
            prompt = self.create_icm_prompt(target, context)
            yes_score, no_score = await self.client.get_label_logprobs(session, prompt)
            
            if yes_score != -100.0 or no_score != -100.0:
                # Get logprob of the assigned label
                if target.predicted_label == 1:
                    logprob = yes_score
                else:
                    logprob = no_score
                
                if logprob > -100.0:
                    total_logprob += logprob
                    valid_count += 1
        
        # Normalize by count
        if valid_count > 0:
            return total_logprob / valid_count
        return -100.0

    async def calculate_objective(
        self, 
        examples: List[Example], 
        session: aiohttp.ClientSession
    ) -> Tuple[float, float, float]:
        """
        Calculate U(D) = α·P_θ(D) - I(D)
        """
        # Mutual predictability
        p_score = await self.calculate_mutual_predictability(examples, session)
        
        # Inconsistency penalty
        i_penalty = self.calculate_inconsistency_penalty(examples)
        
        # Combined objective
        u_score = self.config.alpha * p_score - i_penalty
        
        return u_score, p_score, i_penalty

    async def get_label_scores(
        self, 
        target_ex: Example, 
        context: List[Example],
        session: aiohttp.ClientSession
    ) -> Tuple[float, float]:
        """Get logprob scores for Yes/No for a target example."""
        prompt = self.create_icm_prompt(target_ex, context)
        return await self.client.get_label_logprobs(session, prompt)

    def metropolis_accept(self, delta_u: float, temperature: float) -> bool:
        """
        Metropolis acceptance criterion for Simulated Annealing.
        
        Accept if:
        - delta_u > 0 (improvement), OR
        - random < exp(delta_u / T) (probabilistic acceptance of worse moves)
        """
        if delta_u > 0:
            return True
        
        if temperature <= 0:
            return False
        
        # Probabilistic acceptance
        acceptance_prob = math.exp(delta_u / temperature)
        return np.random.random() < acceptance_prob

    async def search_labels(self, examples: List[Example]) -> List[Example]:
        """
        Run ICM label search using Simulated Annealing.
        
        Algorithm (from paper):
        1. Initialize with K random labels
        2. Iteratively:
           a. Sample a new example
           b. Determine optimal label (fixing inconsistencies)
           c. Accept/reject based on scoring function and temperature
        3. Cool temperature over time
        """
        logger.info(f"Starting ICM search with Simulated Annealing")
        logger.info(f"  Temperature: {self.config.initial_temperature} -> {self.config.final_temperature}")
        logger.info(f"  Cooling rate: {self.config.cooling_rate}")
        logger.info(f"  Max iterations: {self.config.max_iterations}")
        logger.info(f"  Alpha: {self.config.alpha}")
        
        best_examples = None
        best_score = float('-inf')
        
        connector = aiohttp.TCPConnector(limit=20)
        async with aiohttp.ClientSession(connector=connector) as session:
            
            for restart in range(self.config.n_restarts):
                if self.config.n_restarts > 1:
                    logger.info(f"=== Restart {restart + 1}/{self.config.n_restarts} ===")
                
                # Initialize working copy
                working = [copy.deepcopy(ex) for ex in examples]
                
                # Step 1: Random initialization
                for ex in working:
                    ex.predicted_label = np.random.randint(0, 2)
                
                # Calculate initial objective
                current_score, p_score, i_penalty = await self.calculate_objective(working, session)
                logger.info(f"Initial: U={current_score:.2f}, P={p_score:.2f}, I={i_penalty:.2f}")
                
                # Step 2: Simulated Annealing loop
                temperature = self.config.initial_temperature
                
                accepted = 0
                rejected = 0
                
                pbar = tqdm(range(self.config.max_iterations), desc="SA Search", leave=False)
                
                for iteration in pbar:
                    # Pick a random example to potentially flip
                    idx = np.random.randint(len(working))
                    target = working[idx]
                    
                    # Store old label
                    old_label = target.predicted_label
                    
                    # Get context for this example (stable selection)
                    context = [ex for i, ex in enumerate(working) if i != idx]
                    context = context[:self.config.n_shots_context]
                    
                    # Get model's preference
                    yes_score, no_score = await self.get_label_scores(target, context, session)
                    
                    # Determine proposed new label
                    if yes_score == -100.0 and no_score == -100.0:
                        continue  # Skip if API failed
                    
                    proposed_label = 1 if yes_score > no_score else 0
                    
                    if proposed_label == old_label:
                        # No change proposed, continue
                        continue
                    
                    # Tentatively apply the change
                    target.predicted_label = proposed_label
                    
                    # Calculate new objective
                    new_score, new_p, new_i = await self.calculate_objective(working, session)
                    
                    delta_u = new_score - current_score
                    
                    # Metropolis acceptance
                    if self.metropolis_accept(delta_u, temperature):
                        # Accept the change
                        current_score = new_score
                        accepted += 1
                    else:
                        # Reject - revert
                        target.predicted_label = old_label
                        rejected += 1
                    
                    # Cool down temperature
                    temperature = max(
                        self.config.final_temperature,
                        temperature * self.config.cooling_rate
                    )
                    
                    # Update progress bar
                    if iteration % 20 == 0:
                        yes_count = sum(1 for ex in working if ex.predicted_label == 1)
                        pbar.set_postfix({
                            'U': f'{current_score:.1f}',
                            'T': f'{temperature:.3f}',
                            'Yes%': f'{100*yes_count/len(working):.0f}'
                        })
                    
                    # Early stopping if temperature is cold and stable
                    if temperature <= self.config.final_temperature * 2 and rejected > 50:
                        break
                
                # Final statistics
                final_score, final_p, final_i = await self.calculate_objective(working, session)
                yes_count = sum(1 for ex in working if ex.predicted_label == 1)
                gold_match = sum(1 for i, ex in enumerate(working) 
                               if ex.predicted_label == examples[i].label)
                
                logger.info(f"Final: U={final_score:.2f}, P={final_p:.2f}, I={final_i:.2f}")
                logger.info(f"  Accepted: {accepted}, Rejected: {rejected}")
                logger.info(f"  Label distribution: {yes_count}/{len(working)} Yes ({100*yes_count/len(working):.1f}%)")
                logger.info(f"  Gold agreement: {gold_match}/{len(working)} ({100*gold_match/len(working):.1f}%)")
                
                # Track best solution
                if final_score > best_score:
                    best_score = final_score
                    best_examples = [copy.deepcopy(ex) for ex in working]
        
        return best_examples

    def create_random_labels(self, examples: List[Example]) -> List[Example]:
        """Create examples with random labels."""
        random_ex = [copy.deepcopy(ex) for ex in examples]
        for ex in random_ex:
            ex.predicted_label = np.random.randint(0, 2)
        return random_ex

    def create_gold_labels(self, examples: List[Example]) -> List[Example]:
        """Create examples with gold (true) labels."""
        gold_ex = [copy.deepcopy(ex) for ex in examples]
        for ex in gold_ex:
            ex.predicted_label = ex.label
        return gold_ex

    async def evaluate(
        self, 
        train_labeled: List[Example], 
        test_examples: List[Example], 
        n_shots: int,
        use_chat_template: bool = True
    ) -> float:
        """Evaluate accuracy on test set."""
        prompts = []
        
        for test_ex in test_examples:
            context_str = ""
            
            if n_shots > 0 and train_labeled:
                n_available = min(len(train_labeled), n_shots)
                shot_indices = np.random.choice(len(train_labeled), size=n_available, replace=False)
                shots = [train_labeled[i] for i in shot_indices]
                
                for shot in shots:
                    answer = "Yes" if shot.predicted_label == 1 else "No"
                    context_str += f"Statement: {shot.question}\nPersona agrees: {answer}\n\n"
            
            if use_chat_template:
                sys_msg = "You are simulating a specific persona. Based on their previous opinions, predict if they would agree with the following statement. Answer only Yes or No."
                user_msg = f"{context_str}Statement: {test_ex.question}\nPersona agrees:"
                full_prompt = self.apply_llama3_chat_template(sys_msg, user_msg)
            else:
                full_prompt = f"{context_str}Statement: {test_ex.question}\nPersona agrees:"
            
            prompts.append(full_prompt)
        
        correct = 0
        connector = aiohttp.TCPConnector(limit=20)
        
        async with aiohttp.ClientSession(connector=connector) as session:
            for i in tqdm(range(0, len(prompts), self.config.batch_size), desc=f"Eval {n_shots}-shot", leave=False):
                batch_prompts = prompts[i:i + self.config.batch_size]
                tasks = [self.client.get_completion(session, p) for p in batch_prompts]
                responses = await asyncio.gather(*tasks)
                
                batch_ex = test_examples[i:i + len(responses)]
                
                for ex, resp in zip(batch_ex, responses):
                    resp_lower = resp.lower().strip()
                    if resp_lower.startswith('yes') or resp_lower == 'y':
                        pred = 1
                    elif resp_lower.startswith('no') or resp_lower == 'n':
                        pred = 0
                    else:
                        pred = 1 if 'yes' in resp_lower else 0
                    
                    if pred == ex.label:
                        correct += 1
        
        return correct / len(test_examples) if test_examples else 0.0