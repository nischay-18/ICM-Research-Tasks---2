# icm_core.py
"""
Improved ICM (In-Context Matching) implementation for persona elicitation.
Key improvements:
1. Multiple random restarts to find best solution
2. Better prompt formatting
3. Coherence measurement
4. More iterations for convergence
"""
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass
import asyncio
from tqdm import tqdm
import logging
import aiohttp
import copy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ICMConfig:
    n_iterations: int = 10  # More iterations for convergence
    batch_size: int = 10
    n_shots_context: int = 8
    n_restarts: int = 3  # Multiple random restarts


@dataclass
class Example:
    question: str
    choice: str
    label: Optional[int] = None
    predicted_label: int = 0
    id: str = ""
    
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
        logger.info(f"VLLMClient: {self.base_url}, model: {model_name}")
    
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
                    # Handle various tokenizations
                    if 'yes' in t or t == 'y':
                        yes_score = max(yes_score, logprob)
                    if 'no' in t or t == 'n':
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
    """In-Context Matching for persona elicitation."""
    
    def __init__(self, api_key: str, base_url: str, model_name: str, config: ICMConfig):
        self.config = config
        self.client = VLLMClient(api_key, base_url, model_name)

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

    async def update_single_example(
        self, session: aiohttp.ClientSession, 
        target_ex: Example, all_examples: List[Example]
    ) -> Tuple[bool, float]:
        """Update single example label based on context."""
        possible_context = [ex for ex in all_examples if ex.id != target_ex.id]
        if not possible_context:
            return False, 0.0
        
        n_context = min(len(possible_context), self.config.n_shots_context)
        context_indices = np.random.choice(len(possible_context), size=n_context, replace=False)
        context = [possible_context[i] for i in context_indices]
        
        prompt = self.create_icm_prompt(target_ex, context)
        yes_score, no_score = await self.client.get_label_logprobs(session, prompt)
        
        if yes_score == -100.0 and no_score == -100.0:
            return False, 0.0
        
        confidence = abs(yes_score - no_score)
        new_label = 1 if yes_score > no_score else 0
        
        if new_label != target_ex.predicted_label:
            target_ex.predicted_label = new_label
            return True, confidence
        return False, confidence

    async def run_single_icm(self, examples: List[Example], session: aiohttp.ClientSession) -> Tuple[List[Example], int]:
        """Run single ICM optimization from random start."""
        working = [copy.deepcopy(ex) for ex in examples]
        
        # Random init
        for ex in working:
            ex.predicted_label = np.random.randint(0, 2)
        
        total_changes = 0
        prev_labels = None
        stable = 0
        
        for iteration in range(self.config.n_iterations):
            indices = np.random.permutation(len(working))
            changes = 0
            
            for j in range(0, len(indices), self.config.batch_size):
                batch_idx = indices[j:j + self.config.batch_size]
                tasks = [self.update_single_example(session, working[i], working) for i in batch_idx]
                results = await asyncio.gather(*tasks)
                changes += sum(1 for changed, _ in results if changed)
            
            total_changes += changes
            
            # Check convergence
            current = tuple(ex.predicted_label for ex in working)
            if current == prev_labels:
                stable += 1
                if stable >= 2:
                    break
            else:
                stable = 0
            prev_labels = current
            
            if changes == 0:
                break
        
        return working, total_changes

    async def search_labels(self, examples: List[Example]) -> List[Example]:
        """
        Run ICM label search with multiple restarts.
        Returns the solution with highest internal coherence.
        """
        logger.info(f"Starting ICM search with {self.config.n_restarts} restarts, {self.config.n_iterations} iterations each")
        
        best_examples = None
        best_agreement = -1
        
        connector = aiohttp.TCPConnector(limit=20)
        async with aiohttp.ClientSession(connector=connector) as session:
            for restart in range(self.config.n_restarts):
                logger.info(f"=== ICM Restart {restart + 1}/{self.config.n_restarts} ===")
                
                working, changes = await self.run_single_icm(examples, session)
                
                # Measure agreement with gold (for logging, not selection)
                gold_agreement = sum(1 for i, ex in enumerate(working) if ex.predicted_label == examples[i].label)
                gold_pct = gold_agreement / len(examples) * 100
                
                # Measure internal consistency
                consistency = await self.measure_consistency(working, session)
                
                logger.info(f"Restart {restart + 1}: Gold agreement={gold_pct:.1f}%, Consistency={consistency:.1f}%")
                
                # Select based on consistency (not gold agreement - that would be cheating!)
                if consistency > best_agreement:
                    best_agreement = consistency
                    best_examples = [copy.deepcopy(ex) for ex in working]
        
        # Final stats
        final_gold = sum(1 for i, ex in enumerate(best_examples) if ex.predicted_label == examples[i].label)
        logger.info(f"Best solution: Consistency={best_agreement:.1f}%, Gold agreement={final_gold/len(examples)*100:.1f}%")
        
        return best_examples

    async def measure_consistency(self, examples: List[Example], session: aiohttp.ClientSession) -> float:
        """Measure how consistent the labeling is (internal coherence)."""
        if len(examples) < 5:
            return 0.0
        
        consistent = 0
        total = min(25, len(examples))
        test_indices = np.random.choice(len(examples), size=total, replace=False)
        
        for idx in test_indices:
            target = examples[idx]
            context = [ex for i, ex in enumerate(examples) if i != idx][:self.config.n_shots_context]
            
            prompt = self.create_icm_prompt(target, context)
            yes_score, no_score = await self.client.get_label_logprobs(session, prompt)
            
            if yes_score != -100.0 or no_score != -100.0:
                predicted = 1 if yes_score > no_score else 0
                if predicted == target.predicted_label:
                    consistent += 1
        
        return consistent / total * 100

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