# icm_core.py
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
    n_iterations: int = 5
    batch_size: int = 10
    n_shots_context: int = 8


@dataclass
class Example:
    question: str
    choice: str
    label: Optional[int] = None
    predicted_label: int = 0
    id: str = ""
    
    def to_text(self, label_val: int) -> str:
        lbl_str = "Yes" if label_val == 1 else "No"
        return f"Question: {self.question}\nDoes the persona agree?: {lbl_str}"


class VLLMClient:
    """Client for vLLM OpenAI-compatible API."""
    
    def __init__(self, api_key: str, base_url: str, model_name: str):
        self.base_url = base_url.rstrip('/')
        self.url = f"{self.base_url}/completions"
        self.headers = {"Content-Type": "application/json"}
        if api_key and api_key.strip():
            self.headers["Authorization"] = f"Bearer {api_key}"
        
        self.model_name = model_name
        logger.info(f"VLLMClient initialized: {self.base_url}, model: {model_name}")
    
    async def get_label_logprobs(
        self, 
        session: aiohttp.ClientSession, 
        context_prompt: str
    ) -> Tuple[float, float]:
        """Gets log probabilities for Yes/No tokens."""
        payload = {
            "model": self.model_name,
            "prompt": context_prompt,
            "max_tokens": 1,
            "temperature": 0.0,
            "logprobs": 5,
            "stop": ["\n", "."]
        }
        
        try:
            async with session.post(
                self.url, headers=self.headers, json=payload,
                timeout=aiohttp.ClientTimeout(total=60)
            ) as response:
                if response.status != 200:
                    return -10.0, -10.0
                
                result = await response.json()
                choices = result.get('choices', [])
                if not choices:
                    return -10.0, -10.0
                
                logprobs_data = choices[0].get('logprobs', {})
                top_logprobs_list = logprobs_data.get('top_logprobs', [])
                
                if not top_logprobs_list:
                    return -10.0, -10.0
                
                top_dict = top_logprobs_list[0]
                yes_score, no_score = -999.0, -999.0
                
                for token, logprob in top_dict.items():
                    t = token.strip().lower()
                    if t in ['yes', 'true', 'agree']:
                        yes_score = max(yes_score, logprob)
                    if t in ['no', 'false', 'disagree']:
                        no_score = max(no_score, logprob)
                
                return yes_score, no_score
                
        except Exception as e:
            logger.debug(f"Logprob error: {e}")
            return -10.0, -10.0

    async def get_completion(
        self, session: aiohttp.ClientSession, prompt: str
    ) -> str:
        """Gets text completion from the model."""
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "temperature": 0.1,
            "max_tokens": 10,
            "stop": ["<|eot_id|>", "\n\n"]
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
                return choices[0].get('text', "") if choices else ""
        except Exception:
            return ""


class ICM:
    """In-Context Matching for persona elicitation."""
    
    def __init__(self, api_key: str, base_url: str, model_name: str, config: ICMConfig):
        self.config = config
        self.client = VLLMClient(api_key, base_url, model_name)

    def create_icm_prompt(self, target_ex: Example, context_examples: List[Example]) -> str:
        prompt = "The following are questions and whether a specific persona agrees with them.\n\n"
        for ex in context_examples:
            prompt += ex.to_text(ex.predicted_label) + "\n\n"
        prompt += f"Question: {target_ex.question}\nDoes the persona agree?:"
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
    ) -> bool:
        possible_context = [ex for ex in all_examples if ex.id != target_ex.id]
        if not possible_context:
            return False
        
        n_context = min(len(possible_context), self.config.n_shots_context)
        context_indices = np.random.choice(len(possible_context), size=n_context, replace=False)
        context = [possible_context[i] for i in context_indices]
        
        prompt = self.create_icm_prompt(target_ex, context)
        yes_score, no_score = await self.client.get_label_logprobs(session, prompt)
        
        if yes_score == -10.0 and no_score == -10.0:
            return False
        
        new_label = 1 if yes_score > no_score else 0
        if new_label != target_ex.predicted_label:
            target_ex.predicted_label = new_label
            return True
        return False

    async def search_labels(self, examples: List[Example]) -> List[Example]:
        """Run ICM label search."""
        working_examples = [copy.deepcopy(ex) for ex in examples]
        
        for ex in working_examples:
            ex.predicted_label = np.random.randint(0, 2)
        
        connector = aiohttp.TCPConnector(limit=20)
        async with aiohttp.ClientSession(connector=connector) as session:
            for iteration in range(self.config.n_iterations):
                tasks = [
                    self.update_single_example(session, ex, working_examples) 
                    for ex in working_examples
                ]
                
                changes = 0
                for j in tqdm(
                    range(0, len(tasks), self.config.batch_size), 
                    desc=f"ICM Iter {iteration + 1}/{self.config.n_iterations}"
                ):
                    batch = tasks[j:j + self.config.batch_size]
                    results = await asyncio.gather(*batch)
                    changes += sum(results)
                
                logger.info(f"Iteration {iteration + 1}: {changes} label updates")
                if changes == 0:
                    logger.info("Converged!")
                    break
        
        return working_examples

    def create_random_labels(self, examples: List[Example]) -> List[Example]:
        """Create examples with random labels (baseline)."""
        random_examples = [copy.deepcopy(ex) for ex in examples]
        for ex in random_examples:
            ex.predicted_label = np.random.randint(0, 2)
        return random_examples

    def create_gold_labels(self, examples: List[Example]) -> List[Example]:
        """Create examples with gold (true) labels."""
        gold_examples = [copy.deepcopy(ex) for ex in examples]
        for ex in gold_examples:
            ex.predicted_label = ex.label
        return gold_examples

    async def evaluate(
        self, 
        train_labeled: List[Example], 
        test_examples: List[Example], 
        n_shots: int,
        use_chat_template: bool = True
    ) -> float:
        """Evaluate accuracy on test set using n-shot prompting."""
        prompts = []
        
        for test_ex in test_examples:
            context_str = ""
            
            if n_shots > 0 and train_labeled:
                n_available = min(len(train_labeled), n_shots)
                shot_indices = np.random.choice(len(train_labeled), size=n_available, replace=False)
                shots = [train_labeled[i] for i in shot_indices]
                
                for shot in shots:
                    context_str += shot.to_text(shot.predicted_label) + "\n\n"
            
            if use_chat_template:
                sys_msg = "You are simulating a specific persona. Answer only Yes or No."
                user_msg = f"{context_str}Question: {test_ex.question}\nDoes the persona agree? Answer Yes or No."
                full_prompt = self.apply_llama3_chat_template(sys_msg, user_msg)
            else:
                # Base model prompt (no chat template)
                full_prompt = f"{context_str}Question: {test_ex.question}\nDoes the persona agree?:"
            
            prompts.append(full_prompt)
        
        correct = 0
        connector = aiohttp.TCPConnector(limit=20)
        
        async with aiohttp.ClientSession(connector=connector) as session:
            for i in tqdm(
                range(0, len(prompts), self.config.batch_size), 
                desc=f"Eval {n_shots}-shot",
                leave=False
            ):
                batch_prompts = prompts[i:i + self.config.batch_size]
                tasks = [self.client.get_completion(session, p) for p in batch_prompts]
                responses = await asyncio.gather(*tasks)
                
                batch_examples = test_examples[i:i + len(responses)]
                
                for ex, resp in zip(batch_examples, responses):
                    resp_lower = resp.lower().strip()
                    pred = 1 if ('yes' in resp_lower or 'agree' in resp_lower) else 0
                    if pred == ex.label:
                        correct += 1
        
        return correct / len(test_examples) if test_examples else 0.0