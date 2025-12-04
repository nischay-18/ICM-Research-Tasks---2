# icm_core.py
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass
import asyncio
from tqdm import tqdm
import logging
import aiohttp

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
        # vLLM serves OpenAI-compatible endpoints
        self.base_url = base_url.rstrip('/')
        self.url = f"{self.base_url}/completions"
        self.headers = {
            "Content-Type": "application/json"
        }
        # Only add auth header if API key is provided and not empty
        if api_key and api_key.strip():
            self.headers["Authorization"] = f"Bearer {api_key}"
        
        self.model_name = model_name
        logger.info(f"VLLMClient initialized: {self.base_url}, model: {model_name}")
    
    async def get_label_logprobs(
        self, 
        session: aiohttp.ClientSession, 
        context_prompt: str
    ) -> Tuple[float, float]:
        """
        Gets log probabilities for Yes/No tokens.
        Returns (yes_logprob, no_logprob).
        """
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
                self.url, 
                headers=self.headers, 
                json=payload,
                timeout=aiohttp.ClientTimeout(total=60)
            ) as response:
                if response.status != 200:
                    text = await response.text()
                    logger.error(f"API Error {response.status}: {text[:200]}")
                    return -10.0, -10.0
                
                result = await response.json()
                
                # Navigate OpenAI-compatible response
                choices = result.get('choices', [])
                if not choices:
                    return -10.0, -10.0
                
                logprobs_data = choices[0].get('logprobs', {})
                top_logprobs_list = logprobs_data.get('top_logprobs', [])
                
                if not top_logprobs_list:
                    return -10.0, -10.0
                
                # Get first token's logprobs
                top_dict = top_logprobs_list[0]
                
                yes_score = -999.0
                no_score = -999.0
                
                for token, logprob in top_dict.items():
                    t = token.strip().lower()
                    if t in ['yes', 'true', 'agree']:
                        yes_score = max(yes_score, logprob)
                    if t in ['no', 'false', 'disagree']:
                        no_score = max(no_score, logprob)
                
                return yes_score, no_score
                
        except asyncio.TimeoutError:
            logger.error("Request timed out")
            return -10.0, -10.0
        except Exception as e:
            logger.error(f"Logprob Exception: {e}")
            return -10.0, -10.0

    async def get_completion(
        self, 
        session: aiohttp.ClientSession, 
        prompt: str
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
                self.url, 
                headers=self.headers, 
                json=payload,
                timeout=aiohttp.ClientTimeout(total=60)
            ) as response:
                if response.status != 200:
                    return ""
                result = await response.json()
                
                choices = result.get('choices', [])
                if not choices:
                    return ""
                
                return choices[0].get('text', "")
                
        except Exception as e:
            logger.debug(f"Completion error: {e}")
            return ""


class ICM:
    """
    In-Context Matching for persona elicitation.
    Implements the ICM algorithm from the paper.
    """
    
    def __init__(
        self, 
        api_key: str, 
        base_url: str, 
        model_name: str, 
        config: ICMConfig
    ):
        self.config = config
        self.client = VLLMClient(api_key, base_url, model_name)

    def create_icm_prompt(
        self, 
        target_ex: Example, 
        context_examples: List[Example]
    ) -> str:
        """Creates the ICM prompt with context examples."""
        prompt = "The following are questions and whether a specific persona agrees with them.\n\n"
        
        for ex in context_examples:
            prompt += ex.to_text(ex.predicted_label) + "\n\n"
        
        prompt += f"Question: {target_ex.question}\nDoes the persona agree?:"
        return prompt
    
    def apply_llama3_chat_template(self, system_msg: str, user_msg: str) -> str:
        """Applies Llama-3 chat template format."""
        return (
            f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            f"{system_msg}<|eot_id|>"
            f"<|start_header_id|>user<|end_header_id|>\n\n"
            f"{user_msg}<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n"
        )

    async def update_single_example(
        self, 
        session: aiohttp.ClientSession, 
        target_ex: Example, 
        all_examples: List[Example]
    ) -> bool:
        """
        Update a single example's predicted label based on context.
        Returns True if label changed.
        """
        # Get context examples (excluding target)
        possible_context = [ex for ex in all_examples if ex.id != target_ex.id]
        if not possible_context:
            return False
        
        # Sample random context
        n_context = min(len(possible_context), self.config.n_shots_context)
        context_indices = np.random.choice(
            len(possible_context), 
            size=n_context, 
            replace=False
        )
        context = [possible_context[i] for i in context_indices]
        
        # Create prompt and get logprobs
        prompt = self.create_icm_prompt(target_ex, context)
        yes_score, no_score = await self.client.get_label_logprobs(session, prompt)
        
        if yes_score == -10.0 and no_score == -10.0:
            return False
        
        # Update label based on logprobs
        new_label = 1 if yes_score > no_score else 0
        if new_label != target_ex.predicted_label:
            target_ex.predicted_label = new_label
            return True
        return False

    async def search_labels(self, examples: List[Example]) -> List[Example]:
        """
        Run ICM label search - the core unsupervised algorithm.
        Iteratively updates predicted labels until convergence.
        """
        working_examples = examples[:]
        
        # Initialize with random labels
        for ex in working_examples:
            ex.predicted_label = np.random.randint(0, 2)
        
        connector = aiohttp.TCPConnector(limit=20)
        async with aiohttp.ClientSession(connector=connector) as session:
            for iteration in range(self.config.n_iterations):
                # Create tasks for all examples
                tasks = [
                    self.update_single_example(session, ex, working_examples) 
                    for ex in working_examples
                ]
                
                changes = 0
                # Process in batches
                for j in tqdm(
                    range(0, len(tasks), self.config.batch_size), 
                    desc=f"ICM Iteration {iteration + 1}/{self.config.n_iterations}"
                ):
                    batch = tasks[j:j + self.config.batch_size]
                    results = await asyncio.gather(*batch)
                    changes += sum(results)
                
                logger.info(f"Iteration {iteration + 1}: {changes} label updates")
                
                # Early stopping if converged
                if changes == 0:
                    logger.info("Converged - no more label changes")
                    break
        
        return working_examples

    async def evaluate(
        self, 
        train_labeled: List[Example], 
        test_examples: List[Example], 
        n_shots: int
    ) -> float:
        """
        Evaluate accuracy on test set using n-shot prompting.
        """
        prompts = []
        
        for test_ex in test_examples:
            context_str = ""
            
            if n_shots > 0 and train_labeled:
                # Sample shots from training data
                n_available = min(len(train_labeled), n_shots)
                shot_indices = np.random.choice(
                    len(train_labeled), 
                    size=n_available, 
                    replace=False
                )
                shots = [train_labeled[i] for i in shot_indices]
                
                for shot in shots:
                    context_str += shot.to_text(shot.predicted_label) + "\n\n"
            
            sys_msg = (
                "You are a helpful assistant simulating a specific persona. "
                "You must answer only with Yes or No."
            )
            user_msg = (
                f"{context_str}"
                f"Question: {test_ex.question}\n"
                f"Does the persona agree? Answer only Yes or No."
            )
            
            full_prompt = self.apply_llama3_chat_template(sys_msg, user_msg)
            prompts.append(full_prompt)
        
        correct = 0
        connector = aiohttp.TCPConnector(limit=20)
        
        async with aiohttp.ClientSession(connector=connector) as session:
            for i in tqdm(
                range(0, len(prompts), self.config.batch_size), 
                desc=f"Evaluating {n_shots}-shot"
            ):
                batch_prompts = prompts[i:i + self.config.batch_size]
                tasks = [
                    self.client.get_completion(session, p) 
                    for p in batch_prompts
                ]
                responses = await asyncio.gather(*tasks)
                
                batch_examples = test_examples[i:i + len(responses)]
                
                for ex, resp in zip(batch_examples, responses):
                    resp_lower = resp.lower().strip()
                    
                    # Determine predicted label from response
                    if 'yes' in resp_lower or 'agree' in resp_lower:
                        pred = 1
                    else:
                        pred = 0
                    
                    if pred == ex.label:
                        correct += 1
        
        accuracy = correct / len(test_examples) if test_examples else 0.0
        return accuracy