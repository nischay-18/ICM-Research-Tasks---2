import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import asyncio
from tqdm import tqdm
import logging
import aiohttp
import json

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
    def __init__(self, api_key: str, base_url: str, model_name: str):
        # We target the standard OpenAI-compatible completion endpoint provided by vLLM
        self.url = f"{base_url.rstrip('/')}/completions"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        self.model_name = model_name
        
    async def get_label_logprobs(self, session: aiohttp.ClientSession, context_prompt: str) -> Tuple[float, float]:
        """
        Gets log probabilities for Yes/No using the vLLM standard API.
        """
        payload = {
            "model": self.model_name,
            "prompt": context_prompt,
            "max_tokens": 1,
            "temperature": 0.0,
            "logprobs": 5, # OpenAI API format for top 5 logprobs
            "stop": ["\n", "."]
        }
        
        try:
            async with session.post(self.url, headers=self.headers, json=payload) as response:
                if response.status != 200:
                    text = await response.text()
                    logger.error(f"API Error {response.status}: {text}")
                    return -10.0, -10.0
                
                result = await response.json()
                
                # Navigate OpenAI-compatible response structure
                choices = result.get('choices', [])
                if not choices: return -10.0, -10.0
                
                # In vLLM/OpenAI API, logprobs are inside 'logprobs' -> 'top_logprobs' list
                top_logprobs_list = choices[0].get('logprobs', {}).get('top_logprobs', [])
                if not top_logprobs_list: return -10.0, -10.0

                # Get the first token's dict
                top_dict = top_logprobs_list[0]

                true_score = -999.0
                false_score = -999.0
                
                for token, logprob in top_dict.items():
                    t = token.strip().lower()
                    if t in ['yes', 'true', 'agree']: 
                        true_score = max(true_score, logprob)
                    if t in ['no', 'false', 'disagree']: 
                        false_score = max(false_score, logprob)
                    
                return true_score, false_score
            
        except Exception as e:
            logger.error(f"Logprob Exception: {e}")
            return -10.0, -10.0

    async def get_completion(self, session: aiohttp.ClientSession, prompt: str) -> str:
        """
        Gets text generation using the vLLM standard API.
        """
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "temperature": 0.1,
            "max_tokens": 10,
            "stop": ["<|eot_id|>", "\n\n"]
        }
        
        try:
            async with session.post(self.url, headers=self.headers, json=payload) as response:
                if response.status != 200: return ""
                result = await response.json()
                
                choices = result.get('choices', [])
                if not choices: return ""
                
                return choices[0].get('text', "")
        except Exception:
            return ""

class ICM:
    def __init__(self, api_key: str, base_url: str, model_name: str, config: ICMConfig):
        self.config = config
        # Single client for the Pod
        self.client = VLLMClient(api_key, base_url, model_name)

    def create_icm_prompt(self, target_ex: Example, context_examples: List[Example]) -> str:
        prompt = "The following are questions and whether a specific persona agrees with them.\n\n"
        for ex in context_examples:
            prompt += ex.to_text(ex.predicted_label) + "\n\n"
        prompt += f"Question: {target_ex.question}\nDoes the persona agree?:"
        return prompt
    
    def apply_llama3_chat_template(self, system_msg: str, user_msg: str) -> str:
        return (
            f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_msg}<|eot_id|>"
            f"<|start_header_id|>user<|end_header_id|>\n\n{user_msg}<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n"
        )

    async def update_single_example(self, session: aiohttp.ClientSession, target_ex: Example, all_examples: List[Example]) -> bool:
        possible_context = [ex for ex in all_examples if ex.id != target_ex.id]
        if not possible_context: return False
        
        context = np.random.choice(possible_context, size=min(len(possible_context), self.config.n_shots_context), replace=False)
        prompt = self.create_icm_prompt(target_ex, context)
        
        yes_score, no_score = await self.client.get_label_logprobs(session, prompt)
        
        if yes_score == -10.0 and no_score == -10.0: return False
        
        new_label = 1 if yes_score > no_score else 0
        if new_label != target_ex.predicted_label:
            target_ex.predicted_label = new_label
            return True
        return False

    async def search_labels(self, examples: List[Example]) -> List[Example]:
        working_examples = examples[:]
        for ex in working_examples:
            ex.predicted_label = np.random.randint(0, 2)
            
        async with aiohttp.ClientSession() as session:
            for i in range(self.config.n_iterations):
                tasks = [self.update_single_example(session, ex, working_examples) for ex in working_examples]
                
                changes = 0
                for j in tqdm(range(0, len(tasks), self.config.batch_size), desc=f"ICM Iteration {i+1}"):
                    batch = tasks[j:j+self.config.batch_size]
                    results = await asyncio.gather(*batch)
                    changes += sum(results)
                    
                logger.info(f"Iteration {i+1} finished: {changes} label updates")
                if changes == 0: break
                
        return working_examples

    async def evaluate(self, train_labeled: List[Example], test_examples: List[Example], n_shots: int) -> float:
        prompts = []
        for test_ex in test_examples:
            context_str = ""
            if n_shots > 0 and train_labeled:
                shots = np.random.choice(train_labeled, size=min(len(train_labeled), n_shots), replace=False)
                for shot in shots:
                    context_str += shot.to_text(shot.predicted_label) + "\n\n"
            
            sys_msg = "You are a helpful assistant simulating a specific persona. You must answer only with Yes or No."
            user_msg = f"{context_str}Question: {test_ex.question}\nDoes the persona agree? Answer only Yes or No."
            
            full_prompt = self.apply_llama3_chat_template(sys_msg, user_msg)
            prompts.append(full_prompt)
        
        correct = 0
        async with aiohttp.ClientSession() as session:
            for i in tqdm(range(0, len(prompts), self.config.batch_size), desc=f"Evaluating {n_shots}-shot"):
                batch_prompts = prompts[i:i+self.config.batch_size]
                
                tasks = [self.client.get_completion(session, p) for p in batch_prompts]
                responses = await asyncio.gather(*tasks)
                
                batch_ex = test_examples[i:i+len(responses)]
                for ex, resp in zip(batch_ex, responses):
                    resp = resp.lower().strip()
                    pred = 1 if ('yes' in resp or 'true' in resp or 'agree' in resp) else 0
                    if pred == ex.label: correct += 1
            
        return correct / len(test_examples) if test_examples else 0