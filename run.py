import asyncio
import os
import matplotlib.pyplot as plt
import pandas as pd
from icm_core import ICM, ICMConfig, Example
from data_loader import prepare_country_data
from dotenv import load_dotenv

load_dotenv()

async def run_country_eval(country: str, api_key: str, base_url: str, model_name: str):
    print(f"\n=======================================")
    print(f"--- Processing Persona: {country} ---")
    print(f"=======================================")
    
    # 1. Load Data
    data = prepare_country_data(country, n_samples=120) 
    if not data or len(data) < 20:
        print(f"Insufficient data found for {country}")
        return None
        
    # 2. Split Train/Test
    split_idx = int(len(data) * 0.7)
    train_data = data[:split_idx]
    test_data = data[split_idx:]
    print(f"Train Size: {len(train_data)}, Test Size: {len(test_data)}")
    
    config = ICMConfig()
    # Updated to pass URL and Model Name instead of Endpoint IDs
    icm = ICM(api_key, base_url, model_name, config)
    
    # 3. Run ICM on Train (Unsupervised)
    print("\n--- Step 1: Running ICM Label Search ---")
    icm_labeled_train = await icm.search_labels(train_data)
    
    # 4. Prepare Gold Baseline
    gold_labeled_train = []
    for ex in train_data:
        new_ex = Example(question=ex.question, choice=ex.choice, label=ex.label, predicted_label=ex.label, id=ex.id)
        gold_labeled_train.append(new_ex)
        
    # 5. Evaluate
    results = {'shots': [], 'icm': [], 'gold': [], 'zero': []}
    
    print("\n--- Step 2: Evaluating Zero-shot ---")
    zero_acc = await icm.evaluate([], test_data, n_shots=0)
    print(f"Zero-shot Accuracy: {zero_acc:.4f}")
    
    shot_counts = [2, 4, 8, 16]
    
    for k in shot_counts:
        print(f"\n--- Evaluating {k}-shot ---")
        icm_acc = await icm.evaluate(icm_labeled_train, test_data, n_shots=k)
        gold_acc = await icm.evaluate(gold_labeled_train, test_data, n_shots=k)
        
        results['shots'].append(k)
        results['icm'].append(icm_acc)
        results['gold'].append(gold_acc)
        results['zero'].append(zero_acc)
        print(f"ICM: {icm_acc:.4f} | Gold: {gold_acc:.4f}")
        
    plot_results(country, results)
    return results

def plot_results(country, results):
    df = pd.DataFrame(results)
    plt.figure(figsize=(10, 6))
    plt.plot(df['shots'], df['icm'], marker='o', label='ICM Labeled', color='#4ECDC4', linewidth=2)
    plt.plot(df['shots'], df['gold'], marker='o', label='Gold Labeled', color='#FF6B6B', linewidth=2)
    plt.axhline(y=results['zero'][0], color='gray', linestyle='--', label='Zero-Shot')
    plt.xlabel('Number of Shots (k)')
    plt.ylabel('Accuracy')
    plt.title(f'Persona Elicitation Accuracy: {country} (Llama-3.1-70B)')
    plt.xticks(results['shots'])
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    
    filename = f"ICM_Results_{country}.png"
    plt.savefig(filename)
    print(f"\n[Graph Saved] {filename}")

if __name__ == "__main__":
    API_KEY = os.getenv("RUNPOD_API_KEY")
    BASE_URL = os.getenv("VLLM_BASE_URL")
    MODEL_NAME = os.getenv("MODEL_NAME")

    if not API_KEY or not BASE_URL or not MODEL_NAME:
        print("ERROR: Please set .env variables: RUNPOD_API_KEY, VLLM_BASE_URL, MODEL_NAME")
    else:
        # Run for one country (US)
        asyncio.run(run_country_eval("US", API_KEY, BASE_URL, MODEL_NAME))