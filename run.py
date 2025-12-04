# run.py
import asyncio
import os
import matplotlib.pyplot as plt
import pandas as pd
from icm_core import ICM, ICMConfig, Example
from data_loader import prepare_country_data
from dotenv import load_dotenv

load_dotenv()


async def run_country_eval(
    country: str, 
    api_key: str, 
    base_url: str, 
    model_name: str,
    n_samples: int = 120
):
    """Run ICM evaluation for a single country persona."""
    
    print(f"\n{'='*50}")
    print(f"Processing Persona: {country}")
    print(f"{'='*50}")
    
    # 1. Load Data
    print("\n--- Loading Data ---")
    data = prepare_country_data(country, n_samples=n_samples)
    
    if not data or len(data) < 20:
        print(f"ERROR: Insufficient data for {country} (got {len(data) if data else 0} examples)")
        return None
    
    # 2. Split Train/Test (70/30)
    split_idx = int(len(data) * 0.7)
    train_data = data[:split_idx]
    test_data = data[split_idx:]
    
    print(f"Train size: {len(train_data)}, Test size: {len(test_data)}")
    
    # Label distribution
    train_agree = sum(1 for ex in train_data if ex.label == 1)
    test_agree = sum(1 for ex in test_data if ex.label == 1)
    print(f"Train labels: {train_agree} agree / {len(train_data) - train_agree} disagree")
    print(f"Test labels: {test_agree} agree / {len(test_data) - test_agree} disagree")
    
    # 3. Initialize ICM
    config = ICMConfig(
        n_iterations=5,
        batch_size=10,
        n_shots_context=8
    )
    icm = ICM(api_key, base_url, model_name, config)
    
    # 4. Run ICM on Training Data (Unsupervised Label Search)
    print("\n--- Step 1: Running ICM Label Search (Unsupervised) ---")
    icm_labeled_train = await icm.search_labels(train_data)
    
    # Check ICM vs Gold agreement
    icm_matches = sum(
        1 for i, ex in enumerate(icm_labeled_train) 
        if ex.predicted_label == train_data[i].label
    )
    print(f"ICM label agreement with gold: {icm_matches}/{len(train_data)} "
          f"({100*icm_matches/len(train_data):.1f}%)")
    
    # 5. Prepare Gold Baseline (using true labels)
    gold_labeled_train = []
    for ex in train_data:
        new_ex = Example(
            question=ex.question, 
            choice=ex.choice, 
            label=ex.label, 
            predicted_label=ex.label,  # Use gold label
            id=ex.id
        )
        gold_labeled_train.append(new_ex)
    
    # 6. Evaluate at different shot counts
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
    
    # 7. Plot and save results
    plot_results(country, results)
    
    return {
        'country': country,
        'results': results,
        'train_size': len(train_data),
        'test_size': len(test_data),
        'icm_gold_agreement': icm_matches / len(train_data)
    }


def plot_results(country: str, results: dict):
    """Generate and save results plot."""
    
    df = pd.DataFrame(results)
    
    plt.figure(figsize=(10, 6))
    
    plt.plot(
        df['shots'], df['icm'], 
        marker='o', label='ICM Labeled (Unsupervised)', 
        color='#4ECDC4', linewidth=2, markersize=8
    )
    plt.plot(
        df['shots'], df['gold'], 
        marker='s', label='Gold Labeled (Oracle)', 
        color='#FF6B6B', linewidth=2, markersize=8
    )
    plt.axhline(
        y=results['zero'][0], 
        color='gray', linestyle='--', 
        label='Zero-Shot Baseline', linewidth=2
    )
    
    plt.xlabel('Number of Shots (k)', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title(f'Persona Elicitation: {country}\n(Llama-3.1-70B-Instruct)', fontsize=14)
    plt.xticks(results['shots'])
    plt.ylim(0, 1)
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    filename = f"ICM_Results_{country}.png"
    plt.savefig(filename, dpi=150)
    plt.close()
    
    print(f"\n[Graph Saved] {filename}")
    
    # Also save CSV
    csv_filename = f"ICM_Results_{country}.csv"
    df.to_csv(csv_filename, index=False)
    print(f"[CSV Saved] {csv_filename}")


def plot_combined_results(all_results: list):
    """Plot combined results for all countries."""
    
    plt.figure(figsize=(12, 8))
    
    colors = ['#4ECDC4', '#FF6B6B', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']
    
    for i, res in enumerate(all_results):
        if res is None:
            continue
        country = res['country']
        results = res['results']
        color = colors[i % len(colors)]
        
        plt.plot(
            results['shots'], results['icm'],
            marker='o', label=f'{country} (ICM)',
            color=color, linewidth=2, markersize=6
        )
    
    plt.xlabel('Number of Shots (k)', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('ICM Persona Elicitation Across Countries\n(Llama-3.1-70B-Instruct)', fontsize=14)
    plt.legend(loc='lower right', fontsize=9)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    plt.savefig('ICM_Results_Combined.png', dpi=150)
    plt.close()
    print("\n[Combined Graph Saved] ICM_Results_Combined.png")


async def test_api_connection(base_url: str, model_name: str, api_key: str = ""):
    """Test if vLLM API is accessible."""
    import aiohttp
    
    url = f"{base_url.rstrip('/')}/completions"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    
    payload = {
        "model": model_name,
        "prompt": "Hello",
        "max_tokens": 5,
        "temperature": 0.0
    }
    
    print(f"Testing API connection to {url}...")
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url, 
                headers=headers, 
                json=payload,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    print(f"✓ API connection successful!")
                    print(f"  Model: {result.get('model', 'unknown')}")
                    return True
                else:
                    text = await response.text()
                    print(f"✗ API returned status {response.status}")
                    print(f"  Response: {text[:200]}")
                    return False
    except Exception as e:
        print(f"✗ Connection failed: {e}")
        return False


async def main():
    """Main entry point."""
    
    # Load environment variables
    API_KEY = os.getenv("RUNPOD_API_KEY", "")
    BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
    MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Meta-Llama-3.1-70B-Instruct")
    
    print("="*60)
    print("ICM Persona Elicitation Experiment")
    print("="*60)
    print(f"BASE_URL: {BASE_URL}")
    print(f"MODEL_NAME: {MODEL_NAME}")
    print(f"API_KEY: {'[set]' if API_KEY else '[not set - OK for local]'}")
    
    # Test API connection first
    api_ok = await test_api_connection(BASE_URL, MODEL_NAME, API_KEY)
    
    if not api_ok:
        print("\n" + "="*60)
        print("ERROR: Cannot connect to vLLM API")
        print("="*60)
        print("Please check:")
        print("  1. Is vLLM server running? Check with: nvidia-smi")
        print("  2. Check server logs in RunPod")
        print("  3. Try: curl http://localhost:8000/v1/models")
        print("  4. Verify MODEL_NAME matches what vLLM loaded")
        return
    
    # =========================================================
    # CONFIGURE YOUR EXPERIMENT
    # =========================================================
    
    # Option 1: Use a SAMPLE of data (faster, for testing)
    # n_samples = 120  # Use 120 examples per country
    
    # Option 2: Use ALL available data (slower, for full experiment)
    n_samples = None  # None = use ALL available binary questions
    
    # Countries to process
    # Option A: Single country (for quick testing)
    # countries = ["US"]
    
    # Option B: Multiple countries (recommended for submission)
    countries = ["US", "Germany", "Britain", "France", "Japan", "India"]
    
    # Option C: More countries if you have time
    # countries = ["US", "Germany", "Britain", "France", "Japan", "India", 
    #              "Australia", "Canada", "Brazil", "Mexico", "Poland"]
    # =========================================================
    
    print(f"\nConfiguration:")
    print(f"  Countries: {countries}")
    print(f"  Samples per country: {'ALL' if n_samples is None else n_samples}")
    print("="*60)
    
    all_results = []
    
    for country in countries:
        try:
            result = await run_country_eval(
                country, 
                API_KEY, 
                BASE_URL, 
                MODEL_NAME,
                n_samples=n_samples  # Use the configured value (None = all data)
            )
            all_results.append(result)
        except Exception as e:
            print(f"\nERROR processing {country}: {e}")
            import traceback
            traceback.print_exc()
            all_results.append(None)
    
    # Generate combined plot if multiple countries
    if len([r for r in all_results if r is not None]) > 1:
        plot_combined_results(all_results)
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    for res in all_results:
        if res is None:
            continue
        country = res['country']
        results = res['results']
        
        # Get best ICM accuracy
        best_icm = max(results['icm'])
        best_gold = max(results['gold'])
        zero = results['zero'][0]
        
        print(f"{country:12} | Zero: {zero:.3f} | Best ICM: {best_icm:.3f} | "
              f"Best Gold: {best_gold:.3f} | ICM-Gold Agreement: {res['icm_gold_agreement']:.1%}")
    
    print("\nExperiment complete! Check the generated PNG and CSV files.")


if __name__ == "__main__":
    asyncio.run(main())