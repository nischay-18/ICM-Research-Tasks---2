# run.py
"""
ICM Persona Elicitation Experiment
Generates:
- Figure 1: Aggregated bar chart (all personas combined)
- Figure 2: Line chart comparing ICM vs Random vs Gold
- Per-country results
"""

import asyncio
import os
import shutil
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime
from icm_core import ICM, ICMConfig, Example
from data_loader import prepare_country_data, get_available_countries
from dotenv import load_dotenv

load_dotenv()

# Output directory
OUTPUT_DIR = "results"


def setup_output_dir():
    """Create output directory structure."""
    if os.path.exists(OUTPUT_DIR):
        # Backup old results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = f"{OUTPUT_DIR}_backup_{timestamp}"
        shutil.move(OUTPUT_DIR, backup_dir)
        print(f"Previous results backed up to: {backup_dir}")
    
    os.makedirs(f"{OUTPUT_DIR}/figures", exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/csv", exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/per_country", exist_ok=True)
    print(f"Output directory created: {OUTPUT_DIR}/")


async def run_country_eval(
    country: str, 
    icm: ICM,
    n_samples: int = None
) -> dict:
    """Run ICM evaluation for a single country persona."""
    
    print(f"\n{'='*50}")
    print(f"Processing Persona: {country}")
    print(f"{'='*50}")
    
    # Load Data
    data = prepare_country_data(country, n_samples=n_samples)
    
    if not data or len(data) < 20:
        print(f"WARNING: Insufficient data for {country} (got {len(data) if data else 0})")
        return None
    
    # Split Train/Test (70/30)
    split_idx = int(len(data) * 0.7)
    train_data = data[:split_idx]
    test_data = data[split_idx:]
    
    print(f"Data: {len(train_data)} train, {len(test_data)} test")
    
    # Create label sets
    print("\n--- Running ICM Label Search ---")
    icm_labeled = await icm.search_labels(train_data)
    gold_labeled = icm.create_gold_labels(train_data)
    random_labeled = icm.create_random_labels(train_data)
    
    # Calculate ICM-Gold agreement
    icm_matches = sum(1 for i, ex in enumerate(icm_labeled) if ex.predicted_label == train_data[i].label)
    print(f"ICM-Gold agreement: {icm_matches}/{len(train_data)} ({100*icm_matches/len(train_data):.1f}%)")
    
    # Evaluate all conditions
    results = {
        'country': country,
        'train_size': len(train_data),
        'test_size': len(test_data),
        'icm_gold_agreement': icm_matches / len(train_data),
        'shots': [],
        'zero_shot_base': 0,
        'zero_shot_chat': 0,
        'icm': [],
        'gold': [],
        'random': []
    }
    
    # Zero-shot evaluations
    print("\n--- Evaluating Zero-shot (Base) ---")
    results['zero_shot_base'] = await icm.evaluate([], test_data, n_shots=0, use_chat_template=False)
    print(f"Zero-shot (Base): {results['zero_shot_base']:.4f}")
    
    print("--- Evaluating Zero-shot (Chat) ---")
    results['zero_shot_chat'] = await icm.evaluate([], test_data, n_shots=0, use_chat_template=True)
    print(f"Zero-shot (Chat): {results['zero_shot_chat']:.4f}")
    
    # Few-shot evaluations
    shot_counts = [2, 4, 8, 16]
    
    for k in shot_counts:
        print(f"\n--- Evaluating {k}-shot ---")
        
        icm_acc = await icm.evaluate(icm_labeled, test_data, n_shots=k)
        gold_acc = await icm.evaluate(gold_labeled, test_data, n_shots=k)
        random_acc = await icm.evaluate(random_labeled, test_data, n_shots=k)
        
        results['shots'].append(k)
        results['icm'].append(icm_acc)
        results['gold'].append(gold_acc)
        results['random'].append(random_acc)
        
        print(f"ICM: {icm_acc:.4f} | Gold: {gold_acc:.4f} | Random: {random_acc:.4f}")
    
    # Save per-country results
    save_country_results(country, results)
    
    return results


def save_country_results(country: str, results: dict):
    """Save individual country results."""
    # CSV
    df = pd.DataFrame({
        'shots': results['shots'],
        'icm': results['icm'],
        'gold': results['gold'],
        'random': results['random']
    })
    df.to_csv(f"{OUTPUT_DIR}/per_country/{country}_results.csv", index=False)
    
    # Figure - Bar chart for this country
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(results['shots']))
    width = 0.25
    
    bars1 = ax.bar(x - width, results['random'], width, label='Random Labels', color='#FF9800')
    bars2 = ax.bar(x, results['icm'], width, label='ICM (Unsupervised)', color='#00BCD4')
    bars3 = ax.bar(x + width, results['gold'], width, label='Gold Labels', color='#8BC34A')
    
    # Add zero-shot line
    ax.axhline(y=results['zero_shot_chat'], color='gray', linestyle='--', 
               label=f"Zero-shot Chat ({results['zero_shot_chat']:.2f})", linewidth=1.5)
    
    ax.set_xlabel('Number of Few-shot Examples', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title(f'Persona Elicitation: {country}\n(Llama-3.1-70B-Instruct)', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{k}-shot' for k in results['shots']])
    ax.set_ylim(0, 1.1)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/per_country/{country}_bar_chart.png", dpi=150)
    plt.close()
    
    print(f"Saved: {OUTPUT_DIR}/per_country/{country}_*.csv/png")


def generate_figure1(all_results: list):
    """
    Generate Figure 1: Aggregated bar chart across all personas.
    Shows: Zero-shot (Base), Zero-shot (Chat), ICM, Gold for each shot count.
    """
    valid_results = [r for r in all_results if r is not None]
    if not valid_results:
        print("No valid results for Figure 1")
        return
    
    shot_counts = valid_results[0]['shots']
    
    # Aggregate across countries
    agg = {
        'zero_base': np.mean([r['zero_shot_base'] for r in valid_results]),
        'zero_chat': np.mean([r['zero_shot_chat'] for r in valid_results]),
        'icm': [np.mean([r['icm'][i] for r in valid_results]) for i in range(len(shot_counts))],
        'gold': [np.mean([r['gold'][i] for r in valid_results]) for i in range(len(shot_counts))],
        'random': [np.mean([r['random'][i] for r in valid_results]) for i in range(len(shot_counts))]
    }
    
    # Create bar chart
    fig, ax = plt.subplots(figsize=(12, 7))
    
    x = np.arange(len(shot_counts))
    width = 0.18
    
    # Bars for each condition
    bars1 = ax.bar(x - 1.5*width, [agg['zero_base']]*len(shot_counts), width, 
                   label='Zero-shot (Base)', color='#8B4513', hatch='//')
    bars2 = ax.bar(x - 0.5*width, [agg['zero_chat']]*len(shot_counts), width, 
                   label='Zero-shot (Chat)', color='#D2691E', hatch='//')
    bars3 = ax.bar(x + 0.5*width, agg['icm'], width, 
                   label='ICM (Unsupervised)', color='#00BCD4')
    bars4 = ax.bar(x + 1.5*width, agg['gold'], width, 
                   label='Gold Labels', color='#8BC34A')
    
    ax.set_xlabel('Number of Few-shot Examples', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('GlobalOpinionQA Performance: ICM vs Baselines\n(Aggregated across all personas, Llama-3.1-70B-Instruct)', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{k}-shot' for k in shot_counts])
    ax.set_ylim(0, 1.05)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/figures/Figure1_aggregated_bar.png", dpi=150)
    plt.close()
    
    print(f"\nSaved: {OUTPUT_DIR}/figures/Figure1_aggregated_bar.png")
    
    # Save aggregated data
    agg_df = pd.DataFrame({
        'shots': shot_counts,
        'zero_shot_base': [agg['zero_base']]*len(shot_counts),
        'zero_shot_chat': [agg['zero_chat']]*len(shot_counts),
        'icm': agg['icm'],
        'gold': agg['gold'],
        'random': agg['random']
    })
    agg_df.to_csv(f"{OUTPUT_DIR}/csv/aggregated_results.csv", index=False)


def generate_figure2(all_results: list):
    """
    Generate Figure 2: BAR CHART showing accuracy vs number of examples.
    Compares ICM, Random, and Gold labels.
    """
    valid_results = [r for r in all_results if r is not None]
    if not valid_results:
        print("No valid results for Figure 2")
        return
    
    shot_counts = valid_results[0]['shots']
    
    # Aggregate across countries
    agg_icm = [np.mean([r['icm'][i] for r in valid_results]) for i in range(len(shot_counts))]
    agg_gold = [np.mean([r['gold'][i] for r in valid_results]) for i in range(len(shot_counts))]
    agg_random = [np.mean([r['random'][i] for r in valid_results]) for i in range(len(shot_counts))]
    
    # Create grouped bar chart
    fig, ax = plt.subplots(figsize=(12, 7))
    
    x = np.arange(len(shot_counts))
    width = 0.25  # Width of bars
    
    # Create bars - Random, ICM, Gold (left to right)
    bars1 = ax.bar(x - width, agg_random, width, label='Random Labels', color='#FF9800')
    bars2 = ax.bar(x, agg_icm, width, label='ICM (Unsupervised)', color='#00BCD4')
    bars3 = ax.bar(x + width, agg_gold, width, label='Gold Labels', color='#8BC34A')
    
    # Add value labels on top of bars
    def add_bar_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=9)
    
    add_bar_labels(bars1)
    add_bar_labels(bars2)
    add_bar_labels(bars3)
    
    ax.set_xlabel('Number of Few-shot Examples', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Figure 2: Test Accuracy vs Number of In-Context Examples\n(Comparing ICM, Random, and Gold Labels - Llama-3.1-70B-Instruct)', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{k}-shot' for k in shot_counts])
    ax.set_ylim(0, 1.15)
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(True, axis='y', alpha=0.3)
    ax.set_facecolor('#f8f8f8')
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/figures/Figure2_bar_chart.png", dpi=150)
    plt.close()
    
    print(f"Saved: {OUTPUT_DIR}/figures/Figure2_bar_chart.png")


def generate_combined_country_plot(all_results: list):
    """Generate a combined plot showing all countries."""
    valid_results = [r for r in all_results if r is not None]
    if not valid_results:
        return
    
    fig, ax = plt.subplots(figsize=(12, 7))
    colors = plt.cm.tab10(np.linspace(0, 1, len(valid_results)))
    
    for i, r in enumerate(valid_results):
        ax.plot(r['shots'], r['icm'], 'o-', label=f"{r['country']} (ICM)", 
                color=colors[i], linewidth=2, markersize=6)
    
    ax.set_xlabel('Number of In-Context Examples', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('ICM Performance Across Countries\n(Llama-3.1-70B-Instruct)', fontsize=14)
    ax.set_xticks(valid_results[0]['shots'])
    ax.set_ylim(0, 1.05)
    ax.legend(loc='lower right', fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/figures/all_countries_comparison.png", dpi=150)
    plt.close()


def print_summary(all_results: list):
    """Print and save summary table."""
    valid_results = [r for r in all_results if r is not None]
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"{'Country':<20} {'Data':<10} {'Zero-Base':<10} {'Zero-Chat':<10} {'Best ICM':<10} {'Best Gold':<10} {'ICM-Gold%':<10}")
    print("-"*80)
    
    rows = []
    for r in valid_results:
        row = {
            'Country': r['country'],
            'Train': r['train_size'],
            'Test': r['test_size'],
            'Zero_Base': r['zero_shot_base'],
            'Zero_Chat': r['zero_shot_chat'],
            'Best_ICM': max(r['icm']),
            'Best_Gold': max(r['gold']),
            'Best_Random': max(r['random']),
            'ICM_Gold_Agreement': r['icm_gold_agreement']
        }
        rows.append(row)
        print(f"{r['country']:<20} {r['train_size']+r['test_size']:<10} {r['zero_shot_base']:<10.3f} {r['zero_shot_chat']:<10.3f} {max(r['icm']):<10.3f} {max(r['gold']):<10.3f} {r['icm_gold_agreement']*100:<10.1f}%")
    
    # Save summary
    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(f"{OUTPUT_DIR}/csv/summary.csv", index=False)
    print(f"\nSaved: {OUTPUT_DIR}/csv/summary.csv")


async def test_api_connection(base_url: str, model_name: str, api_key: str = ""):
    """Test if vLLM API is accessible."""
    import aiohttp
    
    url = f"{base_url.rstrip('/')}/completions"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    
    payload = {"model": model_name, "prompt": "Hello", "max_tokens": 5, "temperature": 0.0}
    
    print(f"Testing API: {url}")
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload, 
                                   timeout=aiohttp.ClientTimeout(total=30)) as response:
                if response.status == 200:
                    print("✓ API connection successful!")
                    return True
                else:
                    text = await response.text()
                    print(f"✗ API error {response.status}: {text[:100]}")
                    return False
    except Exception as e:
        print(f"✗ Connection failed: {e}")
        return False


async def main():
    """Main entry point."""
    
    API_KEY = os.getenv("RUNPOD_API_KEY", "")
    BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
    MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Meta-Llama-3.1-70B-Instruct")
    
    print("="*60)
    print("ICM Persona Elicitation Experiment")
    print("="*60)
    print(f"Model: {MODEL_NAME}")
    print(f"API: {BASE_URL}")
    
    # Test API
    if not await test_api_connection(BASE_URL, MODEL_NAME, API_KEY):
        print("\nERROR: Cannot connect to vLLM API")
        return
    
    # Setup output directory
    setup_output_dir()
    
    # =========================================================
    # CONFIGURATION
    # =========================================================
    
    # Countries to process (using EXACT dataset names)
    # Run get_available_countries() to see all options
    countries = [
    "Kenya",
    "Ethiopia", 
    "Zimbabwe",
    "Russia",
    "Germany",
    "Pakistan",
    "Turkey",
    "United States",
    "Lebanon",
    "Nigeria",
    ]
    
    # Use all available data (None) or limit samples
    n_samples = None  # None = all data
    
    print(f"\nCountries: {countries}")
    print(f"Samples: {'ALL' if n_samples is None else n_samples}")
    print("="*60)
    
    # Initialize ICM
    config = ICMConfig(n_iterations=5, batch_size=10, n_shots_context=8)
    icm = ICM(API_KEY, BASE_URL, MODEL_NAME, config)
    
    # Run evaluations
    all_results = []
    for country in countries:
        try:
            result = await run_country_eval(country, icm, n_samples)
            all_results.append(result)
        except Exception as e:
            print(f"\nERROR with {country}: {e}")
            import traceback
            traceback.print_exc()
            all_results.append(None)
    
    # Generate figures
    print("\n" + "="*60)
    print("Generating Figures...")
    print("="*60)
    
    generate_figure1(all_results)
    generate_figure2(all_results)
    generate_combined_country_plot(all_results)
    
    # Print summary
    print_summary(all_results)
    
    print("\n" + "="*60)
    print(f"All results saved to: {OUTPUT_DIR}/")
    print("="*60)


if __name__ == "__main__":
    asyncio.run(main())