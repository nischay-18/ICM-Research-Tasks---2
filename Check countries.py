
"""
Script to check which countries are available in the GlobalOpinionQA dataset
and how many binary (Agree/Disagree) questions each has.
"""

from data_loader import get_available_countries

if __name__ == "__main__":
    print("="*60)
    print("GlobalOpinionQA Dataset - Available Countries")
    print("="*60)
    print("\nScanning dataset for binary (Agree/Disagree) questions...")
    print("(This may take a moment)\n")
    
    country_counts = get_available_countries()
    
    # Filter to show only countries with sufficient data (>20 questions)
    sufficient = {k: v for k, v in country_counts.items() if v >= 20}
    insufficient = {k: v for k, v in country_counts.items() if v < 20}
    
    print("="*60)
    print("Countries with SUFFICIENT data (≥20 binary questions):")
    print("="*60)
    print(f"{'Country':<45} {'Questions':>10}")
    print("-"*60)
    
    for country, count in sufficient.items():
        print(f"  {country:<43} {count:>10}")
    
    print(f"\nTotal: {len(sufficient)} countries with sufficient data")
    
    print("\n" + "="*60)
    print("Countries with INSUFFICIENT data (<20 binary questions):")
    print("="*60)
    
    for country, count in insufficient.items():
        print(f"  {country:<43} {count:>10}")
    
    print(f"\nTotal: {len(insufficient)} countries with insufficient data")
    
    print("\n" + "="*60)
    print("RECOMMENDED countries for experiment:")
    print("="*60)
    
    # Get top 10 by data availability
    top_10 = list(sufficient.items())[:10]
    recommended = [c for c, _ in top_10]
    
    print(f"\nTop 10 by data availability:")
    for c, n in top_10:
        print(f"  - {c} ({n} questions)")
    
    print(f"\nCopy this list for run.py:")
    print(f"countries = {recommended}")
