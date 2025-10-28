#!/usr/bin/env python3
"""
Comparative Analysis: 2024 vs 2025 Reddit Visa Discourse
Generates comprehensive comparison between the two years
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import numpy as np

def load_analysis_data():
    """Load both 2024 and 2025 analysis results"""
    
    # Load 2024 analysis
    with open('2024_analysis_results.json', 'r') as f:
        analysis_2024 = json.load(f)
    
    # Load 2025 analysis
    with open('2025_analysis/comprehensive_analysis_2025.json', 'r') as f:
        analysis_2025 = json.load(f)
    
    return analysis_2024, analysis_2025

def create_comparative_analysis():
    """Create comprehensive comparative analysis"""
    
    print(" Loading analysis data...")
    analysis_2024, analysis_2025 = load_analysis_data()
    
    print(" Creating comparative analysis...")
    
    # Extract key metrics
    comparison = {
        "analysis_date": datetime.now().isoformat(),
        "years_compared": [2024, 2025],
        "summary_comparison": {
            "2024": {
                "total_records": analysis_2024["summary_stats"]["total_records"],
                "total_fear": analysis_2024["summary_stats"]["total_fear"],
                "total_qa": analysis_2024["summary_stats"]["total_qa"],
                "fear_rate": analysis_2024["summary_stats"]["fear_rate"],
                "qa_rate": analysis_2024["summary_stats"]["qa_rate"],
                "date_range": analysis_2024["date_range"],
                "total_days": analysis_2024["total_days"]
            },
            "2025": {
                "total_records": analysis_2025["summary_stats"]["total_records"],
                "total_fear": int(analysis_2025["summary_stats"]["total_fear"]),
                "total_qa": int(analysis_2025["summary_stats"]["total_qa"]),
                "fear_rate": analysis_2025["summary_stats"]["fear_rate"],
                "qa_rate": analysis_2025["summary_stats"]["qa_rate"],
                "date_range": analysis_2025["date_range"],
                "total_days": analysis_2025["total_days"]
            }
        },
        "stage_comparison": {
            "2024": {k: v["total"] for k, v in analysis_2024["stage_analysis"].items()},
            "2025": {k: int(v) for k, v in analysis_2025["stage_analysis"].items()}
        }
    }
    
    # Calculate changes
    changes = {}
    for metric in ["total_records", "total_fear", "total_qa"]:
        val_2024 = int(comparison["summary_comparison"]["2024"][metric])
        val_2025 = int(comparison["summary_comparison"]["2025"][metric])
        
        if val_2024 > 0:
            changes[f"{metric}_change"] = ((val_2025 - val_2024) / val_2024) * 100
        else:
            changes[f"{metric}_change"] = 0
    
    for metric in ["fear_rate", "qa_rate"]:
        val_2024 = comparison["summary_comparison"]["2024"][metric]
        val_2025 = comparison["summary_comparison"]["2025"][metric]
        changes[f"{metric}_change"] = ((val_2025 - val_2024) / val_2024) * 100
    
    comparison["changes"] = changes
    
    # Stage changes
    stage_changes = {}
    for stage in comparison["stage_comparison"]["2024"].keys():
        val_2024 = int(comparison["stage_comparison"]["2024"][stage])
        val_2025 = int(comparison["stage_comparison"]["2025"][stage])
        
        if val_2024 > 0:
            stage_changes[f"{stage}_change"] = ((val_2025 - val_2024) / val_2024) * 100
        else:
            stage_changes[f"{stage}_change"] = 0
    
    comparison["stage_changes"] = stage_changes
    
    return comparison

def create_comparative_visualizations(comparison):
    """Create comprehensive comparative visualizations"""
    
    print(" Creating comparative visualizations...")
    
    # Set style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # Create comprehensive comparison dashboard
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('2024 vs 2025 Reddit Visa Discourse Comparative Analysis', fontsize=16, fontweight='bold')
    
    # 1. Overall Metrics Comparison
    metrics = ['total_records', 'total_fear', 'total_qa']
    values_2024 = [comparison["summary_comparison"]["2024"][m] for m in metrics]
    values_2025 = [comparison["summary_comparison"]["2025"][m] for m in metrics]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    axes[0, 0].bar(x - width/2, values_2024, width, label='2024', alpha=0.8)
    axes[0, 0].bar(x + width/2, values_2025, width, label='2025', alpha=0.8)
    axes[0, 0].set_xlabel('Metrics')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title('Overall Metrics Comparison')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(['Records', 'Fear Cases', 'Q&A Cases'])
    axes[0, 0].legend()
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # 2. Rate Comparison
    rates = ['fear_rate', 'qa_rate']
    rates_2024 = [comparison["summary_comparison"]["2024"][r] for r in rates]
    rates_2025 = [comparison["summary_comparison"]["2025"][r] for r in rates]
    
    x = np.arange(len(rates))
    axes[0, 1].bar(x - width/2, rates_2024, width, label='2024', alpha=0.8)
    axes[0, 1].bar(x + width/2, rates_2025, width, label='2025', alpha=0.8)
    axes[0, 1].set_xlabel('Rates')
    axes[0, 1].set_ylabel('Rate')
    axes[0, 1].set_title('Fear & Q&A Rate Comparison')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(['Fear Rate', 'Q&A Rate'])
    axes[0, 1].legend()
    
    # 3. Stage Distribution Comparison
    stages = list(comparison["stage_comparison"]["2024"].keys())
    stage_2024 = [comparison["stage_comparison"]["2024"][s] for s in stages]
    stage_2025 = [comparison["stage_comparison"]["2025"][s] for s in stages]
    
    x = np.arange(len(stages))
    axes[0, 2].bar(x - width/2, stage_2024, width, label='2024', alpha=0.8)
    axes[0, 2].bar(x + width/2, stage_2025, width, label='2025', alpha=0.8)
    axes[0, 2].set_xlabel('Visa Stages')
    axes[0, 2].set_ylabel('Count')
    axes[0, 2].set_title('Visa Stage Distribution')
    axes[0, 2].set_xticks(x)
    axes[0, 2].set_xticklabels(stages, rotation=45)
    axes[0, 2].legend()
    
    # 4. Percentage Changes
    changes = comparison["changes"]
    change_metrics = ['total_records_change', 'total_fear_change', 'total_qa_change', 'fear_rate_change', 'qa_rate_change']
    change_values = [changes[m] for m in change_metrics]
    change_labels = ['Records', 'Fear Cases', 'Q&A Cases', 'Fear Rate', 'Q&A Rate']
    
    colors = ['green' if v >= 0 else 'red' for v in change_values]
    axes[1, 0].bar(change_labels, change_values, color=colors, alpha=0.7)
    axes[1, 0].set_xlabel('Metrics')
    axes[1, 0].set_ylabel('Percentage Change (%)')
    axes[1, 0].set_title('Year-over-Year Changes')
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 0].axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # 5. Stage Changes
    stage_changes = comparison["stage_changes"]
    stage_change_values = [stage_changes[f"{s}_change"] for s in stages]
    stage_colors = ['green' if v >= 0 else 'red' for v in stage_change_values]
    
    axes[1, 1].bar(stages, stage_change_values, color=stage_colors, alpha=0.7)
    axes[1, 1].set_xlabel('Visa Stages')
    axes[1, 1].set_ylabel('Percentage Change (%)')
    axes[1, 1].set_title('Stage Distribution Changes')
    axes[1, 1].tick_params(axis='x', rotation=45)
    axes[1, 1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # 6. Summary Statistics
    axes[1, 2].axis('off')
    
    # Create summary text
    summary_text = f"""
     COMPARATIVE SUMMARY
    
     Total Records:
    2024: {comparison["summary_comparison"]["2024"]["total_records"]:,}
    2025: {comparison["summary_comparison"]["2025"]["total_records"]:,}
    Change: {changes["total_records_change"]:.1f}%
    
     Fear Analysis:
    2024 Rate: {comparison["summary_comparison"]["2024"]["fear_rate"]:.1%}
    2025 Rate: {comparison["summary_comparison"]["2025"]["fear_rate"]:.1%}
    Change: {changes["fear_rate_change"]:.1f}%
    
     Q&A Analysis:
    2024 Rate: {comparison["summary_comparison"]["2024"]["qa_rate"]:.1%}
    2025 Rate: {comparison["summary_comparison"]["2025"]["qa_rate"]:.1%}
    Change: {changes["qa_rate_change"]:.1f}%
    
     Top Stage Changes:
    """
    
    # Add top stage changes
    sorted_stages = sorted(stage_changes.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
    for stage_change, change_val in sorted_stages:
        stage_name = stage_change.replace('_change', '').upper()
        summary_text += f"\n{stage_name}: {change_val:.1f}%"
    
    axes[1, 2].text(0.1, 0.9, summary_text, transform=axes[1, 2].transAxes, 
                    fontsize=10, verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    plt.savefig('2025_analysis/plots/comparative_analysis_2024_vs_2025.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print(" Comparative visualizations created")

def generate_insights(comparison):
    """Generate key insights from the comparison"""
    
    insights = []
    
    # Overall volume insights
    records_change = comparison["changes"]["total_records_change"]
    if records_change > 0:
        insights.append(f" Total discussion volume increased by {records_change:.1f}% from 2024 to 2025")
    else:
        insights.append(f" Total discussion volume decreased by {abs(records_change):.1f}% from 2024 to 2025")
    
    # Fear analysis insights
    fear_rate_change = comparison["changes"]["fear_rate_change"]
    if fear_rate_change > 0:
        insights.append(f" Fear levels increased by {fear_rate_change:.1f}% - indicating higher anxiety in 2025")
    else:
        insights.append(f" Fear levels decreased by {abs(fear_rate_change):.1f}% - indicating lower anxiety in 2025")
    
    # Q&A insights
    qa_rate_change = comparison["changes"]["qa_rate_change"]
    if qa_rate_change > 0:
        insights.append(f" Question-asking behavior increased by {qa_rate_change:.1f}% - more information seeking in 2025")
    else:
        insights.append(f" Question-asking behavior decreased by {abs(qa_rate_change):.1f}% - less information seeking in 2025")
    
    # Stage insights
    stage_changes = comparison["stage_changes"]
    top_increase = max(stage_changes.items(), key=lambda x: x[1])
    top_decrease = min(stage_changes.items(), key=lambda x: x[1])
    
    if top_increase[1] > 0:
        stage_name = top_increase[0].replace('_change', '').upper()
        insights.append(f" {stage_name} discussions increased most significantly by {top_increase[1]:.1f}%")
    
    if top_decrease[1] < 0:
        stage_name = top_decrease[0].replace('_change', '').upper()
        insights.append(f" {stage_name} discussions decreased most significantly by {abs(top_decrease[1]):.1f}%")
    
    return insights

def main():
    """Main execution function"""
    
    print(" Starting 2024 vs 2025 Comparative Analysis")
    print("=" * 60)
    
    # Create comparison
    comparison = create_comparative_analysis()
    
    # Create visualizations
    create_comparative_visualizations(comparison)
    
    # Generate insights
    insights = generate_insights(comparison)
    
    # Add insights to comparison
    comparison["insights"] = insights
    
    # Save comparison results
    with open('2025_analysis/comparative_analysis_2024_vs_2025.json', 'w') as f:
        json.dump(comparison, f, indent=2, default=str)
    
    # Print summary
    print("\n COMPARATIVE ANALYSIS SUMMARY")
    print("=" * 60)
    print(f" Records: {comparison['summary_comparison']['2024']['total_records']:,} → {comparison['summary_comparison']['2025']['total_records']:,} ({comparison['changes']['total_records_change']:.1f}%)")
    print(f" Fear Rate: {comparison['summary_comparison']['2024']['fear_rate']:.1%} → {comparison['summary_comparison']['2025']['fear_rate']:.1%} ({comparison['changes']['fear_rate_change']:.1f}%)")
    print(f" Q&A Rate: {comparison['summary_comparison']['2024']['qa_rate']:.1%} → {comparison['summary_comparison']['2025']['qa_rate']:.1%} ({comparison['changes']['qa_rate_change']:.1f}%)")
    
    print("\n KEY INSIGHTS:")
    for insight in insights:
        print(f"  {insight}")
    
    print("\n Comparative analysis completed!")
    print(" Results saved to: 2025_analysis/comparative_analysis_2024_vs_2025.json")
    print(" Visualization saved to: 2025_analysis/plots/comparative_analysis_2024_vs_2025.png")

if __name__ == "__main__":
    main()
