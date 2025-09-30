#!/usr/bin/env python3
"""
Pre/Post Analysis for Immigration Classification
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import json
from datetime import datetime

class PrePostAnalyzer:
    def __init__(self):
        self.pre_analysis = {}
        self.post_analysis = {}
        self.comparison_metrics = {}
    
    def analyze_pre_classification(self, df):
        """Analyze data before classification"""
        print(" Pre-Classification Analysis...")
        
        analysis = {
            'timestamp': datetime.now().isoformat(),
            'total_samples': len(df),
            'label_distribution': df['label'].value_counts().to_dict(),
            'text_length_stats': {
                'mean': df['text'].str.len().mean(),
                'std': df['text'].str.len().std(),
                'min': df['text'].str.len().min(),
                'max': df['text'].str.len().max()
            },
            'label_percentages': (df['label'].value_counts(normalize=True) * 100).to_dict(),
            'unique_texts': df['text'].nunique(),
            'duplicate_texts': len(df) - df['text'].nunique()
        }
        
        # Text length distribution by label
        try:
            length_by_label = df.groupby('label')['text'].apply(lambda x: x.str.len())
            analysis['length_by_label'] = {
                label: {
                    'mean': float(lengths.mean()) if hasattr(lengths, 'mean') and len(lengths) > 0 else 0.0,
                    'std': float(lengths.std()) if hasattr(lengths, 'std') and len(lengths) > 0 else 0.0,
                    'median': float(lengths.median()) if hasattr(lengths, 'median') and len(lengths) > 0 else 0.0
                }
                for label, lengths in length_by_label.items()
            }
        except Exception as e:
            analysis['length_by_label'] = {'error': str(e)}
        
        self.pre_analysis = analysis
        return analysis
    
    def analyze_post_classification(self, df, predictions, actual_labels, metrics):
        """Analyze data after classification"""
        print(" Post-Classification Analysis...")
        
        # Create results DataFrame
        results_df = df.copy()
        results_df['predicted'] = predictions
        results_df['actual'] = actual_labels
        results_df['correct'] = predictions == actual_labels
        
        analysis = {
            'timestamp': datetime.now().isoformat(),
            'total_samples': len(df),
            'accuracy': metrics.get('accuracy', 0),
            'precision': metrics.get('precision', 0),
            'recall': metrics.get('recall', 0),
            'f1_score': metrics.get('f1', 0),
            'classification_report': classification_report(actual_labels, predictions, output_dict=True),
            'confusion_matrix': confusion_matrix(actual_labels, predictions).tolist(),
            'label_performance': self._analyze_label_performance(results_df),
            'error_analysis': self._analyze_errors(results_df)
        }
        
        self.post_analysis = analysis
        return analysis
    
    def _analyze_label_performance(self, results_df):
        """Analyze performance by label"""
        label_performance = {}
        
        for label in results_df['actual'].unique():
            label_data = results_df[results_df['actual'] == label]
            
            if len(label_data) > 0:
                accuracy = label_data['correct'].mean()
                total_samples = len(label_data)
                correct_predictions = label_data['correct'].sum()
                
                # Most common misclassifications
                misclassifications = label_data[~label_data['correct']]
                if len(misclassifications) > 0:
                    common_errors = misclassifications['predicted'].value_counts().to_dict()
                else:
                    common_errors = {}
                
                label_performance[label] = {
                    'accuracy': accuracy,
                    'total_samples': total_samples,
                    'correct_predictions': correct_predictions,
                    'incorrect_predictions': total_samples - correct_predictions,
                    'common_errors': common_errors
                }
        
        return label_performance
    
    def _analyze_errors(self, results_df):
        """Analyze classification errors"""
        errors = results_df[~results_df['correct']]
        
        if len(errors) == 0:
            return {'total_errors': 0, 'error_patterns': {}}
        
        # Error patterns
        error_patterns = errors.groupby(['actual', 'predicted']).size().to_dict()
        
        # Most problematic samples
        error_samples = errors[['text', 'actual', 'predicted']].head(10).to_dict('records')
        
        return {
            'total_errors': len(errors),
            'error_rate': len(errors) / len(results_df),
            'error_patterns': error_patterns,
            'sample_errors': error_samples
        }
    
    def compare_pre_post(self):
        """Compare pre and post classification metrics"""
        if not self.pre_analysis or not self.post_analysis:
            print(" Need both pre and post analysis data")
            return None
        
        comparison = {
            'timestamp': datetime.now().isoformat(),
            'data_quality': {
                'pre_total_samples': self.pre_analysis['total_samples'],
                'post_total_samples': self.post_analysis['total_samples'],
                'samples_consistent': self.pre_analysis['total_samples'] == self.post_analysis['total_samples']
            },
            'label_distribution': {
                'pre_distribution': self.pre_analysis['label_distribution'],
                'post_distribution': self.post_analysis.get('label_distribution', {}),
                'distribution_changed': self.pre_analysis['label_distribution'] != self.post_analysis.get('label_distribution', {})
            },
            'performance_metrics': {
                'accuracy': self.post_analysis.get('accuracy', 0),
                'precision': self.post_analysis.get('precision', 0),
                'recall': self.post_analysis.get('recall', 0),
                'f1_score': self.post_analysis.get('f1_score', 0)
            },
            'improvement_suggestions': self._generate_improvement_suggestions()
        }
        
        self.comparison_metrics = comparison
        return comparison
    
    def _generate_improvement_suggestions(self):
        """Generate suggestions for improving classification"""
        suggestions = []
        
        if self.post_analysis:
            accuracy = self.post_analysis.get('accuracy', 0)
            
            if accuracy < 0.7:
                suggestions.append("🔴 Low accuracy - consider more training data or feature engineering")
            elif accuracy < 0.85:
                suggestions.append("🟡 Moderate accuracy - fine-tune hyperparameters or add more diverse training data")
            else:
                suggestions.append("🟢 Good accuracy - consider ensemble methods for further improvement")
            
            # Check for class imbalance
            if 'label_performance' in self.post_analysis:
                label_perfs = self.post_analysis['label_performance']
                min_accuracy = min(perf['accuracy'] for perf in label_perfs.values())
                max_accuracy = max(perf['accuracy'] for perf in label_perfs.values())
                
                if max_accuracy - min_accuracy > 0.3:
                    suggestions.append(" Significant performance variation across labels - consider class balancing")
            
            # Check error patterns
            if 'error_analysis' in self.post_analysis:
                error_rate = self.post_analysis['error_analysis'].get('error_rate', 0)
                if error_rate > 0.2:
                    suggestions.append(" High error rate - analyze misclassified samples for pattern recognition")
        
        return suggestions
    
    def generate_visualizations(self, save_path="analysis_plots"):
        """Generate visualization plots"""
        import os
        os.makedirs(save_path, exist_ok=True)
        
        # Pre-classification plots
        if self.pre_analysis:
            self._plot_pre_analysis(save_path)
        
        # Post-classification plots
        if self.post_analysis:
            self._plot_post_analysis(save_path)
        
        # Comparison plots
        if self.comparison_metrics:
            self._plot_comparison(save_path)
    
    def _plot_pre_analysis(self, save_path):
        """Plot pre-classification analysis"""
        # Label distribution
        plt.figure(figsize=(10, 6))
        labels = list(self.pre_analysis['label_distribution'].keys())
        counts = list(self.pre_analysis['label_distribution'].values())
        
        plt.subplot(1, 2, 1)
        plt.bar(labels, counts)
        plt.title('Label Distribution (Pre-Classification)')
        plt.xlabel('Labels')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        
        # Text length distribution
        plt.subplot(1, 2, 2)
        plt.hist([self.pre_analysis['text_length_stats']['mean']], bins=20, alpha=0.7)
        plt.title('Text Length Distribution')
        plt.xlabel('Character Count')
        plt.ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig(f'{save_path}/pre_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_post_analysis(self, save_path):
        """Plot post-classification analysis"""
        if 'confusion_matrix' in self.post_analysis:
            # Confusion Matrix
            plt.figure(figsize=(8, 6))
            cm = np.array(self.post_analysis['confusion_matrix'])
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title('Confusion Matrix')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.savefig(f'{save_path}/confusion_matrix.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # Performance by label
        if 'label_performance' in self.post_analysis:
            plt.figure(figsize=(10, 6))
            labels = list(self.post_analysis['label_performance'].keys())
            accuracies = [perf['accuracy'] for perf in self.post_analysis['label_performance'].values()]
            
            plt.bar(labels, accuracies)
            plt.title('Accuracy by Label')
            plt.xlabel('Labels')
            plt.ylabel('Accuracy')
            plt.xticks(rotation=45)
            plt.ylim(0, 1)
            
            plt.tight_layout()
            plt.savefig(f'{save_path}/label_performance.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def _plot_comparison(self, save_path):
        """Plot comparison analysis"""
        # Performance metrics comparison
        if 'performance_metrics' in self.comparison_metrics:
            plt.figure(figsize=(10, 6))
            metrics = self.comparison_metrics['performance_metrics']
            metric_names = list(metrics.keys())
            metric_values = list(metrics.values())
            
            plt.bar(metric_names, metric_values)
            plt.title('Performance Metrics')
            plt.xlabel('Metrics')
            plt.ylabel('Score')
            plt.ylim(0, 1)
            
            plt.tight_layout()
            plt.savefig(f'{save_path}/performance_metrics.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def save_analysis_report(self, output_file="pre_post_analysis_report.json"):
        """Save complete analysis report"""
        report = {
            'pre_analysis': self.pre_analysis,
            'post_analysis': self.post_analysis,
            'comparison': self.comparison_metrics,
            'generated_at': datetime.now().isoformat()
        }
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f" Complete analysis report saved to: {output_file}")
        return report

def run_pre_post_analysis(df, predictions=None, actual_labels=None, metrics=None):
    """Run complete pre/post analysis"""
    analyzer = PrePostAnalyzer()
    
    # Pre-analysis
    pre_results = analyzer.analyze_pre_classification(df)
    
    # Post-analysis (if data provided)
    if predictions is not None and actual_labels is not None and metrics is not None:
        post_results = analyzer.analyze_post_classification(df, predictions, actual_labels, metrics)
        comparison = analyzer.compare_pre_post()
        
        # Generate visualizations
        analyzer.generate_visualizations()
        
        # Save report
        report = analyzer.save_analysis_report()
        
        return pre_results, post_results, comparison, report
    else:
        return pre_results, None, None, None

if __name__ == "__main__":
    print(" Pre/Post Analysis Module")
    print("This module provides comprehensive pre and post classification analysis.")
