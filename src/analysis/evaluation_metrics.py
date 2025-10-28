"""
Comprehensive Evaluation Metrics for Reddit Visa Discourse Analysis
Implements precision, recall, F1-score, confusion matrix, and other evaluation metrics
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    precision_recall_fscore_support, confusion_matrix,
    classification_report, roc_auc_score, average_precision_score,
    precision_recall_curve, roc_curve, accuracy_score
)
from sklearn.metrics import multilabel_confusion_matrix
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from typing import Dict, List, Tuple, Optional, Any
import json
from pathlib import Path
import warnings
from tqdm import tqdm

from config import config

class RedditEvaluationMetrics:
    """Comprehensive evaluation metrics for Reddit visa discourse classification"""
    
    def __init__(self, labels: List[str] = None):
        self.labels = labels or config.labels
        self.num_labels = len(self.labels)
    
    def calculate_basic_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """Calculate basic classification metrics"""
        metrics = {}
        
        # Overall metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average='macro', zero_division=0
        )
        
        metrics['overall'] = {
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'support': int(support)
        }
        
        # Per-label metrics
        per_label_metrics = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )
        
        metrics['per_label'] = {}
        for i, label in enumerate(self.labels):
            metrics['per_label'][label] = {
                'precision': float(per_label_metrics[0][i]),
                'recall': float(per_label_metrics[1][i]),
                'f1_score': float(per_label_metrics[2][i]),
                'support': int(per_label_metrics[3][i])
            }
        
        return metrics
    
    def calculate_confusion_matrices(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """Calculate confusion matrices for each label"""
        confusion_matrices = {}
        
        for i, label in enumerate(self.labels):
            cm = confusion_matrix(y_true[:, i], y_pred[:, i])
            confusion_matrices[label] = {
                'matrix': cm.tolist(),
                'true_negatives': int(cm[0, 0]),
                'false_positives': int(cm[0, 1]),
                'false_negatives': int(cm[1, 0]),
                'true_positives': int(cm[1, 1])
            }
        
        return confusion_matrices
    
    def calculate_roc_auc_metrics(self, y_true: np.ndarray, y_scores: np.ndarray) -> Dict:
        """Calculate ROC AUC metrics for each label"""
        roc_auc_metrics = {}
        
        for i, label in enumerate(self.labels):
            try:
                # Calculate ROC AUC
                roc_auc = roc_auc_score(y_true[:, i], y_scores[:, i])
                roc_auc_metrics[label] = {
                    'roc_auc': float(roc_auc)
                }
            except ValueError:
                # Handle case where label doesn't exist in test set
                roc_auc_metrics[label] = {
                    'roc_auc': 0.0
                }
        
        # Calculate macro and micro averages
        try:
            macro_auc = roc_auc_score(y_true, y_scores, average='macro')
            micro_auc = roc_auc_score(y_true, y_scores, average='micro')
            roc_auc_metrics['macro_average'] = float(macro_auc)
            roc_auc_metrics['micro_average'] = float(micro_auc)
        except ValueError:
            roc_auc_metrics['macro_average'] = 0.0
            roc_auc_metrics['micro_average'] = 0.0
        
        return roc_auc_metrics
    
    def calculate_precision_recall_metrics(self, y_true: np.ndarray, y_scores: np.ndarray) -> Dict:
        """Calculate precision-recall metrics for each label"""
        pr_metrics = {}
        
        for i, label in enumerate(self.labels):
            try:
                # Calculate average precision
                avg_precision = average_precision_score(y_true[:, i], y_scores[:, i])
                pr_metrics[label] = {
                    'average_precision': float(avg_precision)
                }
            except ValueError:
                pr_metrics[label] = {
                    'average_precision': 0.0
                }
        
        # Calculate macro and micro averages
        try:
            macro_ap = average_precision_score(y_true, y_scores, average='macro')
            micro_ap = average_precision_score(y_true, y_scores, average='micro')
            pr_metrics['macro_average'] = float(macro_ap)
            pr_metrics['micro_average'] = float(micro_ap)
        except ValueError:
            pr_metrics['macro_average'] = 0.0
            pr_metrics['micro_average'] = 0.0
        
        return pr_metrics
    
    def calculate_class_imbalance_metrics(self, y_true: np.ndarray) -> Dict:
        """Calculate class imbalance metrics"""
        imbalance_metrics = {}
        
        for i, label in enumerate(self.labels):
            positive_count = np.sum(y_true[:, i])
            total_count = len(y_true)
            positive_ratio = positive_count / total_count if total_count > 0 else 0
            
            imbalance_metrics[label] = {
                'positive_count': int(positive_count),
                'total_count': int(total_count),
                'positive_ratio': float(positive_ratio),
                'imbalance_ratio': float((total_count - positive_count) / positive_count) if positive_count > 0 else float('inf')
            }
        
        return imbalance_metrics
    
    def calculate_error_analysis(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                texts: List[str] = None) -> Dict:
        """Analyze classification errors"""
        error_analysis = {}
        
        for i, label in enumerate(self.labels):
            # False positives
            false_positives = np.where((y_true[:, i] == 0) & (y_pred[:, i] == 1))[0]
            # False negatives
            false_negatives = np.where((y_true[:, i] == 1) & (y_pred[:, i] == 0))[0]
            
            error_analysis[label] = {
                'false_positives_count': len(false_positives),
                'false_negatives_count': len(false_negatives),
                'false_positive_indices': false_positives.tolist(),
                'false_negative_indices': false_negatives.tolist()
            }
            
            # Add text examples if available
            if texts is not None:
                if len(false_positives) > 0:
                    error_analysis[label]['false_positive_examples'] = [
                        texts[idx] for idx in false_positives[:5]  # Top 5 examples
                    ]
                if len(false_negatives) > 0:
                    error_analysis[label]['false_negative_examples'] = [
                        texts[idx] for idx in false_negatives[:5]  # Top 5 examples
                    ]
        
        return error_analysis
    
    def create_confusion_matrix_plots(self, confusion_matrices: Dict, output_dir: Path):
        """Create confusion matrix visualizations"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for label, cm_data in confusion_matrices.items():
            cm = np.array(cm_data['matrix'])
            
            # Create matplotlib plot
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=['Predicted 0', 'Predicted 1'],
                       yticklabels=['Actual 0', 'Actual 1'])
            plt.title(f'Confusion Matrix - {label}')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.tight_layout()
            plt.savefig(output_dir / f'confusion_matrix_{label}.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # Create plotly plot
            fig = go.Figure(data=go.Heatmap(
                z=cm,
                x=['Predicted 0', 'Predicted 1'],
                y=['Actual 0', 'Actual 1'],
                text=cm,
                texttemplate="%{text}",
                textfont={"size": 20},
                colorscale='Blues'
            ))
            fig.update_layout(
                title=f'Confusion Matrix - {label}',
                xaxis_title='Predicted Label',
                yaxis_title='True Label'
            )
            fig.write_html(output_dir / f'confusion_matrix_{label}.html')
    
    def create_roc_curves(self, y_true: np.ndarray, y_scores: np.ndarray, output_dir: Path):
        """Create ROC curve visualizations"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Individual ROC curves
        for i, label in enumerate(self.labels):
            try:
                fpr, tpr, _ = roc_curve(y_true[:, i], y_scores[:, i])
                roc_auc = roc_auc_score(y_true[:, i], y_scores[:, i])
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=fpr, y=tpr,
                    mode='lines',
                    name=f'{label} (AUC = {roc_auc:.3f})',
                    line=dict(width=2)
                ))
                fig.add_trace(go.Scatter(
                    x=[0, 1], y=[0, 1],
                    mode='lines',
                    name='Random',
                    line=dict(dash='dash', color='red')
                ))
                fig.update_layout(
                    title=f'ROC Curve - {label}',
                    xaxis_title='False Positive Rate',
                    yaxis_title='True Positive Rate',
                    xaxis=dict(range=[0, 1]),
                    yaxis=dict(range=[0, 1])
                )
                fig.write_html(output_dir / f'roc_curve_{label}.html')
                
            except ValueError:
                continue
        
        # Combined ROC curves
        fig = go.Figure()
        for i, label in enumerate(self.labels):
            try:
                fpr, tpr, _ = roc_curve(y_true[:, i], y_scores[:, i])
                roc_auc = roc_auc_score(y_true[:, i], y_scores[:, i])
                
                fig.add_trace(go.Scatter(
                    x=fpr, y=tpr,
                    mode='lines',
                    name=f'{label} (AUC = {roc_auc:.3f})',
                    line=dict(width=2)
                ))
            except ValueError:
                continue
        
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            name='Random',
            line=dict(dash='dash', color='red')
        ))
        fig.update_layout(
            title='ROC Curves - All Labels',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            xaxis=dict(range=[0, 1]),
            yaxis=dict(range=[0, 1])
        )
        fig.write_html(output_dir / 'roc_curves_combined.html')
    
    def create_precision_recall_curves(self, y_true: np.ndarray, y_scores: np.ndarray, output_dir: Path):
        """Create precision-recall curve visualizations"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Individual PR curves
        for i, label in enumerate(self.labels):
            try:
                precision, recall, _ = precision_recall_curve(y_true[:, i], y_scores[:, i])
                avg_precision = average_precision_score(y_true[:, i], y_scores[:, i])
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=recall, y=precision,
                    mode='lines',
                    name=f'{label} (AP = {avg_precision:.3f})',
                    line=dict(width=2)
                ))
                fig.update_layout(
                    title=f'Precision-Recall Curve - {label}',
                    xaxis_title='Recall',
                    yaxis_title='Precision',
                    xaxis=dict(range=[0, 1]),
                    yaxis=dict(range=[0, 1])
                )
                fig.write_html(output_dir / f'pr_curve_{label}.html')
                
            except ValueError:
                continue
        
        # Combined PR curves
        fig = go.Figure()
        for i, label in enumerate(self.labels):
            try:
                precision, recall, _ = precision_recall_curve(y_true[:, i], y_scores[:, i])
                avg_precision = average_precision_score(y_true[:, i], y_scores[:, i])
                
                fig.add_trace(go.Scatter(
                    x=recall, y=precision,
                    mode='lines',
                    name=f'{label} (AP = {avg_precision:.3f})',
                    line=dict(width=2)
                ))
            except ValueError:
                continue
        
        fig.update_layout(
            title='Precision-Recall Curves - All Labels',
            xaxis_title='Recall',
            yaxis_title='Precision',
            xaxis=dict(range=[0, 1]),
            yaxis=dict(range=[0, 1])
        )
        fig.write_html(output_dir / 'pr_curves_combined.html')
    
    def create_metrics_summary_plots(self, metrics: Dict, output_dir: Path):
        """Create metrics summary visualizations"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Precision, Recall, F1 comparison
        labels = list(metrics['per_label'].keys())
        precision = [metrics['per_label'][label]['precision'] for label in labels]
        recall = [metrics['per_label'][label]['recall'] for label in labels]
        f1 = [metrics['per_label'][label]['f1_score'] for label in labels]
        
        fig = go.Figure()
        fig.add_trace(go.Bar(name='Precision', x=labels, y=precision))
        fig.add_trace(go.Bar(name='Recall', x=labels, y=recall))
        fig.add_trace(go.Bar(name='F1 Score', x=labels, y=f1))
        
        fig.update_layout(
            title='Metrics Comparison by Label',
            xaxis_title='Labels',
            yaxis_title='Score',
            barmode='group'
        )
        fig.write_html(output_dir / 'metrics_comparison.html')
        
        # Support distribution
        support = [metrics['per_label'][label]['support'] for label in labels]
        
        fig = go.Figure(data=go.Bar(x=labels, y=support))
        fig.update_layout(
            title='Support Distribution by Label',
            xaxis_title='Labels',
            yaxis_title='Support'
        )
        fig.write_html(output_dir / 'support_distribution.html')
    
    def run_comprehensive_evaluation(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                   y_scores: np.ndarray = None, texts: List[str] = None) -> Dict:
        """Run comprehensive evaluation and generate all metrics and visualizations"""
        print("Running comprehensive evaluation...")
        
        # Calculate all metrics
        basic_metrics = self.calculate_basic_metrics(y_true, y_pred)
        confusion_matrices = self.calculate_confusion_matrices(y_true, y_pred)
        class_imbalance = self.calculate_class_imbalance_metrics(y_true)
        error_analysis = self.calculate_error_analysis(y_true, y_pred, texts)
        
        # ROC and PR metrics if scores available
        roc_auc_metrics = {}
        pr_metrics = {}
        if y_scores is not None:
            roc_auc_metrics = self.calculate_roc_auc_metrics(y_true, y_scores)
            pr_metrics = self.calculate_precision_recall_metrics(y_true, y_scores)
        
        # Combine all metrics
        evaluation_results = {
            'basic_metrics': basic_metrics,
            'confusion_matrices': confusion_matrices,
            'class_imbalance': class_imbalance,
            'error_analysis': error_analysis,
            'roc_auc_metrics': roc_auc_metrics,
            'pr_metrics': pr_metrics,
            'evaluation_summary': self.create_evaluation_summary(basic_metrics)
        }
        
        # Create visualizations
        viz_dir = config.get_metrics_path() / "visualizations"
        self.create_confusion_matrix_plots(confusion_matrices, viz_dir)
        
        if y_scores is not None:
            self.create_roc_curves(y_true, y_scores, viz_dir)
            self.create_precision_recall_curves(y_true, y_scores, viz_dir)
        
        self.create_metrics_summary_plots(basic_metrics, viz_dir)
        
        # Save results
        results_file = config.get_metrics_path() / "comprehensive_evaluation.json"
        config.get_metrics_path().mkdir(parents=True, exist_ok=True)
        
        with open(results_file, 'w') as f:
            json.dump(evaluation_results, f, indent=2, default=str)
        
        print(f"Comprehensive evaluation results saved to {results_file}")
        print(f"Visualizations saved to {viz_dir}")
        
        return evaluation_results
    
    def create_evaluation_summary(self, metrics: Dict) -> Dict:
        """Create a summary of evaluation results"""
        summary = {
            'overall_performance': metrics['overall'],
            'best_performing_label': max(metrics['per_label'].items(), key=lambda x: x[1]['f1_score']),
            'worst_performing_label': min(metrics['per_label'].items(), key=lambda x: x[1]['f1_score']),
            'label_performance_ranking': sorted(
                metrics['per_label'].items(), 
                key=lambda x: x[1]['f1_score'], 
                reverse=True
            )
        }
        
        return summary
    
    def print_evaluation_summary(self, results: Dict):
        """Print a summary of evaluation results"""
        print("\n=== EVALUATION SUMMARY ===")
        
        # Overall performance
        overall = results['basic_metrics']['overall']
        print(f"Overall Performance:")
        print(f"  Precision: {overall['precision']:.4f}")
        print(f"  Recall: {overall['recall']:.4f}")
        print(f"  F1 Score: {overall['f1_score']:.4f}")
        
        # Per-label performance
        print(f"\nPer-label Performance:")
        for label, metrics in results['basic_metrics']['per_label'].items():
            print(f"  {label}:")
            print(f"    Precision: {metrics['precision']:.4f}")
            print(f"    Recall: {metrics['recall']:.4f}")
            print(f"    F1 Score: {metrics['f1_score']:.4f}")
            print(f"    Support: {metrics['support']}")
        
        # Class imbalance
        print(f"\nClass Imbalance:")
        for label, imbalance in results['class_imbalance'].items():
            print(f"  {label}: {imbalance['positive_ratio']:.2%} positive")
        
        # Error analysis
        print(f"\nError Analysis:")
        for label, errors in results['error_analysis'].items():
            print(f"  {label}:")
            print(f"    False Positives: {errors['false_positives_count']}")
            print(f"    False Negatives: {errors['false_negatives_count']}")

def main():
    """Main function for evaluation metrics"""
    import numpy as np
    
    # Example usage with dummy data
    np.random.seed(42)
    n_samples = 1000
    n_labels = len(config.labels)
    
    # Generate dummy data
    y_true = np.random.randint(0, 2, (n_samples, n_labels))
    y_pred = np.random.randint(0, 2, (n_samples, n_labels))
    y_scores = np.random.rand(n_samples, n_labels)
    
    # Initialize evaluator
    evaluator = RedditEvaluationMetrics()
    
    # Run comprehensive evaluation
    results = evaluator.run_comprehensive_evaluation(y_true, y_pred, y_scores)
    
    # Print summary
    evaluator.print_evaluation_summary(results)
    
    return results

if __name__ == "__main__":
    main()
