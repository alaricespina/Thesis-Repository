import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import pandas as pd


class ModelComparison:
    def __init__(self):
        self.results = {}
    
    def add_model_results(self, model_name, y_true, y_pred, y_prob=None, training_time=None):
        """Add model results for comparison"""
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
        cm = confusion_matrix(y_true, y_pred)
        
        self.results[model_name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'predictions': y_pred,
            'probabilities': y_prob,
            'confusion_matrix': cm,
            'training_time': training_time
        }
    
    def plot_accuracy_comparison(self):
        """Plot accuracy comparison between models"""
        models = list(self.results.keys())
        accuracies = [self.results[model]['accuracy'] for model in models]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(models, accuracies, color=['#FF6B6B', '#4ECDC4'])
        plt.title('Model Accuracy Comparison', fontsize=16, fontweight='bold')
        plt.ylabel('Accuracy', fontsize=12)
        plt.xlabel('Models', fontsize=12)
        plt.ylim(0, 1)
        
        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{acc:.4f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.show()
    
    def plot_metrics_comparison(self):
        """Plot comprehensive metrics comparison"""
        models = list(self.results.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.ravel()
        
        colors = ['#FF6B6B', '#4ECDC4']
        
        for i, metric in enumerate(metrics):
            values = [self.results[model][metric] for model in models]
            bars = axes[i].bar(models, values, color=colors)
            axes[i].set_title(f'{metric.replace("_", " ").title()} Comparison', fontweight='bold')
            axes[i].set_ylabel(metric.replace("_", " ").title())
            axes[i].set_ylim(0, 1)
            
            # Add value labels
            for bar, val in zip(bars, values):
                axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{val:.4f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.show()
    
    def plot_confusion_matrices(self, class_names):
        """Plot confusion matrices side by side"""
        models = list(self.results.keys())
        
        fig, axes = plt.subplots(1, len(models), figsize=(15, 6))
        if len(models) == 1:
            axes = [axes]
        
        for i, model in enumerate(models):
            cm = self.results[model]['confusion_matrix']
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=class_names, yticklabels=class_names, ax=axes[i])
            axes[i].set_title(f'{model} Confusion Matrix', fontweight='bold')
            axes[i].set_xlabel('Predicted')
            axes[i].set_ylabel('Actual')
        
        plt.tight_layout()
        plt.show()
    
    def generate_comparison_report(self):
        """Generate a detailed comparison report"""
        print("=" * 80)
        print("MODEL COMPARISON REPORT")
        print("=" * 80)
        
        # Create comparison table
        models = list(self.results.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        
        print(f"\n{'Metric':<15}", end="")
        for model in models:
            print(f"{model:<20}", end="")
        print()
        print("-" * (15 + len(models) * 20))
        
        for metric in metrics:
            print(f"{metric.replace('_', ' ').title():<15}", end="")
            for model in models:
                value = self.results[model][metric]
                print(f"{value:<20.4f}", end="")
            print()
        
        # Training time comparison if available
        if any(self.results[model]['training_time'] is not None for model in models):
            print(f"\n{'Training Time':<15}", end="")
            for model in models:
                time_val = self.results[model]['training_time']
                if time_val is not None:
                    print(f"{time_val:<20.2f}", end="")
                else:
                    print(f"{'N/A':<20}", end="")
            print()
        
        print("\n" + "=" * 80)
        
        # Performance analysis
        best_accuracy_model = max(models, key=lambda x: self.results[x]['accuracy'])
        best_f1_model = max(models, key=lambda x: self.results[x]['f1_score'])
        
        print("PERFORMANCE ANALYSIS:")
        print("-" * 20)
        print(f"Best Accuracy: {best_accuracy_model} ({self.results[best_accuracy_model]['accuracy']:.4f})")
        print(f"Best F1-Score: {best_f1_model} ({self.results[best_f1_model]['f1_score']:.4f})")
        
        # Calculate improvement if DBN+RF vs Vanilla RF
        if len(models) == 2:
            dbn_model = None
            vanilla_model = None
            
            for model in models:
                if 'DBN' in model.upper():
                    dbn_model = model
                elif 'VANILLA' in model.upper() or 'RF' in model.upper():
                    vanilla_model = model
            
            if dbn_model and vanilla_model:
                dbn_acc = self.results[dbn_model]['accuracy']
                vanilla_acc = self.results[vanilla_model]['accuracy']
                improvement = ((dbn_acc - vanilla_acc) / vanilla_acc) * 100
                
                print(f"\nIMPROVEMENT ANALYSIS:")
                print("-" * 20)
                if improvement > 0:
                    print(f"DBN+RF is {improvement:.2f}% better than Vanilla RF")
                    print(f"Absolute improvement: {dbn_acc - vanilla_acc:.4f}")
                elif improvement < 0:
                    print(f"Vanilla RF is {abs(improvement):.2f}% better than DBN+RF")
                    print(f"Absolute difference: {abs(dbn_acc - vanilla_acc):.4f}")
                else:
                    print("Both models perform equally well")
        
        print("=" * 80)
    
    def save_results_to_csv(self, filepath="model_comparison_results.csv"):
        """Save comparison results to CSV"""
        models = list(self.results.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        
        data = []
        for model in models:
            row = {'Model': model}
            for metric in metrics:
                row[metric.replace('_', ' ').title()] = self.results[model][metric]
            if self.results[model]['training_time'] is not None:
                row['Training Time (s)'] = self.results[model]['training_time']
            data.append(row)
        
        df = pd.DataFrame(data)
        df.to_csv(filepath, index=False)
        print(f"Comparison results saved to {filepath}")
        
        return df