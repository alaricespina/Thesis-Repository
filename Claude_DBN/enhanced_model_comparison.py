import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import pandas as pd
import os


class EnhancedModelComparison:
    def __init__(self, save_plots=True, plot_dir="plots"):
        self.results = {}
        self.save_plots = save_plots
        self.plot_dir = plot_dir
        
        if self.save_plots:
            os.makedirs(self.plot_dir, exist_ok=True)
    
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
        
        plt.figure(figsize=(12, 8))
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        bars = plt.bar(models, accuracies, color=colors[:len(models)])
        
        plt.title('Model Accuracy Comparison', fontsize=18, fontweight='bold', pad=20)
        plt.ylabel('Accuracy', fontsize=14)
        plt.xlabel('Models', fontsize=14)
        plt.ylim(0, 1)
        
        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{acc:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=12)
        
        # Add grid for better readability
        plt.grid(axis='y', alpha=0.3, linestyle='--')
        plt.tight_layout()
        
        if self.save_plots:
            plt.savefig(os.path.join(self.plot_dir, 'accuracy_comparison.png'), 
                       dpi=300, bbox_inches='tight')
            print(f"Saved accuracy comparison to {self.plot_dir}/accuracy_comparison.png")
        
        plt.show()
    
    def plot_metrics_comparison(self):
        """Plot comprehensive metrics comparison"""
        models = list(self.results.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.ravel()
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        
        for i, metric in enumerate(metrics):
            values = [self.results[model][metric] for model in models]
            bars = axes[i].bar(models, values, color=colors[:len(models)])
            axes[i].set_title(f'{metric.replace("_", " ").title()} Comparison', 
                            fontweight='bold', fontsize=14)
            axes[i].set_ylabel(metric.replace("_", " ").title(), fontsize=12)
            axes[i].set_ylim(0, 1)
            axes[i].grid(axis='y', alpha=0.3, linestyle='--')
            
            # Add value labels
            for bar, val in zip(bars, values):
                axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{val:.4f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        if self.save_plots:
            plt.savefig(os.path.join(self.plot_dir, 'metrics_comparison.png'), 
                       dpi=300, bbox_inches='tight')
            print(f"Saved metrics comparison to {self.plot_dir}/metrics_comparison.png")
        
        plt.show()
    
    def plot_confusion_matrices(self, class_names):
        """Plot confusion matrices side by side"""
        models = list(self.results.keys())
        
        fig, axes = plt.subplots(1, len(models), figsize=(8*len(models), 6))
        if len(models) == 1:
            axes = [axes]
        
        for i, model in enumerate(models):
            cm = self.results[model]['confusion_matrix']
            
            # Create heatmap
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=class_names, yticklabels=class_names, 
                       ax=axes[i], cbar_kws={'shrink': 0.8})
            
            axes[i].set_title(f'{model} Confusion Matrix', fontweight='bold', fontsize=14)
            axes[i].set_xlabel('Predicted', fontsize=12)
            axes[i].set_ylabel('Actual', fontsize=12)
            
            # Rotate labels for better readability
            axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=45, ha='right')
            axes[i].set_yticklabels(axes[i].get_yticklabels(), rotation=0)
        
        plt.tight_layout()
        
        if self.save_plots:
            plt.savefig(os.path.join(self.plot_dir, 'confusion_matrices.png'), 
                       dpi=300, bbox_inches='tight')
            print(f"Saved confusion matrices to {self.plot_dir}/confusion_matrices.png")
        
        plt.show()
    
    def plot_class_distribution(self, y, class_names, title="Class Distribution"):
        """Plot the distribution of classes in the dataset"""
        unique, counts = np.unique(y, return_counts=True)
        
        plt.figure(figsize=(12, 8))
        colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99']
        bars = plt.bar([class_names[i] for i in unique], counts, color=colors[:len(unique)])
        
        plt.title(title, fontsize=18, fontweight='bold', pad=20)
        plt.xlabel('Weather Conditions', fontsize=14)
        plt.ylabel('Count', fontsize=14)
        plt.xticks(rotation=45, ha='right')
        
        # Add count labels on bars
        for bar, count in zip(bars, counts):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts)*0.01,
                    str(count), ha='center', va='bottom', fontweight='bold', fontsize=12)
        
        # Add percentage labels
        total = sum(counts)
        for bar, count in zip(bars, counts):
            percentage = (count / total) * 100
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height()/2,
                    f'{percentage:.1f}%', ha='center', va='center', 
                    fontweight='bold', fontsize=10, color='white')
        
        plt.grid(axis='y', alpha=0.3, linestyle='--')
        plt.tight_layout()
        
        if self.save_plots:
            plt.savefig(os.path.join(self.plot_dir, 'class_distribution.png'), 
                       dpi=300, bbox_inches='tight')
            print(f"Saved class distribution to {self.plot_dir}/class_distribution.png")
        
        plt.show()
    
    def plot_training_time_comparison(self):
        """Plot training time comparison if available"""
        models = []
        times = []
        
        for model, results in self.results.items():
            if results['training_time'] is not None:
                models.append(model)
                times.append(results['training_time'])
        
        if len(models) < 2:
            print("Training time comparison not available (insufficient data)")
            return
        
        plt.figure(figsize=(10, 6))
        colors = ['#FF6B6B', '#4ECDC4']
        bars = plt.bar(models, times, color=colors[:len(models)])
        
        plt.title('Training Time Comparison', fontsize=16, fontweight='bold')
        plt.ylabel('Training Time (seconds)', fontsize=12)
        plt.xlabel('Models', fontsize=12)
        
        # Add value labels on bars
        for bar, time_val in zip(bars, times):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(times)*0.01,
                    f'{time_val:.2f}s', ha='center', va='bottom', fontweight='bold')
        
        plt.grid(axis='y', alpha=0.3, linestyle='--')
        plt.tight_layout()
        
        if self.save_plots:
            plt.savefig(os.path.join(self.plot_dir, 'training_time_comparison.png'), 
                       dpi=300, bbox_inches='tight')
            print(f"Saved training time comparison to {self.plot_dir}/training_time_comparison.png")
        
        plt.show()
    
    def generate_comparison_report(self):
        """Generate a detailed comparison report"""
        print("=" * 80)
        print("ENHANCED MODEL COMPARISON REPORT")
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
        
        if self.save_plots:
            print(f"\nAll plots have been saved to the '{self.plot_dir}' directory")
    
    def save_results_to_csv(self, filepath="enhanced_model_comparison_results.csv"):
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
        print(f"Enhanced comparison results saved to {filepath}")
        
        return df