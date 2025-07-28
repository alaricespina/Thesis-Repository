import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import os


class VanillaRandomForestClassifier:
    def __init__(self, n_estimators=100, max_depth=None, random_state=42):
        """
        Vanilla Random Forest Classifier without DBN preprocessing
        
        Parameters:
        -----------
        n_estimators : int, default=100
            Number of trees in the forest
        max_depth : int or None, default=None
            Maximum depth of the trees
        random_state : int, default=42
            Random state for reproducibility
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        
        self.rf_classifier = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1
        )
    
    def fit(self, X, y):
        """Fit the Random Forest classifier"""
        print("Training Vanilla Random Forest Classifier...")
        print(f"Input shape: {X.shape}")
        print(f"Number of classes: {len(np.unique(y))}")
        
        # Train the classifier
        self.rf_classifier.fit(X, y)
        
        # Store classes for prediction
        self.classes_ = self.rf_classifier.classes_
        
        return self
    
    def predict(self, X):
        """Make predictions using the trained classifier"""
        return self.rf_classifier.predict(X)
    
    def predict_proba(self, X):
        """Get prediction probabilities"""
        return self.rf_classifier.predict_proba(X)
    
    def score(self, X, y):
        """Return the mean accuracy on the given test data and labels"""
        return self.rf_classifier.score(X, y)
    
    def evaluate(self, X_test, y_test, class_names=None):
        """Comprehensive evaluation of the model"""
        predictions = self.predict(X_test)
        probabilities = self.predict_proba(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, predictions)
        
        print(f"\nVanilla Random Forest Evaluation Results:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"\nClassification Report:")
        print(classification_report(y_test, predictions, target_names=class_names))
        
        print(f"\nConfusion Matrix:")
        cm = confusion_matrix(y_test, predictions)
        print(cm)
        
        # Feature importance
        if hasattr(self.rf_classifier, 'feature_importances_'):
            print(f"\nRandom Forest Feature Importances (top 10):")
            importances = self.rf_classifier.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            for i in range(min(10, len(importances))):
                print(f"Feature {indices[i]}: {importances[indices[i]]:.4f}")
        
        return {
            'accuracy': accuracy,
            'predictions': predictions,
            'probabilities': probabilities,
            'confusion_matrix': cm
        }
    
    def save_model(self, filepath):
        """Save the trained model"""
        # os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self.rf_classifier, filepath)
        print(f"Vanilla RF Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a pre-trained model"""
        self.rf_classifier = joblib.load(filepath)
        self.classes_ = self.rf_classifier.classes_
        print(f"Vanilla RF Model loaded from {filepath}")
        return self
    
    def get_model_info(self):
        """Get information about the trained model"""
        info = {
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'random_state': self.random_state,
            'n_classes': len(self.classes_) if hasattr(self, 'classes_') else None,
            'classes': self.classes_.tolist() if hasattr(self, 'classes_') else None
        }
        return info