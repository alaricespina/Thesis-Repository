import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import os
from dbn import DeepBeliefNetwork


class DBNRandomForestClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, dbn_hidden_layers=[100, 50], dbn_learning_rate=0.1, 
                 dbn_epochs=100, dbn_batch_size=32, rf_n_estimators=100, 
                 rf_max_depth=None, rf_random_state=42):
        
        # DBN parameters
        self.dbn_hidden_layers = dbn_hidden_layers
        self.dbn_learning_rate = dbn_learning_rate
        self.dbn_epochs = dbn_epochs
        self.dbn_batch_size = dbn_batch_size
        
        # Random Forest parameters
        self.rf_n_estimators = rf_n_estimators
        self.rf_max_depth = rf_max_depth
        self.rf_random_state = rf_random_state
        
        # Initialize components
        self.dbn = DeepBeliefNetwork(
            hidden_layers=dbn_hidden_layers,
            learning_rate=dbn_learning_rate,
            n_epochs=dbn_epochs,
            batch_size=dbn_batch_size
        )
        
        self.rf_classifier = RandomForestClassifier(
            n_estimators=rf_n_estimators,
            max_depth=rf_max_depth,
            random_state=rf_random_state,
            n_jobs=-1
        )
        
        # Pipeline combining DBN and Random Forest
        self.pipeline = Pipeline([
            ('dbn', self.dbn),
            ('rf', self.rf_classifier)
        ])
    
    def fit(self, X, y):
        """Fit the DBN-RF pipeline"""
        print("Training DBN-Random Forest Classifier...")
        print(f"Input shape: {X.shape}")
        print(f"Number of classes: {len(np.unique(y))}")
        
        # Train the pipeline
        self.pipeline.fit(X, y)
        
        # Store classes for prediction
        self.classes_ = self.rf_classifier.classes_
        
        return self
    
    def predict(self, X):
        """Make predictions using the trained pipeline"""
        return self.pipeline.predict(X)
    
    def predict_proba(self, X):
        """Get prediction probabilities"""
        return self.pipeline.predict_proba(X)
    
    def score(self, X, y):
        """Return the mean accuracy on the given test data and labels"""
        return self.pipeline.score(X, y)
    
    def evaluate(self, X_test, y_test, class_names=None):
        """Comprehensive evaluation of the model"""
        predictions = self.predict(X_test)
        probabilities = self.predict_proba(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, predictions)
        
        print(f"\nModel Evaluation Results:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"\nClassification Report:")
        print(classification_report(y_test, predictions, target_names=class_names))
        
        print(f"\nConfusion Matrix:")
        cm = confusion_matrix(y_test, predictions)
        print(cm)
        
        # Feature importance from Random Forest
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
        joblib.dump(self.pipeline, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a pre-trained model"""
        self.pipeline = joblib.load(filepath)
        self.rf_classifier = self.pipeline.named_steps['rf']
        self.dbn = self.pipeline.named_steps['dbn']
        self.classes_ = self.rf_classifier.classes_
        print(f"Model loaded from {filepath}")
        return self
    
    def get_dbn_features(self, X):
        """Extract features from the DBN layers"""
        return self.dbn.transform(X)
    
    def get_model_info(self):
        """Get information about the trained model"""
        info = {
            'dbn_layers': self.dbn_hidden_layers,
            'dbn_learning_rate': self.dbn_learning_rate,
            'dbn_epochs': self.dbn_epochs,
            'rf_n_estimators': self.rf_n_estimators,
            'rf_max_depth': self.rf_max_depth,
            'n_classes': len(self.classes_) if hasattr(self, 'classes_') else None,
            'classes': self.classes_.tolist() if hasattr(self, 'classes_') else None
        }
        return info