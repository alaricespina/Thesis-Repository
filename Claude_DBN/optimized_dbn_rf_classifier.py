import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import GridSearchCV
import joblib
import os
from dbn import DeepBeliefNetwork


class OptimizedDBNRandomForestClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, dbn_hidden_layers=[256, 128, 64], dbn_learning_rate=0.001, 
                 dbn_epochs=150, dbn_batch_size=128, rf_n_estimators=500, 
                 rf_max_depth=20, rf_min_samples_split=5, rf_min_samples_leaf=2,
                 rf_random_state=42, use_ensemble=True):
        
        # Optimized DBN parameters
        self.dbn_hidden_layers = dbn_hidden_layers
        self.dbn_learning_rate = dbn_learning_rate
        self.dbn_epochs = dbn_epochs
        self.dbn_batch_size = dbn_batch_size
        
        # Optimized Random Forest parameters
        self.rf_n_estimators = rf_n_estimators
        self.rf_max_depth = rf_max_depth
        self.rf_min_samples_split = rf_min_samples_split
        self.rf_min_samples_leaf = rf_min_samples_leaf
        self.rf_random_state = rf_random_state
        self.use_ensemble = use_ensemble
        
        # Initialize components
        self.dbn = DeepBeliefNetwork(
            hidden_layers=dbn_hidden_layers,
            learning_rate=dbn_learning_rate,
            n_epochs=dbn_epochs,
            batch_size=dbn_batch_size
        )
        
        if use_ensemble:
            # Create ensemble of classifiers
            self.rf_classifier = RandomForestClassifier(
                n_estimators=rf_n_estimators,
                max_depth=rf_max_depth,
                min_samples_split=rf_min_samples_split,
                min_samples_leaf=rf_min_samples_leaf,
                random_state=rf_random_state,
                n_jobs=-1,
                class_weight='balanced',
                bootstrap=True,
                max_features='sqrt'
            )
            
            self.gb_classifier = GradientBoostingClassifier(
                n_estimators=200,
                max_depth=10,
                learning_rate=0.1,
                random_state=rf_random_state
            )
            
            # Voting classifier combining RF and GB
            self.final_classifier = VotingClassifier(
                estimators=[
                    ('rf', self.rf_classifier),
                    ('gb', self.gb_classifier)
                ],
                voting='soft'
            )
        else:
            self.rf_classifier = RandomForestClassifier(
                n_estimators=rf_n_estimators,
                max_depth=rf_max_depth,
                min_samples_split=rf_min_samples_split,
                min_samples_leaf=rf_min_samples_leaf,
                random_state=rf_random_state,
                n_jobs=-1,
                class_weight='balanced',
                bootstrap=True,
                max_features='sqrt'
            )
            self.final_classifier = self.rf_classifier
        
        # Pipeline combining DBN and classifier
        self.pipeline = Pipeline([
            ('dbn', self.dbn),
            ('classifier', self.final_classifier)
        ])
    
    def fit(self, X, y):
        """Fit the optimized DBN-RF pipeline"""
        print("Training Optimized DBN-Random Forest Classifier...")
        print(f"Input shape: {X.shape}")
        print(f"Number of classes: {len(np.unique(y))}")
        print(f"DBN layers: {self.dbn_hidden_layers}")
        print(f"Using ensemble: {self.use_ensemble}")
        
        # Train the pipeline
        self.pipeline.fit(X, y)
        
        # Store classes for prediction
        self.classes_ = self.final_classifier.classes_
        
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
        
        print(f"\nOptimized DBN-RF Model Evaluation Results:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"\nClassification Report:")
        print(classification_report(y_test, predictions, target_names=class_names))
        
        print(f"\nConfusion Matrix:")
        cm = confusion_matrix(y_test, predictions)
        print(cm)
        
        # Feature importance from Random Forest (if available)
        if hasattr(self.rf_classifier, 'feature_importances_'):
            print(f"\nRandom Forest Feature Importances (top 15):")
            importances = self.rf_classifier.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            for i in range(min(15, len(importances))):
                print(f"Feature {indices[i]}: {importances[indices[i]]:.4f}")
        
        return {
            'accuracy': accuracy,
            'predictions': predictions,
            'probabilities': probabilities,
            'confusion_matrix': cm
        }
    
    def save_model(self, filepath):
        """Save the trained model"""
        joblib.dump(self.pipeline, filepath)
        print(f"Optimized model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a pre-trained model"""
        self.pipeline = joblib.load(filepath)
        if self.use_ensemble:
            self.final_classifier = self.pipeline.named_steps['classifier']
            self.rf_classifier = self.final_classifier.named_estimators_['rf']
        else:
            self.rf_classifier = self.pipeline.named_steps['classifier']
            self.final_classifier = self.rf_classifier
        self.dbn = self.pipeline.named_steps['dbn']
        self.classes_ = self.final_classifier.classes_
        print(f"Optimized model loaded from {filepath}")
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
            'use_ensemble': self.use_ensemble,
            'n_classes': len(self.classes_) if hasattr(self, 'classes_') else None,
            'classes': self.classes_.tolist() if hasattr(self, 'classes_') else None
        }
        return info


class OptimizedVanillaRandomForestClassifier:
    def __init__(self, n_estimators=500, max_depth=20, min_samples_split=5, 
                 min_samples_leaf=2, random_state=42, use_ensemble=True):
        """
        Optimized Vanilla Random Forest Classifier
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.use_ensemble = use_ensemble
        
        if use_ensemble:
            # Create ensemble of classifiers
            self.rf_classifier = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                random_state=random_state,
                n_jobs=-1,
                class_weight='balanced',
                bootstrap=True,
                max_features='sqrt'
            )
            
            self.gb_classifier = GradientBoostingClassifier(
                n_estimators=200,
                max_depth=10,
                learning_rate=0.1,
                random_state=random_state
            )
            
            # Voting classifier combining RF and GB
            self.final_classifier = VotingClassifier(
                estimators=[
                    ('rf', self.rf_classifier),
                    ('gb', self.gb_classifier)
                ],
                voting='soft'
            )
        else:
            self.rf_classifier = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                random_state=random_state,
                n_jobs=-1,
                class_weight='balanced',
                bootstrap=True,
                max_features='sqrt'
            )
            self.final_classifier = self.rf_classifier
    
    def fit(self, X, y):
        """Fit the optimized Random Forest classifier"""
        print("Training Optimized Vanilla Random Forest Classifier...")
        print(f"Input shape: {X.shape}")
        print(f"Number of classes: {len(np.unique(y))}")
        print(f"Using ensemble: {self.use_ensemble}")
        
        # Train the classifier
        self.final_classifier.fit(X, y)
        
        # Store classes for prediction
        self.classes_ = self.final_classifier.classes_
        
        return self
    
    def predict(self, X):
        """Make predictions using the trained classifier"""
        return self.final_classifier.predict(X)
    
    def predict_proba(self, X):
        """Get prediction probabilities"""
        return self.final_classifier.predict_proba(X)
    
    def score(self, X, y):
        """Return the mean accuracy on the given test data and labels"""
        return self.final_classifier.score(X, y)
    
    def evaluate(self, X_test, y_test, class_names=None):
        """Comprehensive evaluation of the model"""
        predictions = self.predict(X_test)
        probabilities = self.predict_proba(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, predictions)
        
        print(f"\nOptimized Vanilla Random Forest Evaluation Results:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"\nClassification Report:")
        print(classification_report(y_test, predictions, target_names=class_names))
        
        print(f"\nConfusion Matrix:")
        cm = confusion_matrix(y_test, predictions)
        print(cm)
        
        # Feature importance
        if hasattr(self.rf_classifier, 'feature_importances_'):
            print(f"\nRandom Forest Feature Importances (top 15):")
            importances = self.rf_classifier.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            for i in range(min(15, len(importances))):
                print(f"Feature {indices[i]}: {importances[indices[i]]:.4f}")
        
        return {
            'accuracy': accuracy,
            'predictions': predictions,
            'probabilities': probabilities,
            'confusion_matrix': cm
        }
    
    def save_model(self, filepath):
        """Save the trained model"""
        joblib.dump(self.final_classifier, filepath)
        print(f"Optimized Vanilla RF Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a pre-trained model"""
        self.final_classifier = joblib.load(filepath)
        if self.use_ensemble:
            self.rf_classifier = self.final_classifier.named_estimators_['rf']
        else:
            self.rf_classifier = self.final_classifier
        self.classes_ = self.final_classifier.classes_
        print(f"Optimized Vanilla RF Model loaded from {filepath}")
        return self
    
    def get_model_info(self):
        """Get information about the trained model"""
        info = {
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'min_samples_split': self.min_samples_split,
            'min_samples_leaf': self.min_samples_leaf,
            'use_ensemble': self.use_ensemble,
            'random_state': self.random_state,
            'n_classes': len(self.classes_) if hasattr(self, 'classes_') else None,
            'classes': self.classes_.tolist() if hasattr(self, 'classes_') else None
        }
        return info