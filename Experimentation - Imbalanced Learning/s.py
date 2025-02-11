import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline
from sklearn.datasets import make_classification
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns  # for better visualization
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import BorderlineSMOTE # Changed the oversampling Technique
from sklearn.model_selection import GridSearchCV # For Parameter Optimization
from sklearn.decomposition import PCA

# 1. Normal Dataset, Random Forest Classifier
def normal_rf():
    # Generate a balanced dataset
    X, y = make_classification(n_samples=1000, n_features=20,
                               n_informative=15, n_redundant=5,
                               random_state=42, weights=[0.5, 0.5])  # balanced

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    # Evaluate
    y_pred = rf.predict(X_test)
    print("\nNormal Dataset - Random Forest:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    # Display Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix - Normal RF")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()


# 2. Normal Dataset, Random Forest + Deep Belief Network
def normal_dbn_rf():
    # Generate a balanced dataset
    X, y = make_classification(n_samples=1000, n_features=20,
                               n_informative=15, n_redundant=5,
                               random_state=42, weights=[0.5, 0.5])  # balanced

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create DBN + RF pipeline
    dbn_rf = Pipeline([
        ('rbm', BernoulliRBM(n_components=10, learning_rate=0.06, n_iter=10, random_state=42)),
        ('rf', RandomForestClassifier(n_estimators=100, random_state=42))
    ])

    # Train
    dbn_rf.fit(X_train, y_train)

    # Evaluate
    y_pred = dbn_rf.predict(X_test)
    print("\nNormal Dataset - DBN + Random Forest:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    # Display Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix - Normal DBN + RF")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()


# 3. Imbalanced Dataset, Random Forest (Before Balancing)
def imbalanced_rf():
    # Generate an imbalanced dataset
    X, y = make_classification(n_samples=1000, n_features=20,
                               n_informative=15, n_redundant=5,
                               random_state=42, weights=[0.9, 0.1])  # Imbalanced

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Random Forest (without balancing)
    rf = RandomForestClassifier(n_estimators=100, random_state=42)  # No balancing
    rf.fit(X_train, y_train)

    # Evaluate
    y_pred = rf.predict(X_test)
    print("\nImbalanced Dataset - Random Forest (Without Balancing):")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    # Display Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix - Imbalanced RF (No Balancing)")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()


# 4. Imbalanced Dataset, Random Forest + Deep Belief Network (Oversampling with SMOTE)
def imbalanced_dbn_rf():
    # Generate an imbalanced dataset
    X, y = make_classification(n_samples=1000, n_features=20,
                               n_informative=15, n_redundant=5,
                               random_state=42, weights=[0.9, 0.1])  # Imbalanced

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Oversample with SMOTE
    if SMOTE is not None:
        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    else:
        print("SMOTE is not available. Please install imbalanced-learn.")
        return

    # Create DBN + RF pipeline
    dbn_rf = Pipeline([
        ('rbm', BernoulliRBM(n_components=10, learning_rate=0.06, n_iter=10, random_state=42)),
        ('rf', RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')) # IMPORTANT
    ])

    # Train on the *resampled* data
    dbn_rf.fit(X_train_resampled, y_train_resampled)

    # Evaluate
    y_pred = dbn_rf.predict(X_test)
    print("\nImbalanced Dataset - DBN + Random Forest (Oversampling with SMOTE):")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    # Display Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix - Imbalanced DBN + RF (SMOTE)")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()

# 5. Imbalanced Dataset, Random Forest + Deep Belief Network (Oversampling with BorderlineSMOTE)
def imbalanced_dbn_rf_borderline():
    # Generate an imbalanced dataset
    X, y = make_classification(n_samples=5000, n_features=60,
                               n_informative=45, n_redundant=15,
                               n_classes=4,
                               random_state=42, weights=[0.4, 0.4, 0.1, 0.1])  # Imbalanced

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Oversample with BorderlineSMOTE
    smote = BorderlineSMOTE(random_state=42) # CHANGED HERE
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)


    # Create DBN + RF pipeline
    dbn_rf = Pipeline([
        ('rbm', BernoulliRBM(random_state=42)), # Removed the parameters and just define the basic functions
        ('rf', RandomForestClassifier(random_state=42)) #Removed the parameters and just define the basic functions
    ])

    # Parameter Grid for Tuning - ADJUST TO THE AVAILABLE COMPUTATION
    param_grid = {
        'rbm__n_components': [5, 10, 15], # Hidden Units
        'rbm__learning_rate': [0.05, 0.1],
        'rf__n_estimators': [50, 100, 200],
        'rf__class_weight': ['balanced', 'balanced_subsample'] # Used one of the class weights
    }

    # Grid Search for Parameter Optimization
    grid_search = GridSearchCV(dbn_rf, param_grid, cv=3, scoring='f1_weighted') # Using F1-Weighted as Metric for best parameters
    grid_search.fit(X_train_resampled, y_train_resampled)

    # Best Model from Grid Search
    best_dbn_rf = grid_search.best_estimator_

    # Evaluate
    y_pred = best_dbn_rf.predict(X_test)
    print("\nImbalanced Dataset - DBN + Random Forest (Oversampling with BorderlineSMOTE):")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    # Display Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix - Imbalanced DBN + RF (BorderlineSMOTE)")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()

# 6. Imbalanced Dataset, PCA + Random Forest + Deep Belief Network (Oversampling with BorderlineSMOTE)
def imbalanced_pca_dbn_rf():
    # Generate an imbalanced dataset
    X, y = make_classification(n_samples=5000, n_features=60,
                               n_informative=45, n_redundant=15,
                               n_classes=4,
                               random_state=42, weights=[0.4, 0.4, 0.1, 0.1])  # Imbalanced

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Oversample with BorderlineSMOTE
    smote = BorderlineSMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    # X_train_resampled, y_train_resampled = X_train, y_train

    # Create Pipeline including PCA before DBN
    pca_dbn_rf = Pipeline([
        # ('pca', PCA(n_components=10)),  # Reduce to 10 components
        # ('rbm', BernoulliRBM(n_components=5, learning_rate=0.06, n_iter=10, random_state=42)),
        ('rf', RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'))
    ])

    # Train
    pca_dbn_rf.fit(X_train_resampled, y_train_resampled)

    # Evaluate
    y_pred = pca_dbn_rf.predict(X_test)
    print("\nImbalanced Dataset - PCA + DBN + Random Forest (Oversampling with BorderlineSMOTE):")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    # Display Confusion Matrix (omitted for brevity)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix - Imbalanced Dataset - PCA + DBN + Random Forest (Oversampling with BorderlineSMOTE)")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()

def imbalanced_dbn_rf_2():
    # Generate an imbalanced dataset (as before)
    X, y = make_classification(n_samples=5000, n_features=60,
                               n_informative=45, n_redundant=15,
                               n_classes=4,
                               random_state=42, weights=[0.4, 0.4, 0.1, 0.1])  # Imbalanced

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Oversample with BorderlineSMOTE (as before)
    smote = BorderlineSMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    # --- DBN as Feature Extractor ---
    # 1. Train the DBN on the resampled training data
    rbm = BernoulliRBM(n_components=200, learning_rate=0.06, n_iter=10, random_state=42) # Adjust n_components as needed
    rbm.fit(X_train_resampled)

    # 2. Transform the training and testing data using the trained DBN
    X_train_transformed = rbm.transform(X_train_resampled)
    X_test_transformed = rbm.transform(X_test)

    # 3. PCA (Optional: Use PCA after the RBM - you said you might still want to test it. Adjust n_components if needed
    # pca = PCA(n_components = 30)
    # X_train_transformed = pca.fit_transform(X_train_transformed)
    # X_test_transformed = pca.transform(X_test_transformed)

    # --- Random Forest Classifier ---
    # Create the Random Forest (now trained on DBN-extracted features)
    rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')

    # Train the Random Forest on the transformed data
    rf.fit(X_train_transformed, y_train_resampled)

    # Make predictions on the transformed test data
    y_pred = rf.predict(X_test_transformed)

    # --- Evaluate the model ---
    print("\nImbalanced Dataset - DBN as Feature Extractor + Random Forest (Oversampling with BorderlineSMOTE):")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    # Display Confusion Matrix (as before)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix - DBN as Feature Extractor + RF")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()

def imbalanced_stacked_dbn_rf():
    # Generate an imbalanced dataset
    X, y = make_classification(n_samples=5000, n_features=60,
                               n_informative=45, n_redundant=15,
                               n_classes=4,
                               random_state=42, weights=[0.4, 0.4, 0.1, 0.1])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Oversample with BorderlineSMOTE
    smote = BorderlineSMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    # --- Stacked DBN as Feature Extractor ---
    # 1. Define the DBN architecture (number of layers, number of components in each layer)
    n_rbm_components = [40, 20, 10]  # Example: 3 layers with 40, 20, and 10 components respectively. Adjust this!

    # 2. Create a list to store the RBM layers
    rbm_layers = []

    # 3. Train each RBM layer sequentially
    X_train_transformed = X_train_resampled.copy()  # Start with the original training data
    for n_components in n_rbm_components:
        rbm = BernoulliRBM(n_components=n_components, learning_rate=0.06, n_iter=10, random_state=42)
        rbm.fit(X_train_transformed)  # Train the RBM on the output of the previous layer
        rbm_layers.append(rbm)  # Store the trained RBM layer
        X_train_transformed = rbm.transform(X_train_transformed)  # Transform the training data for the next layer

    # 4. Transform the test data using the trained DBN layers
    X_test_transformed = X_test.copy()
    for rbm in rbm_layers:
        X_test_transformed = rbm.transform(X_test_transformed)

    # --- Random Forest Classifier ---
    # Create the Random Forest (trained on DBN-extracted features)
    rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')

    # Train the Random Forest on the transformed data
    rf.fit(X_train_transformed, y_train_resampled)

    # Make predictions on the transformed test data
    y_pred = rf.predict(X_test_transformed)

    # --- Evaluate the model ---
    print("\nImbalanced Dataset - Stacked DBN as Feature Extractor + Random Forest (Oversampling with BorderlineSMOTE):")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    # Display Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix - Stacked DBN as Feature Extractor + RF")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()

# Run the functions
# normal_rf()
# normal_dbn_rf()
# imbalanced_rf()
# imbalanced_dbn_rf()
# imbalanced_dbn_rf_borderline()
# imbalanced_pca_dbn_rf()
# imbalanced_dbn_rf_2()
imbalanced_stacked_dbn_rf()