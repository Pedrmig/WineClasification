import numpy as np
import wandb
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize

# Load the dataset
wine = load_wine()

# Define features and target variables
X, y = wine.data, wine.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=357)

# hyperparameters
hyperparameters = {
    'learning_rates': [0.01, 0.1, 0.2, 0.25],
    'max_depths': [2, 3, 4, 5],
    'n_estimators': [50, 100, 150],
    'loss_functions': ['log_loss'],
    'subsamples': [0.8, 1.0],
    'min_samples_splits': [2, 4],
    'min_samples_leafs': [1, 2]
}

# Bucle
for lr in hyperparameters['learning_rates']:
    for max_depth in hyperparameters['max_depths']:
        for n_estimator in hyperparameters['n_estimators']:
            for loss_function in hyperparameters['loss_functions']:
                for subsample in hyperparameters['subsamples']:
                    for min_samples_split in hyperparameters['min_samples_splits']:
                        for min_samples_leaf in hyperparameters['min_samples_leafs']:

                            # Initialize wandb
                            experiment_name = f"gbm_lr{lr}_depth{max_depth}_est{n_estimator}_loss{loss_function}_subsample{subsample}_minsplit{min_samples_split}_minleaf{min_samples_leaf}"
                            wandb.init(project="vinitoclasi", name=experiment_name, config=hyperparameters)

                            # Train the model
                            clf = GradientBoostingClassifier(learning_rate=lr, max_depth=max_depth, n_estimators=n_estimator,
                                                             loss=loss_function, subsample=subsample, min_samples_split=min_samples_split,
                                                             min_samples_leaf=min_samples_leaf, random_state=357, validation_fraction=0.1,
                                                             n_iter_no_change=5, tol=0.01)
                            
                            clf.fit(X_train, y_train) 
                            
                            # Make predictions
                            y_pred = clf.predict(X_test) #comentario prueba
                            y_pred_proba = clf.predict_proba(X_test)

                            # Log hyperparameters to wandb
                            wandb.config.learning_rate = lr
                            wandb.config.max_depth = max_depth
                            wandb.config.n_estimators = n_estimator
                            wandb.config.loss_function = loss_function
                            wandb.config.subsample = subsample
                            wandb.config.min_samples_split = min_samples_split
                            wandb.config.min_samples_leaf = min_samples_leaf
                            
                            #  Calculate performance metrics
                            accuracy = accuracy_score(y_test, y_pred)
                            f1_macro = f1_score(y_test, y_pred, average="macro")
                            y_test_bin = label_binarize(y_test, classes=np.unique(y))
                            y_pred_bin = y_pred_proba.reshape(-1, len(np.unique(y)))
                            roc_auc_macro = roc_auc_score(y_test_bin, y_pred_bin, average="macro", multi_class="ovr")

                            # Log metrics to wandb
                            wandb.log({"accuracy": accuracy, "f1_macro": f1_macro, "roc_auc_macro": roc_auc_macro,
                                       "validation_score": clf.train_score_[-1]})
                            
                            # Finish the experiment
                            wandb.finish()
                            
# Cambiar hiperpárametros, test size y comprobar el mejor resultado