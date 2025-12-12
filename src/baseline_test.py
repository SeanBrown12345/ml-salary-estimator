from sklearn.metrics import make_scorer, f1_score
from sklearn.model_selection import cross_val_score
import pandas as pd

def baseline_test(pipelines, X_train, y_train):
    """
    Performs 5 fold cross-validation on a dictionary of sklearn pipelines and report the mean F1 score
    Args:
        pipelines: A dictionary where keys are model names and values are pipeline objects
        X_train: The training data
        y_train: The target values

    Returns:
        pd.DataFrame: A summary DataFrame containing the name of the model and the mean F1 score across folds
    """
    f1_scorer = make_scorer(f1_score, pos_label='>50K')
    
    cv_results = {}
    
    for model_name, pipeline in pipelines.items():
        print(f'Running baseline test for {model_name}... ', end='', flush=True)
        cv_results[model_name] = cross_val_score(pipeline, X_train, y_train, cv=5, scoring=f1_scorer).mean()
        print('Done')
        
    return pd.DataFrame(list(cv_results.items()), columns=['Model', 'Mean f1 score'])
