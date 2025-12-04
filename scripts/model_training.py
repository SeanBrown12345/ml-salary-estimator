import click
import os
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import make_column_transformer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, cross_val_predict, RandomizedSearchCV
from sklearn.metrics import make_scorer, f1_score, precision_recall_curve, classification_report, ConfusionMatrixDisplay
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import loguniform
import matplotlib.pyplot as plt

@click.command()
@click.option('-tr', '--train', type=str, required=True, help="path to the data file holding the training data")
@click.option('-te', '--test', type=str, required=True, help="path to the data file holding the test data")
@click.option('-o', '--output_path', type=str, required=True, help="Path to directory where the results will be saved to")
def main(train, test, output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    if not os.path.exists(os.path.join(output_path, 'tables')):
        os.makedirs(os.path.join(output_path, 'tables'))
    if not os.path.exists(os.path.join(output_path, 'figures')):
        os.makedirs(os.path.join(output_path, 'figures'))

    try:
        train_df = pd.read_csv(train)
        print('Successfully loaded training data')
        test_df = pd.read_csv(test)
        print('Successfully loaded test data')
    except Exception as e:
        print(f'Error when reading files: {e}')
        return(1)
    
    X_train = train_df.drop(columns="income")
    y_train = train_df["income"]
    X_test = test_df.drop(columns="income")
    y_test = test_df["income"]

    numeric_features = ['age', 'capital_gain', 'capital_loss', 'hours_per_week']
    categorical_features = ['workclass', 'marital_status', 'occupation', 'relationship', 'native_country']
    ordinal_features = ['education']
    binary_features = ['sex']
    drop_features = ['fnlwgt', 'education_num', 'race']
    order = ['Preschool', '1st-4th', '5th-6th', '7th-8th', '9th', '10th', 
             '11th', '12th', "HS-grad", "Prof-school", "Assoc-voc", "Assoc-acdm", 
             "Some-college", "Bachelors", 'Masters', 'Doctorate']
    
    ordinal_transformer = make_pipeline(
        SimpleImputer(strategy='most_frequent'),
        OrdinalEncoder(categories=[order], dtype = int, handle_unknown='use_encoded_value', unknown_value=-1)
    )
    binary_transformer = make_pipeline(
        SimpleImputer(strategy='most_frequent'),
        OneHotEncoder(drop="if_binary", dtype=int)
    )
    numeric_transformer = make_pipeline(
        SimpleImputer(strategy='median'),
        StandardScaler()
    )
    categorical_transformer = make_pipeline(
        SimpleImputer(strategy="constant", fill_value="missing"),
        OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    )
    
    
    preprocessor = make_column_transformer(
        (numeric_transformer, numeric_features),
        (ordinal_transformer, ordinal_features),
        (binary_transformer, binary_features),
        (categorical_transformer, categorical_features),
        ('drop', drop_features)
    )

    print('Starting baseline test')

    f1_scorer = make_scorer(f1_score, pos_label='>50K')
    
    cv_results = {}
    
    dummy = DummyClassifier(strategy='constant', constant='>50K')
    dummy_pipe = make_pipeline(preprocessor, dummy)
    cv_results['Dummy'] = cross_val_score(dummy_pipe, X_train, y_train, cv=5, scoring=f1_scorer).mean()

    # logistic regression
    logReg = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=456)
    logReg_pipe = make_pipeline(preprocessor, logReg)
    cv_results['LogisticRegression'] = cross_val_score(logReg_pipe, X_train, y_train, cv=5, scoring=f1_scorer).mean()

    # RBF SVM
    svm = SVC(kernel='rbf', class_weight='balanced', random_state=456)
    svm_pipe = make_pipeline(preprocessor, svm)
    cv_results['SVC'] = cross_val_score(svm_pipe, X_train, y_train, cv=5, scoring=f1_scorer).mean()

    # random forest
    rf = RandomForestClassifier(class_weight='balanced', random_state=456)
    rf_pipe = make_pipeline(preprocessor, rf)
    cv_results['RandomForest'] = cross_val_score(rf_pipe, X_train, y_train, cv=5, scoring=f1_scorer).mean()

    results_df = pd.DataFrame(list(cv_results.items()), columns=['Model', 'Mean f1 score'])
    results_df.to_csv(os.path.join(output_path, 'tables/baseline_comparison.csv'), index=False)
    
    print('Starting hyperparameter tuning')
    # C tuning for best performing model (logistic regression)
    param_dist = {
        "logisticregression__C": loguniform(0.01, 10)
    }
    random_search = RandomizedSearchCV(
        logReg_pipe,
        param_distributions = param_dist, 
        n_iter=50, 
        n_jobs=-1,
        return_train_score=True,
        scoring=f1_scorer,
        cv=5,
        random_state=114514
    )
    random_search.fit(X_train, y_train)

    # threshold tuning
    best_model = random_search.best_estimator_
    pred_y = cross_val_predict(
        best_model, 
        X_train, 
        y_train, 
        cv=5, 
        method="predict_proba"
    )[:, 1]
    pre, rec, thr = precision_recall_curve(y_train, pred_y, pos_label='>50K')
    f1 = 2 * (pre * rec) / (pre + rec)
    best_thr = thr[f1.argmax()]

    tuning_df = pd.DataFrame([random_search.best_params_])
    tuning_df['best_cv_score'] = random_search.best_score_
    tuning_df['best_threshold'] = best_thr
    tuning_df.to_csv(os.path.join(output_path, 'tables/tuning_results.csv'), index=False)

    print('Evaluating on test set')
    best_model.fit(X_train, y_train)
    y_test_pred = best_model.predict_proba(X_test)
    pred_result = (y_test_pred[:,1] >= best_thr)
    y_test_binary = (y_test == '>50K').astype(int)
    report = classification_report(y_test_binary, pred_result)

    report_dict = classification_report(y_test_binary, pred_result, target_names=['<=50k', '>50K'], output_dict=True)
    report_df = pd.DataFrame(report_dict).T
    report_df.to_csv(os.path.join(output_path, 'tables/classification_report.csv'), index_label='metric_category')

    fig, ax = plt.subplots(figsize=(8, 6))
    ConfusionMatrixDisplay.from_predictions(y_test_binary, pred_result, display_labels=['<=50k', '>50K'], ax=ax)
    plt.title("Confusion matrix on test set")
    plt.savefig(os.path.join(output_path, 'figures/confusion_matrix.png'))
    plt.close()

    plt.figure(figsize=(6, 6))
    plt.plot(rec, pre)
    plt.scatter(rec[f1.argmax()], pre[f1.argmax()], color='red', label='Best F1 Score', s=50, zorder=100)
    plt.title(f'Precision-Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    plt.savefig(os.path.join(output_path, 'figures/precision_recall_curve.png'))
    plt.close() 

if __name__ == "__main__":
    main() 
