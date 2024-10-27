import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, roc_curve, auc
from lightgbm import LGBMClassifier
from scipy.stats import randint as sp_randint
from sklearn.utils import resample
import shap

# Load data
def load_data(filepath):
    df = pd.read_csv(filepath)
    X = df.drop("Grade", axis=1)
    y = df["Grade"].copy()
    return train_test_split(X, y, test_size=0.20, random_state=13)

# Train classifier and print accuracy
def train_lgbm(X_train, y_train, X_test, y_test):
    clf = LGBMClassifier(random_state=42, verbose=-1)
    clf.fit(X_train, y_train)
    print("LGBM Classifier Test Score:", clf.score(X_test, y_test))
    return clf

# Evaluate multiple classifiers
def evaluate_classifiers(X_train, y_train, X_test, y_test):
    classifiers = [
        RandomForestClassifier(random_state=42),
        GaussianNB(),
        DecisionTreeClassifier(),
        SVC(C=10, probability=True),  # Enable probability for ROC curve
        KNeighborsClassifier()
    ]
    for clf in classifiers:
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        print(f"\n{clf.__class__.__name__} Classification Report:\n", classification_report(y_test, y_pred, digits=4))
    return classifiers

# Perform randomized hyperparameter search
def hyperparameter_search(clf, X_train, y_train):
    param_dist = {
        'n_estimators': sp_randint(50, 500),
        'max_depth': sp_randint(3, 15),
        'learning_rate': np.linspace(0.01, 0.3, 100),
        'subsample': np.linspace(0.2, 1.0, 100),
        'colsample_bytree': np.linspace(0.2, 1.0, 100),
    }
    random_search = RandomizedSearchCV(
        clf, param_distributions=param_dist, n_iter=100, scoring='accuracy', cv=10, n_jobs=-1, random_state=42
    )
    random_search.fit(X_train, y_train)
    return random_search

# Explain predictions with SHAP
def explain_shap(model, X_test):
    explainer = shap.Explainer(model.predict, X_test)
    shap_values = explainer(X_test)
    shap.plots.bar(shap_values, max_display=6, show=False)
    plt.savefig("figure_2.svg", format='svg')  # Save SHAP bar plot
    plt.close()
    
    shap.plots.beeswarm(shap_values, max_display=6, show=False)
    plt.savefig("figure_3.svg", format='svg')  # Save SHAP beeswarm plot
    plt.close()

# Plot ROC curves with confidence intervals
def plot_roc_with_ci(classifiers, classifier_names, X_train, y_train, X_test, y_test, n_bootstraps=1000):
    plt.figure(figsize=(10, 5))

    for clf, name in zip(classifiers, classifier_names):
        # Fit the classifier
        clf.fit(X_train, y_train)

        # Get the predicted scores
        y_score = clf.decision_function(X_test) if hasattr(clf, "decision_function") else clf.predict_proba(X_test)[:, 1]

        # Calculate ROC curve
        fpr, tpr, _ = roc_curve(y_test, y_score)
        roc_auc = auc(fpr, tpr)

        # Bootstrap for confidence intervals
        bootstrapped_aucs = []
        for _ in range(n_bootstraps):
            # Resample the test set
            X_bs, y_bs = resample(X_test, y_test)
            y_score_bs = clf.decision_function(X_bs) if hasattr(clf, "decision_function") else clf.predict_proba(X_bs)[:, 1]

            # Calculate ROC curve and AUC for bootstrap sample
            fpr_bs, tpr_bs, _ = roc_curve(y_bs, y_score_bs)
            bootstrapped_aucs.append(auc(fpr_bs, tpr_bs))

        # Calculate the confidence intervals
        ci_lower, ci_upper = np.percentile(bootstrapped_aucs, [2.5, 97.5])

        # Plot ROC curve
        plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f}, 95% CI = [{ci_lower:.2f}, {ci_upper:.2f}])')
        
        # Fill area for confidence intervals
        plt.fill_between(fpr, 
                         np.interp(fpr, fpr, tpr) - (ci_upper - roc_auc), 
                         np.interp(fpr, fpr, tpr) + (ci_upper - roc_auc), 
                         alpha=0.1)

    # Plot the diagonal line
    plt.plot([0, 1], [0, 1], 'k--', lw=2)  # Diagonal line
    plt.xlim([-0.02, 1.0])
    plt.ylim([0.0, 1.02])
    plt.xlabel('False Positive Rate', fontsize=13)
    plt.ylabel('True Positive Rate', fontsize=13)
    plt.legend(loc="lower right", fontsize=12)
    plt.grid()
    plt.tight_layout()
    plt.savefig("roc_with_ci.svg", format='svg')  # Save ROC curve figure
    plt.close()

# Main workflow
if __name__ == "__main__":
    filepath = "datasets/TCGA_InfoWithGrade.csv"
    X_train, X_test, y_train, y_test = load_data(filepath)

    # LightGBM Model
    lgbm_clf = train_lgbm(X_train, y_train, X_test, y_test)

    # Evaluate other classifiers
    classifiers = evaluate_classifiers(X_train, y_train, X_test, y_test)
    classifier_names = [clf.__class__.__name__ for clf in classifiers]

    # Hyperparameter search on LGBM
    optimized_model = hyperparameter_search(lgbm_clf, X_train, y_train)
    print("\nOptimized Model Report:\n", classification_report(y_test, optimized_model.predict(X_test), digits=4))

    # # SHAP Explanation
    # explain_shap(optimized_model, X_test)

    # ROC Plot with CI for classifiers
    plot_roc_with_ci(classifiers, classifier_names, X_train, y_train, X_test, y_test)
    
    print("DONE")
