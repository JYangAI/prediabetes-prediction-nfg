from sklearn.metrics import precision_recall_curve, auc
import matplotlib.pyplot as plt
from joblib import load
from collections import Counter


def draw_pr_curve(model_path, x_test, y_test, name):
    print(f'Test dataset class distribution: {Counter(y_test)}')
    model = load(model_path)
    y_prob = model.predict_proba(x_test)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_test, y_prob)

    pr_auc = auc(recall, precision)
    print(f"PR-AUC: {pr_auc:.4f}")
    plt.plot(recall, precision, label=f'{name} (PR-AUC = {pr_auc:.4f})')

    return name
