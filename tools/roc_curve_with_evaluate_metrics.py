import numpy as np
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, \
    confusion_matrix, roc_auc_score, matthews_corrcoef, roc_curve
import matplotlib.pyplot as plt
from joblib import load
from collections import Counter


def evaluate_model(model_path, x_test, y_test, name):
    print(f'Test dataset class distribution: {Counter(y_test)}')
    model = load(model_path)
    y_pred = model.predict(x_test)

    print("Classification report:")
    print(classification_report(y_test, y_pred))

    accuracy = accuracy_score(y_test, y_pred)

    y_prob = model.predict_proba(x_test)[:, 1]

    roc_auc = roc_auc_score(y_test, y_prob)

    precision = precision_score(y_test, y_pred)

    recall = recall_score(y_test, y_pred)

    f1 = f1_score(y_test, y_pred)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    fpr = fp / (fp + tn)
    fnr = fn / (fn + tp)
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    ppv = tp / (tp + fp)
    npv = tn / (tn + fn)

    mcc = matthews_corrcoef(y_test, y_pred)

    n_bootstrap = 1000
    accuracy_bootstraps = []
    roc_auc_bootstraps = []
    precision_bootstraps = []
    recall_bootstraps = []
    f1_bootstraps = []
    fpr_bootstraps = []
    fnr_bootstraps = []
    sensitivity_bootstraps = []
    specificity_bootstraps = []
    mcc_bootstraps = []
    ppv_bootstraps = []
    npv_bootstraps = []

    for _ in range(n_bootstrap):
        indices = np.random.choice(len(x_test), size=len(x_test), replace=True)
        boot_x_test = x_test.iloc[indices]
        boot_y_test = y_test.iloc[indices]

        boot_y_pred = model.predict(boot_x_test)
        boot_y_prob = model.predict_proba(boot_x_test)[:, 1]

        boot_accuracy = accuracy_score(boot_y_test, boot_y_pred)
        accuracy_bootstraps.append(boot_accuracy)

        boot_roc_auc = roc_auc_score(boot_y_test, boot_y_prob)
        roc_auc_bootstraps.append(boot_roc_auc)

        boot_precision = precision_score(boot_y_test, boot_y_pred)
        precision_bootstraps.append(boot_precision)

        boot_recall = recall_score(boot_y_test, boot_y_pred)
        recall_bootstraps.append(boot_recall)

        boot_f1 = f1_score(boot_y_test, boot_y_pred)
        f1_bootstraps.append(boot_f1)

        boot_tn, boot_fp, boot_fn, boot_tp = confusion_matrix(boot_y_test, boot_y_pred).ravel()
        boot_fpr = boot_fp / (boot_fp + boot_tn)
        fpr_bootstraps.append(boot_fpr)

        boot_fnr = boot_fn / (boot_fn + boot_tp)
        fnr_bootstraps.append(boot_fnr)

        boot_ppv = boot_tp / (boot_tp + boot_fp)
        ppv_bootstraps.append(boot_ppv)

        boot_npv = boot_tn / (boot_tn + boot_fn)
        npv_bootstraps.append(boot_npv)

        boot_sensitivity = boot_tp / (boot_tp + boot_fn)
        sensitivity_bootstraps.append(boot_sensitivity)

        boot_specificity = boot_tn / (boot_tn + boot_fp)
        specificity_bootstraps.append(boot_specificity)

        boot_mcc = matthews_corrcoef(boot_y_test, boot_y_pred)
        mcc_bootstraps.append(boot_mcc)

    accuracy_lower = np.percentile(accuracy_bootstraps, 2.5)
    accuracy_upper = np.percentile(accuracy_bootstraps, 97.5)
    roc_auc_lower = np.percentile(roc_auc_bootstraps, 2.5)
    roc_auc_upper = np.percentile(roc_auc_bootstraps, 97.5)
    precision_lower = np.percentile(precision_bootstraps, 2.5)
    precision_upper = np.percentile(precision_bootstraps, 97.5)
    recall_lower = np.percentile(recall_bootstraps, 2.5)
    recall_upper = np.percentile(recall_bootstraps, 97.5)
    f1_lower = np.percentile(f1_bootstraps, 2.5)
    f1_upper = np.percentile(f1_bootstraps, 97.5)
    fpr_lower = np.percentile(fpr_bootstraps, 2.5)
    fpr_upper = np.percentile(fpr_bootstraps, 97.5)
    fnr_lower = np.percentile(fnr_bootstraps, 2.5)
    fnr_upper = np.percentile(fnr_bootstraps, 97.5)
    ppv_lower = np.percentile(ppv_bootstraps, 2.5)
    ppv_upper = np.percentile(ppv_bootstraps, 97.5)
    npv_lower = np.percentile(npv_bootstraps, 2.5)
    npv_upper = np.percentile(npv_bootstraps, 97.5)
    sensitivity_lower = np.percentile(sensitivity_bootstraps, 2.5)
    sensitivity_upper = np.percentile(sensitivity_bootstraps, 97.5)
    specificity_lower = np.percentile(specificity_bootstraps, 2.5)
    specificity_upper = np.percentile(specificity_bootstraps, 97.5)
    mcc_lower = np.percentile(mcc_bootstraps, 2.5)
    mcc_upper = np.percentile(mcc_bootstraps, 97.5)

    metrics_dict = {
        'Test AUC': roc_auc,
        'Test AUC 95%CI lower bound': roc_auc_lower,
        'Test AUC 95%CI upper bound': roc_auc_upper,
        'Test accuracy': accuracy,
        'Test accuracy 95%CI lower bound': accuracy_lower,
        'Test accuracy 95%CI upper bound': accuracy_upper,
        'Test precision': precision,
        'Test precision 95%CI lower bound': precision_lower,
        'Test precision 95%CI upper bound': precision_upper,
        'Test recall': recall,
        'Test recall 95%CI lower bound': recall_lower,
        'Test recall 95%CI upper bound': recall_upper,
        'Test FPR': fpr,
        'Test FPR 95%CI lower bound': fpr_lower,
        'Test FPR 95%CI upper bound': fpr_upper,
        'Test FNR': fnr,
        'Test FNR 95%CI lower bound': fnr_lower,
        'Test FNR 95%CI upper bound': fnr_upper,
        'Test Sensitivity': sensitivity,
        'Test Sensitivity 95%CI lower bound': sensitivity_lower,
        'Test Sensitivity 95%CI upper bound': sensitivity_upper,
        'Test Specificity': specificity,
        'Test Specificity 95%CI lower bound': specificity_lower,
        'Test Specificity 95%CI upper bound': specificity_upper,
        'Test PPV': ppv,
        'Test PPV 95%CI lower bound': ppv_lower,
        'Test PPV 95%CI upper bound': ppv_upper,
        'Test NPV': npv,
        'Test NPV 95%CI lower bound': npv_lower,
        'Test NPV 95%CI upper bound': npv_upper,
        'Test F1 score': f1,
        'Test F1 score 95%CI lower bound': f1_lower,
        'Test F1 score 95%CI upper bound': f1_upper,
        'Test MCC': mcc,
        'Test MCC 95%CI lower bound': mcc_lower,
        'Test MCC 95%CI upper bound': mcc_upper,
    }
    print(metrics_dict)

    fpr_roc, tpr_roc, _ = roc_curve(y_test, y_prob)
    plt.plot(fpr_roc, tpr_roc, label=f'{name} (AUC = {roc_auc:.4f})')

    return metrics_dict, name
