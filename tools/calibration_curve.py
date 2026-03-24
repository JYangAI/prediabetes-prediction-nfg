from sklearn.metrics import classification_report
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
from joblib import load
from collections import Counter


def draw_calibration_curve(model_path, x_test, y_test, name):
    print(f'Test dataset class distribution: {Counter(y_test)}')
    model = load(model_path)
    y_pred = model.predict(x_test)

    print("Classification report:")
    print(classification_report(y_test, y_pred))

    y_prob = model.predict_proba(x_test)[:, 1]

    prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=10)
    plt.plot(prob_pred, prob_true, marker='o', label=f'{name}')

    return name
