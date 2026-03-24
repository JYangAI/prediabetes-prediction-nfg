import matplotlib.pyplot as plt
import pandas as pd

from tools.calibration_curve import draw_calibration_curve
from tools.scaler import scaler

NHANES_test_df = pd.read_excel('../datasets/NHANES/NHANES_2017_2020_3_test_data_for_prediabetes_prediction.xlsx')
NHANES_test_df = scaler(NHANES_test_df, 'Prediabetes', '../tools/scaler.joblib')

x_NHANES_test = NHANES_test_df.drop(columns='Prediabetes')
y_NHANES_test = NHANES_test_df['Prediabetes']

plt.figure(figsize=(10, 8))
print(draw_calibration_curve('../models/model_files/xgboost_classifier_model.joblib', x_NHANES_test, y_NHANES_test,
                             name='NHANES test'))

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 20
plt.rcParams['axes.titlesize'] = 22
plt.rcParams['axes.labelsize'] = 20
plt.rcParams['xtick.labelsize'] = 20
plt.rcParams['ytick.labelsize'] = 20
plt.rcParams['legend.fontsize'] = 20

plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlim([0.0, 1.05])
plt.ylim([0.0, 1.05])
plt.xlabel('Predicted probability', font={'family': 'Times New Roman'})
plt.ylabel('True probability', font={'family': 'Times New Roman'})
plt.title('Test calibration curve', font={'family': 'Times New Roman'})
plt.yticks(fontproperties='Times New Roman')
plt.xticks(fontproperties='Times New Roman')
plt.legend(loc='lower right')
plt.grid(False)
# plt.savefig('NHANES_test_calibration_curve.png', dpi=600, bbox_inches='tight')
plt.show()
