import numpy as np
from Model_Selection import best_model_result

# coef for age as predictor
b1 = best_model_result.params[1]

# coef for sysBP as predictor
b2 = best_model_result.params[2]

# coef for gender as predictor
b3 = best_model_result.params[3]

odds_ratio_age = np.exp(b1)
print(f"OR for additional 1 year in age = {odds_ratio_age:.4f}")
odds_ratio_sysBP = np.exp(b2*10)
print(f"OR for additional 10 unit in sysBP = {odds_ratio_sysBP:.4f}")
odds_ratio_gender = np.exp(b3)
print(f"OR for male = {odds_ratio_gender:.4f}")

# predictive performance evaluation
model_matrix = best_model_result.pred_table(threshold=0.5)
TP = model_matrix[1][1]
TN = model_matrix[0][0]
FP = model_matrix[0][1]
FN = model_matrix[1][0]
print(f'There are {TP} true positive, {TN} true negative, {FP} false positive, and {FN} false negative')

accuracy = (TP+TN)/(TP+FP+TN+FN)
sensitivity = TP / (TP + FN)
specificity = TN / (TN + FP)
print(f'The model can correctly predict {accuracy*100:.2f}% of the outcomes')
print(f'The model can correctly predict {sensitivity*100:.2f}% of heart disease actually occurring')
print(f'The model can correctly predict {specificity*100:.2f}% of heart disease not occurring')
