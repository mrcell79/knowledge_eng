import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, average_precision_score, precision_recall_curve, \
    roc_curve, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from inspect import signature

columns_to_drop = ["ID", "ZIP Code"]
categorical_columns = ["Family", "Education", "Personal Loan", "Securities Account", "CD Account", "Online",
                       "CreditCard"]
numeric_columns = ["Age", "Experience", "Income", "CCAvg", "Mortgage"]


def get_rndm_balanced_data():
    # reading data
    bank = pd.read_csv('../Bank_loan.csv')
    bank.drop(columns_to_drop, axis=1, inplace=True)
    bank = pd.get_dummies(bank, columns=categorical_columns, drop_first=True)

    # splitting train data from test data
    samples = bank.copy().drop("Personal Loan_1", axis=1)
    labels = bank["Personal Loan_1"]

    # handling oversampling data
    sm = SMOTE()

    return sm.fit_sample(samples, labels.ravel())


# getting random balanced data
samples, labels = get_rndm_balanced_data()

# splitting train and test samples
train_samples, test_samples, train_labels, test_labels = train_test_split(samples, labels, test_size=0.20)

""" Discovering the best value for K(num of neighbors)
# Calculating error rates for K values between 1 and 20 over 5 iterates and random train/test sets
iterates = range(1, 5)
k_range = range(1, 15)
errors = []
samples, labels = get_rndm_balanced_data()

for i in iterates:
    train_samples, test_samples, train_labels, test_labels = train_test_split(samples, labels, test_size=0.20)
    for j in k_range:  
        knn = KNeighborsClassifier(n_neighbors = j)
        knn.fit(train_samples, train_labels)
        prediction = knn.predict(test_samples)
        if (i == 1):
            errors.append(np.mean(prediction != test_labels))
        else:
            prev_error = errors[j - 1]
            curr_error = np.mean(prediction != test_labels)
            errors[j - 1] = np.mean([prev_error, curr_error])

# plotting mean error rates in function of K
plt.plot(k_range, errors, color='green', linestyle='dashed', marker='o', markerfacecolor='black')
plt.title('mean error rate in function of K neighbors over 5 iterates')  
plt.xlabel('K value')  
plt.ylabel('mean error')
plt.savefig('mean-errors.svg')
plt.clf()
"""

# building model with 2 neighbors
knn_model = KNeighborsClassifier(n_neighbors=2)
knn_model.fit(train_samples, train_labels)

# calculating predictions on test set
predictions = knn_model.predict(test_samples)

# classification report
print('\nClassification report:\n', classification_report(test_labels, predictions))

# plotting heatmap
confusion_matrix = confusion_matrix(test_labels, predictions)
df_cm = pd.DataFrame(confusion_matrix)
sn.heatmap(df_cm, annot=True, yticklabels=2, fmt='g')
plt.savefig('heatmap.svg')
plt.clf()

# plotting precision-recall graph
average_precision = average_precision_score(test_labels, predictions)
precision, recall, _ = precision_recall_curve(test_labels, predictions)
step_kwargs = ({'step': 'post'}
               if 'step' in signature(plt.fill_between).parameters
               else {})
plt.step(recall, precision, color='b', alpha=0.2, where='post')
plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('Precision-Recall curve: AP = {0:0.2f}'.format(average_precision))
plt.savefig('precision-recall-curve.svg')
plt.clf()

# calculating probabilities
probs = knn_model.predict_proba(test_samples)[:, 1]

# calculating AUC score
auc = roc_auc_score(test_labels, probs)

# plotting roc curve and AUC score
fpr, tpr, thresholds = roc_curve(test_labels, probs)
plt.plot([0, 1], [0, 1], linestyle='--')
plt.plot(fpr, tpr, marker='.')
plt.title('ROC curve: AUC = %.3f' % auc)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.savefig('roc-curve.svg')
plt.clf()

""" CROSS VALIDATION TECHNIQUE """
# fetching random balanced data
samples, labels = get_rndm_balanced_data()

# building cross validation model
cv_knn_model = KNeighborsClassifier(n_neighbors=2)

# collecting cross validation scores
cv_scores = cross_val_score(cv_knn_model, samples, labels, cv=10)
print('cv_scores mean: {}'.format(np.mean(cv_scores)))
print('cv_score variance: {}'.format(np.var(cv_scores)))
print('cv_score dev standard: {}'.format(np.std(cv_scores)))

# plotting cross validation model variance and standard deviation
data = {'variance': np.var(cv_scores), 'standard dev': np.std(cv_scores)}
names = list(data.keys())
values = list(data.values())
fig, axs = plt.subplots(1, 1, figsize=(6, 3), sharey=True)
axs.bar(names, values)
fig.savefig('cv-var-std-dev.svg')
plt.clf()
