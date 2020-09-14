import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, classification_report, average_precision_score, precision_recall_curve
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

# training model
k_means = KMeans(n_clusters=2, init='k-means++', max_iter=500)
predictions = k_means.fit_predict(samples)

# classification report
print('\nClassification report:\n', classification_report(labels, predictions))

# plotting heatmap
confusion_matrix = confusion_matrix(labels, predictions)
df_cm = pd.DataFrame(confusion_matrix)
sn.heatmap(df_cm, annot=True, yticklabels=2, fmt='g')
plt.savefig('heatmap.svg')
plt.clf()

# plotting precision-recall curve
average_precision = average_precision_score(labels, predictions)
precision, recall, _ = precision_recall_curve(labels, predictions)
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