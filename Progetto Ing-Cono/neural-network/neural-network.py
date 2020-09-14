import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, average_precision_score, precision_recall_curve, roc_curve, roc_auc_score
from inspect import signature
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from keras.layers import Dense
from keras import models

columns_to_drop = ["ID", "ZIP Code"]
categorical_columns = ["Family", "Education", "Personal Loan", "Securities Account", "CD Account", "Online", "CreditCard"]
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

def build_model():
    network = Sequential()
    network.add(Dense(32, input_shape=(14,), activation='relu'))
    network.add(Dense(1, activation='sigmoid'))
    network.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['acc'])

    return network

# getting random balanced data
samples, labels = get_rndm_balanced_data()

# splitting train and test samples
train_samples, test_samples, train_labels, test_labels = train_test_split(samples, labels, test_size=0.20)

# building and training model
epochs = 100
model = build_model()
history = model.fit(train_samples, train_labels, epochs=epochs, batch_size=20)

# plotting the training and validation loss
history_dictionary = history.history
loss_values = history_dictionary['loss']
epochs_range = range(1, epochs + 1)
plt.plot(epochs_range, loss_values, 'b', label='Training loss')
plt.title('Training loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('training-validation-loss.svg')
plt.clf()

# plotting the training and validation accuracy
acc_values = history_dictionary['acc']
plt.plot(epochs_range, acc_values, 'b', label='Training accuracy')
plt.title('Training accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('training-accuracy.svg')
plt.clf()

# calculating predictions
predictions = model.predict(test_samples)
rounded_predictions = [round(x[0]) for x in predictions]

# classification report
print('\nClassification report:\n', classification_report(test_labels, rounded_predictions))

# plotting heatmap
confusion_matrix = confusion_matrix(test_labels, rounded_predictions)
df_cm = pd.DataFrame(confusion_matrix)
sn.heatmap(df_cm, annot=True, yticklabels=2, fmt='g')
plt.savefig('heatmap.svg')
plt.clf()

# plotting precision-recall
average_precision = average_precision_score(test_labels, rounded_predictions)
precision, recall, _ = precision_recall_curve(test_labels, rounded_predictions)
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
probs = model.predict_proba(test_samples)

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
# getting random balanced data
samples, labels = get_rndm_balanced_data()

# model with cross validation technique
cv_model = KerasClassifier(build_fn=build_model, epochs=epochs, batch_size=20, verbose=0)
cv_scores = cross_val_score(cv_model, samples, labels, cv=10)
print('cv_scores mean: {}'.format(np.mean(cv_scores)))
print('cv_score variance: {}'.format(np.var(cv_scores)))
print('cv_score dev standard: {}'.format(np.std(cv_scores)))

# cv model variance and standard deviation
data = {'variance': np.var(cv_scores), 'standard dev': np.std(cv_scores)}
names = list(data.keys())
values = list(data.values())
fig, axs = plt.subplots(1, 1, figsize=(6, 3), sharey=True)
axs.bar(names, values)
fig.savefig('cv-var-std-dev.svg')
plt.clf()