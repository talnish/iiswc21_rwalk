import sklearn
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import numpy as np
import scikitplot as skplt
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.preprocessing import OneHotEncoder

# Uncomment last few lines in cpu-rwalk/nodeclass_classifier.h to print these files...
test_label = np.genfromtxt("out_test_label.txt", delimiter=' ')
predicted_label_prob = np.genfromtxt("out_test_prob.txt", delimiter=' ')

y_test = test_label[0]
predicted = test_label[1]

print('--------------------')
accuracy = accuracy_score(y_test, predicted)
print('Global accuracy: {}'.format(accuracy))

print('--------------------')
precision, recall, fscore, _ = score(y_test, predicted, beta=1, average='macro')
print('Macro precision: {}'.format(precision))
print('Macro recall: {}'.format(recall))
print('Macro f1 score: {}'.format(fscore))

print('--------------------')
precision, recall, fscore, _ = score(y_test, predicted, beta=1, average='micro')
print('Micro precision: {}'.format(precision))
print('Micro recall: {}'.format(recall))
print('Micro f1 score: {}'.format(fscore))

print('--------------------')

one_hot_encoder = OneHotEncoder(sparse=False)
y_true_label_exp = one_hot_encoder.fit_transform(y_test.reshape(-1, 1))

AUC = metrics.roc_auc_score(y_true=y_true_label_exp, y_score=predicted_label_prob)
print("AUC:", AUC)
print('--------------------')

predicted = np.genfromtxt("out_test_prob.txt", delimiter=' ')
skplt.metrics.plot_roc_curve(y_test, predicted)
plt.savefig("auc.jpg")
