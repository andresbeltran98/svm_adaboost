from libsvm.svmutil import *
from libsvm.commonutil import *

y, x = svm_read_problem('DogsVsCats.train')

# 10-fold cross validation
print('10-fold Linear Kernel')
m = svm_train(y, x, '-t 0 -v 10 -q')

print('\n10-fold Polynomial Kernel')
m = svm_train(y, x, '-t 1 -d 5 -v 10 -q')

# Training accuracy
print('\nLinear Kernel - Training accuracy')
m = svm_train(y, x, '-t 0 -q')
_, p_acc, _ = svm_predict(y, x, m)

print('\nPolynomial Kernel - Training accuracy')
m = svm_train(y, x, '-t 1 -d 5 -q')
_, p_acc, _ = svm_predict(y, x, m)

y_test, x_test = svm_read_problem('DogsVsCats.test')

# Test accuracy
print('\nLinear Kernel - Test accuracy')
m = svm_train(y, x, '-t 0 -q')
_, p_acc_test, _ = svm_predict(y_test, x_test, m)

print('\nPolynomial Kernel - Test accuracy')
m = svm_train(y, x, '-t 1 -d 5 -q')
_, p_acc_test, _ = svm_predict(y_test, x_test, m)

