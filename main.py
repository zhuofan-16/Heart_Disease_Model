from sklearn import datasets

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
iris=datasets.load_iris()
digits = datasets.load_digits()
x=iris.data
y=iris.target
clf=svm.SVC(gamma=0.001,C=100.)
print(clf.fit(digits.data[:-1], digits.target[:-1]).score(digits.data[:-1], digits.target[:-1]))
print(clf.predict(digits.data[-1:]))

