from tensorflow.keras.datasets import reuters
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

(X_train, y_train), (X_test, y_test) = reuters.load_data(num_words=None, test_split=0.2)

print(y_train[1])