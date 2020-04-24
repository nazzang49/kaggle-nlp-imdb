# next-time success_amount prediction model by LSTM (one of the most famous model of RNN)
import pandas as pd
import numpy as np
import matplotlib.pylab as plt

df = pd.read_csv('D:/success_amount_prediction/hourFactData_1.csv', parse_dates=['date_hour_key'], index_col='date_hour_key', delimiter=',', quoting=3)
# check called csv file
print(df.head())

# first of all, let's test this model by only success_amount
# del df['success_count']
# del df['fail_count']
# del df['fail_amount']

print(df.head())
plt.title('success_count_pattern')
plt.plot(df['success_count'])
plt.show()

# split train, test dataset
print(pd.Timestamp('2020-02-25T05'))
split_criteria = pd.Timestamp('2020-02-25T05')

# 3000row = 2020-02-25 05:00:00
train = df.loc[:split_criteria, ['success_amount', 'success_count', 'fail_count', 'fail_amount']]
test = df.loc[split_criteria:, ['success_amount', 'success_count', 'fail_count', 'fail_amount']]

print('================="below is 10 values of train series"=================')
print(train[:10])

# for normalization of value from zero to one
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0, 1))

# rearrange from 0 to 1
train_sc = sc.fit_transform(train)
test_sc = sc.transform(test)

print(train_sc[:10])
print(test_sc[:10])

# series -> dataframe
train_sc_df = pd.DataFrame(train_sc, columns=['success_amount', 'success_count', 'fail_count', 'fail_amount'], index=train.index)
test_sc_df = pd.DataFrame(test_sc, columns=['success_amount', 'success_count', 'fail_count', 'fail_amount'], index=test.index)

print(train_sc_df.head())

# refer to 24 previous data to predict next-hour success_amount
for s in range(4, 28):
    train_sc_df['success_amount_{}'.format(s)] = train_sc_df['success_count'].shift(s)
    test_sc_df['success_amount_{}'.format(s)] = test_sc_df['success_count'].shift(s)

print(train_sc_df.head(25))

# remove all NaN value (axis = 1 -> row / axis = 0 -> col)
X_train = train_sc_df.dropna().drop('success_count', axis=1)
# y_train -> present value
y_train = train_sc_df.dropna()[['success_count']]

X_test = test_sc_df.dropna().drop('success_count', axis=1)
y_test = test_sc_df.dropna()[['success_count']]

print(X_train.head(10))

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

X_train = X_train.values
y_train = y_train.values
X_test = X_test.values
y_test = y_test.values

# (array) 2dim -> 3dim
X_train_t = X_train.reshape(X_train.shape[0], 27, 1)
X_test_t = X_test.reshape(X_test.shape[0], 27, 1)

print('================="now, we prepared whole dataset (train, test)"=================')
print(X_train_t.shape)
print(X_train_t[:3])

from keras.layers import LSTM
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
import keras.backend as K
from keras.callbacks import EarlyStopping

K.clear_session()

# actual deep learning process
model = Sequential()
# input_shape agrs -> timestep, feature(col number, unit line)
model.add(LSTM(32, input_shape=(27, 1)))
model.add(Dropout(0.3))
# result(=prediction) -> one value(=next-time success_amount)
model.add(Dense(1))
# loss function -> the most basic function
model.compile(loss='mean_squared_error', optimizer='adam')

early_stop = EarlyStopping(monitor='loss', patience=1, verbose=1)
hist = model.fit(X_train_t, y_train, epochs=50, batch_size=20, verbose=1, callbacks=[early_stop])
model.summary()

loss, acc = model.evaluate(X_test_t, y_test)
print('acc -> ', acc)
print('loss -> ', loss)

# plt.title('accuracy of model')
# plt.plot(model.metrics)
# plt.show()