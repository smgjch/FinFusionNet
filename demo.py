import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping,TensorBoard
from sklearn.preprocessing import MinMaxScaler
from src.model.FFNet import FFN_withDilation,FFN_without_dilation,FFN_withStep

input_window_size = 96
output_window_size = 96
model = FFN_without_dilation(input_window_size = input_window_size)

data = pd.read_csv('data/ETTh2.csv')
target_col = 'OT'
X = data[['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL']].values
Y = data[target_col].values

scaler = MinMaxScaler()
X = scaler.fit_transform(X)


def create_sliding_windows(data, target, input_window_size, output_window_size):
    X = []
    Y = []
    for i in range(len(data) - input_window_size - output_window_size + 1):
        X.append(data[i:i + input_window_size])
        Y.append(target[i + input_window_size:i + input_window_size + output_window_size])
    return np.array(X), np.array(Y)




x, y = create_sliding_windows(X, Y, input_window_size, output_window_size)

X_train, X_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.8,
                                                    random_state=1)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
                                                  train_size=0.8,
                                                  random_state=1)

input_shape = (None, input_window_size, X_train.shape[2])

model.build(input_shape)
model.compile(optimizer='adam', loss='mse')
model.summary()

early_stopping = EarlyStopping(monitor='val_loss', patience=15)
tensorboard_callback = TensorBoard(log_dir="logs")

history = model.fit(x=X_train,
                    y=y_train,
                    batch_size=64,
                    callbacks=[early_stopping,tensorboard_callback],
                    epochs=10000,
                    validation_data=[X_val, y_val])
