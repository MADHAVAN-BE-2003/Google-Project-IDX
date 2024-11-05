import pandas as pd
import numpy as np
from digit_recognizer import DigitRecognizer

data = pd.read_csv('./data.csv').to_numpy()
m, n = data.shape
np.random.shuffle(data)

data_train = data[1000:m].T
Y_train = data_train[0]
X_train = data_train[1:n] / 255.0
_, m_train = X_train.shape

model = DigitRecognizer()
model.gradient_descent(X_train, Y_train, alpha=0.10, iterations=5000)

model.save_model("digit_recognizer_weights.npz")
