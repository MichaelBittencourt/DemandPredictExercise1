import sys
from nneural import *

df = pd.read_csv(str(sys.argv[1]), sep=';')
df.drop(['TEMPO'], axis=1, inplace=True)
data = np.array(df)
data = tf.keras.utils.normalize(data, axis=1)

x_train = data[0:216, 0:5]
y_train = data[0:216, 5:7]

x_test = data[216:308,0:5]
y_test = data[216:308, 5:7]

metrics = ['mean_squared_error', "mean_absolute_error", "mape"]
epochs = 1

nnModelSigmoid = NNeural("Sigmoid", metrics = metrics, verbosity=True)
nnModelRelu = NNeural("Relu", activation="relu", metrics = metrics, verbosity=True)
nnModelTanh = NNeural("TanH", activation="tanh", metrics = metrics, verbosity=True)

modelList = [nnModelSigmoid, nnModelRelu, nnModelTanh]

for model in modelList:
    model.train(x_train, y_train, epochs

print("Calling method to evaluate_models")
NNeural.evaluate_models(modelList, x_test, y_test)
