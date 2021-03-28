# mlp for multi-label classification
from numpy import mean
from numpy import std
from sklearn.model_selection import RepeatedKFold
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import accuracy_score
import pandas as pd
from keras.utils.vis_utils import plot_model
from sklearn import preprocessing



# get the dataset
def get_dataset():
    data = pd.read_csv("treino.csv")
    y = pd.DataFrame(data["y"], index=range(0,3176), columns=["y"])
    x = data.drop(columns=["y", "id"])
    scaler = preprocessing.MinMaxScaler()
    x_scaled = scaler.fit_transform(x.values)
    #x_new = SelectKBest(chi2, k=15).fit_transform(x_scaled, y)
    #print(x_new.shape)
    return x_scaled, y.values


# get the model
def get_model(n_inputs, n_outputs):
    model = Sequential()
    #model.add(Dense(10, input_dim=n_inputs, kernel_initializer='uniform', activation='selu'))
    model.add(Dense(35, input_dim=n_inputs, kernel_initializer='uniform', activation='tanh', use_bias=True))
    model.add(Dense(n_outputs, activation='tanh', use_bias=True))
    #opt = keras.optimizers.SGD(
    #    learning_rate=0.01, momentum=0.05, nesterov=False, name="SGD")
    model.compile(loss='mean_squared_error', optimizer="Adam")
    plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
    return model


# evaluate a model using repeated k-fold cross-validation
def evaluate_model(X, y):
    results = list()
    n_inputs, n_outputs = X.shape[1], y.shape[1]
    # define evaluation procedure
    cv = RepeatedKFold(n_splits=10, n_repeats=1, random_state=1)
    # enumerate folds
    for train_ix, test_ix in cv.split(X):
        # prepare data
        X_train, X_test = X[train_ix], X[test_ix]
        y_train, y_test = y[train_ix], y[test_ix]
        # define model
        model = get_model(n_inputs, n_outputs)
        # fit model
        history = model.fit(X_train, y_train, verbose=0, epochs=3000)
        # make a prediction on the test set
        yhat = model.predict(X_test)
        yt = (1 * (yhat >= 0) - 0.5) * 2
        # round probabilities to class labels
        yhat = yt.round()
        # calculate accuracy
        acc = accuracy_score(y_test, yhat)
        # store result
        print('>%.3f' % acc)
        results.append(acc)
    return results

# load dataset
X, Y = get_dataset()
x = X
y = Y
results = evaluate_model(x, y)
# summarize performance
print('Accuracy: %.3f (%.3f)' % (mean(results), std(results)))
