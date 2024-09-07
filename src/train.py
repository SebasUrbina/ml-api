import pickle
from sklearn.datasets import load_iris

# ---------------------------------------------------------------------------- #
#                                   Load Data                                  #
# ---------------------------------------------------------------------------- #

iris_df = load_iris(as_frame=True) # cargar dataset
X = iris_df["data"] # features para predecir
y = iris_df["target"] # variable target, 0: setosa, 1: versicolor, 2: viginica

# # features
# X
# # target
# y 

# ---------------------------------------------------------------------------- #
#                                Train RF Model                                #
# ---------------------------------------------------------------------------- #

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

seed = 1997

# separamos los datos
X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, test_size=0.3, random_state = seed)

model = RandomForestClassifier(random_state = seed) # instanciar modelo
model.fit(X_train, y_train) # fit

y_pred = model.predict(X_test) # predict sobre X_test
accuracy_score(y_test, y_pred) # performance

# ---------------------------------------------------------------------------- #
#                                  Save Model                                  #
# ---------------------------------------------------------------------------- #

with open('../models/rf_model.pkl', 'wb') as file:
    pickle.dump(model, file)