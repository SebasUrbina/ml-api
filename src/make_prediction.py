import pickle

# -------------------------------- Load Model -------------------------------- #
with open("models/rf_model.pkl", "rb") as f:
    model = pickle.load(f)

labels_mapper = {0: "Setosa", 1: "Versicolor", 2: "Virginica"}

def make_prediction(
    sepal_length: float, 
    sepal_width: float, 
    petal_length: float, 
    petal_width: float):
    """
    Function  that returns the prediction given a set of features
    """

    features = [[
        sepal_length, sepal_width, petal_length, petal_width]]

    prediction = model.predict(features).item()
    label = labels_mapper[prediction]

    return label