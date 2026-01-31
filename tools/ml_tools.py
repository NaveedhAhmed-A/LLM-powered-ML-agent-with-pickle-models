import pickle
import numpy as np
import random

def predict(values, model_choice):

    if model_choice == "auto":
        model_choice = random.choice(["linear", "random"])
        print("Auto-selected model:", model_choice)

    if model_choice == "linear":
        model = pickle.load(open("Linear_model.pkl", "rb"))
    elif model_choice == "random":
        model = pickle.load(open("Random_forest.pkl", "rb"))
    else:
        raise ValueError(f"Invalid model choice after cleanup: {model_choice}")

    return model.predict(values)
