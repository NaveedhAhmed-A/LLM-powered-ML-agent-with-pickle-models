import os
import glob
import pickle
import random
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()
llm = ChatGroq(model="qwen/qwen3-32b", temperature=0, api_key=os.getenv("GROQ_API_KEY"))

# Utility: find a .pkl that contains the keyword (case-insensitive)
def find_model_file(keyword):
    keyword = keyword.lower()
    pkl_files = glob.glob("**/*.pkl", recursive=True)
    for p in pkl_files:
        if keyword in os.path.basename(p).lower():
            return p
    return None, pkl_files

def predict(values, model_choice):
    # handle auto
    if model_choice == "auto":
        model_choice = random.choice(["linear", "random"])
        print("Auto-selected model:", model_choice)

    # find model file
    model_file = None
    pkl_files = glob.glob("**/*.pkl", recursive=True)
    for p in pkl_files:
        if model_choice in os.path.basename(p).lower():
            model_file = p
            break

    if not model_file:
        # helpful error message: list discovered .pkl files
        files_list = "\n".join(pkl_files) if pkl_files else "(no .pkl files found)"
        raise FileNotFoundError(
            f"Could not find a .pkl for model '{model_choice}'.\n"
            f"Expected a filename containing '{model_choice}'.\n"
            f"PKL files found in project:\n{files_list}\n\n"
            "Solution: put your model .pkl files into the project (e.g. ./models/) "
            "or rename them to include 'linear' or 'random'."
        )

    print("Loading model file:", model_file)
    with open(model_file, "rb") as f:
        model = pickle.load(f)

    # if values is a DataFrame, convert properly
    import numpy as np
    if hasattr(values, "values"):
        vals = values.values
    else:
        vals = values
    vals = np.array(vals).reshape(1, -1)

    pred = model.predict(vals)
    return pred

def choose_model():
    prompt = """
Reply with ONLY ONE WORD.
No explanation.

Options:
linear
random
auto
"""
    response = llm.invoke(prompt)
    raw = response.content.lower()
    for option in ["linear", "random", "auto"]:
        if option in raw:
            return option
    return "auto"  # fallback

if __name__ == "__main__":
    import pandas as pd
    values = pd.DataFrame([[8, 307, 130, 3504, 12, 70, 1]])
    model_choice = choose_model()
    print("LLM chose:", model_choice)
    prediction = predict(values, model_choice)
    print("Prediction:", prediction)