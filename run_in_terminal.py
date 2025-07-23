# This script is used to run in a terminal with the virtual environment activated.

# . 'C:\Users\dylan\PythonScripts\MachineLearning\FederatedTesting\.venv\Scripts\Activate.ps1'
# cd.venv


import torch
print(torch.cuda.is_available())  # Should be True
print(torch.cuda.get_device_name(0))  # Should print your GPU name

from datasets import load_dataset

from dotenv import load_dotenv
import os

load_dotenv()  # Loads from .env by default

hf_token = os.getenv("hf_token")

os.environ["HF_TOKEN"] = hf_token
from datasets import DatasetDict, Dataset as HFDataset
import random

def load_custom_cleveland_dataset() -> DatasetDict:
    with open("processed_cleveland.txt", "r") as file:
        data = [line.strip().split(",") for line in file.readlines()]
    random.shuffle(data)

    texts, labels, label_ids = [], [], []
    label_map = {
        ' Prediction: No Presence of Heart Disease (<50 %diameter narrowing)': 0,
        ' Prediction: Low Presence of Heart Disease (>50 %diameter narrowing)': 1,
        ' Prediction: Slight Presence of Heart Disease': 2,
        ' Prediction: Moderate Presence of Heart Disease': 3,
        ' Prediction: High Presence of Heart Disease': 4,
    }

    for row in data:
        if not row or row[-1] not in label_map:
            continue
        features = row[1:-1]
        prompt = "Patient data: " + ", ".join([f"attr{i}={v}" for i, v in enumerate(features)])
        texts.append(prompt)
        labels.append(row[-1])
        label_ids.append((label_map[row[-1]]))


    full_dataset = HFDataset.from_dict({"text": texts, "label": label_ids})

    return full_dataset

dataset = load_custom_cleveland_dataset()

dataset.push_to_hub("dylanfuan03/cleveland_hd", private = True)
