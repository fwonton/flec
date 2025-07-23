from transformers import AutoModelForSequenceClassification

from dotenv import load_dotenv
import os

load_dotenv()  # Loads from .env by default

hf_token = os.getenv("hf_token")

os.environ["HF_TOKEN"] = hf_token

model = AutoModelForSequenceClassification.from_pretrained("allenai/biomed_roberta_base",num_labels=5)

