"""flec: A Flower / HuggingFace app."""

import warnings
from collections import OrderedDict

import torch
import transformers
from datasets.utils.logging import disable_progress_bar
from evaluate import load as load_metric
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorWithPadding
from datasets import load_dataset

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

transformers.logging.set_verbosity_error()


fds = None  # Cache FederatedDataset

from dotenv import load_dotenv
import os

load_dotenv()  # Loads from .env by default

hf_token = os.getenv("hf_token")

os.environ["HF_TOKEN"] = hf_token

def load_data(partition_id: int, num_partitions: int, model_name: str):
    """Load partition data."""
    
    # Only initialize `FederatedDataset` once
    global fds
    if fds is None:
        partitioner = IidPartitioner(num_partitions=num_partitions)
        fds = FederatedDataset(
            dataset = "dylanfuan03/mimic_hd",
            partitioners={"train": partitioner},
            shuffle = True,
            )
    partition = fds.load_partition(partition_id)
    # Divide data: 80% train, 20% test
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_function(examples):
        return tokenizer(
            examples["text"], padding = True, truncation=True, add_special_tokens=True, max_length=512
        )

    partition_train_test = partition_train_test.map(tokenize_function, batched=True)
    partition_train_test = partition_train_test.remove_columns("text")
    partition_train_test = partition_train_test.rename_column("label", "labels")

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    trainloader = DataLoader(
        partition_train_test["train"],
        shuffle=True,
        batch_size=16,
        collate_fn=data_collator,
    )

    testloader = DataLoader(
        partition_train_test["test"], batch_size=16, collate_fn=data_collator
    )

    return trainloader, testloader


def train(net, trainloader, epochs, device):
    optimizer = AdamW(net.parameters(), lr=5e-6, weight_decay=0.01)
    net.train()
    for _ in range(epochs):
        for batch in trainloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = net(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()


from evaluate import load as load_metric
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt




def test(net, testloader, device):
    accuracy_metric = load_metric("accuracy")
    f1_metric = load_metric("f1")
    
    total_loss = 0.0
    total_samples = 0
    net.eval()

    false_positives = 0
    false_negatives = 0
    true_positives = 0
    true_negatives = 0

    all_preds = []
    all_labels = []

    for batch in testloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = net(**batch)
        
        logits = outputs.logits
        loss = outputs.loss.item()
        batch_size = batch["labels"].size(0)

        total_loss += loss * batch_size
        total_samples += batch_size

        predictions = torch.argmax(logits, dim=-1)
        accuracy_metric.add_batch(predictions=predictions, references=batch["labels"])
        f1_metric.add_batch(predictions=predictions, references=batch["labels"])

        # Count FP and FN
        for i in range(batch_size):
            true_label = batch["labels"][i].item()
            pred_label = predictions[i].item()

            if pred_label != true_label:
                if pred_label == 1 and true_label == 0:
                    false_positives += 1
                elif pred_label == 0 and true_label == 1:
                    false_negatives += 1
            elif pred_label == 1 and true_label == 1:
                true_positives += 1 
            elif pred_label == 0 and true_label == 0:
                true_negatives += 1 
        
        predictions = torch.argmax(logits, dim=-1)
        all_preds.extend(predictions.cpu().numpy())
        all_labels.extend(batch["labels"].cpu().numpy())

    avg_loss = total_loss / total_samples
    accuracy = accuracy_metric.compute()["accuracy"]
    f1 = f1_metric.compute(average="weighted")["f1"]

    '''cm = confusion_matrix(all_labels, all_preds, labels=[0,1])

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[0,1], yticklabels=[0,1])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix (Aggregated)")
    plt.show()'''


    return avg_loss, total_samples, {
        "accuracy": float(accuracy),
        "f1": float(f1),
        "loss": float(avg_loss),
        "false_positives": int(false_positives),
        "false_negatives": int(false_negatives),
        "true_positives": int(true_positives),
        "true_negatives": int(true_negatives)
    }


def get_weights(net):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_weights(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)
