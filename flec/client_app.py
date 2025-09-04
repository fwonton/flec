"""flec: A Flower / HuggingFace app."""

import torch
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from transformers import AutoModelForSequenceClassification
import gc
from flec.task import get_weights, load_data, set_weights, test, train
from dotenv import load_dotenv
import os

load_dotenv()  # Loads from .env by default

hf_token = os.getenv("hf_token")

os.environ["HF_TOKEN"] = hf_token
# Flower client
class FlowerClient(NumPyClient):
    def __init__(self, net, trainloader, testloader, local_epochs):
        self.net = net
        self.trainloader = trainloader
        self.testloader = testloader
        self.local_epochs = local_epochs
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)

    def fit(self, parameters, config):
        set_weights(self.net, parameters)
        train(self.net, self.trainloader, epochs=self.local_epochs, device=self.device)

        # Evaluate on validation/test data
        with torch.no_grad():
            loss, num_samples, metrics = test(self.net, self.testloader, self.device)

        torch.cuda.empty_cache()
        gc.collect()

        return get_weights(self.net), len(self.trainloader), metrics

    
    def evaluate(self, parameters, config):
        set_weights(self.net, parameters)
        with torch.no_grad():
            loss, num_samples, metrics = test(self.net, self.testloader, self.device)
        return float(loss), num_samples, metrics



def client_fn(context: Context):

    # Get this client's dataset partition
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    model_name = context.run_config["model-name"]
    trainloader, valloader = load_data(partition_id, num_partitions, model_name)

    # Load model
    num_labels = context.run_config["num-labels"]
    net = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels
    )
    net.gradient_checkpointing_enable()

    local_epochs = context.run_config["local-epochs"]

    # Return Client instance
    return FlowerClient(net, trainloader, valloader, local_epochs).to_client()


# Flower ClientApp
app = ClientApp(
    client_fn,
)
