"""flec: A Flower / HuggingFace app."""

from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from transformers import AutoModelForSequenceClassification

from flec.task import get_weights

from dotenv import load_dotenv
import os

load_dotenv()  # Loads from .env by default

hf_token = os.getenv("hf_token")

os.environ["HF_TOKEN"] = hf_token
def aggregate_fit_metrics(metrics_list):
    total_examples = sum(num_examples for num_examples, _ in metrics_list)
    agg = {}
    for key in ["accuracy", "loss"]:
        total = sum(
            num_examples * metrics.get(key, 0)
            for num_examples, metrics in metrics_list
        )
        agg[key] = total / total_examples if total_examples > 0 else 0.0
    return agg

def aggregate_eval_metrics(metrics_list):
    total_examples = 0
    weighted_accuracy_sum = 0.0
    print("Received evaluation metrics:", metrics_list)  # Debug line
    for num_examples, metrics in metrics_list:
        if "accuracy" in metrics:
            total_examples += num_examples
            weighted_accuracy_sum += num_examples * metrics["accuracy"]

    if total_examples == 0:
        return {"accuracy": 0.0}

    return {"accuracy": weighted_accuracy_sum / total_examples}


def server_fn(context: Context):
    # Read from config
    num_rounds = context.run_config["num-server-rounds"]
    fraction_fit = context.run_config["fraction-fit"]

    # Initialize global model
    model_name = context.run_config["model-name"]
    num_labels = context.run_config["num-labels"]
    net = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels
    )

    weights = get_weights(net)
    initial_parameters = ndarrays_to_parameters(weights)

    # Define strategy
    strategy = FedAvg(
        fraction_fit=fraction_fit,
        fraction_evaluate=0.5,
        initial_parameters=initial_parameters,
        fit_metrics_aggregation_fn=aggregate_fit_metrics,
        evaluate_metrics_aggregation_fn=aggregate_eval_metrics
        )
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


# Create ServerApp
app = ServerApp(server_fn=server_fn)
