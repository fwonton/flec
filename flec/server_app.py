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


import torch
from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters

# Convert Flower Parameters to HuggingFace model weights
def set_weights(model, weights):
    state_dict = model.state_dict()
    new_state_dict = {}
    for (key, value), weight in zip(state_dict.items(), weights):
        new_state_dict[key] = torch.tensor(weight)
    model.load_state_dict(new_state_dict, strict=True)

# Convert HuggingFace model weights to Flower Parameters
def get_weights(model):
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


class SaveModelStrategy(FedAvg):
    def __init__(self, model_name, num_labels, num_rounds, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_name = model_name
        self.num_labels = num_labels
        self.num_rounds = num_rounds

    def aggregate_fit(self, rnd, results, failures):
        aggregated_parameters, metrics_aggregated = super().aggregate_fit(rnd, results, failures)

        if aggregated_parameters is not None:
            # Rebuild Hugging Face model
            model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name, num_labels=self.num_labels
            )
            set_weights(model, parameters_to_ndarrays(aggregated_parameters))

            # Save only final global model
            if rnd == self.num_rounds:
                save_path = "global_model_final"
                os.makedirs(save_path, exist_ok=True)
                model.save_pretrained(save_path)
                print(f"✅ Saved final global model at {save_path}")

        return aggregated_parameters, metrics_aggregated

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
    weighted_f1_sum = 0.0
    weighted_loss_sum = 0.0


    total_false_positives = 0
    total_false_negatives = 0
    total_true_positives = 0
    total_true_negatives = 0


    print("Received evaluation metrics:", metrics_list)  # Debug line

    for num_examples, metrics in metrics_list:
        total_examples += num_examples
        weighted_accuracy_sum += num_examples * metrics.get("accuracy", 0.0)
        weighted_f1_sum += num_examples * metrics.get("f1", 0.0)
        weighted_loss_sum += num_examples * metrics.get("loss", 0.0)
        total_false_positives += metrics.get("false_positives", 0)
        total_false_negatives += metrics.get("false_negatives", 0)
        total_true_positives += metrics.get("true_positives", 0)
        total_true_negatives += metrics.get("true_negatives", 0)


    if total_examples == 0:
        return {"accuracy": 0.0, "f1": 0.0, "loss": 0.0, "false_positives": 0, "false_negatives": 0, "confusion_matrix":None}


    return {
        "accuracy": float(weighted_accuracy_sum / total_examples),
        "f1": float(weighted_f1_sum / total_examples),
        "loss": float(weighted_loss_sum / total_examples),
        "false_positives": int(total_false_positives),
        "false_negatives": int(total_false_negatives),
        "true_positives": int(total_true_positives),
        "true_negatives": int(total_true_negatives)
    }


def server_fn(context: Context):
    # Read config
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

    # Strategy with evaluation + saving
    strategy = SaveModelStrategy(
        model_name=model_name,
        num_labels=num_labels,
        num_rounds=num_rounds,
        fraction_fit=fraction_fit,
        fraction_evaluate=0.5,
        initial_parameters=initial_parameters,
        fit_metrics_aggregation_fn=aggregate_fit_metrics,
        evaluate_metrics_aggregation_fn=aggregate_eval_metrics,
    )

    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


# Create ServerApp
app = ServerApp(server_fn=server_fn)
