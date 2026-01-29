import os
import json
import pandas as pd

EMBEDDINGS_PATH = "embeddings/"
OUTPUT_FILE = "test.predicted-scores.json"

MODELS = [
    "sdvae_hear_api",
    "ezaudio_hear_api",
    "audioldm2_hear_api",
]

results = {}

for model in MODELS:
    tasks = os.listdir(os.path.join(EMBEDDINGS_PATH, model))
    model_results = {}
    for task in tasks:
        embeddings_dir = os.path.join(EMBEDDINGS_PATH, model, task)
        output_path = os.path.join(embeddings_dir, OUTPUT_FILE)
        if os.path.exists(output_path):
            with open(output_path, "r") as f:
                task_results = json.load(f)
            # try ["test"]["test_score"]
            test_score = task_results.get("test", {}).get("test_score", None)
            if test_score is not None:
                model_results[task] = test_score
            else:
                # try ["aggregated_scores"]["test_score_mean"]
                test_score = task_results.get("aggregated_scores", {}).get("test_score_mean", None)
                if test_score is not None:
                    model_results[task] = test_score
                else:
                    print(f"Warning: No test score found for model {model} on task {task}")
        else:
            print(f"Warning: Output file not found for model {model} on task {task}")
    results[model] = model_results

# export to csv
df = pd.DataFrame(results)
df.to_csv("model_scores.csv")
print("Scores saved to model_scores.csv")
