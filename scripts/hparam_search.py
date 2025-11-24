# scripts/hparam_search.py
import optuna
import subprocess
import re

def objective(trial):
    hidden_dim = trial.suggest_categorical("hidden_dim", [384, 512, 640])
    num_layers = trial.suggest_int("num_layers", 2, 4)
    attn_heads = trial.suggest_categorical("attn_heads", [4, 6, 8])
    if hidden_dim % attn_heads != 0:
        return float("inf")
    block_size = trial.suggest_categorical("block_size", [64, 96, 128])
    dropout = trial.suggest_float("dropout", 0.1, 0.3)
    lr = trial.suggest_float("lr", 3e-4, 1e-3, log=True)

    cmd = [
        "python3", "scripts/train_language_model.py",
        "--data_file", "dummy.txt",
        "--tokenizer_file", "models/wiki_tokenizer.json",
        "--brain_dataset", "data/brain_ctx_pairs_100k.npz",
        "--epochs", "4",  # keep short per trial
        "--batch_size", "32",
        "--block_size", str(block_size),
        "--hidden_dim", str(hidden_dim),
        "--num_layers", str(num_layers),
        "--attn_heads", str(attn_heads),
        "--dropout", str(dropout),
        "--lr", str(lr),
        "--device", "cuda",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    # Parse the last reported validation loss from stdout
    val_loss = None
    for line in result.stdout.splitlines():
        m = re.search(r"Validation Loss:\s*([0-9.]+)", line)
        if m:
            val_loss = float(m.group(1))
    if val_loss is None:
        print("Failed to parse val loss. Stdout:\n", result.stdout)
        print("Stderr:\n", result.stderr)
        return float("inf")
    return val_loss

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=20)
print("Best params:", study.best_params)
print("Best val_loss:", study.best_value)

# Optionally persist trials to CSV
out_csv = "hparam_trials.csv"
import pandas as pd  # noqa: E402
df = study.trials_dataframe(attrs=("number", "value", "params", "state"))
df.to_csv(out_csv, index=False)
print(f"Wrote trial history to {out_csv}")
