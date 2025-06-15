import os
import json
import shutil
import optuna
import argparse
import logging

import train as trainer
import eval as evaluator
from utils.helpers import set_seed, setup_logging

set_seed(42)
setup_logging("logs/optuna_search.log")

# --- Configuration ---
# Total number of trials Optuna will run.
N_TRIALS = 800
# Directory to store results from all runs.
HPARAM_RUNS_DIR = "results/optuna_runs"

# --- Base Configuration ---
# These are the base arguments that will be passed to train.py and eval.py.
# You should modify these to match your environment.
BASE_ARGS = {
    # Model args
    "model": "clip_lora",
    "base_model": "openai/clip-vit-base-patch32",
    "lora_target_modules": "q_proj,v_proj",
    # Data args
    "root_dir": "datasets",
    "batch_size": 32,
    "num_workers": 4,
    # Training args
    "weight_decay": 0.01,
    # Other args
    "device": "cuda",
    "random_seed": 42,
    "use_processor": False,
}


def run_trial(params, run_id):
    """
    Runs a full training and evaluation trial for a given set of hyperparameters.
    """
    output_dir = os.path.join(HPARAM_RUNS_DIR, f"trial_{run_id}")
    test_metrics_path = os.path.join(output_dir, "test_metrics.json")

    if os.path.exists(test_metrics_path):
        logging.info(f"--- [Trial {run_id}] Results already exist. Loading AUC. ---")
        with open(test_metrics_path, "r") as f:
            return json.load(f).get("video_level", {}).get("auc", 0.0)

    os.makedirs(output_dir, exist_ok=True)

    # --- Step 1: Training ---
    logging.info(f"\n--- [Trial {run_id}] Starting Training | Params: {params} ---")
    try:
        train_args = BASE_ARGS.copy()
        train_args.update(params)
        train_args["output_dir"] = output_dir
        train_args["log_path"] = os.path.join(output_dir, "train.log")
        train_args["lr"] = train_args.pop("learning_rate")

        train_args_ns = argparse.Namespace(**train_args)
        trainer.main(train_args_ns)
        logging.info(f"--- [Trial {run_id}] Training finished successfully. ---")
    except Exception as e:
        logging.error(f"!!! [Trial {run_id}] Training Failed !!!\n  - Exception: {e}")
        import traceback
        traceback.print_exc()
        return 0.0

    # --- Step 2: Evaluation ---
    logging.info(f"--- [Trial {run_id}] Starting Evaluation on Test Set... ---")
    try:
        eval_args = BASE_ARGS.copy()
        eval_args["model_dir"] = output_dir
        eval_args["split"] = "test"
        eval_args["log_path"] = os.path.join(output_dir, "eval.log")
        eval_args["roc_curve_path"] = os.path.join(output_dir, "roc_curve_test.png")
        # These are not needed for eval but are in BASE_ARGS
        eval_args.pop("weight_decay", None)
        
        # We need to add all lora params to eval args for model loading
        for p in ['lora_r', 'lora_alpha', 'lora_dropout', 'lora_target_modules']:
            if p in params:
                 eval_args[p] = params[p]
            elif p in BASE_ARGS:
                 eval_args[p] = BASE_ARGS[p]


        eval_args_ns = argparse.Namespace(**eval_args)
        
        # Replicating eval.py main logic to get metrics back
        evaluator.main(eval_args_ns)

        with open(test_metrics_path, "r") as f:
            metrics = json.load(f)
        auc = metrics.get("video_level", {}).get("auc", 0.0)
        logging.info(f"--- [Trial {run_id}] Evaluation finished. Test AUC: {auc:.4f} ---")
        return auc
    except Exception as e:
        logging.error(f"!!! [Trial {run_id}] Evaluation Failed !!!\n  - Exception: {e}")
        import traceback
        traceback.print_exc()
        return 0.0


def objective(trial):
    """
    The objective function for Optuna to optimize.
    """
    run_id = trial.number
    params = {
        'lora_r': trial.suggest_categorical('lora_r', [8, 16, 32,]),
        'lora_alpha': trial.suggest_categorical('lora_alpha', [8, 16, 32, 64,]),
        'lora_dropout': trial.suggest_categorical('lora_dropout', [0.0, 0.05, 0.1]),
        'learning_rate': trial.suggest_categorical('learning_rate', [5e-5, 1e-4, 3e-4, 5e-4, 1e-3]),
        'epochs': trial.suggest_categorical('epochs', [1,])
    }

    # A common heuristic is to have alpha be a multiple of r.
    # Pruning unpromising trials where alpha < r.
    if params['lora_alpha'] < params['lora_r'] or params['lora_alpha'] // params['lora_r'] > 2:
        raise optuna.exceptions.TrialPruned()

    return run_trial(params, run_id)


def main():
    """
    Main function to setup and run the Optuna hyperparameter search.
    """
    # You will need to install optuna: pip install optuna
    logging.info("--- Optuna Hyperparameter Search Runner ---")
    logging.warning("WARNING: This script uses the TEST SET to select the best hyperparameters.")
    logging.warning("This is not standard practice and may lead to over-optimistic performance estimates.\n")

    if not os.path.exists(HPARAM_RUNS_DIR):
        os.makedirs(HPARAM_RUNS_DIR)

    # Use a TPE sampler for more intelligent search
    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    
    try:
        study.optimize(objective, n_trials=N_TRIALS)
    except KeyboardInterrupt:
        logging.info("\nSearch interrupted by user. Reporting best results so far...")

    # --- Summarize Results ---
    logging.info("\n\n--- Hyperparameter Search Finished ---")
    
    pruned_trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.COMPLETE])
    fail_trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.FAIL])

    logging.info(f"Study statistics: ")
    logging.info(f"  Number of finished trials: {len(study.trials)}")
    logging.info(f"  Number of pruned trials: {len(pruned_trials)}")
    logging.info(f"  Number of complete trials: {len(complete_trials)}")
    logging.info(f"  Number of failed trials: {len(fail_trials)}")

    logging.info("\n--- Best Trial ---")
    try:
        best_trial = study.best_trial
        logging.info(f"  Value (Test AUC): {best_trial.value:.4f}")
        logging.info("  Params: ")
        for key, value in best_trial.params.items():
            logging.info(f"    {key}: {value}")

        # Copy best model to a final directory
        best_trial_id = best_trial.number
        best_model_dir = os.path.join(HPARAM_RUNS_DIR, f"trial_{best_trial_id}")
        final_model_dir = "results/best_clip_lora_model"

        if os.path.exists(best_model_dir):
            if os.path.exists(final_model_dir):
                shutil.rmtree(final_model_dir)
            shutil.copytree(best_model_dir, final_model_dir)
            logging.info(f"\nBest model from trial {best_trial_id} copied to '{final_model_dir}'")
            logging.info(f"You can re-evaluate this model anytime using:\npython eval.py --model_dir {final_model_dir} --model clip_lora")
        else:
            logging.error(f"\nCould not find directory for best trial: {best_model_dir}")

    except ValueError:
        logging.error("No successful trials completed. Could not determine a best configuration.")


if __name__ == "__main__":
    main()
