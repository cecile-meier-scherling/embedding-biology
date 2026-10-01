"""Evaluate annotations on an unseen batch, excluding overlapping compounds."""
from annotation_model_utils import run_analysis


if __name__ == "__main__":
    run_analysis(["held_out_batch"], fit_final_models=False)
