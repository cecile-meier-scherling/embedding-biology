"""Evaluate annotations for unseen compounds, using familiar assay batches."""
from annotation_model_utils import run_analysis


if __name__ == "__main__":
    run_analysis(["held_out_compound"])
