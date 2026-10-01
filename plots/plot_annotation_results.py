"""Compatibility runner for both annotation-result figures."""
from plot_annotation_predictions import plot_metrics
from plot_cluster_enrichment import plot_enrichment


if __name__ == "__main__":
    print(f"Saved {plot_metrics()}")
    print(f"Saved {plot_enrichment()}")
