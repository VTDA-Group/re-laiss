"""
Configuration parameters for anomaly detection module.

Streamlined version containing only essential parameters used in the minimal pipeline.
"""

# k-NN Distance Parameters
DEFAULT_K = 20

# Memory and Performance Parameters
TRAINING_SAMPLE_SIZE = 5000         # Maximum training features to store

# Training Data Quality Parameters
HIGH_NAN_WARNING_THRESHOLD = 20    # Percentage of NaN samples to trigger warning
VERY_HIGH_NAN_ERROR_THRESHOLD = 50 # Percentage of NaN samples to trigger error

# File Handling Parameters
MODEL_FILENAME_TEMPLATE = "kNN_scaler_lc={num_lc}_host={num_host}.pkl" 