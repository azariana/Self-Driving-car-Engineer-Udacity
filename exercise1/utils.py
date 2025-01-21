import json
import numpy as np

def check_results(ious):
    """Check if the IoU calculations are correct."""
    solution = np.load('exercise1/data/exercise1_check.npy')  # Updated path
    assert (ious == solution).sum() == 40, 'The IoU calculation is wrong!'
    print('Congrats, the IoU calculation is correct!')

def get_data():
    """Load ground truth and prediction data."""
    with open('exercise1/data/ground_truth.json') as f:  # Updated path
        ground_truth = json.load(f)
    
    with open('exercise1/data/predictions.json') as f:  # Updated path
        predictions = json.load(f)

    return ground_truth, predictions
