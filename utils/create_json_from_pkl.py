import pickle
import json
import numpy as np

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        if isinstance(obj, set):
            return list(obj)
        if isinstance(obj, bytes):
            return obj.decode('utf-8', errors='ignore')
        return super().default(obj)

# Load pickle file with Latin-1 encoding
with open('session1_right/gt_data.pkl', 'rb') as f:
    data = pickle.load(f, encoding='latin1')

# Save to JSON with NumPy encoder
with open('session1_right/gt_data.json', 'w') as f:
    json.dump(data, f, indent=4, cls=NumpyEncoder)

print("Successfully converted to JSON")