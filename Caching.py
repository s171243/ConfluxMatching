import json
import os
import numpy as np
from json import JSONEncoder

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def get_or_calculate_cache(outfile_name, func_result, **kwargs):
    try:
    # Import similarity matrix
        with open(outfile_name) as input_file:
            print("Cached")
            return json.load(input_file)
    except (IOError, FileNotFoundError):
        print("File not found. Calculating (might take a while)")
        output_matrix = func_result(**kwargs)
        with open(outfile_name, "w+") as outfile:
            print(type(output_matrix))
            json.dump(output_matrix, outfile, cls=NumpyArrayEncoder)
            print("Calculation done")
            return output_matrix

