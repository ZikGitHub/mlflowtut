import json
import requests
import pandas as pd
import numpy as np

# Prepare input data
data = pd.DataFrame({
    "input": [np.int32(15)], 
    "model_name": ["model_1"]  # Keep model_name inside an array
})

# Convert DataFrame to JSON using "dataframe_split" format
json_data = json.dumps({"dataframe_split": data.to_dict(orient="split")})

headers = {"Content-Type": "application/json"}
endpoint = "http://127.0.0.1:5000/invocations"

# Send request
response = requests.post(endpoint, data=json_data, headers=headers)

# Print response
print(response.text)
print(response.status_code)
