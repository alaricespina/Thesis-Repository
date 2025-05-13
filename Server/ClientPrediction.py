import requests

# Sample Iris input data (e.g., from test set)
sample_input = [5.1, 3.5, 1.4, 0.2]  # Likely setosa

response = requests.post(
    "http://127.0.0.1:8000/predict/",
    json = {"data": sample_input}
)

if response.status_code == 200:
    result = response.json()
    print("✅ Prediction received:")
    print(f"Class Index: {result['class_index']}")
    print(f"Class Name: {result['class_name']}")
    print(f"Confidence: {result['confidence']:.2f}")
else:
    print("❌ Failed to get prediction")
