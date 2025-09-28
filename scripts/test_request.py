import requests

resp = requests.post(
    "http://127.0.0.1:5000/predict",
    json={"URL_Length": 120, "Has_IP": 1, "Prefix_Suffix": 0}
)

print(resp.json())