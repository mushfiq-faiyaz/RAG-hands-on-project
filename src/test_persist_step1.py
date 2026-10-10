import json
import requests

r = requests.post(
    "http://127.0.0.1:8000/chat",
    json={"question": "What is the latest version of PidiFie?"},
)
data = r.json()
print("Answer:", data["answer"])
print("Session:", data["session_id"])

with open("session_test.json", "w") as f:
    json.dump({"session_id": data["session_id"]}, f)