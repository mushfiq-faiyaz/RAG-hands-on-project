import json
import requests

with open("session_test.json") as f:
    sid = json.load(f)["session_id"]

print("Reusing session:", sid)

r = requests.post(
    "http://127.0.0.1:8000/chat",
    json={"question": "When was that released?", "session_id": sid},
)
print("Answer:", r.json()["answer"])