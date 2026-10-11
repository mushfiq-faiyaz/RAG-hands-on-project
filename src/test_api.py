import config
import requests

headers = {"X-Auth-Token": config.AUTH_TOKEN}

r1 = requests.post(
    "http://127.0.0.1:8000/chat",
    json={"question": "What is the latest version of PidiFie?"},
    headers=headers,
)
print("Turn 1:", r1.json()["answer"])

sid = r1.json()["session_id"]

r2 = requests.post(
    "http://127.0.0.1:8000/chat",
    json={"question": "When was that released?", "session_id": sid},
    headers=headers,
)
print("Turn 2:", r2.json()["answer"])
print("Same session:", r2.json()["session_id"] == sid)