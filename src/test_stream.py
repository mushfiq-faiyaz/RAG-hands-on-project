import config
import requests

r = requests.post(
    "http://127.0.0.1:8000/chat/stream",
    json={"question": "What is the latest version of PidiFie?"},
    headers={"X-Auth-Token": config.AUTH_TOKEN},
    stream=True,
)

for line in r.iter_lines(decode_unicode=True):
    if line:
        print(line)