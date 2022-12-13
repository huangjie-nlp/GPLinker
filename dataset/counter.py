import json

file = "./CMeIE/train_data.json"
with open(file, "r", encoding='utf-8') as f:
    data = json.load(f)

print(len(data))