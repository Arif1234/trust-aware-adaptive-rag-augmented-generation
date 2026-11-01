import json

def load_jsonl(path):
    data = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))

    return data


responses = load_jsonl(
    "data/response.jsonl"
)

sources = load_jsonl(
    "data/source_info.jsonl"
)

print(len(responses))
print(len(sources))