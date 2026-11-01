import json

with open(
    "data/source_info.jsonl",
    "r",
    encoding="utf-8"
) as f:

    first = json.loads(next(f))

print(first.keys())
print(first)