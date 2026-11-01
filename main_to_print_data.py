import json
from pathlib import Path

data_file = Path(__file__).parent / "data" / "source_info.jsonl"

with data_file.open(encoding="utf-8") as file:
    for number, line in enumerate(file, start=1):
        record = json.loads(line)
        print(f"\nRecord {number}")
        print("source_id:", record.get("source_id"))
        print("source_info:", str(record.get("source_info", ""))[:500])

        if number == 3:
            break