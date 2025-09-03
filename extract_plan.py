import json
import re

input_path = "results/CoT_GPT-4o_42.json"
output_path = "results/CoT_GPT-4o_42_action.json"

results = []
with open(input_path, "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        sample_id = item.get("sample_id")
        response = item.get("model_response", "")
        # 提取 "3." 之后的内容
        match = re.search(r'3\.\s*(.*)', response, re.DOTALL)
        action = match.group(1).strip() if match else ""
        results.append({
            "sample_id": sample_id,
            "action": action
        })

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)