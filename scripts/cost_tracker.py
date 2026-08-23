import json
import os

COST_FILE = "cost_tracker.json"

def track_cost(response, is_embedding=False):
    """Track API costs and save to cost_tracker.json."""
    usage = response.usage
    if is_embedding:
        cost = usage.total_tokens * 0.02 / 1_000_000
    else:  # chat (gpt-4o-mini)
        cost = (usage.prompt_tokens * 0.15 + usage.completion_tokens * 0.60) / 1_000_000

    if os.path.exists(COST_FILE):
        with open(COST_FILE) as f:
            data = json.load(f)
    else:
        data = {"total": 0.0, "calls": 0}

    data["total"] += cost
    data["calls"] = data.get("calls", 0) + 1

    with open(COST_FILE, "w") as f:
        json.dump(data, f, indent=2)

    print(f"💰 This call: ${cost:.6f} | Team total: ${data['total']:.4f} / $5.00")
    return cost
