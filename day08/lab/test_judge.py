import json
from rag_answer import call_llm

def ask_judge_llm(prompt):
    try:
        res = call_llm(prompt)
        res_json = json.loads(res.strip().strip("").removeprefix("json").strip())
        return {"score": res_json.get("score"), "notes": str(res_json.get("reason", res_json.get("missing_points", "")))}
    except Exception as e:
        return {"score": 3, "notes": f"LLM error: {e}"}

# This is just testing if we can parse the json
