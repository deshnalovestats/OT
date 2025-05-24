import json
from task import Task, Processor

def load_input_from_json(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)

    tasks = [Task(t["id"], t["c_m"], t["c_o"], t["p"]) for t in data["tasks"]]
    processors = [Processor(p["id"], p["frequencies"]) for p in data["processors"]]
    
    return tasks, processors
