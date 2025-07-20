import json
import os
import random

# === Settings ===
num_instances_required = 50
fixed_num_tasks = 10 #change a
fixed_num_processors = 5
num_freqs_per_processor = 5  # number of frequency levels per processor
fixed_proc_dir = "Tasks_20" #change the folder name as 10, 20, 30, etc. based on the number of tasks
os.makedirs(fixed_proc_dir, exist_ok=True)

# === Shared Frequency Pool (same for all processors)
shared_freq_pool = sorted({round(i * 0.1, 1) for i in range(1, 11)})  # 0.1 to 1.0
print("Shared Frequency Pool:", shared_freq_pool)

# === Helper Functions ===
def generate_tasks(n):
    tasks = []
    for i in range(n):
        period = random.randint(10, 50)
        cm = random.randint(1, int(0.4 * period))  # discrete cm
        co = random.randint(1, int(0.3 * period))  # discrete co
        tasks.append({
            "id": i + 1,
            "period": period,
            "cm": cm,
            "co": co
        })
    return tasks

def calculate_utilization(tasks):
    return sum((t["cm"] + t["co"]) / t["period"] for t in tasks)

def generate_fixed_processors(num_processors, freq_pool, num_freqs):
    processors = []
    for i in range(num_processors):
        # Sample without replacement from pool
        freqs = sorted(random.sample(freq_pool, num_freqs))
        processors.append({"id": i + 1, "frequencies": freqs})
    return processors

# === Step 1: Generate the Processor List Once ===
fixed_processors = generate_fixed_processors(fixed_num_processors, shared_freq_pool, num_freqs_per_processor)
print("Fixed Processor Frequency Configurations:")
for p in fixed_processors:
    print(f"Processor {p['id']}: {p['frequencies']}")

# === Step 2: Generate Task Sets and Use the Same Processor Configs ===
valid_count = 0
attempt = 0
while valid_count < num_instances_required:
    attempt += 1
    tasks = generate_tasks(fixed_num_tasks)
    utilization = calculate_utilization(tasks)

    if utilization > fixed_num_processors:
        print(f"[Attempt {attempt}] Skipped: utilization = {utilization:.2f}")
        continue

    instance = {
        "tasks": tasks,
        "processors": fixed_processors  # Reuse same processors
    }

    filename = f"instance_fixedproc_{valid_count+1:03d}.json"
    filepath = os.path.join(fixed_proc_dir, filename)
    with open(filepath, "w") as f:
        json.dump(instance, f, indent=4)

    print(f"[Attempt {attempt}] Saved: {filename} (utilization = {utilization:.2f})")
    valid_count += 1
