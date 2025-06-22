import os
import json
import random

# Output directories
base_dir = "data"
fixed_proc_dir = os.path.join(base_dir, "fixed_processors")
fixed_freq_dir = os.path.join(base_dir, "fixed_frequencies")
os.makedirs(fixed_proc_dir, exist_ok=True)
os.makedirs(fixed_freq_dir, exist_ok=True)

def generate_random_period():
    return random.choice([i for i in range(50, 200, 50)])

def generate_tasks(num_tasks):
    tasks = []
    for tid in range(1, num_tasks + 1):
        c_m = round(random.uniform(1, 20), 2)
        c_o = round(random.uniform(1, 15), 2)
        period = generate_random_period()
        tasks.append({
            "id": tid,
            "c_m": c_m,
            "c_o": c_o,
            "p": period
        })
    return tasks

def generate_processors(num_processors, freq_levels_per_proc=None):
    processors = []
    for pid in range(1, num_processors + 1):
        if freq_levels_per_proc is None:
            f = random.randint(3, 7)
            freqs = sorted([round(random.uniform(0.05, 0.95), 2) for _ in range(f - 1)])
            freqs = freqs + [1.0]
        else:
            freqs = freq_levels_per_proc[:]  # Use the fixed frequency levels
        processors.append({
            "id": pid,
            "frequencies": freqs
        })
    return processors

# Settings
num_instances = 5
task_range = (3, 7)
proc_range = (2, 10)
fixed_num_processors = 4
fixed_freq_levels = sorted([round(random.uniform(0.05, 0.95), 2) for _ in range(4)]) + [1.0]

# Generate instances with fixed number of processors, random frequencies
for i in range(1, num_instances + 1):
    num_tasks = random.randint(*task_range)
    tasks = generate_tasks(num_tasks)
    processors = generate_processors(fixed_num_processors)
    instance = {"tasks": tasks, "processors": processors}
    filename = f"instance_fixedproc_{i:03d}.json"
    filepath = os.path.join(fixed_proc_dir, filename)
    with open(filepath, "w") as f:
        json.dump(instance, f, indent=4)

# Generate instances with fixed frequency levels, random number of processors
for i in range(1, num_instances + 1):
    num_tasks = random.randint(*task_range)
    num_processors = random.randint(*proc_range)
    tasks = generate_tasks(num_tasks)
    processors = generate_processors(num_processors, freq_levels_per_proc=fixed_freq_levels)
    instance = {"tasks": tasks, "processors": processors}
    filename = f"instance_fixedfreq_{i:03d}.json"
    filepath = os.path.join(fixed_freq_dir, filename)
    with open(filepath, "w") as f:
        json.dump(instance, f, indent=4)

print("Instances generated in 'data/fixed_processors' and 'data/fixed_frequencies'.")