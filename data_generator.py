import os
import json
import random
from math import gcd
from functools import reduce

# Directory to save generated JSON files
output_dir = "data"
os.makedirs(output_dir, exist_ok=True)

def lcm(x, y):
    return x * y // gcd(x, y)

def generate_random_period():
    # Period should be multiples of 50, between 50 and 200
    return random.choice([i for i in range(50, 200, 50)])

def generate_instance(instance_id):
    # Number of tasks (n) between 3 to 7
    num_tasks = random.randint(3, 7)
    tasks = []
    for tid in range(1, num_tasks + 1):
        # precision of 2 decimal points
        c_m = round(random.uniform(1, 20), 2)  # Random float between 1 and 20 for c_m_i
        c_o = round(random.uniform(1, 15), 2) # Random float between 1 and 15 for c_o_i
        period = generate_random_period()
        tasks.append({
            "id": tid,
            "c_m": c_m,
            "c_o": c_o,
            "p": period
        })

    # Number of processors (h) between  
    num_processors = random.randint(2, 10)
    processors = []

    for pid in range(1, num_processors + 1):
        # Randomly decide how many frequency levels for this processor (between 3 and 7)
        f = random.randint(3, 7)
        
        # Generate f random frequency levels between 0 and 1 (excluding 0 and 1)
        freqs = sorted([round(random.uniform(0.05, 0.95), 2) for _ in range(f - 1)])
        
        # Add 1.0 as the maximum frequency
        freqs = freqs + [1.0]
    
        # Append processor with its frequency levels
        processors.append({
            "id": pid,
            "frequencies": freqs
        })


    return {
        "tasks": tasks,
        "processors": processors
    }

# Generate 100 instances
for i in range(1, 101):
    instance_data = generate_instance(i)
    filename = f"instance_{i:03d}.json"
    filepath = os.path.join(output_dir, filename)
    with open(filepath, "w") as f:
        json.dump(instance_data, f, indent=4)

print(" 100 JSON files created in the 'data' folder.")
