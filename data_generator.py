import json
import os
import random
import math

# === Settings ===
num_instances_required = 50
fixed_num_tasks = 20
fixed_num_processors = 5
num_freqs_per_processor = 5
fixed_proc_dir = "Tasks_20_5"
os.makedirs(fixed_proc_dir, exist_ok=True)

# === Normalized Frequency Pool (0.1 = 10% performance, 1.0 = 100% performance)
# As per problem statement: frequency values are normalized, Fjf ≤ 1.0
shared_freq_pool = sorted({round(i * 0.1, 1) for i in range(1, 11)})  # 0.1 to 1.0
print("Shared Frequency Pool (Normalized):", shared_freq_pool)

# === Helper Functions ===
def generate_harmonic_periods():
    """Generate harmonic periods to ensure LCM doesn't explode"""
    base_periods = [50, 100, 200]  # Harmonic relationship
    return random.choice([i for i in range(50, 300, 50)])


def generate_normalized_feasible_tasks(n, processors):
    """
    Generate tasks with execution times normalized relative to periods.
    Much more conservative approach to ensure feasibility.
    """
    tasks = []
    min_freq = min(min(p["frequencies"]) for p in processors)
    
    # MUCH more conservative utilization budget
    # Target only 40% total system utilization in worst case
    total_util_budget = fixed_num_processors * 1
    util_per_task = total_util_budget / n  # 0.1 per task for 20 tasks
    
    for i in range(n):
        period = generate_harmonic_periods()
        
        # Calculate max execution time based on very conservative budget
        # worst_case_util = (cm + co) / (min_freq * period) = util_per_task
        # Therefore: (cm + co) = util_per_task * period * min_freq
        max_total_exec = util_per_task * period * min_freq
        
        # Very conservative distribution
        cm_fraction = random.uniform(0.3, 0.6)  # 30-60% of budget
        co_fraction = random.uniform(0.2, 0.4)  # 20-40% of budget
        
        cm = round(max_total_exec * cm_fraction, 2)
        co = round(max_total_exec * co_fraction, 2)
        
        # Ensure absolute minimums
        cm = max(cm, 0.1)
        co = max(co, 0.1)
        
        tasks.append({
            "id": i + 1,
            "period": period,
            "cm": cm,
            "co": co
        })
    
    return tasks

def calculate_detailed_utilization(tasks, processors):
    """Calculate utilization with detailed breakdown"""
    min_freq = min(min(p["frequencies"]) for p in processors)
    max_freq = max(max(p["frequencies"]) for p in processors)
    
    # Best case: only mandatory at max frequency
    min_util = sum(t["cm"] / (max_freq * t["period"]) for t in tasks)
    
    # Worst case: mandatory + optional at min frequency
    max_util = sum((t["cm"] + t["co"]) / (min_freq * t["period"]) for t in tasks)
    
    # Individual feasibility
    infeasible_tasks = []
    for task in tasks:
        worst_time = (task["cm"] + task["co"]) / min_freq
        if worst_time > task["period"]:
            infeasible_tasks.append({
                'id': task['id'],
                'worst_time': worst_time,
                'period': task['period'],
                'cm': task['cm'],
                'co': task['co'],
                'ratio': worst_time / task['period']
            })
    
    return min_util, max_util, len(infeasible_tasks) == 0, infeasible_tasks

def validate_system_feasibility(tasks, processors):
    """Validate with VERY relaxed constraints"""
    min_util, max_util, individual_ok, bad_tasks = calculate_detailed_utilization(tasks, processors)
    
    # Check individual feasibility
    if not individual_ok:
        return False, f"Individual tasks infeasible: {len(bad_tasks)} tasks"
    
    # VERY relaxed system constraints
    if min_util > fixed_num_processors * 1.0:  # Even mandatory parts too much
        return False, f"Min utilization too high: {min_util:.2f}"
    
    # Allow very high max utilization due to frequency scaling
    if max_util > fixed_num_processors * 5.0:  # 25.0 for 5 processors
        return False, f"Max utilization too high: {max_util:.2f}"
    
    return True, f"OK (util: {min_util:.2f}-{max_util:.2f})"

def generate_simple_processors(num_processors, freq_pool, num_freqs):
    """Generate processors with guaranteed variety"""
    processors = []
    
    for i in range(num_processors):
        # Ensure each processor has a good spread of frequencies
        freqs = []
        
        # Always include high frequency for feasibility
        high_freqs = [f for f in freq_pool if f >= 0.7]
        freqs.append(random.choice(high_freqs))
        
        # Add remaining frequencies
        remaining = [f for f in freq_pool if f not in freqs]
        freqs.extend(random.sample(remaining, min(num_freqs - 1, len(remaining))))
        
        # Fill to required count
        while len(freqs) < num_freqs:
            freqs.append(random.choice(freq_pool))
        
        # Sort and deduplicate
        freqs = sorted(list(set(freqs)))[:num_freqs]
        
        processors.append({"id": i, "frequencies": freqs})
    
    return processors

# === Generate Simple Processors ===
fixed_processors = generate_simple_processors(fixed_num_processors, shared_freq_pool, num_freqs_per_processor)
print("\nProcessor Configurations:")
for p in fixed_processors:
    print(f"Processor {p['id']}: {p['frequencies']}")

# === Generate Feasible Task Sets ===
valid_count = 0
attempt = 0
max_attempts = 25000

print(f"\nGenerating {num_instances_required} feasible instances...")

while valid_count < num_instances_required and attempt < max_attempts:
    attempt += 1
    
    # Generate conservative tasks
    tasks = generate_normalized_feasible_tasks(fixed_num_tasks, fixed_processors)
    
    # Validate feasibility
    is_feasible, reason = validate_system_feasibility(tasks, fixed_processors)
    
    if not is_feasible:
        if attempt % 100 == 0:
            print(f"[Attempt {attempt}] Skipped: {reason}")
        continue
    
    # Additional checks with very relaxed bounds
    min_util, max_util, _, _ = calculate_detailed_utilization(tasks, fixed_processors)
    
    # Very relaxed quality checks
    if min_util < 0.1:  # Too trivial
        continue
    if max_util > fixed_num_processors * 4.0:  # Still too tight
        continue
    
    # Save the instance
    instance = {
        "tasks": tasks,
        "processors": fixed_processors
    }
    
    filename = f"instance_fixedproc_{valid_count+1:03d}.json"
    filepath = os.path.join(fixed_proc_dir, filename)
    with open(filepath, "w") as f:
        json.dump(instance, f, indent=4)
    
    if valid_count % 10 == 0 or valid_count < 5:
        print(f"[Attempt {attempt}] Saved: {filename}")
        print(f"  Utilization: {min_util:.2f} - {max_util:.2f}")
        print(f"  Status: {reason}")
    
    valid_count += 1

# === Summary ===
if attempt >= max_attempts:
    print(f"\nWarning: Reached maximum attempts ({max_attempts}). Generated {valid_count} instances.")
else:
    print(f"\nSuccessfully generated {valid_count} feasible instances in {attempt} attempts.")

print(f"\nGeneration Summary:")
print(f"- Frequency range: {min(shared_freq_pool)} - {max(shared_freq_pool)} (normalized)")
print(f"- Processors: {fixed_num_processors}")
print(f"- Tasks per instance: {fixed_num_tasks}")
print(f"- Valid instances: {valid_count}")
print(f"- Directory: {fixed_proc_dir}/")

# === Debug: Test single instance ===
print("\n=== DEBUG: Testing single instance ===")
test_tasks = generate_normalized_feasible_tasks(fixed_num_tasks, fixed_processors)
min_util, max_util, individual_ok, bad_tasks = calculate_detailed_utilization(test_tasks, fixed_processors)

print(f"Test instance utilization: {min_util:.3f} - {max_util:.3f}")
print(f"Individual feasibility: {individual_ok}")
if not individual_ok:
    print("Problematic tasks:")
    for task in bad_tasks[:3]:  # Show first 3
        print(f"  Task {task['id']}: {task['worst_time']:.2f} > {task['period']} (ratio: {task['ratio']:.2f})")

is_feasible, reason = validate_system_feasibility(test_tasks, fixed_processors)
print(f"System feasibility: {is_feasible} - {reason}")