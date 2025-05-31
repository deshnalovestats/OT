from test import *
import os
import pandas as pd

def export_results_to_csv(result, problem: EnergyPerformanceOptimizationProblem, 
                         algorithm_name: str, save_dir: str = "results"):
    """Export optimization results to CSV for further analysis"""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    if len(result.F) == 0:
        print("No results to export!")
        return
    
    # Prepare data for export
    data = []
    
    for i, (energy_norm, penalty_norm) in enumerate(result.F):
        assignments = problem._decode_solution(result.X[i])
        
        # Calculate actual values
        actual_energy = problem._calculate_energy(assignments)
        actual_penalty = problem._calculate_performance_penalty(assignments)
        
        # Calculate processor utilizations
        proc_utils = {}
        for proc in problem.processors:
            proc_load = 0
            for j, assignment in enumerate(assignments):
                if assignment['processor_id'] == proc.id:
                    job = problem.jobs[j]
                    exec_time = job['c_m']
                    if assignment['execute_optional']:
                        exec_time += job['c_o']
                    proc_load += exec_time / assignment['frequency']
            proc_utils[f'proc_{proc.id}_utilization'] = (proc_load / problem.hyperperiod) * 100
        
        # Count optional executions
        optional_executed = sum(1 for a in assignments if a['execute_optional'])
        optional_percentage = (optional_executed / len(assignments)) * 100
        
        # Check feasibility
        is_feasible = problem._check_timing_constraints(assignments)
        
        row_data = {
            'solution_id': i,
            'algorithm': algorithm_name,
            'normalized_energy': energy_norm,
            'normalized_penalty': penalty_norm,
            'actual_energy': actual_energy,
            'actual_penalty': actual_penalty,
            'optional_executed_count': optional_executed,
            'optional_executed_percentage': optional_percentage,
            'is_feasible': is_feasible,
            **proc_utils
        }
        
        data.append(row_data)
    
    # Create DataFrame and save
    df = pd.DataFrame(data)
    filename = f'{save_dir}/results_{algorithm_name.lower()}.csv'
    df.to_csv(filename, index=False)
    print(f"Results exported to '{filename}'")
    
    return df

# Additional utility functions
def create_custom_system():
    """Allow user to create a custom task system"""
    print("\nCreating custom task system...")
    
    # Get number of tasks
    try:
        n_tasks = int(input("Enter number of tasks: "))
        if n_tasks <= 0:
            print("Invalid number of tasks!")
            return None, None
    except ValueError:
        print("Invalid input!")
        return None, None
    
    tasks = []
    for i in range(n_tasks):
        print(f"\nTask {i}:")
        try:
            c_m = float(input(f"  Mandatory execution time: "))
            c_o = float(input(f"  Optional execution time: "))
            period = float(input(f"  Period: "))
            
            if c_m <= 0 or c_o < 0 or period <= 0 or period < c_m + c_o:
                print("Invalid task parameters!")
                return None, None
            
            tasks.append(Task(id=i, c_m=c_m, c_o=c_o, period=period))
        except ValueError:
            print("Invalid input!")
            return None, None
    
    # Get number of processors
    try:
        n_processors = int(input(f"\nEnter number of processors: "))
        if n_processors <= 0:
            print("Invalid number of processors!")
            return None, None
    except ValueError:
        print("Invalid input!")
        return None, None
    
    processors = []
    for i in range(n_processors):
        print(f"\nProcessor {i}:")
        freq_input = input(f"  Enter frequency levels (comma-separated, e.g., 0.5,0.7,1.0): ")
        try:
            frequencies = [float(f.strip()) for f in freq_input.split(',')]
            frequencies = sorted([f for f in frequencies if 0 < f <= 1.0])  # Filter and sort
            
            if not frequencies:
                print("No valid frequencies!")
                return None, None
            
            processors.append(Processor(id=i, frequencies=frequencies))
        except ValueError:
            print("Invalid frequency input!")
            return None, None
    
    return tasks, processors

def summarize_pareto_front(result, problem: EnergyPerformanceOptimizationProblem):
    """Provide a summary of the Pareto front results"""
    print(f"\n{'='*60}")
    print("PARETO FRONT SUMMARY")
    print(f"{'='*60}")
    
    if len(result.F) == 0:
        print("No feasible solutions found!")
        return
    
    # Find extreme solutions
    energy_values = result.F[:, 0]
    penalty_values = result.F[:, 1]
    
    min_energy_idx = np.argmin(energy_values)
    min_penalty_idx = np.argmin(penalty_values)
    
    print(f"Total solutions in Pareto front: {len(result.F)}")
    
    print(f"\nBest Energy Solution (Index {min_energy_idx}):")
    print(f"  - Normalized Energy: {energy_values[min_energy_idx]:.4f}")
    print(f"  - Normalized Penalty: {penalty_values[min_energy_idx]:.4f}")
    
    print(f"\nBest Performance Solution (Index {min_penalty_idx}):")
    print(f"  - Normalized Energy: {energy_values[min_penalty_idx]:.4f}")
    print(f"  - Normalized Penalty: {penalty_values[min_penalty_idx]:.4f}")
    
    # Calculate some statistics
    energy_range = np.max(energy_values) - np.min(energy_values)
    penalty_range = np.max(penalty_values) - np.min(penalty_values)
    
    print(f"\nPareto Front Characteristics:")
    print(f"  - Energy range: {energy_range:.4f}")
    print(f"  - Performance penalty range: {penalty_range:.4f}")
    print(f"  - Average energy: {np.mean(energy_values):.4f}")
    print(f"  - Average penalty: {np.mean(penalty_values):.4f}")
    
    # Denormalize and show actual values
    e_min, e_max = problem.energy_bounds
    j_min, j_max = problem.performance_bounds
    
    actual_energies = energy_values * (e_max - e_min) + e_min
    actual_penalties = penalty_values * (j_max - j_min) + j_min
    
    print(f"\nActual Value Ranges:")
    print(f"  - Energy: {np.min(actual_energies):.2f} to {np.max(actual_energies):.2f}")
    print(f"  - Penalty: {np.min(actual_penalties):.2f} to {np.max(actual_penalties):.2f}")
