from test import *
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.operators.crossover.pntx import TwoPointCrossover
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.operators.sampling.rnd import BinaryRandomSampling
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.algorithms.moo.rvea import RVEA
from pymoo.algorithms.moo.spea2 import SPEA2
from pymoo.algorithms.moo.age import AGEMOEA
from pymoo.util.ref_dirs import get_reference_directions
import pandas as pd
from typing import Dict, List, Optional
import os
from visualisation import *
import json
import numpy as np
from CustFeasibleSampling import FeasibleBinarySampling
from CustJobLevelCrossover import JobLevelUniformCrossover
from CustJobLevelMutation import JobLevelMutation


def create_sample_system():
    """Create a sample system for testing"""
    json_file = 'Tasks_30/instance_fixedproc_032.json'
    # json_file = 'sample_data/deshna_test.json'
    with open(json_file, 'r') as f:
        data = json.load(f)

    tasks = [Task(t["id"], t["cm"], t["co"], t["period"]) for t in data["tasks"]]
    processors = [Processor(p["id"], p["frequencies"]) for p in data["processors"]]
    
    return tasks, processors


def get_available_algorithms():
    """Get dictionary of available multi-objective algorithms"""
    algorithms = {
        'NSGA2': {
            'class': NSGA2,
            'params': {
                'pop_size': 100,
                'sampling': FeasibleBinarySampling(),
                'crossover': JobLevelUniformCrossover(),
                'mutation': JobLevelMutation(prob=0.2),
                'eliminate_duplicates': True
            },
            'description': 'Non-dominated Sorting Genetic Algorithm II'
        },
        'NSGA3': {
            'class': NSGA3,
            'params': {
                'pop_size': 100,
                'ref_dirs': get_reference_directions("das-dennis", 2, n_partitions=12),
                'sampling': FeasibleBinarySampling(),
                'crossover': TwoPointCrossover(),
                'mutation': BitflipMutation(prob=0.1),
                'eliminate_duplicates': True
            },
            'description': 'Non-dominated Sorting Genetic Algorithm III'
        },
        'MOEAD': {
            'class': MOEAD,
            'params': {
                'ref_dirs': get_reference_directions("das-dennis", 2, n_partitions=12),
                'n_neighbors': 15,
                'prob_neighbor_mating': 0.7,
                'sampling': BinaryRandomSampling(),
                'crossover': TwoPointCrossover(),
                'mutation': BitflipMutation(prob=0.1)
            },
            'description': 'Multi-Objective Evolutionary Algorithm based on Decomposition'
        },
        'RVEA': {
            'class': RVEA,
            'params': {
                'ref_dirs': get_reference_directions("das-dennis", 2, n_partitions=12),
                'pop_size': 100,
                'sampling': BinaryRandomSampling(),
                'crossover': TwoPointCrossover(),
                'mutation': BitflipMutation(prob=0.1)
            },
            'description': 'Reference Vector Guided Evolutionary Algorithm'
        },
        'SPEA2': {
            'class': SPEA2,
            'params': {
                'pop_size': 100,
                'archive_size': 100,
                'sampling': BinaryRandomSampling(),
                'crossover': TwoPointCrossover(),
                'mutation': BitflipMutation(prob=0.1)
            },
            'description': 'Strength Pareto Evolutionary Algorithm 2'
        },
        'AGEMOEA': {
            'class': AGEMOEA,
            'params': {
                'pop_size': 100,
                'sampling': BinaryRandomSampling(),
                'crossover': TwoPointCrossover(),
                'mutation': BitflipMutation(prob=0.1)
            },
            'description': 'Age-based Multi-Objective Evolutionary Algorithm'
        }
    }
    return algorithms

def run_optimization(algorithm_name: str = 'NSGA2', n_generations: int = 100, verbose: bool = True):
    """Run optimization with specified algorithm"""
    print("Creating sample system...")
    tasks, processors = create_sample_system()
    
    print(f"System configuration:")
    print(f"- Tasks: {len(tasks)}")
    print(f"- Processors: {len(processors)}")
    
    # Create the optimization problem
    print("Setting up optimization problem")
    problem = EnergyPerformanceOptimizationProblem(tasks, processors)
    
    print(f"- Jobs in hyperperiod: {problem.n_jobs}")
    print(f"- Decision variables: {problem.n_var}")
    print(f"- Hyperperiod: {problem.hyperperiod}")
    
    # Get algorithm configuration
    algorithms = get_available_algorithms()
    if algorithm_name not in algorithms:
        print(f"Algorithm '{algorithm_name}' not available. Available algorithms:")
        for name, info in algorithms.items():
            print(f"  - {name}: {info['description']}")
        return None, None
    
    # Configure selected algorithm
    alg_config = algorithms[algorithm_name]
    algorithm = alg_config['class'](**alg_config['params'])
    
    print(f"Running {algorithm_name} optimization...")
    print(f"Description: {alg_config['description']}")
    
    # Run optimization
    result = minimize(
        problem,
        algorithm,
        ('n_gen', n_generations),
        verbose=verbose,
        save_history=True,
        callback=lambda algorithm: problem._evaluate_callback(algorithm)
    )
    
    print(f"\nOptimization completed!")
    print(f"Number of solutions in Pareto front: {len(result.F)}")
    
    return result, problem

def compare_algorithms(algorithms_to_compare: List[str], n_generations: int = 50):
    """Compare multiple algorithms and visualize results"""
    print("Comparing multiple algorithms...")
    
    results = {}
    problems = {}
    
    for alg_name in algorithms_to_compare:
        print(f"\n{'='*50}")
        print(f"Running {alg_name}")
        print(f"{'='*50}")
        
        result, problem = run_optimization(alg_name, n_generations, verbose=False)
        if result is not None:
            results[alg_name] = result
            problems[alg_name] = problem
            
            # Individual algorithm visualization
            visualize_pareto_front(result, alg_name)
        else:
            print(f"Failed to run {alg_name}")
    
    # Comparison visualization
    if len(results) > 1:
        plt.figure(figsize=(12, 8))
        colors = plt.cm.tab10(np.linspace(0, 1, len(results)))
        
        for i, (alg_name, result) in enumerate(results.items()):
            if len(result.F) > 0:
                plt.scatter(result.F[:, 0], result.F[:, 1], 
                           c=[colors[i]], s=50, alpha=0.7, label=alg_name)
        
        plt.xlabel('Normalized Energy Consumption')
        plt.ylabel('Normalized Performance Penalty')
        plt.title('Algorithm Comparison - Pareto Fronts')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if not os.path.exists("results"):
            os.makedirs("results")
        plt.savefig('results/algorithm_comparison.png', dpi=300, bbox_inches='tight')
        print("Algorithm comparison saved as 'results/algorithm_comparison.png'")
        plt.close()
        
        # Performance metrics comparison
        metrics_data = []
        for alg_name, result in results.items():
            if len(result.F) > 0:
                energy_vals = result.F[:, 0]
                penalty_vals = result.F[:, 1]
                
                metrics_data.append({
                    'Algorithm': alg_name,
                    'Solutions': len(result.F),
                    'Best Energy': np.min(energy_vals),
                    'Best Performance': np.min(penalty_vals),
                    'Avg Energy': np.mean(energy_vals),
                    'Avg Performance': np.mean(penalty_vals),
                    'Energy Range': np.max(energy_vals) - np.min(energy_vals),
                    'Performance Range': np.max(penalty_vals) - np.min(penalty_vals)
                })
        
        # Create metrics table
        if metrics_data:
            df = pd.DataFrame(metrics_data)
            print("\n" + "="*80)
            print("ALGORITHM COMPARISON METRICS")
            print("="*80)
            print(df.to_string(index=False, float_format='%.4f'))
    
    return results, problems

def analyze_solution(solution_idx: int, result, problem: EnergyPerformanceOptimizationProblem):
    """Analyze a specific solution from the Pareto front"""
    if solution_idx >= len(result.X):
        print(f"Invalid solution index. Available solutions: 0-{len(result.X)-1}")
        return None
    
    x = result.X[solution_idx]
    assignments = problem._decode_solution(x)
    
    print(f"\nAnalysis of Solution {solution_idx}:")
    print(f"Normalized Energy: {result.F[solution_idx, 0]:.4f}")
    print(f"Normalized Performance Penalty: {result.F[solution_idx, 1]:.4f}")
    
    # Calculate actual values
    energy = problem._calculate_energy(assignments)
    penalty = problem._calculate_performance_penalty(assignments)
    
    print(f"Actual Energy Consumption: {energy:.2f}")
    print(f"Actual Performance Penalty: {penalty:.2f}")
    
    # Count optional executions by task
    task_optional_counts = {}
    task_total_counts = {}
    for task in problem.tasks:
        task_optional_counts[task.id] = 0
        task_total_counts[task.id] = 0
    
    for i, assignment in enumerate(assignments):
        job = problem.jobs[i]
        task_id = job['task_id']
        task_total_counts[task_id] += 1
        if assignment['execute_optional']:
            task_optional_counts[task_id] += 1
    
    print(f"\nOptional Execution Statistics:")
    print("Task ID | Total Jobs | Optional Executed | Percentage")
    print("-" * 50)
    for task in problem.tasks:
        total = task_total_counts[task.id]
        executed = task_optional_counts[task.id]
        percentage = (executed / total) * 100 if total > 0 else 0
        print(f"{task.id:7d} | {total:10d} | {executed:17d} | {percentage:8.1f}%")
    
    # Show processor utilization
    processor_loads = {p.id: 0 for p in problem.processors}
    processor_job_counts = {p.id: 0 for p in problem.processors}
    
    for i, assignment in enumerate(assignments):
        job = problem.jobs[i]
        exec_time = job['c_m']
        if assignment['execute_optional']:
            exec_time += job['c_o']
        actual_exec_time = exec_time / assignment['frequency']
        processor_loads[assignment['processor_id']] += actual_exec_time
        processor_job_counts[assignment['processor_id']] += 1
    
    print(f"\nProcessor Utilization:")
    print("Processor | Jobs Assigned | Total Load | Utilization")
    print("-" * 48)
    for proc_id in range(problem.n_processors):
        load = processor_loads[proc_id]
        job_count = processor_job_counts[proc_id]
        utilization = (load / problem.hyperperiod) * 100
        print(f"{proc_id:9d} | {job_count:13d} | {load:10.2f} | {utilization:10.2f}%")
    
    # Check if solution is feasible
    is_feasible = problem._check_timing_constraints(assignments)
    print(f"\nSolution Feasibility: {'✓ FEASIBLE' if is_feasible else '✗ INFEASIBLE'}")
    
    return assignments

def main():
    """Main function with menu-driven interface"""
    algorithms = get_available_algorithms()
    
    while True:
        print("\n" + "="*60)
        print("MULTI-OBJECTIVE REAL-TIME TASK SCHEDULING OPTIMIZER")
        print("="*60)
        print("1. Run single algorithm")
        print("2. Compare multiple algorithms")
        print("3. Exit")
        
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == '1':
            print("\nAvailable algorithms:")
            alg_names = list(algorithms.keys())
            for i, name in enumerate(alg_names, 1):
                print(f"{i}. {name}: {algorithms[name]['description']}")
            
            try:
                alg_idx = int(input(f"\nSelect algorithm (1-{len(alg_names)}): ")) - 1
                if 0 <= alg_idx < len(alg_names):
                    selected_alg = alg_names[alg_idx]
                    n_gen = int(input("Enter number of generations (default 100): ") or "100")
                    
                    # Run optimization
                    result, problem = run_optimization(selected_alg, n_gen)
                    
                    if result is not None and len(result.F) > 0:
                        # Create visualizations
                        visualize_pareto_front(result, selected_alg)
                        # Analyze and plot hypervolume/diversity progress
                        analyze_hypervolume_and_spread(result, selected_alg)
                        # Print and save Pareto front summary table
                        summarize_pareto_table(result, selected_alg)
                        # For boxplot over multiple runs:
                        algorithm_class = algorithms[selected_alg]['class']
                        algorithm_params = algorithms[selected_alg]['params']
                        multiple_runs_boxplot(problem, selected_alg,algorithm_params, algorithm_class, n_runs=20, n_gen=100)
                        # For Pareto front animation:
                        animate_pareto_front(result, savefile=f"pareto_evolution_{selected_alg}.gif")

                        # Analyze best solutions
                        print("\n" + "="*60)
                        print("SOLUTION ANALYSIS")
                        print("="*60)
                        
                        # Best energy solution
                        best_energy_idx = np.argmin(result.F[:, 0])
                        assignments_energy = analyze_solution(best_energy_idx, result, problem)
                        if assignments_energy:
                            visualize_schedule(assignments_energy, problem, f"{selected_alg} Best Energy")
                            visualize_processor_utilization(assignments_energy, problem, f"{selected_alg} Best Energy")
                        
                        # Best performance solution
                        if len(result.F) > 1:
                            best_perf_idx = np.argmin(result.F[:, 1])
                            if best_perf_idx != best_energy_idx:
                                print("\n" + "-"*60)
                                assignments_perf = analyze_solution(best_perf_idx, result, problem)
                                if assignments_perf:
                                    visualize_schedule(assignments_perf, problem, f"{selected_alg} Best Performance")
                                    visualize_processor_utilization(assignments_perf, problem, f"{selected_alg} Best Performance")
                        
                        # Interactive solution selection
                        while True:
                            try:
                                sol_idx = input(f"\nAnalyze specific solution (0-{len(result.F)-1}) or 'q' to quit: ").strip()
                                if sol_idx.lower() == 'q':
                                    break
                                sol_idx = int(sol_idx)
                                assignments = analyze_solution(sol_idx, result, problem)
                                if assignments:
                                    visualize_schedule(assignments, problem, f"{selected_alg} Solution {sol_idx}")
                                    visualize_processor_utilization(assignments, problem, f"{selected_alg} Solution {sol_idx}")
                            except (ValueError, IndexError):
                                print("Invalid input. Please enter a valid solution index or 'q'.")
                                continue
                    else:
                        print("No feasible solutions found!")
                else:
                    print("Invalid algorithm selection!")
            except ValueError:
                print("Invalid input! Please enter a number.")
        
        elif choice == '2':
            print("\nAvailable algorithms:")
            alg_names = list(algorithms.keys())
            for i, name in enumerate(alg_names, 1):
                print(f"{i}. {name}")
            
            try:
                selected_indices = input("\nEnter algorithm numbers to compare (e.g., 1,2,3): ").strip()
                indices = [int(x.strip()) - 1 for x in selected_indices.split(',')]
                
                selected_algs = []
                for idx in indices:
                    if 0 <= idx < len(alg_names):
                        selected_algs.append(alg_names[idx])
                    else:
                        print(f"Invalid algorithm index: {idx + 1}")
                
                if len(selected_algs) >= 2:
                    n_gen = int(input("Enter number of generations (default 50): ") or "50")
                    results, problems = compare_algorithms(selected_algs, n_gen)
                    
                    # Allow detailed analysis of best algorithm
                    if results:
                        print(f"\nAlgorithms compared: {', '.join(results.keys())}")
                        selected_alg = input("Enter algorithm name for detailed analysis (or press Enter to skip): ").strip()
                        
                        if selected_alg in results:
                            result = results[selected_alg]
                            problem = problems[selected_alg]
                            
                            if len(result.F) > 0:
                                best_energy_idx = np.argmin(result.F[:, 0])
                                assignments = analyze_solution(best_energy_idx, result, problem)
                                if assignments:
                                    visualize_schedule(assignments, problem, f"{selected_alg} Best Energy (Comparison)")
                                    visualize_processor_utilization(assignments, problem, f"{selected_alg} Best Energy (Comparison)")
                else:
                    print("Please select at least 2 algorithms for comparison!")
            except ValueError:
                print("Invalid")
        
        
        elif choice == '3':
            print("END")
            break
        
        else:
            print("Invalid choice")



if __name__ == "__main__":
    main()