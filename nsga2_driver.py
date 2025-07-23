import json
import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter
import matplotlib.pyplot as plt
import os
from visualisation import *

# Import with logging
from test import EnergyPerformanceOptimizationProblem, Task, Processor
from CustFeasibleSampling import FeasibleBinarySampling
from CustJobLevelCrossover import JobLevelUniformCrossover
from CustJobLevelMutation import JobLevelMutation
from logging_config import LoggingFlags, log_if

def load_test_data(json_file: str):
    """Load test data from JSON file"""
    log_if(LoggingFlags.MAIN_MENU, f"Loading data from {json_file}")
    
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    tasks = [Task(t["id"], t["cm"], t["co"], t["period"]) for t in data["tasks"]]
    processors = [Processor(p["id"], p["frequencies"]) for p in data["processors"]]
    
    log_if(LoggingFlags.MAIN_MENU, f"Loaded {len(tasks)} tasks and {len(processors)} processors")
    return tasks, processors

def configure_nsga2(pop_size: int = 100):
    """Configure NSGA2 algorithm with custom operators"""
    algorithm = NSGA2(
        pop_size=pop_size,
        sampling=FeasibleBinarySampling(),
        crossover=JobLevelUniformCrossover(),
        mutation=JobLevelMutation(prob=0.2),
        eliminate_duplicates=True
    )
    
    log_if(LoggingFlags.OPTIMIZATION_PROGRESS, f"NSGA2 configured with population size: {pop_size}")
    return algorithm

def run_nsga2_optimization(json_file: str, n_generations: int = 100, pop_size: int = 100):
    """Run NSGA2 optimization on the given problem instance"""
    
    # Load data
    tasks, processors = load_test_data(json_file)

    log_if(LoggingFlags.OPTIMIZATION_PROGRESS, f"System configuration:")
    log_if(LoggingFlags.OPTIMIZATION_PROGRESS, f"- Tasks: {len(tasks)}")
    log_if(LoggingFlags.OPTIMIZATION_PROGRESS, f"- Processors: {len(processors)}")
    
    # Create problem
    log_if(LoggingFlags.OPTIMIZATION_PROGRESS, "Creating optimization problem...")
    problem = EnergyPerformanceOptimizationProblem(tasks, processors)

    log_if(LoggingFlags.OPTIMIZATION_PROGRESS, f"- Jobs in hyperperiod: {problem.n_jobs}")
    log_if(LoggingFlags.OPTIMIZATION_PROGRESS, f"- Decision variables: {problem.n_var}")
    log_if(LoggingFlags.OPTIMIZATION_PROGRESS, f"- Hyperperiod: {problem.hyperperiod}")

    # Configure algorithm
    algorithm = configure_nsga2(pop_size)
    
    # Run optimization
    log_if(LoggingFlags.OPTIMIZATION_PROGRESS, f"Starting NSGA2 optimization for {n_generations} generations...")
    
    result = minimize(
        problem,
        algorithm,
        ('n_gen', n_generations),
        verbose=LoggingFlags.OPTIMIZATION_PROGRESS,
        save_history=True,
        callback=lambda algorithm: problem._evaluate_callback(algorithm)
    )
    
    log_if(LoggingFlags.OPTIMIZATION_PROGRESS, f"Optimization completed!")
    log_if(LoggingFlags.OPTIMIZATION_PROGRESS, f"Pareto front size: {len(result.F)}")
    
    return result, problem

def visualize_results(result, problem, save_dir):
    """Create and save visualizations"""
    if len(result.F) == 0:
        log_if(LoggingFlags.SOLUTION_ANALYSIS, "No solutions to visualize")
        return
    
    # Create results directory
    os.makedirs(save_dir, exist_ok=True)

    log_if(LoggingFlags.SOLUTION_ANALYSIS, f"\n{'='*60}")
    log_if(LoggingFlags.SOLUTION_ANALYSIS, f"GENERATING VISUALIZATIONS")
    log_if(LoggingFlags.SOLUTION_ANALYSIS, f"{'='*60}")
    
    # 1. Pareto front visualization
    log_if(LoggingFlags.SOLUTION_ANALYSIS, "Creating Pareto front visualization...")
    visualize_pareto_front(result, "NSGA2", save_dir=save_dir)

    # 2. Hypervolume and diversity analysis
    log_if(LoggingFlags.SOLUTION_ANALYSIS, "Analyzing hypervolume and spread...")
    analyze_hypervolume_and_spread(result, "NSGA2", save_dir=save_dir)
    
    # 3. Pareto front summary table
    log_if(LoggingFlags.SOLUTION_ANALYSIS, "Generating Pareto front summary table...")
    summarize_pareto_table(result, "NSGA2", save_dir=save_dir)

    # 4. Multiple runs boxplot comparison
    log_if(LoggingFlags.SOLUTION_ANALYSIS, "Running multiple optimization runs for statistical analysis...")
    try:
        # Get algorithm configuration for multiple runs
        algorithm_class = NSGA2
        algorithm_params = {
            'pop_size': 100,
            'sampling': FeasibleBinarySampling(),
            'crossover': JobLevelUniformCrossover(),
            'mutation': JobLevelMutation(prob=0.2),
            'eliminate_duplicates': True
        }
        multiple_runs_boxplot(problem, "NSGA2", algorithm_params, algorithm_class, n_runs=20, n_gen=100, save_dir=save_dir)
        log_if(LoggingFlags.SOLUTION_ANALYSIS, "Multiple runs analysis completed")
    except Exception as e:
        log_if(LoggingFlags.SOLUTION_ANALYSIS, f"Warning: Multiple runs analysis failed: {e}")
    
    # 5. Pareto front evolution animation
    log_if(LoggingFlags.SOLUTION_ANALYSIS, "Creating Pareto front evolution animation...")
    try:
        animate_pareto_front(result, save_dir=save_dir)
        log_if(LoggingFlags.SOLUTION_ANALYSIS, "Pareto evolution animation saved")
    except Exception as e:
        log_if(LoggingFlags.SOLUTION_ANALYSIS, f"Warning: Animation creation failed: {e}")




def analyze_solution(solution_idx: int, result, problem, solution_type: str = "Solution", solution_name: str = None):
    """Analyze a specific solution from the Pareto front with logging"""
    if solution_idx >= len(result.X):
        log_if(LoggingFlags.SOLUTION_ANALYSIS, f"Invalid solution index. Available solutions: 0-{len(result.X)-1}")
        return None
    
    x = result.X[solution_idx]
    assignments = problem._decode_solution(x)
    if solution_name is None:
        solution_name = f"{solution_type} {solution_idx + 1}"
    log_if(LoggingFlags.SOLUTION_ANALYSIS, f"\nAnalysis of {solution_name}:")
    log_if(LoggingFlags.SOLUTION_ANALYSIS, f"Normalized Energy: {result.F[solution_idx, 0]:.4f}")
    log_if(LoggingFlags.SOLUTION_ANALYSIS, f"Normalized Performance Penalty: {result.F[solution_idx, 1]:.4f}")
    
    # Calculate actual values
    energy = problem._calculate_energy(assignments)
    penalty = problem._calculate_performance_penalty(assignments)
    
    log_if(LoggingFlags.SOLUTION_ANALYSIS, f"Actual Energy Consumption: {energy:.2f}")
    log_if(LoggingFlags.SOLUTION_ANALYSIS, f"Actual Performance Penalty: {penalty:.2f}")
    
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
    
    log_if(LoggingFlags.SOLUTION_ANALYSIS, f"\nOptional Execution Statistics:")
    log_if(LoggingFlags.SOLUTION_ANALYSIS, "Task ID | Total Jobs | Optional Executed | Percentage")
    log_if(LoggingFlags.SOLUTION_ANALYSIS, "-" * 50)
    for task in problem.tasks:
        total = task_total_counts[task.id]
        executed = task_optional_counts[task.id]
        percentage = (executed / total) * 100 if total > 0 else 0
        log_if(LoggingFlags.SOLUTION_ANALYSIS, f"{task.id:7d} | {total:10d} | {executed:17d} | {percentage:8.1f}%")
    
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
    
    log_if(LoggingFlags.SOLUTION_ANALYSIS, f"\nProcessor Utilization:")
    log_if(LoggingFlags.SOLUTION_ANALYSIS, "Processor | Jobs Assigned | Total Load | Utilization")
    log_if(LoggingFlags.SOLUTION_ANALYSIS, "-" * 48)
    total_system_load = 0
    for proc_id in range(problem.n_processors):
        load = processor_loads[proc_id]
        job_count = processor_job_counts[proc_id]
        utilization = (load / problem.hyperperiod) * 100
        total_system_load += load
        log_if(LoggingFlags.SOLUTION_ANALYSIS, f"{proc_id:9d} | {job_count:13d} | {load:10.2f} | {utilization:10.2f}%")
    
    avg_utilization = (total_system_load / (problem.n_processors * problem.hyperperiod)) * 100
    log_if(LoggingFlags.SOLUTION_ANALYSIS, f"Average system utilization: {avg_utilization:.2f}%")
    
    # Check if solution is feasible
    is_feasible = problem._check_timing_constraints(assignments)
    log_if(LoggingFlags.SOLUTION_ANALYSIS, f"\nSolution Feasibility: {'✓ FEASIBLE' if is_feasible else '✗ INFEASIBLE'}")
    
    return assignments

def analyze_best_solutions(result, problem, save_dir):
    """Analyze the best energy and performance solutions with detailed analysis"""
    if len(result.F) == 0:
        return
    
    # Find best solutions
    best_energy_idx = np.argmin(result.F[:, 0])
    best_perf_idx = np.argmin(result.F[:, 1])
    
    log_if(LoggingFlags.SOLUTION_ANALYSIS, f"\n{'='*80}")
    log_if(LoggingFlags.SOLUTION_ANALYSIS, f"DETAILED ANALYSIS OF BEST SOLUTIONS")
    log_if(LoggingFlags.SOLUTION_ANALYSIS, f"{'='*80}")
    
    # Analyze best energy solution
    log_if(LoggingFlags.SOLUTION_ANALYSIS, f"\n{'='*60}")
    log_if(LoggingFlags.SOLUTION_ANALYSIS, f"BEST ENERGY SOLUTION")
    log_if(LoggingFlags.SOLUTION_ANALYSIS, f"{'='*60}")
    assignments_energy = analyze_solution(best_energy_idx, result, problem, solution_name="Best Energy Solution")
    visualize_schedule(assignments_energy, problem, f"Best Energy", save_dir=save_dir)
    visualize_processor_utilization(assignments_energy, problem, f"Best Energy", save_dir=save_dir)

    
    # Check if performance solution is different
    if best_perf_idx != best_energy_idx:
        log_if(LoggingFlags.SOLUTION_ANALYSIS, f"\n{'='*60}")
        log_if(LoggingFlags.SOLUTION_ANALYSIS, f"BEST PERFORMANCE SOLUTION")
        log_if(LoggingFlags.SOLUTION_ANALYSIS, f"{'='*60}")
        analyze_solution(best_perf_idx, result, problem, solution_name="Best Performance Solution")
        visualize_schedule(assignments_energy, problem, f"Best Performance", save_dir=save_dir)
        visualize_processor_utilization(assignments_energy, problem, f"Best Performance", save_dir=save_dir)
    else:
        log_if(LoggingFlags.SOLUTION_ANALYSIS, f"Note: Best energy and best performance solutions are the same!")
        log_if(LoggingFlags.SOLUTION_ANALYSIS, f"This indicates a dominant solution that excels in both objectives.")

def main():
    """Main driver function"""
    print("="*60)
    print("NSGA2 REAL-TIME TASK SCHEDULING OPTIMIZER")
    print("="*60)
    
    # Configuration options
    print("\nLogging Configuration:")
    print("1. Production mode (minimal logging)")
    print("2. Debug mode (detailed logging)")
    print("3. Custom configuration")
    
    log_choice = input("Select logging mode (1-3): ").strip()
    
    if log_choice == '1':
        LoggingFlags.set_production_mode()
        print("Production mode enabled")
    elif log_choice == '2':
        LoggingFlags.enable_all_debug()
        print("Debug mode enabled")
    elif log_choice == '3':
        print("\nCustom Configuration:")
        print("Available flags:", [attr for attr in dir(LoggingFlags) if not attr.startswith('_') and attr.isupper()])
        # You can add interactive flag setting here
    
    # Problem configuration
    json_file = input("Enter JSON file path (default: Tasks_20_5/instance_fixedproc_001.json): ").strip()
    if not json_file:
        json_file = 'Tasks_20_5/instance_fixedproc_001.json'
    
    n_generations = int(input("Enter number of generations (default: 100): ") or "100")
    pop_size = int(input("Enter population size (default: 100): ") or "100")
    
    try:
        # Run optimization
        result, problem = run_nsga2_optimization(json_file, n_generations, pop_size)
        
        # Visualize and analyze results
        visualize_results(result, problem)
        analyze_best_solutions(result, problem)
        
        log_if(LoggingFlags.MAIN_MENU, "\n Visualisation and Analysis completed successfully!")
        log_if(LoggingFlags.MAIN_MENU, "Results saved in nsga2_results/")
        
    except FileNotFoundError:
        print(f"Error: File {json_file} not found")
    except Exception as e:
        print(f"Error during optimization: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()