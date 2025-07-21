import json
import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter
import matplotlib.pyplot as plt
import os

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
    
    # Create problem
    log_if(LoggingFlags.OPTIMIZATION_PROGRESS, "Creating optimization problem...")
    problem = EnergyPerformanceOptimizationProblem(tasks, processors)
    
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

def visualize_results(result, problem, title: str = "NSGA2 Results"):
    """Create and save visualizations"""
    if len(result.F) == 0:
        log_if(LoggingFlags.SOLUTION_ANALYSIS, "No solutions to visualize")
        return
    
    # Create results directory
    os.makedirs("nsga2_results", exist_ok=True)
    
    # Pareto front plot
    plt.figure(figsize=(10, 8))
    plt.scatter(result.F[:, 0], result.F[:, 1], c='blue', s=50, alpha=0.7)
    plt.xlabel('Normalized Energy Consumption')
    plt.ylabel('Normalized Performance Penalty')
    plt.title(f'{title} - Pareto Front')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('nsga2_results/pareto_front.png', dpi=300, bbox_inches='tight')
    log_if(LoggingFlags.SOLUTION_ANALYSIS, "Pareto front saved to nsga2_results/pareto_front.png")
    plt.close()
    
    # Statistics
    if len(result.F) > 0:
        energy_vals = result.F[:, 0]
        penalty_vals = result.F[:, 1]
        
        log_if(LoggingFlags.SOLUTION_ANALYSIS, f"\n=== NSGA2 Results Summary ===")
        log_if(LoggingFlags.SOLUTION_ANALYSIS, f"Number of solutions: {len(result.F)}")
        log_if(LoggingFlags.SOLUTION_ANALYSIS, f"Energy range: [{np.min(energy_vals):.4f}, {np.max(energy_vals):.4f}]")
        log_if(LoggingFlags.SOLUTION_ANALYSIS, f"Performance range: [{np.min(penalty_vals):.4f}, {np.max(penalty_vals):.4f}]")
        log_if(LoggingFlags.SOLUTION_ANALYSIS, f"Best energy: {np.min(energy_vals):.4f}")
        log_if(LoggingFlags.SOLUTION_ANALYSIS, f"Best performance: {np.min(penalty_vals):.4f}")

def analyze_best_solutions(result, problem):
    """Analyze the best energy and performance solutions"""
    if len(result.F) == 0:
        return
    
    # Best energy solution
    best_energy_idx = np.argmin(result.F[:, 0])
    assignments = problem._decode_solution(result.X[best_energy_idx])
    
    log_if(LoggingFlags.SOLUTION_ANALYSIS, f"\n=== Best Energy Solution ===")
    log_if(LoggingFlags.SOLUTION_ANALYSIS, f"Energy: {result.F[best_energy_idx, 0]:.4f}")
    log_if(LoggingFlags.SOLUTION_ANALYSIS, f"Performance: {result.F[best_energy_idx, 1]:.4f}")
    
    # Count optional executions
    optional_count = sum(1 for assign in assignments if assign['execute_optional'])
    log_if(LoggingFlags.SOLUTION_ANALYSIS, f"Optional executions: {optional_count}/{len(assignments)}")
    
    # Best performance solution (if different)
    best_perf_idx = np.argmin(result.F[:, 1])
    if best_perf_idx != best_energy_idx:
        assignments_perf = problem._decode_solution(result.X[best_perf_idx])
        optional_count_perf = sum(1 for assign in assignments_perf if assign['execute_optional'])
        
        log_if(LoggingFlags.SOLUTION_ANALYSIS, f"\n=== Best Performance Solution ===")
        log_if(LoggingFlags.SOLUTION_ANALYSIS, f"Energy: {result.F[best_perf_idx, 0]:.4f}")
        log_if(LoggingFlags.SOLUTION_ANALYSIS, f"Performance: {result.F[best_perf_idx, 1]:.4f}")
        log_if(LoggingFlags.SOLUTION_ANALYSIS, f"Optional executions: {optional_count_perf}/{len(assignments_perf)}")

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
        
        log_if(LoggingFlags.MAIN_MENU, "\nOptimization completed successfully!")
        log_if(LoggingFlags.MAIN_MENU, "Results saved in nsga2_results/ directory")
        
    except FileNotFoundError:
        print(f"Error: File {json_file} not found")
    except Exception as e:
        print(f"Error during optimization: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()