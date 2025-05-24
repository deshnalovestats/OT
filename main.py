from test import *
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.algorithms.moo.rvea import RVEA
from pymoo.algorithms.moo.spea2 import SPEA2
from pymoo.algorithms.moo.age import AGEMOEA
from pymoo.util.ref_dirs import get_reference_directions
import pandas as pd
from typing import Dict, List, Optional
import os

def create_sample_system():
    """Create a sample system for testing"""
    # Create tasks (similar to ACC example in the paper)
    tasks = [
        Task(id=0, c_m=20, c_o=15, period=100),  # Obstacle Detection
        Task(id=1, c_m=10, c_o=10, period=100),  # Speed Profile Adjustment
        Task(id=2, c_m=5, c_o=8, period=50),     # Sensor Data Processing
        Task(id=3, c_m=15, c_o=5, period=200),   # Path Planning
    ]
    
    # Create processors with different frequency levels
    processors = [
        Processor(id=0, frequencies=[0.5, 0.7, 0.9, 1.0]),  # High-performance processor
        Processor(id=1, frequencies=[0.3, 0.5, 0.7]),       # Energy-efficient processor
        Processor(id=2, frequencies=[0.4, 0.6, 0.8, 1.0]),  # Balanced processor
    ]
    
    return tasks, processors

def get_available_algorithms():
    """Get dictionary of available multi-objective algorithms"""
    algorithms = {
        'NSGA2': {
            'class': NSGA2,
            'params': {
                'pop_size': 100,
                'sampling': BinaryRandomSampling(),
                'crossover': TwoPointCrossover(),
                'mutation': BitflipMutation(prob=0.1),
                'eliminate_duplicates': True
            },
            'description': 'Non-dominated Sorting Genetic Algorithm II'
        },
        'NSGA3': {
            'class': NSGA3,
            'params': {
                'pop_size': 100,
                'ref_dirs': get_reference_directions("das-dennis", 2, n_partitions=12),
                'sampling': BinaryRandomSampling(),
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
    print("Setting up optimization problem...")
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
        save_history=True
    )
    
    print(f"\nOptimization completed!")
    print(f"Number of solutions in Pareto front: {len(result.F)}")
    
    return result, problem

def visualize_pareto_front(result, algorithm_name: str, save_dir: str = "results"):
    """Create comprehensive Pareto front visualization"""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Create subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Pareto Front Analysis - {algorithm_name}', fontsize=16, fontweight='bold')
    
    if len(result.F) == 0:
        print("No solutions to visualize!")
        return
    
    energy_vals = result.F[:, 0]
    penalty_vals = result.F[:, 1]
    
    # 1. Basic Pareto Front Scatter Plot
    scatter = ax1.scatter(energy_vals, penalty_vals, c='red', s=60, alpha=0.7, edgecolors='black', linewidth=0.5)
    ax1.set_xlabel('Normalized Energy Consumption')
    ax1.set_ylabel('Normalized Performance Penalty')
    ax1.set_title('Pareto Front')
    ax1.grid(True, alpha=0.3)
    
    # Highlight extreme points
    min_energy_idx = np.argmin(energy_vals)
    min_penalty_idx = np.argmin(penalty_vals)
    ax1.scatter(energy_vals[min_energy_idx], penalty_vals[min_energy_idx], 
               c='blue', s=100, marker='*', label='Best Energy', edgecolors='black')
    ax1.scatter(energy_vals[min_penalty_idx], penalty_vals[min_penalty_idx], 
               c='green', s=100, marker='^', label='Best Performance', edgecolors='black')
    ax1.legend()
    
    # 2. Objective Distribution Histograms
    ax2.hist(energy_vals, bins=20, alpha=0.7, color='red', label='Energy', density=True)
    ax2.hist(penalty_vals, bins=20, alpha=0.7, color='blue', label='Penalty', density=True)
    ax2.set_xlabel('Normalized Objective Values')
    ax2.set_ylabel('Density')
    ax2.set_title('Objective Value Distributions')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Trade-off Analysis
    if len(result.F) > 1:
        # Calculate dominated hypervolume approximation
        sorted_indices = np.argsort(energy_vals)
        sorted_energy = energy_vals[sorted_indices]
        sorted_penalty = penalty_vals[sorted_indices]
        
        ax3.plot(sorted_energy, sorted_penalty, 'ro-', alpha=0.7, markersize=4)
        ax3.fill_between(sorted_energy, 0, sorted_penalty, alpha=0.3, color='lightblue')
        ax3.set_xlabel('Normalized Energy Consumption')
        ax3.set_ylabel('Normalized Performance Penalty')
        ax3.set_title('Trade-off Curve')
        ax3.grid(True, alpha=0.3)
    
    # 4. Solution Quality Metrics
    diversity = np.std(energy_vals) + np.std(penalty_vals)
    convergence = np.mean(np.sqrt(energy_vals**2 + penalty_vals**2))
    
    metrics_text = f"""Solution Quality Metrics:
    
    Total Solutions: {len(result.F)}
    Energy Range: {np.max(energy_vals) - np.min(energy_vals):.4f}
    Penalty Range: {np.max(penalty_vals) - np.min(penalty_vals):.4f}
    Diversity: {diversity:.4f}
    Convergence: {convergence:.4f}
    
    Best Energy: {np.min(energy_vals):.4f}
    Best Performance: {np.min(penalty_vals):.4f}"""
    
    ax4.text(0.05, 0.95, metrics_text, transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.axis('off')
    ax4.set_title('Quality Metrics')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/pareto_analysis_{algorithm_name.lower()}.png', dpi=300, bbox_inches='tight')
    print(f"Pareto front analysis saved as '{save_dir}/pareto_analysis_{algorithm_name.lower()}.png'")
    plt.close()

def visualize_schedule(assignments: List[Dict], problem: EnergyPerformanceOptimizationProblem, 
                      solution_name: str, save_dir: str = "results"):
    """Visualize the schedule as Gantt chart"""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Group assignments by processor
    processor_schedules = {p.id: [] for p in problem.processors}
    
    for i, assignment in enumerate(assignments):
        job = problem.jobs[i]
        execution_time = job['c_m']
        if assignment['execute_optional']:
            execution_time += job['c_o']
        
        actual_execution_time = execution_time / assignment['frequency']
        
        schedule_item = {
            'job_id': job['id'],
            'task_id': job['task_id'],
            'instance': job['instance'],
            'release_time': job['release_time'],
            'deadline': job['deadline'],
            'execution_time': actual_execution_time,
            'frequency': assignment['frequency'],
            'execute_optional': assignment['execute_optional'],
            'processor_id': assignment['processor_id']
        }
        
        processor_schedules[assignment['processor_id']].append(schedule_item)
    
    # Calculate actual start and finish times using EDF
    for proc_id, schedule in processor_schedules.items():
        if not schedule:
            continue
        
        # Sort by deadline (EDF)
        schedule.sort(key=lambda x: x['deadline'])
        
        current_time = 0
        for job in schedule:
            start_time = max(current_time, job['release_time'])
            finish_time = start_time + job['execution_time']
            job['start_time'] = start_time
            job['finish_time'] = finish_time
            current_time = finish_time
    
    # Create Gantt chart
    fig, axes = plt.subplots(len(problem.processors), 1, figsize=(16, 4 * len(problem.processors)))
    if len(problem.processors) == 1:
        axes = [axes]
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(problem.tasks)))
    task_colors = {task.id: colors[i] for i, task in enumerate(problem.tasks)}
    
    for proc_idx, (proc_id, schedule) in enumerate(processor_schedules.items()):
        ax = axes[proc_idx]
        
        if not schedule:
            ax.set_title(f'Processor {proc_id} (Empty)')
            ax.set_xlim(0, problem.hyperperiod)
            continue
        
        # Draw jobs
        for job in schedule:
            task_id = job['task_id']
            color = task_colors[task_id]
            
            # Main execution bar
            rect = patches.Rectangle(
                (job['start_time'], -0.4), 
                job['execution_time'], 0.8,
                linewidth=1, edgecolor='black', facecolor=color, alpha=0.8
            )
            ax.add_patch(rect)
            
            # Add job label
            label = f"T{task_id}J{job['instance']}"
            if job['execute_optional']:
                label += "*"  # Mark optional execution
            
            ax.text(job['start_time'] + job['execution_time']/2, 0, label,
                   ha='center', va='center', fontsize=8, fontweight='bold')
            
            # Draw deadline marker
            ax.axvline(x=job['deadline'], color='red', linestyle='--', alpha=0.5)
            ax.text(job['deadline'], 0.6, f"D{job['job_id']}", rotation=90, 
                   fontsize=6, ha='center', va='bottom', color='red')
        
        # Processor info
        total_utilization = sum(job['execution_time'] for job in schedule)
        utilization_percent = (total_utilization / problem.hyperperiod) * 100
        
        ax.set_title(f'Processor {proc_id} (Utilization: {utilization_percent:.1f}%)')
        ax.set_xlim(0, problem.hyperperiod)
        ax.set_ylim(-0.8, 0.8)
        ax.set_ylabel('Jobs')
        ax.grid(True, alpha=0.3)
        
        # Remove y-axis ticks
        ax.set_yticks([])
    
    # Set common x-label for bottom subplot
    axes[-1].set_xlabel('Time')
    
    # Add legend
    legend_elements = [patches.Patch(facecolor=task_colors[task.id], 
                                   label=f'Task {task.id}') for task in problem.tasks]
    legend_elements.append(patches.Patch(facecolor='none', label='* = Optional Executed'))
    fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98))
    
    plt.suptitle(f'Schedule Visualization - {solution_name}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{save_dir}/schedule_{solution_name.lower().replace(" ", "_")}.png', 
                dpi=300, bbox_inches='tight')
    print(f"Schedule visualization saved as '{save_dir}/schedule_{solution_name.lower().replace(' ', '_')}.png'")
    plt.close()

def visualize_processor_utilization(assignments: List[Dict], problem: EnergyPerformanceOptimizationProblem,
                                  solution_name: str, save_dir: str = "results"):
    """Visualize processor utilization and energy consumption"""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Calculate processor metrics
    processor_data = []
    
    for proc in problem.processors:
        proc_assignments = [a for i, a in enumerate(assignments) if a['processor_id'] == proc.id]
        
        total_load = 0
        total_energy = 0
        job_count = len(proc_assignments)
        optional_count = 0
        freq_usage = {}
        
        for i, assignment in enumerate(proc_assignments):
            job = problem.jobs[assignments.index(assignment)]
            exec_time = job['c_m']
            if assignment['execute_optional']:
                exec_time += job['c_o']
                optional_count += 1
            
            actual_exec_time = exec_time / assignment['frequency']
            energy = (assignment['frequency'] ** 2) * exec_time
            
            total_load += actual_exec_time
            total_energy += energy
            
            # Track frequency usage
            freq = assignment['frequency']
            freq_usage[freq] = freq_usage.get(freq, 0) + 1
        
        utilization = (total_load / problem.hyperperiod) * 100
        
        processor_data.append({
            'processor_id': proc.id,
            'utilization': utilization,
            'energy': total_energy,
            'job_count': job_count,
            'optional_count': optional_count,
            'freq_usage': freq_usage,
            'available_freqs': proc.frequencies
        })
    
    # Create visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Processor Analysis - {solution_name}', fontsize=16, fontweight='bold')
    
    proc_ids = [p['processor_id'] for p in processor_data]
    utilizations = [p['utilization'] for p in processor_data]
    energies = [p['energy'] for p in processor_data]
    job_counts = [p['job_count'] for p in processor_data]
    optional_counts = [p['optional_count'] for p in processor_data]
    
    # 1. Utilization bar chart
    bars1 = ax1.bar(proc_ids, utilizations, color='skyblue', alpha=0.8, edgecolor='black')
    ax1.set_xlabel('Processor ID')
    ax1.set_ylabel('Utilization (%)')
    ax1.set_title('Processor Utilization')
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, util in zip(bars1, utilizations):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{util:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # 2. Energy consumption
    bars2 = ax2.bar(proc_ids, energies, color='lightcoral', alpha=0.8, edgecolor='black')
    ax2.set_xlabel('Processor ID')
    ax2.set_ylabel('Energy Consumption')
    ax2.set_title('Energy Consumption by Processor')
    ax2.grid(True, alpha=0.3)
    
    # 3. Job distribution
    width = 0.35
    x = np.arange(len(proc_ids))
    bars3a = ax3.bar(x - width/2, job_counts, width, label='Total Jobs', 
                     color='lightgreen', alpha=0.8, edgecolor='black')
    bars3b = ax3.bar(x + width/2, optional_counts, width, label='Optional Executed',
                     color='orange', alpha=0.8, edgecolor='black')
    ax3.set_xlabel('Processor ID')
    ax3.set_ylabel('Number of Jobs')
    ax3.set_title('Job Distribution')
    ax3.set_xticks(x)
    ax3.set_xticklabels(proc_ids)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Frequency usage heatmap
    all_freqs = sorted(set(freq for p in processor_data for freq in p['available_freqs']))
    freq_matrix = np.zeros((len(proc_ids), len(all_freqs)))
    
    for i, proc_data in enumerate(processor_data):
        for j, freq in enumerate(all_freqs):
            if freq in proc_data['freq_usage']:
                freq_matrix[i, j] = proc_data['freq_usage'][freq]
    
    im = ax4.imshow(freq_matrix, cmap='YlOrRd', aspect='auto')
    ax4.set_xlabel('Frequency Levels')
    ax4.set_ylabel('Processor ID')
    ax4.set_title('Frequency Usage Heatmap')
    ax4.set_xticks(range(len(all_freqs)))
    ax4.set_xticklabels([f'{f:.1f}' for f in all_freqs])
    ax4.set_yticks(range(len(proc_ids)))
    ax4.set_yticklabels(proc_ids)
    
    # Add colorbar
    plt.colorbar(im, ax=ax4, label='Usage Count')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/processor_analysis_{solution_name.lower().replace(" ", "_")}.png',
                dpi=300, bbox_inches='tight')
    print(f"Processor analysis saved as '{save_dir}/processor_analysis_{solution_name.lower().replace(' ', '_')}.png'")
    plt.close()

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
        print("3. List available algorithms")
        print("4. Exit")
        
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
                        best_alg = input("Enter algorithm name for detailed analysis (or press Enter to skip): ").strip()
                        
                        if best_alg in results:
                            result = results[best_alg]
                            problem = problems[best_alg]
                            
                            if len(result.F) > 0:
                                best_energy_idx = np.argmin(result.F[:, 0])
                                assignments = analyze_solution(best_energy_idx, result, problem)
                                if assignments:
                                    visualize_schedule(assignments, problem, f"{best_alg} Best Energy (Comparison)")
                                    visualize_processor_utilization(assignments, problem, f"{best_alg} Best Energy (Comparison)")
                else:
                    print("Please select at least 2 algorithms for comparison!")
            except ValueError:
                print("Invalid input! Please enter valid numbers.")
        
        elif choice == '3':
            print("\nAvailable Multi-Objective Algorithms:")
            print("-" * 80)
            for name, info in algorithms.items():
                print(f"{name:12} | {info['description']}")
                # Show key parameters
                key_params = []
                if 'pop_size' in info['params']:
                    key_params.append(f"Pop: {info['params']['pop_size']}")
                if 'ref_dirs' in info['params']:
                    key_params.append("Uses reference directions")
                if key_params:
                    print(f"{'':<12} | Parameters: {', '.join(key_params)}")
                print()
        
        elif choice == '4':
            print("Goodbye!")
            break
        
        else:
            print("Invalid choice! Please enter 1, 2, 3, or 4.")

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

if __name__ == "__main__":
    main()