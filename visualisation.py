import os
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict
from test import *
import matplotlib.patches as patches
from pymoo.optimize import minimize
import imageio # type: ignore
import pandas as pd
from pymoo.vendor.hv import HyperVolume 

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

def analyze_hypervolume_and_spread(result, algorithm_name: str, ref_point=[1.2, 1.2], save_dir: str = "results"):
    """Analyze and plot hypervolume and diversity (spread) progress over generations."""
    if not hasattr(result, "history") or not result.history:
        print("No generation history available for hypervolume/diversity analysis.")
        return None, None

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    hypervolume_progress = []
    spread_progress = []

    hv = HyperVolume(referencePoint=ref_point)

    for gen in result.history:
        F = gen.pop.get("F")
        hypervolume_progress.append(hv.compute(F))
        pairwise_dist = np.linalg.norm(F[:, None, :] - F[None, :, :], axis=2)
        diversity = np.std(pairwise_dist)
        spread_progress.append(diversity)

    # Plotting
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(hypervolume_progress, marker='o')
    plt.title(f"Hypervolume Progress - {algorithm_name}")
    plt.xlabel("Generation")
    plt.ylabel("Hypervolume")

    plt.subplot(1, 2, 2)
    plt.plot(spread_progress, marker='s', color='orange')
    plt.title(f"Diversity Progress (Spread) - {algorithm_name}")
    plt.xlabel("Generation")
    plt.ylabel("Std. of Pairwise Distances")

    plt.tight_layout()
    save_path = os.path.join(save_dir, f"hv_spread_{algorithm_name.lower().replace(' ', '_')}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Hypervolume and diversity progress saved as '{save_path}'")
    plt.close()

    return hypervolume_progress, spread_progress

def summarize_pareto_table(result, algorithm_name: str, save_dir: str = "results"):
    """Print and save a summary table of the Pareto front."""
    if not hasattr(result, "F") or len(result.F) == 0:
        print("No Pareto front solutions to summarize.")
        return None

    summary = []
    for i in range(len(result.F)):
        summary.append({
            'ID': i,
            'Energy': result.F[i, 0],
            'Performance': result.F[i, 1]
        })
    df = pd.DataFrame(summary)
    print(f"\nPareto Front Summary for {algorithm_name}:")
    print(df.to_string(index=False, float_format='%.4f'))

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, f"pareto_summary_{algorithm_name.lower().replace(' ', '_')}.csv")
    df.to_csv(save_path, index=False)
    print(f"Pareto front summary saved as '{save_path}'")
    return df

def multiple_runs_boxplot(problem, algorithm_name, algorithm_params, algorithm_class, n_runs=20, n_gen=100, save_dir="results"):
    """Run multiple seeds and plot a boxplot of final hypervolume values."""
    hv_values = []
    ref_point = [1.2, 1.2]
    hv = HyperVolume(referencePoint=ref_point)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print(f"\n{'='*60}")
    print(f"Running {n_runs} runs for {algorithm_name} to analyze hypervolume distribution")
    print(f"{'='*60}")

    for seed in range(n_runs):
        np.random.seed(seed)
        random.seed(seed)
        algorithm = algorithm_class(**algorithm_params)
        result = minimize(problem, algorithm, ('n_gen', n_gen), verbose=False)
        hv_val = hv.compute(result.F)
        hv_values.append(hv_val)
        print(f"  Run {seed+1}/{n_runs}: Hypervolume = {hv_val:.4f}")

    plt.boxplot(hv_values)
    plt.title(f"Hypervolume Boxplot over {n_runs} seeds ({algorithm_name})")
    plt.ylabel("Hypervolume")
    plt.grid()
    save_path = os.path.join(save_dir, f"hv_boxplot_{algorithm_name.lower().replace(' ', '_')}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Hypervolume boxplot saved as '{save_path}'")
    plt.close()

    return hv_values

def animate_pareto_front(result, savefile="pareto_evolution.gif", save_dir="results"):
    """Create and save an animation of Pareto front evolution over generations."""
    import imageio
    if not hasattr(result, "history") or not result.history:
        print("No generation history available for animation.")
        return

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    images = []
    tmp_files = []
    for i, gen in enumerate(result.history):
        F = gen.pop.get("F")
        fig, ax = plt.subplots()
        ax.scatter(F[:, 0], F[:, 1], color='red')
        ax.set_title(f"Generation {i}")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Energy")
        ax.set_ylabel("Penalty")
        plt.tight_layout()
        fname = os.path.join(save_dir, f"tmp_{i}.png")
        plt.savefig(fname)
        plt.close()
        images.append(imageio.v2.imread(fname))
        tmp_files.append(fname)
    gif_path = os.path.join(save_dir, savefile)
    imageio.mimsave(gif_path, images, fps=2)
    # cleanup
    for fname in tmp_files:
        os.remove(fname)
    print(f"Saved Pareto front evolution animation to {gif_path}")