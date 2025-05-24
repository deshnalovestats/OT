from modules import plt,defaultdict,sns,pd,np


# Visualization functions
def plot_pareto_front(front, title="Pareto Front"):
    """Plot Pareto front showing energy-performance trade-off"""
    energy_values = [p.energy for p in front]
    perf_values = [p.performance for p in front]
    
    plt.figure(figsize=(10, 6))
    plt.scatter(energy_values, perf_values, c='blue', s=50)
    plt.xlabel('Energy Consumption')
    plt.ylabel('Performance Penalty')
    plt.title(title)
    plt.grid(True)
    
    # Add point labels
    for i, point in enumerate(front):
        plt.annotate(f"Solution {i+1}", 
                     (point.energy, point.performance),
                     textcoords="offset points",
                     xytext=(5, 5),
                     ha='left')
    
    plt.tight_layout()
    return plt

def plot_pareto_evolution(pareto_history, step=10):
    """Plot evolution of Pareto front over generations"""
    plt.figure(figsize=(12, 8))
    
    # Plot initial, some intermediate, and final fronts
    gens_to_plot = [0, 
                    min(step, len(pareto_history)-1), 
                    min(2*step, len(pareto_history)-1),
                    min(3*step, len(pareto_history)-1),
                    len(pareto_history)-1]
    gens_to_plot = sorted(list(set(gens_to_plot)))
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(gens_to_plot)))
    
    for i, gen_idx in enumerate(gens_to_plot):
        front = pareto_history[gen_idx]
        energy_values = [p.energy for p in front]
        perf_values = [p.performance for p in front]
        
        plt.scatter(energy_values, perf_values, 
                   c=[colors[i]], 
                   label=f"Generation {gen_idx+1}",
                   s=30)
    
    plt.xlabel('Energy Consumption')
    plt.ylabel('Performance Penalty')
    plt.title('Evolution of Pareto Front Over Generations')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    return plt

def plot_schedule(solution, processors, hyperperiod):
    """Visualize the schedule for a selected solution"""
    schedule = solution.get_schedule()
    
    plt.figure(figsize=(14, len(processors) * 1.5))
    
    for proc in processors:
        proc_jobs = [job for job in schedule if job['processor'] == proc.id]
        
        for job in proc_jobs:
            # Mandatory part
            color = 'blue' if job['optional'] == 0 else 'green'
            plt.barh(f"P{proc.id}", 
                    width=job['end'] - job['start'], 
                    left=job['start'], 
                    height=0.5, 
                    color=color,
                    alpha=0.7)
            
            # Show job ID
            plt.text(job['start'] + (job['end'] - job['start'])/2, 
                     f"P{proc.id}", 
                     f"T{job['task_id']}_{job['job_id'].split('_')[1]}", 
                     ha='center', 
                     va='center',
                     fontsize=8)
            
            # Show deadline
            plt.axvline(x=job['deadline'], color='red', alpha=0.3, linestyle='--', linewidth=0.5)
    
    plt.xlim(0, hyperperiod)
    plt.xlabel('Time')
    plt.ylabel('Processor')
    plt.title(f'Schedule (Energy: {solution.energy:.2f}, Performance Penalty: {solution.performance:.2f})')
    plt.grid(True, axis='x')
    plt.tight_layout()
    return plt

def plot_task_metrics(tasks, solution):
    """Plot metrics for each task in the solution"""
    schedule = solution.get_schedule()
    
    # Extract task-level metrics
    task_metrics = defaultdict(lambda: {'mandatory': 0, 'optional': 0, 'skipped': 0})
    
    for job in schedule:
        task_id = job['task_id']
        if job['optional'] == 1:
            task_metrics[task_id]['optional'] += 1
        else:
            task_metrics[task_id]['skipped'] += 1
        task_metrics[task_id]['mandatory'] += 1
    
    # Prepare data for plotting
    task_ids = sorted(task_metrics.keys())
    mandatory_counts = [task_metrics[t]['mandatory'] for t in task_ids]
    optional_counts = [task_metrics[t]['optional'] for t in task_ids]
    skipped_counts = [task_metrics[t]['skipped'] for t in task_ids]
    
    # Plot
    plt.figure(figsize=(10, 6))
    
    x = range(len(task_ids))
    width = 0.3
    
    plt.bar([i - width for i in x], mandatory_counts, width, label='Mandatory', color='blue')
    plt.bar(x, optional_counts, width, label='Optional Executed', color='green')
    plt.bar([i + width for i in x], skipped_counts, width, label='Optional Skipped', color='red')
    
    plt.xlabel('Task ID')
    plt.ylabel('Count')
    plt.title('Task Execution Metrics')
    plt.xticks(x, [f"Task {t}" for t in task_ids])
    plt.legend()
    plt.grid(True, axis='y')
    plt.tight_layout()
    return plt

def plot_processor_utilization(solution, processors, hyperperiod,tasks):
    """Plot processor utilization"""
    schedule = solution.get_schedule()
    
    # Calculate processor utilization
    proc_util = defaultdict(lambda: {'mandatory': 0, 'optional': 0, 'total_time': 0})
    
    for job in schedule:
        proc_id = job['processor']
        job_time = job['end'] - job['start']
        
        if job['optional'] == 1:
            # Approximate split between mandatory and optional
            task_id = job['task_id']
            task = next(t for t in tasks if t.id == task_id)
            freq = job['frequency']
            
            m_time = task.c_m / freq
            o_time = task.c_o / freq
            
            proc_util[proc_id]['mandatory'] += m_time
            proc_util[proc_id]['optional'] += o_time
        else:
            proc_util[proc_id]['mandatory'] += job_time
        
        proc_util[proc_id]['total_time'] += job_time
    
    # Prepare data for plotting
    proc_ids = sorted(proc_util.keys())
    mandatory_times = [proc_util[p]['mandatory'] for p in proc_ids]
    optional_times = [proc_util[p]['optional'] for p in proc_ids]
    total_times = [proc_util[p]['total_time'] for p in proc_ids]
    
    # Calculate percentage of hyperperiod
    utilization = [t / hyperperiod * 100 for t in total_times]
    
    # Plot
    plt.figure(figsize=(10, 6))
    
    x = range(len(proc_ids))
    width = 0.4
    
    plt.bar(x, mandatory_times, width, label='Mandatory', color='blue')
    plt.bar(x, optional_times, width, bottom=mandatory_times, label='Optional', color='green')
    
    # Add utilization percentage
    for i, v in enumerate(utilization):
        plt.text(i, total_times[i] + 1, f"{v:.1f}%", ha='center')
    
    plt.xlabel('Processor ID')
    plt.ylabel('Time Units')
    plt.title('Processor Utilization')
    plt.xticks(x, [f"P{p}" for p in proc_ids])
    plt.legend()
    plt.grid(True, axis='y')
    plt.tight_layout()
    return plt

def plot_frequency_distribution(solution):
    """Plot frequency distribution across processors"""
    schedule = solution.get_schedule()
    
    # Group jobs by processor and frequency
    freq_dist = defaultdict(lambda: defaultdict(int))
    
    for job in schedule:
        proc_id = job['processor']
        freq = job['frequency']
        job_time = job['end'] - job['start']
        
        freq_dist[proc_id][freq] += job_time
    
    # Prepare data for plotting
    proc_ids = sorted(freq_dist.keys())
    unique_freqs = sorted(set(freq for proc_freqs in freq_dist.values() for freq in proc_freqs))
    
    # Create DataFrame for easier plotting
    data = []
    for proc_id in proc_ids:
        for freq in unique_freqs:
            data.append({
                'Processor': f"P{proc_id}",
                'Frequency': freq,
                'Usage Time': freq_dist[proc_id][freq]
            })
    
    df = pd.DataFrame(data)
    
    # Create pivot table
    pivot = df.pivot(index='Processor', columns='Frequency', values='Usage Time').fillna(0)
    
    # Plot
    plt.figure(figsize=(12, 8))
    ax = sns.heatmap(pivot, annot=True, cmap='YlGnBu', fmt='.1f')
    plt.title('Processor Frequency Usage (Time Units)')
    plt.tight_layout()
    return plt

def evaluate_solution(solution, hyperperiod):
    """Evaluate a solution with detailed metrics"""
    schedule = solution.get_schedule()
    
    # 1. Calculate energy efficiency
    total_energy = solution.energy
    energy_per_time = total_energy / hyperperiod
    
    # 2. Calculate performance metrics
    total_optional = sum(1 for job in schedule if job['optional'] == 1)
    total_jobs = len(schedule)
    optional_ratio = total_optional / total_jobs
    
    # 3. Calculate processor utilization
    proc_util = defaultdict(float)
    for job in schedule:
        proc_id = job['processor']
        job_time = job['end'] - job['start']
        proc_util[proc_id] += job_time
    
    avg_util = sum(proc_util.values()) / (len(proc_util) * hyperperiod)
    max_util = max(proc_util.values()) / hyperperiod
    min_util = min(proc_util.values()) / hyperperiod
    
    # 4. Calculate slack (time between job completion and deadline)
    total_slack = sum(job['deadline'] - job['end'] for job in schedule)
    avg_slack = total_slack / len(schedule)
    
    # 5. Calculate frequency distribution
    freq_dist = defaultdict(int)
    weighted_time = 0
    for job in schedule:
        freq = job['frequency']
        job_time = job['end'] - job['start']
        freq_dist[freq] += job_time
        weighted_time += freq * job_time
    
    avg_freq = weighted_time / sum(freq_dist.values())
    
    # Return comprehensive metrics
    return {
        'energy_total': total_energy,
        'energy_per_time': energy_per_time,
        'performance_penalty': solution.performance,
        'optional_executed_ratio': optional_ratio,
        'avg_processor_util': avg_util,
        'max_processor_util': max_util,
        'min_processor_util': min_util,
        'avg_slack': avg_slack,
        'avg_frequency': avg_freq,
        'frequency_distribution': freq_dist
    }

def compare_solutions(solutions, hyperperiod):
    """Compare multiple solutions from the Pareto front"""
    metrics = []
    
    for i, sol in enumerate(solutions):
        metric = evaluate_solution(sol, hyperperiod)
        metric['solution_id'] = i
        metrics.append(metric)
    
    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(metrics)
    
    # Plot comparison
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Energy vs Performance
    axes[0, 0].scatter(df['energy_total'], df['performance_penalty'])
    axes[0, 0].set_xlabel('Energy Consumption')
    axes[0, 0].set_ylabel('Performance Penalty')
    axes[0, 0].set_title('Energy vs Performance Trade-off')
    axes[0, 0].grid(True)
    
    # Energy vs Optional Ratio
    axes[0, 1].scatter(df['energy_total'], df['optional_executed_ratio'])
    axes[0, 1].set_xlabel('Energy Consumption')
    axes[0, 1].set_ylabel('Optional Tasks Executed Ratio')
    axes[0, 1].set_title('Energy vs Optional Task Execution')
    axes[0, 1].grid(True)
    
    # Avg Frequency vs Utilization
    axes[1, 0].scatter(df['avg_frequency'], df['avg_processor_util'])
    axes[1, 0].set_xlabel('Average Frequency')
    axes[1, 0].set_ylabel('Average Processor Utilization')
    axes[1, 0].set_title('Frequency vs Utilization')
    axes[1, 0].grid(True)
    
    # Energy vs Slack
    axes[1, 1].scatter(df['energy_total'], df['avg_slack'])
    axes[1, 1].set_xlabel('Energy Consumption')
    axes[1, 1].set_ylabel('Average Slack Time')
    axes[1, 1].set_title('Energy vs Slack Time')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    return plt, df
