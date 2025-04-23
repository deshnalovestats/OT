import modules
import task
import chromosome
import NSGA
import vis

if __name__ == "__main__":
    # Example from the paper: Adaptive Cruise Control tasks
    tasks = [
        task.Task(1, 20, 15, 100),  # Task τ1: Obstacle Detection, c_m=20ms, c_o=15ms, period=100ms
        task.Task(2, 10, 10, 100),  # Task τ2: Speed Profile Adjustment, c_m=10ms, c_o=10ms, period=100ms
    ]
    
    # Let's add some additional tasks for a more complex scenario
    tasks.extend([
        task.Task(3, 5, 5, 50),     # Task τ3: Sensor Data Collection, c_m=5ms, c_o=5ms, period=50ms
        task.Task(4, 15, 10, 200),  # Task τ4: Path Planning, c_m=15ms, c_o=10ms, period=200ms
        task.Task(5, 8, 7, 150),    # Task τ5: User Interface Update, c_m=8ms, c_o=7ms, period=150ms
    ])
    
    # Define processors with different frequency levels
    processors = [
        task.Processor(1, [0.2,0.5, 0.8, 1.0]),  # Processor 1 with 3 frequency levels
        task.Processor(2, [0.1,0.5, 0.9]),  # Processor 2 with 3 frequency levels
        task.Processor(3, [0.2,0.4, 0.6,0.7, 0.8]),  # Processor 3 with 3 frequency levels
    ]
    
    # # Tasks with their respective computation and optional execution times and periods
    # tasks = [
    #     task.Task(1, 2.98, 4.81, 100),  # Task τ1: id=1, c_m=2.98ms, c_o=4.81ms, period=100ms
    #     task.Task(2, 19.19, 13.41, 50),  # Task τ2: id=2, c_m=19.19ms, c_o=13.41ms, period=50ms
    #     task.Task(3, 15.86, 3.12, 100),  # Task τ3: id=3, c_m=15.86ms, c_o=3.12ms, period=100ms
    #     task.Task(4, 18.24, 2.86, 50),   # Task τ4: id=4, c_m=18.24ms, c_o=2.86ms, period=50ms
    #     task.Task(5, 5.9, 6.3, 100),     # Task τ5: id=5, c_m=5.9ms, c_o=6.3ms, period=100ms
    #     task.Task(6, 15.14, 14.54, 150), # Task τ6: id=6, c_m=15.14ms, c_o=14.54ms, period=150ms
    # ]

    # # Processors with their respective frequency levels
    # processors = [
    #     task.Processor(1, [0.19, 0.26, 0.45, 0.52, 1.0]),  # Processor 1 with 5 frequency levels
    #     task.Processor(2, [0.09, 0.17, 0.49, 0.8, 0.87, 1.0]),  # Processor 2 with 6 frequency levels
    # ]


    # Calculate hyperperiod (LCM of all periods)
    def lcm(a, b):
        return a * b // modules.np.gcd(a, b)
    
    def lcm_multiple(numbers):
        result = 1
        for num in numbers:
            result = lcm(result, num)
        return result
    
    hyperperiod = lcm_multiple([task.period for task in tasks])
    print(f"Hyperperiod: {hyperperiod}")
    
    # Initialize and run NSGA-II
    nsga = NSGA.NSGAII(tasks, processors, hyperperiod)
    final_front, pareto_history = nsga.run(debug=False)
    
    # Sort solutions for easier analysis
    final_front.sort(key=lambda x: x.energy)
    
    # Visualize results
    
    # 1. Plot final Pareto front
    pareto_plot = vis.plot_pareto_front(final_front, "Final Pareto Front")
    pareto_plot.savefig('pareto_front.png')
    
    # 2. Plot evolution of Pareto front
    evolution_plot = vis.plot_pareto_evolution(pareto_history)
    evolution_plot.savefig('pareto_evolution.png')
    
    # 3. Select and visualize some representative solutions
    # Low energy solution (first in sorted front)
    low_energy_sol = final_front[0]
    
    # High performance solution (last in sorted front)
    high_perf_sol = final_front[-1]
    
    # Balanced solution (middle of front)
    balanced_sol = final_front[len(final_front)//2]
    
    # Plot schedules
    low_energy_schedule = vis.plot_schedule(low_energy_sol, processors, hyperperiod)
    low_energy_schedule.savefig('low_energy_schedule.png')
    
    high_perf_schedule = vis.plot_schedule(high_perf_sol, processors, hyperperiod)
    high_perf_schedule.savefig('high_perf_schedule.png')
    
    balanced_schedule = vis.plot_schedule(balanced_sol, processors, hyperperiod)
    balanced_schedule.savefig('balanced_schedule.png')
    
    # 4. Plot task metrics for balanced solution
    task_metrics_plot = vis.plot_task_metrics(tasks, balanced_sol)
    task_metrics_plot.savefig('task_metrics.png')
    
    # 5. Plot processor utilization for balanced solution
    proc_util_plot = vis.plot_processor_utilization(balanced_sol, processors,hyperperiod,tasks)
    proc_util_plot.savefig('processor_utilization.png')
    
    # 6. Plot frequency distribution for balanced solution
    freq_dist_plot = vis.plot_frequency_distribution(balanced_sol)
    freq_dist_plot.savefig('frequency_distribution.png')
    
    # 7. Compare representative solutions
    comp_plot, metrics_df = vis.compare_solutions([low_energy_sol, balanced_sol, high_perf_sol], hyperperiod)
    comp_plot.savefig('solution_comparison.png')
    
    # Print detailed metrics
    print("\nDetailed Metrics Comparison:")
    print(metrics_df.round(3))
    
    print("\nNSGA-II Simulation Complete!")
    
    # Show plots if in interactive environment
    #modules.plt.show()