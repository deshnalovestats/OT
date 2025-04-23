import os
from loader import load_input_from_json
import modules  # for lcm
import MOPSO  # Assuming MOPSO class is in MOPSO.py
import vis

DATA_DIR = "data"
RESULT_DIR = "results_MOPSO"
os.makedirs(RESULT_DIR, exist_ok=True)

def lcm(a, b):
    return a * b // modules.np.gcd(a, b)

def lcm_multiple(numbers):
    result = 1
    for num in numbers:
        result = lcm(result, num)
    return result

for filename in sorted(os.listdir(DATA_DIR))[0:1]:  # Adjust the range as needed
    
    if filename.endswith(".json"):
        filepath = os.path.join(DATA_DIR, filename)
        print(f"\n Running optimization on: {filename}")
        
        # Load the task and processor data
        tasks, processors = load_input_from_json(filepath)

        # Compute hyperperiod
        hyperperiod = lcm_multiple([t.period for t in tasks])

        # Initialize and run the MOPSO algorithm
        mopso = MOPSO.MOPSO(tasks, processors, hyperperiod)  # Assuming MOPSO class is in MOPSO.py
        final_front, pareto_history = mopso.run(debug=True)
        final_front.sort(key=lambda x: x.energy)

        # Select solutions
        low_energy_sol = final_front[0]
        high_perf_sol = final_front[-1]
        balanced_sol = final_front[len(final_front)//2]

        # Output folder per instance
        instance_name = os.path.splitext(filename)[0]
        out_path = os.path.join(RESULT_DIR, instance_name)
        os.makedirs(out_path, exist_ok=True)

        # Save all plots
        vis.plot_pareto_front(final_front, f"Pareto Front - {instance_name}").savefig(f"{out_path}/pareto_front.png")
        vis.plt.close()
        vis.plot_pareto_evolution(pareto_history).savefig(f"{out_path}/pareto_evolution.png")
        vis.plt.close()
        vis.plot_schedule(low_energy_sol, processors, hyperperiod).savefig(f"{out_path}/low_energy_schedule.png")
        vis.plt.close()
        vis.plot_schedule(high_perf_sol, processors, hyperperiod).savefig(f"{out_path}/high_perf_schedule.png")
        vis.plt.close()
        vis.plot_schedule(balanced_sol, processors, hyperperiod).savefig(f"{out_path}/balanced_schedule.png")
        vis.plt.close()
        vis.plot_task_metrics(tasks, balanced_sol).savefig(f"{out_path}/task_metrics.png") 
        vis.plt.close()
        vis.plot_processor_utilization(balanced_sol, processors, hyperperiod, tasks).savefig(f"{out_path}/processor_utilization.png")
        vis.plt.close()
        vis.plot_frequency_distribution(balanced_sol).savefig(f"{out_path}/frequency_distribution.png") 
        vis.plt.close()

        comp_plot, metrics_df = vis.compare_solutions([low_energy_sol, balanced_sol, high_perf_sol], hyperperiod)
        comp_plot.savefig(f"{out_path}/solution_comparison.png")
        vis.plt.close()
        metrics_df.round(3).to_csv(f"{out_path}/metrics.csv")

        # Print completion message
        print(f"MOPSO optimization completed for {filename}.")
        print(f"Results saved to {out_path}")
