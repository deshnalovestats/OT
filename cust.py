from logging_config import LoggingFlags, log_if
import os
from nsga2_driver import run_nsga2_optimization, visualize_results, analyze_best_solutions
import json
import traceback
import sys
import time

def customisedLogging():
    # Main execution flags
    LoggingFlags.MAIN_MENU = True
    LoggingFlags.OPTIMIZATION_PROGRESS = True
    LoggingFlags.SOLUTION_ANALYSIS = True
    
    # Problem evaluation flags
    LoggingFlags.EVALUATION_DETAILS = False
    LoggingFlags.GENERATION_STATS = False
    LoggingFlags.FEASIBILITY_CHECKS = False
    LoggingFlags.REPAIR_OPERATIONS = False
    
    # Sampling and genetic operators flags
    LoggingFlags.SAMPLING_PROGRESS = False
    LoggingFlags.CROSSOVER_DETAILS = False
    LoggingFlags.MUTATION_DETAILS = False
    
    # Performance and debugging flags
    LoggingFlags.TIMING_CONSTRAINTS = False
    LoggingFlags.ENERGY_CALCULATION = False
    LoggingFlags.PERFORMANCE_CALCULATION = False

# Configure logging
# LoggingFlags.silent_run()
customisedLogging()

class LogCapture:
    """Capture console output to both file and console"""
    def __init__(self, log_file_path):
        self.terminal = sys.stdout
        self.log_file = open(log_file_path, 'w', encoding='utf-8')
    
    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)
        self.log_file.flush()
    
    def flush(self):
        self.terminal.flush()
        self.log_file.flush()
    
    def close(self):
        self.log_file.close()

def get_all_instances(base_folders=['Tasks_10', 'Tasks_20', 'Tasks_30']):
    """Get all JSON instance files from the specified folders"""
    all_instances = []
    
    for folder in base_folders:
        if os.path.exists(folder):
            json_files = [f for f in os.listdir(folder) if f.endswith('.json')]
            for json_file in sorted(json_files):
                instance_path = os.path.join(folder, json_file)
                all_instances.append((folder, json_file, instance_path))
            log_if(LoggingFlags.MAIN_MENU, f"Found {len(json_files)} instances in {folder}")
        else:
            log_if(LoggingFlags.MAIN_MENU, f"Warning: Folder {folder} not found!")
    
    return all_instances

def run_single_instance(instance_path, output_dir, n_generations=100, pop_size=100):
    """Run optimization for a single instance with comprehensive logging and error handling"""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    instance_name = os.path.splitext(os.path.basename(instance_path))[0]

    
    # Setup logging to file
    log_file_path = os.path.join(output_dir, "optimization_log.txt")
    log_capture = LogCapture(log_file_path)
    
    try:
        # Redirect stdout to capture all prints
        original_stdout = sys.stdout
        sys.stdout = log_capture
        
        log_if(LoggingFlags.MAIN_MENU, f"Running NSGA-II on instance: {instance_name}")

        # Record optimization start time
        opt_start_time = time.time()

           # Run optimization with error handling for repair failures
        try:
            result, problem = run_nsga2_optimization(
                instance_path,
                n_generations=n_generations,
                pop_size=pop_size
            )
            
            # Record optimization end time
            opt_end_time = time.time()
            optimization_time = opt_end_time - opt_start_time
            
            log_if(LoggingFlags.MAIN_MENU, f"\nOptimization completed successfully!")
            log_if(LoggingFlags.MAIN_MENU, f"Optimization time: {optimization_time:.2f} seconds")
            log_if(LoggingFlags.MAIN_MENU, f"Number of solutions in Pareto front: {len(result.F)}")
            
            # Save Pareto solutions data
            pareto_data = {
                'objectives': result.F.tolist(),
                'variables': result.X.tolist(),
                'optimization_time': optimization_time,
                'n_generations': n_generations,
                'pop_size': pop_size,
                'instance_file': instance_path
            }
            
            pareto_file = os.path.join(output_dir, "pareto_solutions.json")
            with open(pareto_file, 'w') as f:
                json.dump(pareto_data, f, indent=2)
            log_if(LoggingFlags.MAIN_MENU, f"Pareto solutions saved to: {pareto_file}")
            
            # Record visualization start time
            vis_start_time = time.time()
            
            visualize_results(result, problem, save_dir=output_dir)
            analyze_best_solutions(result, problem, save_dir=output_dir)
            
            # Record visualization end time
            vis_end_time = time.time()
            visualization_time = vis_end_time - vis_start_time
            
            # Restore logging
            sys.stdout = log_capture
            log_if(LoggingFlags.MAIN_MENU, f"Visualization time: {visualization_time:.2f} seconds")
            log_if(LoggingFlags.MAIN_MENU, f"Total time: {(opt_end_time - opt_start_time) + visualization_time:.2f} seconds")
            log_if(LoggingFlags.MAIN_MENU, f"All files saved to: {output_dir}")
            log_if(LoggingFlags.MAIN_MENU, f"Completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
            
            return {
                'success': True,
                'optimization_time': optimization_time,
                'visualization_time': visualization_time,
                'total_time': optimization_time + visualization_time,
                'n_solutions': len(result.F),
                'error_type': None
            }
            
        except ValueError as val_error:
            opt_end_time = time.time()
            optimization_time = opt_end_time - opt_start_time

            error_msg = str(val_error)

            if "Unable to generate enough feasible solutions" in error_msg:
                log_if(LoggingFlags.MAIN_MENU, f"\nREPAIR FAILURE - FEASIBILITY ISSUE:")
                log_if(LoggingFlags.MAIN_MENU, f"Error: {error_msg}")
                log_if(LoggingFlags.MAIN_MENU, f"Instance flagged as having repair/feasibility issues.")
                log_if(LoggingFlags.MAIN_MENU, f"This indicates the problem constraints are too tight or the instance is infeasible.")
                error_type = 'repair_failure'
            
            return {
                'success': False,
                'optimization_time': optimization_time,
                'visualization_time': 0,
                'total_time': optimization_time,
                'n_solutions': 0,
                'error_type': error_type
            }
    
    except Exception as e:
        log_if(LoggingFlags.MAIN_MENU, f"\nGENERAL ERROR: {str(e)}")
        traceback.print_exc()
        return {
            'success': False,
            'optimization_time': 0,
            'visualization_time': 0,
            'total_time': 0,
            'n_solutions': 0,
            'error_type': 'general_error'
        }
    
    finally:
        # Restore original stdout
        sys.stdout = original_stdout
        log_capture.close()

def run_all_instances(base_folders=['Tasks_10', 'Tasks_20', 'Tasks_30'], 
                          n_generations=100, pop_size=100):
    """Run batch optimization on all instances"""
    
    # Configure logging
    customisedLogging()
    
    log_if(LoggingFlags.MAIN_MENU, "="*80)
    log_if(LoggingFlags.MAIN_MENU, "NSGA2 BATCH OPTIMIZATION - REAL-TIME TASK SCHEDULING")
    log_if(LoggingFlags.MAIN_MENU, "="*80)
    
    # Create main results directory
    results_dir = "nsga2_results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Get all instances
    all_instances = get_all_instances(base_folders)
    
    if not all_instances:
        log_if(LoggingFlags.MAIN_MENU, "No instances found! Please check the folder paths.")
        return
    
    log_if(LoggingFlags.MAIN_MENU, f"Found {len(all_instances)} instances to process")
    
    # Track results
    successful_runs = []
    failed_runs = []
    repair_failures = []
    optimization_errors = []
    general_errors = []
    
    total_start_time = time.time()
    
    # Process each instance
    for i, (folder, filename, instance_path) in enumerate(all_instances, 1):
        log_if(LoggingFlags.MAIN_MENU, f"\n{'='*60}")
        log_if(LoggingFlags.MAIN_MENU, f"Processing {i}/{len(all_instances)}: {folder}/{filename}")
        log_if(LoggingFlags.MAIN_MENU, f"{'='*60}")
        
        # Create output directory structure
        instance_name = os.path.splitext(filename)[0]
        output_dir = os.path.join(results_dir, folder, instance_name).replace("data/"," ")
        
        # Run optimization for this instance
        result = run_single_instance(
            instance_path, output_dir, n_generations, pop_size
        )
        
        # Track results
        if result['success']:
            successful_runs.append({
                'folder': folder,
                'instance': filename,
                'optimization_time': result['optimization_time'],
                'visualization_time': result['visualization_time'],
                'total_time': result['total_time'],
                'solutions': result['n_solutions']
            })
            log_if(LoggingFlags.MAIN_MENU, f"✓ SUCCESS: {folder}/{filename} ({result['total_time']:.2f}s, {result['n_solutions']} solutions)")
        else:
            failed_runs.append({
                'folder': folder,
                'instance': filename,
                'error_type': result['error_type'],
                'time': result['total_time']
            })
            
            # Categorize failures
            if result['error_type'] == 'repair_failure':
                repair_failures.append(f"{folder}/{filename}")
            elif result['error_type'] == 'optimization_error':
                optimization_errors.append(f"{folder}/{filename}")
            else:
                general_errors.append(f"{folder}/{filename}")
            
            log_if(LoggingFlags.MAIN_MENU, f"✗ FAILED: {folder}/{filename} ({result['error_type']}, {result['total_time']:.2f}s)")
    
    # Final summary
    total_end_time = time.time()
    total_time = total_end_time - total_start_time
    
    log_if(LoggingFlags.MAIN_MENU, f"\n{'='*80}")
    log_if(LoggingFlags.MAIN_MENU, f"BATCH PROCESSING COMPLETED")
    log_if(LoggingFlags.MAIN_MENU, f"{'='*80}")
    log_if(LoggingFlags.MAIN_MENU, f"Total time: {total_time:.2f} seconds ({total_time/60:.1f} minutes)")
    log_if(LoggingFlags.MAIN_MENU, f"Total instances: {len(all_instances)}")
    log_if(LoggingFlags.MAIN_MENU, f"Successful: {len(successful_runs)}")
    log_if(LoggingFlags.MAIN_MENU, f"Failed: {len(failed_runs)}")
    log_if(LoggingFlags.MAIN_MENU, f"  - Repair failures: {len(repair_failures)}")
    log_if(LoggingFlags.MAIN_MENU, f"  - Optimization errors: {len(optimization_errors)}")
    log_if(LoggingFlags.MAIN_MENU, f"  - General errors: {len(general_errors)}")
    
    # Save comprehensive summary report
    summary_file = os.path.join(results_dir, "batch_summary.json")
    summary_data = {
        'total_instances': len(all_instances),
        'successful_runs': len(successful_runs),
        'failed_runs': len(failed_runs),
        'repair_failures': len(repair_failures),
        'optimization_errors': len(optimization_errors),
        'general_errors': len(general_errors),
        'total_time_seconds': total_time,
        'configuration': {
            'n_generations': n_generations,
            'pop_size': pop_size,
            'folders_processed': base_folders
        },
        'successful_details': successful_runs,
        'failed_details': failed_runs,
        'repair_failure_instances': repair_failures,
        'optimization_error_instances': optimization_errors,
        'general_error_instances': general_errors,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open(summary_file, 'w') as f:
        json.dump(summary_data, f, indent=2)
    
    log_if(LoggingFlags.MAIN_MENU, f"\nDetailed summary saved to: {summary_file}")
    
    # Print categorized failures
    if repair_failures:
        log_if(LoggingFlags.MAIN_MENU, f"\nRepair failure instances:")
        for instance in repair_failures:
            log_if(LoggingFlags.MAIN_MENU, f"  - {instance}")
    
    if optimization_errors:
        log_if(LoggingFlags.MAIN_MENU, f"\nOptimization error instances:")
        for instance in optimization_errors:
            log_if(LoggingFlags.MAIN_MENU, f"  - {instance}")
    
    if general_errors:
        log_if(LoggingFlags.MAIN_MENU, f"\nGeneral error instances:")
        for instance in general_errors:
            log_if(LoggingFlags.MAIN_MENU, f"  - {instance}")
    
    # Statistics for successful runs
    if successful_runs:
        avg_opt_time = sum(run['optimization_time'] for run in successful_runs) / len(successful_runs)
        avg_vis_time = sum(run['visualization_time'] for run in successful_runs) / len(successful_runs)
        avg_total_time = sum(run['total_time'] for run in successful_runs) / len(successful_runs)
        avg_solutions = sum(run['solutions'] for run in successful_runs) / len(successful_runs)
        
        log_if(LoggingFlags.MAIN_MENU, f"\nSuccessful runs statistics:")
        log_if(LoggingFlags.MAIN_MENU, f"  - Average optimization time: {avg_opt_time:.2f} seconds")
        log_if(LoggingFlags.MAIN_MENU, f"  - Average visualization time: {avg_vis_time:.2f} seconds")
        log_if(LoggingFlags.MAIN_MENU, f"  - Average total time: {avg_total_time:.2f} seconds")
        log_if(LoggingFlags.MAIN_MENU, f"  - Average Pareto solutions: {avg_solutions:.1f}")

if __name__ == "__main__":
    # Configuration
    folders_to_process = ['data/Tasks_10']
    generations = 100
    population_size = 100

    # Run batch optimization
    run_all_instances(
        base_folders=folders_to_process,
        n_generations=generations,
        pop_size=population_size
    )
