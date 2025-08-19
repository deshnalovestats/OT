import json
import numpy as np
import pickle
import os
import time
from tqdm import tqdm
from pymoo.algorithms.moo.nsga2 import NSGA2
from visualisation import *
from pymoo.optimize import minimize

# Import with logging disabled
from logging_config import LoggingFlags
from test import EnergyPerformanceOptimizationProblem, Task, Processor
from CustFeasibleSampling import FeasibleBinarySampling
from CustJobLevelCrossover import JobLevelUniformCrossover
from CustJobLevelMutation import JobLevelMutation

# Disable ALL logging
LoggingFlags.disable_all_debug()

class SilentMultiRunDriver:
    """Minimal driver for running multiple optimization runs with only progress bar output"""
    
    def __init__(self, n_runs=20, n_generations=100, pop_size=100):
        self.n_runs = n_runs
        self.n_generations = n_generations
        self.pop_size = pop_size
    
    def load_test_data(self, json_file: str):
        """Load test data from JSON file silently"""
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        tasks = [Task(t["id"], t["cm"], t["co"], t["period"]) for t in data["tasks"]]
        processors = [Processor(p["id"], p["frequencies"]) for p in data["processors"]]
        
        return tasks, processors
    
    def configure_nsga2(self):
        """Configure NSGA2 algorithm with custom operators"""
        algorithm = NSGA2(
            pop_size=self.pop_size,
            sampling=FeasibleBinarySampling(),
            crossover=JobLevelUniformCrossover(),
            mutation=JobLevelMutation(prob=0.2),
            eliminate_duplicates=True
        )
        return algorithm
    
    def run_single_optimization(self, problem, seed):
        """Run a single optimization and return the result object"""
        try:
            algorithm = self.configure_nsga2()
            
            result = minimize(
                problem,
                algorithm,
                ('n_gen', self.n_generations),
                verbose=False,
                save_history=True,  # changed 
                seed=seed
            )
            
            return result, None
            
        except Exception as e:
            return None, str(e)
    
    def run_instance_multiple_times(self, instance_path, output_dir):
        """Run optimization multiple times for a single instance"""
        
        # Load problem data
        tasks, processors = self.load_test_data(instance_path)
        problem = EnergyPerformanceOptimizationProblem(tasks, processors)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate seeds for reproducibility
        base_seed = hash(instance_path) % (2**32)  # Instance-specific base seed
        seeds = [base_seed + i for i in range(self.n_runs)]
        
        # Save seed mapping
        seed_mapping = {
            'instance_path': instance_path,
            'base_seed': base_seed,
            'seeds': seeds,
            'run_mapping': {f'run_{i+1:03d}': seeds[i] for i in range(self.n_runs)}
        }
        
        with open(os.path.join(output_dir, 'seeds.pkl'), 'wb') as f:
            pickle.dump(seed_mapping, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        # Progress bar for this instance
        instance_name = os.path.basename(instance_path).replace('.json', '')
        pbar = tqdm(range(self.n_runs), 
                desc=f"  {instance_name}", 
                unit="run",
                leave=False,
                ncols=100,
                position=1,
                ascii=True)
        
        successful_runs = 0
        failed_runs = 0
        
        # Run multiple optimizations
        for run_id in pbar:
            seed = seeds[run_id]
            result, error = self.run_single_optimization(problem, seed)
            
            if result is not None:
                successful_runs += 1
                
                # Save successful result with seed information
                result_data = {
                    'result': result,
                    'seed': seed,
                    'run_id': run_id + 1,
                    'instance_path': instance_path,
                    'status': 'success',
                    'error': None
                }
                
                pickle_filename = f"run_{run_id+1:03d}_result.pkl"
                pickle_path = os.path.join(output_dir, pickle_filename)
                
                with open(pickle_path, 'wb') as f:
                    pickle.dump(result_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            else:
                failed_runs += 1
                
                # Save failed run information
                failed_data = {
                    'result': None,
                    'seed': seed,
                    'run_id': run_id + 1,
                    'instance_path': instance_path,
                    'status': 'failed',
                    'error': error
                }
                
                pickle_filename = f"run_{run_id+1:03d}_failed.pkl"
                pickle_path = os.path.join(output_dir, pickle_filename)
                
                with open(pickle_path, 'wb') as f:
                    pickle.dump(failed_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            # Update progress bar postfix
            pbar.set_postfix_str(f"OK:{successful_runs} FAIL:{failed_runs}")

        
        pbar.close()
        return successful_runs
    
    def process_folder(self, folder_path, base_output_dir="results_nsga2"):
        """Process all instances in a folder with minimal output"""
        
        if not os.path.exists(folder_path):
            print(f"Error: Folder {folder_path} not found!")
            return
        
        # Get all JSON files in the folder
        json_files = [f for f in os.listdir(folder_path) if f.endswith('.json')]
        json_files.sort()
        
        if not json_files:
            print(f"No JSON files found in {folder_path}")
            return
        
        folder_name = os.path.basename(folder_path)
        folder_output_dir = os.path.join(base_output_dir, folder_name)
        os.makedirs(folder_output_dir, exist_ok=True)
        
        # Main progress bar for instances
        main_pbar = tqdm(json_files, 
                    desc=f"{folder_name}", 
                    unit="instance",
                    ncols=120,
                    position=0,
                    ascii=True)
        
        total_successful = 0
        total_failed = 0
        total_runs = 0
        
        # Process each instance
        for json_file in main_pbar:
            instance_path = os.path.join(folder_path, json_file)
            instance_name = os.path.splitext(json_file)[0]
            instance_output_dir = os.path.join(folder_output_dir, instance_name)
            
            try:
                successful_runs = self.run_instance_multiple_times(instance_path, instance_output_dir)
                failed_runs = self.n_runs - successful_runs
                
                total_successful += successful_runs
                total_failed += failed_runs
                total_runs += self.n_runs
                
                # Update main progress bar
                success_rate = (total_successful / total_runs) * 100 if total_runs > 0 else 0
                main_pbar.set_postfix_str(f"Success:{total_successful}/{total_runs} ({success_rate:.1f}%) Failed:{total_failed}")

                
            except Exception as e:
                # Silent error handling - count entire instance as failed
                total_failed += self.n_runs
                total_runs += self.n_runs
                success_rate = (total_successful / total_runs) * 100 if total_runs > 0 else 0
                main_pbar.set_postfix_str(f"Success:{total_successful}/{total_runs} ({success_rate:.1f}%) Failed:{total_failed}")

        
        main_pbar.close()
        
        # Save final summary
        final_summary = {
            'folder_name': folder_name,
            'folder_output_dir': folder_output_dir,
            'total_instances': len(json_files),
            'total_runs': total_runs,
            'successful_runs': total_successful,
            'failed_runs': total_failed,
            'success_rate': (total_successful / total_runs) * 100 if total_runs > 0 else 0,
            'config': {
                'n_runs': self.n_runs,
                'n_generations': self.n_generations,
                'pop_size': self.pop_size
            },
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        summary_path = os.path.join(folder_output_dir, "final_summary.pkl")
        with open(summary_path, 'wb') as f:
            pickle.dump(final_summary, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        print(f"Completed: {total_successful}/{total_runs} successful runs ({final_summary['success_rate']:.1f}%)")
        print(f"Failed runs: {total_failed}")
        print(f"Results saved to: {folder_output_dir}")
        
        return final_summary

def load_result_with_seed(pickle_path):
    """Utility function to load a result with its seed information"""
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)
    return data['result'], data['seed']

def get_seeds_for_instance(instance_dir):
    """Get seed mapping for an instance"""
    seeds_path = os.path.join(instance_dir, 'seeds.pkl')
    if os.path.exists(seeds_path):
        with open(seeds_path, 'rb') as f:
            return pickle.load(f)
    return None

def main():
    print("NSGA2 Multi-Run Driver")
    print("=" * 40)
    
    # Simple configuration
    # folder_path = input("Folder path (e.g., Tasks_10): ").strip() or "Tasks_10"
    # n_runs = int(input("Runs per instance (default: 20): ") or "20")
    # n_generations = int(input("Generations (default: 100): ") or "100")
    # pop_size = int(input("Population size (default: 100): ") or "100")

    folder_path = "data/Tasks_20"
    n_runs = 20
    n_generations = 100
    pop_size = 100
    
    print(f"\nRunning {n_runs} × {len([f for f in os.listdir(folder_path) if f.endswith('.json')])} optimizations...")
    
    # Initialize and run
    driver = SilentMultiRunDriver(n_runs=n_runs, n_generations=n_generations, pop_size=pop_size)
    
    try:
        start_time = time.time()
        summary = driver.process_folder(folder_path)
        end_time = time.time()
        
        print(f"Total time: {end_time - start_time:.1f} seconds")
        
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()