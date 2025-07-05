import numpy as np
import random
from typing import List, Tuple, Dict
from dataclasses import dataclass
from pymoo.core.problem import Problem
from pymoo.core.sampling import Sampling

@dataclass
class Task:
    """Represents a real-time task with mandatory and optional components"""
    id: int
    c_m: float  
    c_o: float  
    period: float  
    
    def __post_init__(self):
        self.deadline = self.period  

@dataclass
class Processor:
    """Represents a DVFS-enabled processor"""
    id: int
    frequencies: List[float] 
    
    def get_min_frequency(self):
        return min(self.frequencies)
    
    def get_max_frequency(self):
        return max(self.frequencies)

class EnergyPerformanceOptimizationProblem(Problem):
    """
    Multi-objective optimization problem for energy and performance trade-offs
    in real-time task scheduling with hard deadlines
    """
    
    def __init__(self, tasks: List[Task], processors: List[Processor]):
        self.tasks = tasks
        self.processors = processors
        self.n_tasks = len(tasks)
        self.n_processors = len(processors)
        
        # Calculate hyperperiod
        self.hyperperiod = self._calculate_hyperperiod()
        
        # Generate all jobs in hyperperiod
        self.jobs = self._generate_jobs()
        self.n_jobs = len(self.jobs)
        """
        Calculate decision variable dimensions
        For each job: processor assignment (log2(n_processors) bits) + 
                      frequency assignment (log2(max_frequencies) bits) + 
                      optional execution (1 bit)
        """
        max_freq_levels = max(len(p.frequencies) for p in processors)
        self.bits_per_processor = int(np.ceil(np.log2(self.n_processors)))
        self.bits_per_frequency = int(np.ceil(np.log2(max_freq_levels)))
        self.bits_per_job = self.bits_per_processor + self.bits_per_frequency + 1  # +1 for optional execution
        
        n_var = self.n_jobs * self.bits_per_job
        
        super().__init__(n_var=n_var, n_obj=2, n_constr=0, xl=0, xu=1, vtype=bool)
        
        
        self.energy_bounds = self._calculate_energy_bounds()
        self.performance_bounds = self._calculate_performance_bounds()
    
    def _calculate_hyperperiod(self) -> float:
        """Calculate the hyperperiod (LCM of all task periods)"""
        def gcd(a, b):
            while b:
                a, b = b, a % b
            return a
        
        def lcm(a, b):
            return abs(a * b) // gcd(a, b)
        
        periods = [int(task.period) for task in self.tasks]
        result = periods[0]
        for period in periods[1:]:
            result = lcm(result, period)
        return float(result)
    
    def _generate_jobs(self) -> List[Dict]:
        """Generate all job instances in the hyperperiod"""
        jobs = []
        job_id = 0
        
        for task in self.tasks:
            n_instances = int(self.hyperperiod / task.period)
            for instance in range(n_instances):
                job = {
                    'id': job_id,
                    'task_id': task.id,
                    'instance': instance,
                    'release_time': instance * task.period,
                    'deadline': (instance + 1) * task.period,
                    'c_m': task.c_m,
                    'c_o': task.c_o
                }
                jobs.append(job)
                job_id += 1
        
        return jobs
    
    def _calculate_energy_bounds(self) -> Tuple[float, float]:
        """Calculate theoretical energy consumption bounds for normalization"""
        
        min_freq = min(p.get_min_frequency() for p in self.processors)
        e_min = sum(job['c_m'] * (min_freq ** 2) for job in self.jobs)
        
        max_freq = 1.0  
        e_max = sum((job['c_m'] + job['c_o']) * (max_freq ** 2) for job in self.jobs)
        
        return e_min, e_max
    
    def _calculate_performance_bounds(self) -> Tuple[float, float]:
        """Calculate theoretical performance bounds for normalization"""
        # Best performance: no jobs skipped (S_i = 0 for all tasks)
        j_min = 0.0
        
        # Worst performance: all optional jobs skipped
        j_max = sum((task.c_o ** 2) * (self.hyperperiod / task.period) for task in self.tasks)
        
        return j_min, j_max
    
    def _decode_solution(self, x: np.ndarray) -> List[Dict]:
        """Decode binary solution vector into job assignments"""
        assignments = []
        
        for job_idx in range(self.n_jobs):
            start_idx = job_idx * self.bits_per_job
            
            # Extract processor assignment bits
            proc_bits = x[start_idx:start_idx + self.bits_per_processor]
            processor_id = int(''.join(map(str, proc_bits.astype(int))), 2) 
            
            # Extract frequency assignment bits
            freq_start = start_idx + self.bits_per_processor
            freq_bits = x[freq_start:freq_start + self.bits_per_frequency]
            freq_idx = int(''.join(map(str, freq_bits.astype(int))), 2) 
            frequency = self.processors[processor_id].frequencies[freq_idx]
            
            # Extract optional execution bit
            optional_bit = x[start_idx + self.bits_per_processor + self.bits_per_frequency]
            execute_optional = bool(optional_bit)
            
            assignment = {
                'job_id': job_idx,
                'processor_id': processor_id,
                'frequency': frequency,
                'execute_optional': execute_optional
            }
            assignments.append(assignment)
        
        return assignments

    def _encode_solution(self, assignments: List[Dict]) -> np.ndarray:
        """
        Encode job assignments back into a binary solution vector.
        """
        x = np.zeros(self.n_jobs * self.bits_per_job, dtype=bool)
        for job_idx, assignment in enumerate(assignments):
            start_idx = job_idx * self.bits_per_job

            # Processor assignment
            proc_bits = np.array(list(np.binary_repr(assignment['processor_id'], width=self.bits_per_processor)), dtype=int)
            x[start_idx:start_idx + self.bits_per_processor] = proc_bits

            # Frequency assignment (find index in processor's frequency list)
            proc_id = assignment['processor_id']
            freq_idx = self.processors[proc_id].frequencies.index(assignment['frequency'])
            freq_bits = np.array(list(np.binary_repr(freq_idx, width=self.bits_per_frequency)), dtype=int)
            freq_start = start_idx + self.bits_per_processor
            x[freq_start:freq_start + self.bits_per_frequency] = freq_bits

            # Optional execution bit
            x[start_idx + self.bits_per_processor + self.bits_per_frequency] = int(assignment['execute_optional'])

        return x

    
    def _check_timing_constraints(self, assignments: List[Dict]) -> bool:
        """Check if all timing constraints are satisfied"""
        processor_schedules = {p.id: [] for p in self.processors}
        
        for i, assignment in enumerate(assignments):
            job = self.jobs[i]
            execution_time = job['c_m']
            if assignment['execute_optional']:
                execution_time += job['c_o']
            
            actual_execution_time = execution_time / assignment['frequency']
            schedule_item = {
                'job_id': job['id'],
                'release_time': job['release_time'],
                'deadline': job['deadline'],
                'execution_time': actual_execution_time,
                'assignment': assignment
            }
            processor_schedules[assignment['processor_id']].append(schedule_item)
        
        for proc_id, schedule in processor_schedules.items():
            if not self._is_schedulable_edf(schedule):
                # print(f"Processor {proc_id} schedule is not feasible.")
                return False
        
        return True
    
    def _is_schedulable_edf(self, jobs: List[Dict]) -> bool:
        """Check if jobs are schedulable using Earliest Deadline First (EDF)"""
        if not jobs:
            return True
        
        """use pair to complete teh comments"""
        sorted_jobs = sorted(jobs, key=lambda x: x['deadline'])
        
        current_time = 0
        for job in sorted_jobs:
            
            start_time = max(current_time, job['release_time'])
            finish_time = start_time + job['execution_time']
            
            if finish_time > job['deadline']:
                return False
            
            current_time = finish_time
        
        return True
    
    def _calculate_energy(self, assignments: List[Dict]) -> float:
        """Calculate total energy consumption"""
        total_energy = 0
        
        for i, assignment in enumerate(assignments):
            job = self.jobs[i]
            execution_time = job['c_m']
            if assignment['execute_optional']:
                execution_time += job['c_o']
            """
            Energy = frequency^2 * execution_time
            """
            energy = (assignment['frequency'] ** 2) * execution_time
            total_energy += energy
        
        return total_energy
    
    def _calculate_performance_penalty(self, assignments: List[Dict]) -> float:
        """Calculate performance penalty due to skipped optional jobs"""
        penalty = 0
        
        
        task_skips = {}
        for task in self.tasks:
            task_skips[task.id] = 0
        
        for i, assignment in enumerate(assignments):
            job = self.jobs[i]
            task_id = job['task_id']
            
            if not assignment['execute_optional']:
                task_skips[task_id] += 1
        
        # Calculate quadratic penalty
        for task in self.tasks:
            skipped = task_skips[task.id]
            penalty += (task.c_o ** 2) * skipped
        
        return penalty
       
    def repair(self, x, **kwargs):
        """
        Repair infeasible solutions in the population x.
        For each individual, if infeasible, attempt repair strategies before resampling.
        """
        #print("Repairing infeasible solutions")
        x_repaired = x.copy()
        
        for i in range(x.shape[0]):
            assignments = self._decode_solution(x_repaired[i])           
            if self._check_timing_constraints(assignments):
                continue  # Feasible
            
            # --- Strategy 1: Fix frequency assignments ---
            for job in self.jobs:
                job_id = job['id']
                proc_id = assignments[job_id]['processor_id']
                freq = assignments[job_id]['frequency']
                proc = next(p for p in self.processors if p.id == proc_id)
                # If frequency is not the maximum, increase it
                current_freq_idx = proc.frequencies.index(freq)
                if current_freq_idx < len(proc.frequencies) - 1:
                    assignments[job_id]['frequency'] = proc.frequencies[current_freq_idx + 1]
            
            # Check feasibility after Strategy 1
            if self._check_timing_constraints(assignments):
                #print(f"Solution {i} repaired successfully using Strategy 1.")
                x_repaired[i] = self._encode_solution(assignments)
                continue

            # --- Strategy 2: Drop optional executions ---
            for job in self.jobs:
                job_id = job['id']
                if assignments[job_id]['execute_optional']:
                    assignments[job_id]['execute_optional'] = 0

            # Check feasibility after Strategy 2
            if self._check_timing_constraints(assignments):
                #print(f"Solution {i} repaired successfully using Strategy 1 and 2.")
                x_repaired[i] = self._encode_solution(assignments)
                continue

            # if self._check_timing_constraints(assignments):
            #     print(f"Solution {i} repaired successfully.")
            # else:
            #     print(f"Solution {i} could not be repaired.")

        return x_repaired

    
    def _evaluate(self, x, out, *args, **kwargs):
        """Evaluate the multi-objective optimization problem"""
        # import traceback
        # print("Traceback for _evaluate call:")
        # traceback.print_stack()  # Print the call stack

        # Log the shape of x and its source
        #print(f"Evaluating solutions with shape: {x.shape}")
        # if hasattr(self, 'current_generation'):
        #     print(f"Current generation: {self.current_generation}")
        # else:
        #     print("Generation information not available.")
        generation = getattr(self, 'current_generation', 'unknown')
        n_solutions = x.shape[0]
        # print(f"Evaluating generation {generation} with {n_solutions} solutions.") # Log the number of solutions
        f1_values = np.zeros(n_solutions)  
        f2_values = np.zeros(n_solutions)  
        feasible_count = 0  # Counter for feasible solutions
        infeasible_count = 0  # Counter for infeasible solutions
        
        for i in range(n_solutions):
            assignments = self._decode_solution(x[i])
            
            if not self._check_timing_constraints(assignments):
                infeasible_count += 1  # Increment infeasible solution count
                f1_values[i] = 1.0
                f2_values[i] = 1.0
            else:
                feasible_count += 1  # Increment feasible solution count
                energy = self._calculate_energy(assignments)
                penalty = self._calculate_performance_penalty(assignments)
                
                e_min, e_max = self.energy_bounds
                j_min, j_max = self.performance_bounds
                
                f1_values[i] = (energy - e_min) / (e_max - e_min) if e_max > e_min else 0
                f2_values[i] = (penalty - j_min) / (j_max - j_min) if j_max > j_min else 0
        
        # print(f"Number of infeasible solutions in this generation: {infeasible_count}")
        # print(f"Number of feasible solutions in this generation: {feasible_count}")
        
        out["F"] = np.column_stack([f1_values, f2_values])
        out["is_feasible"] = np.array([self._check_timing_constraints(self._decode_solution(x[i])) for i in range(n_solutions)])
    
    def _evaluate_callback(self, algorithm):
        """Callback to pass the generation number to _evaluate."""
        self.current_generation = algorithm.n_gen

        # Get the population after ranking and selection
        ranked_population = algorithm.pop

        # Count the number of feasible solutions
        feasible_solutions = np.sum(ranked_population.get("is_feasible"))
        total_solutions = len(ranked_population)

        # Log the number of feasible solutions
        #print(f"Generation {self.current_generation}: Total solutions after ranking: {total_solutions}")
        #print(f"Generation {self.current_generation}: Feasible solutions after ranking: {feasible_solutions}")

        # Log parent and offspring populations (optional, for debugging)
        parent_population = algorithm.pop.get("X")  # Parent population
        offspring_population = algorithm.off.get("X") if hasattr(algorithm, 'off') else None  # Offspring population
        #print(f"Generation {self.current_generation}: Parent population size: {parent_population.shape[0]}")
        # if offspring_population is not None:
        #     print(f"Generation {self.current_generation}: Offspring population size: {offspring_population.shape[0]}")
        # else:
        #     print(f"Generation {self.current_generation}: No offspring population available.")
