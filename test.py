import numpy as np
import random
from typing import List, Tuple, Dict
from dataclasses import dataclass
from pymoo.core.problem import Problem
from pymoo.core.sampling import Sampling
from logging_config import LoggingFlags, log_if

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
        
        max_freq_levels = max(len(p.frequencies) for p in processors)
        self.bits_per_processor = int(np.ceil(np.log2(self.n_processors)))
        self.bits_per_frequency = int(np.ceil(np.log2(max_freq_levels)))
        self.bits_per_job = self.bits_per_processor + self.bits_per_frequency + 1
        
        n_var = self.n_jobs * self.bits_per_job
        
        super().__init__(n_var=n_var, n_obj=2, n_constr=0, xl=0, xu=1, vtype=bool)
        
        self.energy_bounds = self._calculate_energy_bounds()
        self.performance_bounds = self._calculate_performance_bounds()
        
        log_if(LoggingFlags.OPTIMIZATION_PROGRESS, 
               f"Problem initialized: {self.n_tasks} tasks, {self.n_processors} processors, {self.n_jobs} jobs")
    
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
        
        log_if(LoggingFlags.ENERGY_CALCULATION, f"Energy bounds: [{e_min:.2f}, {e_max:.2f}]")
        return e_min, e_max
    
    def _calculate_performance_bounds(self) -> Tuple[float, float]:
        """Calculate theoretical performance bounds for normalization"""
        j_min = 0.0
        j_max = sum((task.c_o ** 2) * (self.hyperperiod / task.period) for task in self.tasks)
        
        log_if(LoggingFlags.PERFORMANCE_CALCULATION, f"Performance bounds: [{j_min:.2f}, {j_max:.2f}]")
        return j_min, j_max
    
    def _decode_solution(self, x: np.ndarray) -> List[Dict]:
        """Decode binary solution vector into job assignments"""
        assignments = []
        
        for job_idx in range(self.n_jobs):
            start_idx = job_idx * self.bits_per_job
            
            proc_bits = x[start_idx:start_idx + self.bits_per_processor]
            processor_id = int(''.join(map(str, proc_bits.astype(int))), 2) 
            
            freq_start = start_idx + self.bits_per_processor
            freq_bits = x[freq_start:freq_start + self.bits_per_frequency]
            freq_idx = int(''.join(map(str, freq_bits.astype(int))), 2) 
            frequency = self.processors[processor_id].frequencies[freq_idx]
            
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
        """Encode job assignments back into a binary solution vector"""
        x = np.zeros(self.n_jobs * self.bits_per_job, dtype=bool)
        for job_idx, assignment in enumerate(assignments):
            start_idx = job_idx * self.bits_per_job

            proc_bits = np.array(list(np.binary_repr(assignment['processor_id'], width=self.bits_per_processor)), dtype=int)
            x[start_idx:start_idx + self.bits_per_processor] = proc_bits

            proc_id = assignment['processor_id']
            freq_idx = self.processors[proc_id].frequencies.index(assignment['frequency'])
            freq_bits = np.array(list(np.binary_repr(freq_idx, width=self.bits_per_frequency)), dtype=int)
            freq_start = start_idx + self.bits_per_processor
            x[freq_start:freq_start + self.bits_per_frequency] = freq_bits

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
                log_if(LoggingFlags.TIMING_CONSTRAINTS, f"Processor {proc_id} schedule is not feasible")
                return False
        
        return True
    
    def _is_schedulable_edf(self, jobs: List[Dict]) -> bool:
        """Check if jobs are schedulable using Earliest Deadline First (EDF)"""
        if not jobs:
            return True
        
        sorted_jobs = sorted(jobs, key=lambda x: x['deadline'])
        
        current_time = 0
        for job in sorted_jobs:
            start_time = max(current_time, job['release_time'])
            finish_time = start_time + job['execution_time']
            
            if finish_time > job['deadline']:
                log_if(LoggingFlags.TIMING_CONSTRAINTS, 
                       f"Job {job['job_id']} misses deadline: {finish_time} > {job['deadline']}")
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
            
            energy = (assignment['frequency'] ** 2) * execution_time
            total_energy += energy
        
        log_if(LoggingFlags.ENERGY_CALCULATION, f"Total energy calculated: {total_energy:.4f}")
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
        
        for task in self.tasks:
            skipped = task_skips[task.id]
            penalty += (task.c_o ** 2) * skipped
        
        log_if(LoggingFlags.PERFORMANCE_CALCULATION, f"Performance penalty calculated: {penalty:.4f}")
        return penalty
       
    def repair(self, x, **kwargs):
        """Repair infeasible solutions in the population x"""
        log_if(LoggingFlags.REPAIR_OPERATIONS, "Starting repair process...")
        x_repaired = x.copy()
        repair_count = 0
        
        for i in range(x.shape[0]):
            assignments = self._decode_solution(x_repaired[i])           
            if self._check_timing_constraints(assignments):
                continue
            
            # Strategy 1: Fix frequency assignments
            for job in self.jobs:
                job_id = job['id']
                proc_id = assignments[job_id]['processor_id']
                freq = assignments[job_id]['frequency']
                proc = next(p for p in self.processors if p.id == proc_id)
                current_freq_idx = proc.frequencies.index(freq)
                if current_freq_idx < len(proc.frequencies) - 1:
                    assignments[job_id]['frequency'] = proc.frequencies[current_freq_idx + 1]
            
            if self._check_timing_constraints(assignments):
                log_if(LoggingFlags.REPAIR_OPERATIONS, f"Solution {i} repaired with Strategy 1")
                x_repaired[i] = self._encode_solution(assignments)
                repair_count += 1
                continue

            # Strategy 2: Drop optional executions
            for job in self.jobs:
                job_id = job['id']
                if assignments[job_id]['execute_optional']:
                    assignments[job_id]['execute_optional'] = False

            if self._check_timing_constraints(assignments):
                log_if(LoggingFlags.REPAIR_OPERATIONS, f"Solution {i} repaired with Strategy 1+2")
                x_repaired[i] = self._encode_solution(assignments)
                repair_count += 1
            else:
                log_if(LoggingFlags.REPAIR_OPERATIONS, f"Solution {i} could not be repaired")

        log_if(LoggingFlags.REPAIR_OPERATIONS, f"Repair completed: {repair_count}/{x.shape[0]} solutions repaired")
        return x_repaired
    
    def _evaluate(self, x, out, *args, **kwargs):
        """Evaluate the multi-objective optimization problem"""
        generation = getattr(self, 'current_generation', 'unknown')
        n_solutions = x.shape[0]
        
        log_if(LoggingFlags.EVALUATION_DETAILS, f"Evaluating generation {generation} with {n_solutions} solutions")
        
        f1_values = np.zeros(n_solutions)  
        f2_values = np.zeros(n_solutions)  
        feasible_count = 0
        infeasible_count = 0
        
        for i in range(n_solutions):
            assignments = self._decode_solution(x[i])
            
            if not self._check_timing_constraints(assignments):
                infeasible_count += 1
                f1_values[i] = 1.0
                f2_values[i] = 1.0
            else:
                feasible_count += 1
                energy = self._calculate_energy(assignments)
                penalty = self._calculate_performance_penalty(assignments)
                
                e_min, e_max = self.energy_bounds
                j_min, j_max = self.performance_bounds
                
                f1_values[i] = (energy - e_min) / (e_max - e_min) if e_max > e_min else 0
                f2_values[i] = (penalty - j_min) / (j_max - j_min) if j_max > j_min else 0
        
        log_if(LoggingFlags.FEASIBILITY_CHECKS, 
               f"Generation {generation}: {feasible_count} feasible, {infeasible_count} infeasible")
        
        out["F"] = np.column_stack([f1_values, f2_values])
        out["is_feasible"] = np.array([self._check_timing_constraints(self._decode_solution(x[i])) for i in range(n_solutions)])
    
    def _evaluate_callback(self, algorithm):
        """Callback to pass the generation number to _evaluate"""
        self.current_generation = algorithm.n_gen

        ranked_population = algorithm.pop
        feasible_solutions = np.sum(ranked_population.get("is_feasible"))
        total_solutions = len(ranked_population)

        log_if(LoggingFlags.GENERATION_STATS, 
               f"Generation {self.current_generation}: {feasible_solutions}/{total_solutions} feasible solutions")