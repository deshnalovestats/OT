

from pymoo.core.problem import Problem
import task as tsk
from modules import random, time
import numpy as np
from chromosome import Chromosome

class TaskSchedulingProblem(Problem):
    def __init__(self, tasks, processors, hyperperiod):
        self.tasks = tasks
        self.processors = processors
        self.hyperperiod = hyperperiod
        
        # Generate all jobs in the hyperperiod
        self.jobs = self.generate_jobs()

        self.n_jobs = len(self.jobs)
        self.n_processors = len(processors)
        self.freq_table = {p.id: p.freq_levels for p in processors}
        
        n_var = self.n_jobs * 3  # 3 decision variables per job: processor, frequency, optional execution

        
        # Bounds for normalization
        self.e_min = self.calculate_e_min()
        self.e_max = self.calculate_e_max()
        self.j_min = 0  # Best performance = 0 penalty
        self.j_max = self.calculate_j_max()
        self.debug = False
        super().__init__(n_var=n_var, n_obj=2, n_ieq_constr=0, xl=0.0, xu=1.0)
        
        
    def generate_jobs(self):
        """Generate all jobs within the hyperperiod"""
        jobs = []
        
        for task in self.tasks:
            n_i = self.hyperperiod // task.period  # Number of jobs for task i
            
            for x in range(1, n_i + 1):
                arrival_time = (x - 1) * task.period
                job_id = f"{task.id}_{x}"
                jobs.append(tsk.Job(task, job_id, arrival_time))
        
        return jobs
    
    def calculate_e_min(self):
        """Calculate theoretical minimum energy (all mandatory at min freq)"""
        e_min = 0
        
        for job in self.jobs:
            # Find minimum frequency across all processors
            min_freq = min(min(p.freq_levels) for p in self.processors)
            e_min += (min_freq ** 2) * job.task.c_m
        
        return e_min
    
    def calculate_e_max(self):
        """Calculate theoretical maximum energy (all mandatory+optional at max freq)"""
        e_max = 0
        
        for job in self.jobs:
            # Use maximum frequency (1.0 after normalization)
            e_max += (1.0 ** 2) * (job.task.c_m + job.task.c_o)
        
        return e_max
    
    def calculate_j_max(self):
        """Calculate maximum performance penalty (all optional skipped)"""
        j_max = 0
        
        for job in self.jobs:
            j_max += (job.task.c_o ** 2)
        
        return j_max
    
    def normalize_objectives(self, energy, performance):
        """Normalize objectives to [0,1] range"""
        norm_energy = (energy - self.e_min) / (self.e_max - self.e_min) if self.e_max > self.e_min else 0
        norm_perf = (performance - self.j_min) / (self.j_max - self.j_min) if self.j_max > self.j_min else 0
        
        return norm_energy, norm_perf