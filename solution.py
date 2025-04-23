from modules import np

class Solution:
    def __init__(self, jobs, processors):
        self.jobs = jobs
        self.processors = processors
        
        # Decision variables
        self.processor_assignment = {}  # job_id -> processor_id
        self.frequency_assignment = {}  # job_id -> frequency
        self.optional_execution = {}    # job_id -> binary (0/1)
        self.start_times = np.zeros(len(jobs), dtype=float)
        
        # Metrics
        self.energy = float('inf')
        self.performance = float('inf')
        self.feasible = True
        self.rank = None  # Rank in the Pareto front
        
    def copy(self):
        """Create and return a copy of this solution."""
        new_solution = Solution(self.jobs, self.processors)
        new_solution.processor_assignment = self.processor_assignment.copy()
        new_solution.frequency_assignment = self.frequency_assignment.copy()
        new_solution.optional_execution = self.optional_execution.copy()
        new_solution.start_times = self.start_times.copy()
        new_solution.energy = self.energy
        new_solution.performance = self.performance
        new_solution.feasible = self.feasible
        return new_solution
        
    def dominates(self, other):
        """Check if this solution dominates the other solution."""
        if not self.feasible:
            return False
        if not other.feasible:
            return True
        return (self.energy <= other.energy and self.performance <= other.performance and 
                (self.energy < other.energy or self.performance < other.performance))

    def evaluate(self):
        """Evaluate the energy consumption and performance of the solution."""
        energy = 0     # Energy consumption calculation
        performance_penalty = 0   # Performance penalty for skipped optional parts
        
        for i in range(len(self.jobs)):
            job = self.jobs[i]
            freq = self.frequency_assignment[i]
            
            # Calculate energy for mandatory execution
            energy += (freq ** 2) * job.task.c_m
            
            # Add energy for optional execution if it's included
            if self.optional_execution[i] == 1:
                energy += (freq ** 2) * job.task.c_o
                
            # Penalty for performance if optional execution is skipped
            if self.optional_execution[i] == 0:
                performance_penalty += (job.task.c_o ** 2)
        
        self.energy = energy
        self.performance = performance_penalty
        
        return energy, performance_penalty

    def get_schedule(self):
        """Convert assignments to a complete schedule"""
        schedule = []
        
        for job in self.jobs:
            proc_id = self.processor_assignment[job.id]
            freq = self.frequency_assignment[job.id]
            
            exec_time = job.task.c_m / freq
            if self.optional_execution[job.id] == 1:
                exec_time += job.task.c_o / freq
                
            start_time = job.arrival_time
            end_time = start_time + exec_time
            
            schedule.append({
                'job_id': job.id,
                'task_id': job.task.id,
                'processor': proc_id,
                'frequency': freq,
                'optional': self.optional_execution[job.id],
                'start': start_time,
                'end': end_time,
                'arrival': job.arrival_time,
                'deadline': job.deadline
            })
        
        return schedule
    