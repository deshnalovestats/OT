from modules import random,defaultdict
# Chromosome representation
class Chromosome:
    def __init__(self, jobs, processors):
        self.jobs = jobs
        self.processors = processors
        
        # Decision variables
        self.proc_assignment = {}  # job_id -> processor_id
        self.freq_assignment = {}  # job_id -> frequency
        self.opt_execution = {}    # job_id -> binary (0/1)
        
        # Metrics
        self.energy = None
        self.performance = None
        self.feasible = True
        self.rank = None
        self.crowding_distance = 0
        
    def initialize_random(self):
        """Initialize with random feasible assignments"""
        for job in self.jobs:
            # Randomly assign processor
            proc = random.choice(self.processors)
            self.proc_assignment[job.id] = proc.id
            
            # Randomly assign frequency (ensuring feasibility for mandatory part)
            valid_freqs = [f for f in proc.freq_levels if (job.task.c_m / f) <= job.task.period]
            if not valid_freqs:  # If no feasible frequency for mandatory
                self.feasible = False
                self.freq_assignment[job.id] = max(proc.freq_levels)
            else:
                self.freq_assignment[job.id] = random.choice(valid_freqs)
            
            # Randomly decide optional execution
            # Check if there's room for optional execution
            exec_time_mandatory = job.task.c_m / self.freq_assignment[job.id]
            if exec_time_mandatory + (job.task.c_o / self.freq_assignment[job.id]) <= job.task.period:
                self.opt_execution[job.id] = random.choice([0, 1])
            else:
                self.opt_execution[job.id] = 0  # No room for optional part
        
        # After initialization, check full schedule feasibility
        self.check_feasibility()
    
    def check_feasibility(self):
        """Check if the schedule is feasible considering processor overlap"""
        processor_schedule = defaultdict(list)
        
        for job in self.jobs:
            proc_id = self.proc_assignment[job.id]
            freq = self.freq_assignment[job.id]
            
            exec_time = job.task.c_m / freq
            if self.opt_execution[job.id] == 1:
                exec_time += job.task.c_o / freq
                
            # Check timing constraint
            if exec_time > job.task.period:
                self.feasible = False
                return
                
            start_time = job.arrival_time
            end_time = start_time + exec_time
            
            # Add job to processor schedule
            processor_schedule[proc_id].append((job.id, start_time, end_time))
        
        # Check for overlaps in each processor
        for proc_id, schedule in processor_schedule.items():
            # Sort by start time
            schedule.sort(key=lambda x: x[1])
            
            # Check for overlaps
            for i in range(1, len(schedule)):
                prev_end = schedule[i-1][2]
                curr_start = schedule[i][1]
                
                if prev_end > curr_start:
                    # Overlap detected - try to fix by moving current job
                    new_start = prev_end
                    new_end = new_start + (schedule[i][2] - schedule[i][1])
                    
                    # Check if moving violates deadline
                    job_id = schedule[i][0]
                    job = next(j for j in self.jobs if j.id == job_id)
                    
                    if new_end > job.deadline:
                        self.feasible = False
                        return
                    
                    # Update schedule
                    schedule[i] = (job_id, new_start, new_end)
    
    def calculate_objectives(self):
        """Calculate energy consumption and performance penalty"""
        energy = 0
        performance_penalty = 0
        
        for job in self.jobs:
            freq = self.freq_assignment[job.id]
            opt_exec = self.opt_execution[job.id]
            
            # Energy calculation: (f_ix)² × (c^m_i + y_ix × c^o_i)
            energy += (freq ** 2) * (job.task.c_m + opt_exec * job.task.c_o)
            
            # Performance penalty: (c^o_i)² · (1 - y_ix)
            performance_penalty += (job.task.c_o ** 2) * (1 - opt_exec)
        
        self.energy = energy
        self.performance = performance_penalty
        
        return energy, performance_penalty
    
    def get_schedule(self):
        """Convert assignments to a complete schedule"""
        schedule = []
        
        for job in self.jobs:
            proc_id = self.proc_assignment[job.id]
            freq = self.freq_assignment[job.id]
            
            exec_time = job.task.c_m / freq
            if self.opt_execution[job.id] == 1:
                exec_time += job.task.c_o / freq
                
            start_time = job.arrival_time
            end_time = start_time + exec_time
            
            schedule.append({
                'job_id': job.id,
                'task_id': job.task.id,
                'processor': proc_id,
                'frequency': freq,
                'optional': self.opt_execution[job.id],
                'start': start_time,
                'end': end_time,
                'arrival': job.arrival_time,
                'deadline': job.deadline
            })
        
        return schedule
    
    def copy(self):
        """Create a deep copy of the chromosome"""
        new_chrom = Chromosome(self.jobs, self.processors)
        new_chrom.proc_assignment = self.proc_assignment.copy()
        new_chrom.freq_assignment = self.freq_assignment.copy()
        new_chrom.opt_execution = self.opt_execution.copy()
        new_chrom.energy = self.energy
        new_chrom.performance = self.performance
        new_chrom.feasible = self.feasible
        new_chrom.rank = self.rank
        new_chrom.crowding_distance = self.crowding_distance
        return new_chrom
