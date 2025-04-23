# Data structures for task scheduling
class Task:
    def __init__(self, task_id, mandatory_exec_time, optional_exec_time, period):
        self.id = task_id
        self.c_m = mandatory_exec_time  # Mandatory execution time
        self.c_o = optional_exec_time   # Optional execution time
        self.period = period            # Period and relative deadline
        
    def __repr__(self):
        return f"Task {self.id}: c_m={self.c_m}, c_o={self.c_o}, period={self.period}"

class Processor:
    def __init__(self, proc_id, freq_levels):
        self.id = proc_id
        self.freq_levels = sorted(freq_levels)  # Available frequency levels (normalized)
        
    def __repr__(self):
        return f"Processor {self.id}: freq_levels={self.freq_levels}"

class Job:
    def __init__(self, task, job_id, arrival_time):
        self.task = task
        self.id = job_id
        self.arrival_time = arrival_time
        self.deadline = arrival_time + task.period
        
    def __repr__(self):
        return f"Job {self.id} (Task {self.task.id}): arrival={self.arrival_time}, deadline={self.deadline}"
