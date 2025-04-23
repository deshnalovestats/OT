from chromosome import Chromosome

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