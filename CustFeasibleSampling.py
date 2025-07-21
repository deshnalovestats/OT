import numpy as np
from pymoo.core.sampling import Sampling
from logging_config import LoggingFlags, log_if

# class FeasibleBinarySampling(Sampling):
#     def _do(self, problem, n_samples, **kwargs):
#         X = np.random.random((n_samples, problem.n_var))
#         X = (X < 0.5).astype(bool)
#         X = problem.repair(X)
#         return X

class FeasibleBinarySampling(Sampling):
    def _do(self, problem, n_samples, **kwargs):
        log_if(LoggingFlags.SAMPLING_PROGRESS, f"Starting feasible sampling for {n_samples} solutions...")
        
        feasible_solutions = []
        max_attempts = 1000
        attempt_count = 0
        
        while len(feasible_solutions) < n_samples and max_attempts > 0:
            X = []
            n_jobs = problem.n_jobs
            bits_per_processor = problem.bits_per_processor
            bits_per_frequency = problem.bits_per_frequency
            remaining = n_samples - len(feasible_solutions)

            for i in range(remaining):
                bits = []
                for job_idx in range(n_jobs):
                    proc_id = np.random.randint(0, problem.n_processors)
                    proc_bits = np.array(list(np.binary_repr(proc_id, width=bits_per_processor)), dtype=int)

                    freq_levels = problem.processors[proc_id].frequencies
                    freq_idx = np.random.randint(0, len(freq_levels))
                    freq_bits = np.array(list(np.binary_repr(freq_idx, width=bits_per_frequency)), dtype=int)

                    opt_bit = np.random.randint(0, 2)

                    bits.extend(proc_bits)
                    bits.extend(freq_bits)
                    bits.append(opt_bit)
                X.append(bits)
            X = np.array(X, dtype=bool)

            X_repaired = problem.repair(X)

            for i in range(X_repaired.shape[0]):
                assignments = problem._decode_solution(X_repaired[i])
                if problem._check_timing_constraints(assignments):
                    feasible_solutions.append(X_repaired[i])
                    log_if(LoggingFlags.SAMPLING_PROGRESS, 
                           f"Solution {len(feasible_solutions)}/{n_samples} accepted (attempt {attempt_count+1})")
                    if len(feasible_solutions) == n_samples:
                        break

            max_attempts -= 1
            attempt_count += 1
            
        log_if(LoggingFlags.SAMPLING_PROGRESS, 
               f"Sampling completed: {len(feasible_solutions)} solutions in {attempt_count} attempts")
        
        if len(feasible_solutions) < n_samples:
            log_if(LoggingFlags.SAMPLING_PROGRESS, 
                   f"Warning: Only {len(feasible_solutions)} feasible solutions generated")
            raise ValueError("Unable to generate enough feasible solutions")

        return np.array(feasible_solutions)