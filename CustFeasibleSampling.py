import numpy as np
from pymoo.core.sampling import Sampling

# class FeasibleBinarySampling(Sampling):
#     def _do(self, problem, n_samples, **kwargs):
#         X = np.random.random((n_samples, problem.n_var))
#         X = (X < 0.5).astype(bool)
#         X = problem.repair(X)
#         return X

class FeasibleBinarySampling(Sampling):
    def _do(self, problem, n_samples, **kwargs):
        feasible_solutions = []
        max_attempts = 1000  # Limit to avoid infinite loops
        while len(feasible_solutions) < n_samples and max_attempts > 0:
            X = [] # leftover solutions to sample
            n_jobs = problem.n_jobs
            bits_per_processor = problem.bits_per_processor
            bits_per_frequency = problem.bits_per_frequency
            remaining = n_samples - len(feasible_solutions)

            for i in range(remaining):
                bits = []
                for job_idx in range(n_jobs):
                    # Processor assignment
                    proc_id = np.random.randint(0, problem.n_processors)
                    proc_bits = np.array(list(np.binary_repr(proc_id, width=bits_per_processor)), dtype=int)

                    # Frequency assignment (valid for the selected processor)
                    freq_levels = problem.processors[proc_id].frequencies
                    freq_idx = np.random.randint(0, len(freq_levels))
                    freq_bits = np.array(list(np.binary_repr(freq_idx, width=bits_per_frequency)), dtype=int)

                    # Optional execution bit
                    opt_bit = np.random.randint(0, 2)

                    bits.extend(proc_bits)
                    bits.extend(freq_bits)
                    bits.append(opt_bit)
                X.append(bits)
            X = np.array(X, dtype=bool)

            #print(f"Sampling shape: {X.shape}")

            # Repair solutions
            X_repaired = problem.repair(X)

            # Check feasibility of repaired solutions
            for i in range(X_repaired.shape[0]):
                assignments = problem._decode_solution(X_repaired[i])
                if problem._check_timing_constraints(assignments):
                    feasible_solutions.append(X_repaired[i])
                    #print(f"[FeasibleBinarySampling] Solution {len(feasible_solutions)}/{n_samples} accepted (attempt {1000-max_attempts+1})")
                    if len(feasible_solutions) == n_samples:
                        break
                # else:
                #     print(f"[FeasibleBinarySampling] Infeasible even after repair: solution {i}")

            max_attempts -= 1
        #print(f"[FeasibleBinarySampling] Completed: {len(feasible_solutions)} feasible solutions generated in {1000-max_attempts} attempts.")
        if len(feasible_solutions) < n_samples:
            print(f"Only {len(feasible_solutions)} feasible solutions generated after {1000 - max_attempts} attempts.")
            raise ValueError("Unable to generate enough feasible solutions within the maximum attempts.")

        return np.array(feasible_solutions)
