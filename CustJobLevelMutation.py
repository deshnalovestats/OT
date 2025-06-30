from pymoo.core.mutation import Mutation
import numpy as np

class JobLevelMutation(Mutation):
    def __init__(self, prob=0.1):
        super().__init__()
        self.prob = prob

    def _do(self, problem, X, **kwargs):
        n_individuals, n_var = X.shape
        n_jobs = problem.n_jobs
        n_processors = problem.n_processors
        bits_per_processor = problem.bits_per_processor
        bits_per_frequency = problem.bits_per_frequency
        bits_per_job = problem.bits_per_job

        X_mut = X.copy()

        for i in range(n_individuals):
            for j in range(n_jobs):
                if np.random.rand() < self.prob:
                    # Mutate processor/frequency/optional
                    proc_id = np.random.randint(0, n_processors)
                    freq_id = np.random.randint(0, len(problem.processors[proc_id].frequencies))
                    opt_bit = np.random.randint(0, 2)

                    proc_bits = format(proc_id, f'0{bits_per_processor}b')
                    freq_bits = format(freq_id, f'0{bits_per_frequency}b')
                    opt_bit_str = str(opt_bit)

                    start = j * bits_per_job
                    end = start + bits_per_job
                    new_bits = [int(b) for b in proc_bits + freq_bits + opt_bit_str]
                    X_mut[i, start:end] = new_bits

        return X_mut