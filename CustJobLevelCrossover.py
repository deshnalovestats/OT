from pymoo.core.crossover import Crossover
import numpy as np
from tabulate import tabulate
from logging_config import LoggingFlags, log_if

class JobLevelUniformCrossover(Crossover):
    def __init__(self, **kwargs):
        """
        Job-level uniform crossover operator.
        """
        super().__init__(2, 2, **kwargs)

    def _do(self, problem, X, **kwargs):
        """
        Perform job-level uniform crossover and print parents and children.

        Parameters
        ----------
        problem : Problem
            The problem instance.
        X : np.ndarray
            The population matrix (parents).

        Returns
        -------
        np.ndarray
            The offspring population after crossover.
        """
        _, n_matings, n_var = X.shape
        bits_per_job = problem.bits_per_job
        n_jobs = n_var // bits_per_job

        Y = np.empty_like(X)

        log_if(LoggingFlags.CROSSOVER_DETAILS, f"Performing crossover on {n_matings} mating pairs")

        for k in range(n_matings):
            parent1, parent2 = X[0, k], X[1, k]

            child1 = np.zeros_like(parent1)
            child2 = np.zeros_like(parent2)

            for job_idx in range(n_jobs):
                start = job_idx * bits_per_job
                end = start + bits_per_job

                if np.random.rand() < 0.5:
                    child1[start:end] = parent1[start:end]
                    child2[start:end] = parent2[start:end]
                else:
                    child1[start:end] = parent2[start:end]
                    child2[start:end] = parent1[start:end]

            Y[0, k] = child1
            Y[1, k] = child2

            log_if(LoggingFlags.CROSSOVER_DETAILS, f"Crossover pair {k+1} completed")
            # Print parents and children in tabular format
            # self._print_crossover_results(parent1, parent2, child1, child2, k)

        return Y
    
    def _print_crossover_results(self, parent1, parent2, child1, child2, mating_index):
        """
        Print the parents and children in a tabular format with binary representation.

        Parameters
        ----------
        parent1 : np.ndarray
            The first parent chromosome.
        parent2 : np.ndarray
            The second parent chromosome.
        child1 : np.ndarray
            The first child chromosome.
        child2 : np.ndarray
            The second child chromosome.
        mating_index : int
            The index of the current mating pair.
        """
        # Convert boolean arrays to binary (0/1)
        parent1_binary = "".join(map(str, parent1.astype(int)))
        parent2_binary = "".join(map(str, parent2.astype(int)))
        child1_binary = "".join(map(str, child1.astype(int)))
        child2_binary = "".join(map(str, child2.astype(int)))

        # Prepare the table
        table = [
            ["Parent 1", parent1_binary],
            ["Parent 2", parent2_binary],
            ["Child 1", child1_binary],
            ["Child 2", child2_binary],
        ]

        print(f"\nMating Pair {mating_index + 1}")
        print(tabulate(table, headers=["Type", "Chromosome"], tablefmt="grid"))