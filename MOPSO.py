
from copy import deepcopy

from modules import random, np, time
from task import Job, Processor, Task
from solution import Solution
from particle import Particle

# MOPSO class - implements the Multi-Objective PSO algorithm
class MOPSO:
    def __init__(self, tasks, processors, hyperperiod, num_particles=100, max_iterations=100):
        self.tasks = tasks
        self.processors = processors
        self.hyperperiod = hyperperiod
        self.num_particles = num_particles
        self.max_iterations = max_iterations
        
        # Generate all jobs within the hyperperiod
        self.jobs = self.generate_jobs()
        self.num_jobs = len(self.jobs)
        
        # MOPSO parameters
        self.w = 0.3 # Inertia weight
        self.c1 = 1.0  # Cognitive weight
        self.c2 = 1.5  # Social weight
        
        # Repository for Pareto front solutions
        self.repository = []
        self.repository_capacity = 50
        
    def generate_jobs(self):
        """Generate all jobs within the hyperperiod for all tasks"""
        jobs = []
        for task in self.tasks:
            num_instances = self.hyperperiod // task.period
            for i in range(num_instances):
                arrival_time = i * task.period
                job = Job(task, f"{task.id}_{i}", arrival_time)
                jobs.append(job)
        return sorted(jobs, key=lambda j: j.arrival_time)
    
    def initialize_particles(self):
        """Initialize particles with random positions and zero velocity"""
        particles = []
        for _ in range(self.num_particles):
            solution = self.generate_random_solution()
            if not solution.feasible:
                # Try to repair the solution
                solution = self.repair_solution(solution)
            
            if solution.feasible:
                self.evaluate_solution(solution)
                # Add feasible solution to repository right away
                self.update_repository(solution)
                
            particle = Particle(solution)
            particles.append(particle)
        return particles
    
    # def generate_random_solution(self):
    #     """Generate a random solution"""
    #     solution = Solution(self.jobs, self.processors)
        
    #     # Randomly assign processors and frequencies
    #     for i in range(self.num_jobs):

    #         job_id = self.jobs[i].id

    #         # Random processor assignment
    #         solution.processor_assignment[job_id] = random.randint(0, len(self.processors) - 1)
            
    #         # Random frequency assignment
    #         proc = self.processors[solution.processor_assignment[job_id]]
    #         freq_index = random.randint(0, len(proc.freq_levels) - 1)
    #         solution.frequency_assignment[job_id] = proc.freq_levels[freq_index]
            
    #         # Random optional execution (0 or 1)
    #         solution.optional_execution[job_id] = random.randint(0, 1)
        
    #     # # Determine start times using a simple greedy approach
    #     # solution = self.determine_start_times(solution)
        
    #     return solution
    
    def generate_random_solution(self):
        """Generate a random solution with feasibility-aware initialization"""
        solution = Solution(self.jobs, self.processors)

        for job in self.jobs:
            job_id = job.id

            # Randomly assign a processor
            proc = random.choice(self.processors)
            solution.processor_assignment[job_id] = proc.id

            # Filter frequencies that allow mandatory execution to fit in the period
            valid_freqs = [f for f in proc.freq_levels if (job.task.c_m / f) <= job.task.period]
            if not valid_freqs:
                # No feasible frequency for mandatory part, mark as infeasible
                solution.feasible = False
                # Still assign highest frequency for fallback
                solution.frequency_assignment[job_id] = max(proc.freq_levels)
            else:
                # Assign a random valid frequency
                freq = random.choice(valid_freqs)
                solution.frequency_assignment[job_id] = freq

            # Decide optional execution only if there's room for it
            exec_time_mandatory = job.task.c_m / solution.frequency_assignment[job_id]
            exec_time_optional = job.task.c_o / solution.frequency_assignment[job_id]
            if exec_time_mandatory + exec_time_optional <= job.task.period:
                solution.optional_execution[job_id] = random.choice([0, 1])
            else:
                solution.optional_execution[job_id] = 0  # No room for optional part

        # # Attempt to determine start times and re-check feasibility
        # solution = self.determine_start_times(solution)

        return solution

        
    # def determine_start_times(self, solution: Solution):
    #     """Determine start times for jobs using earliest deadline first"""
    #     # Sort jobs by deadline
    #     job_indices = list(range(self.num_jobs))
    #     job_indices.sort(key=lambda i: self.jobs[i].deadline)
        
    #     # Initialize processor availability times
    #     processor_available = [0] * len(self.processors)
        
    #     # Feasible flag
    #     solution.feasible = True
        
    #     for job_idx in job_indices:
    #         job = self.jobs[job_idx]
    #         job_id= job.id
    #         proc_idx = solution.processor_assignment[job_id]
    #         freq = solution.frequency_assignment[job_id]
            
    #         # Calculate execution time with the assigned frequency
    #         exec_time = job.task.c_m / freq
    #         if solution.optional_execution[job_id] == 1:
    #             exec_time += job.task.c_o / freq
            
    #         # Earliest start time is max of job's arrival time and processor's availability
    #         earliest_start = max(job.arrival_time, processor_available[proc_idx])
            
    #         # Check if job can finish before its deadline
    #         if earliest_start + exec_time > job.deadline:
    #             solution.feasible = False
    #             # Assign the start time anyway, but mark the solution as infeasible
    #             # solution.start_times[job_id] = earliest_start
    #         else:
    #             # solution.start_times[job_id] = earliest_start
    #             processor_available[proc_idx] = earliest_start + exec_time
        
    #     return solution
    
    def repair_solution(self, solution):
        """Try to repair an infeasible solution"""
        # Make a copy to work with
        new_solution = solution.copy()
        
        # Sort jobs by deadline
        job_indices = list(range(self.num_jobs))
        job_indices.sort(key=lambda i: self.jobs[i].deadline)
        
        # Try to fix by removing optional executions
        for job_idx in job_indices:
            job_id= self.jobs[job_idx].id
            if not new_solution.feasible:
                # If the solution is still infeasible, try to remove optional execution
                if new_solution.optional_execution[job_id] == 1:
                    new_solution.optional_execution[job_id] = 0
                    # new_solution = self.determine_start_times(new_solution)
            else:
                break
        
        # If still infeasible, try to increase frequencies
        if not new_solution.feasible:
            for job_idx in job_indices:
                job_id= self.jobs[job_idx].id
                proc_idx = new_solution.processor_assignment[job_id]
                proc = self.processors[proc_idx]
                
                # Get current frequency index
                current_freq = new_solution.frequency_assignment[job_id]
                current_freq_idx = np.where(np.array(proc.freq_levels) == current_freq)[0][0]
                
                # Try all higher frequencies
                for freq_idx in range(current_freq_idx+1, len(proc.freq_levels)):
                    new_solution.frequency_assignment[job_id] = proc.freq_levels[freq_idx]
                    # new_solution = self.determine_start_times(new_solution)
                    if new_solution.feasible:
                        break
                
                if new_solution.feasible:
                    break

        # Try changing processor assignment if still infeasible
        if not new_solution.feasible:
            for job_idx in job_indices:
                job_id= self.jobs[job_idx].id
                current_proc = new_solution.processor_assignment[job_id]
                
                # Try all other processors
                for p in range(len(self.processors)):
                    if p != current_proc:
                        new_solution.processor_assignment[job_id] = p
                        # Use highest frequency
                        new_solution.frequency_assignment[job_id] = self.processors[p].freq_levels[-1]
                        # new_solution = self.determine_start_times(new_solution)
                        if new_solution.feasible:
                            break
                
                if new_solution.feasible:
                    break
        
        return new_solution
    
    def evaluate_solution(self, solution):
        """Evaluate the energy consumption and performance (optional jobs penalty)"""
        if not solution.feasible:
            solution.energy = float('inf')
            solution.performance = float('inf')
            return
        
        # Calculate energy consumption
        energy = 0
        for i in range(self.num_jobs):
            job = self.jobs[i]
            job_id = job.id
            freq = solution.frequency_assignment[job_id]
            
            # Mandatory execution
            energy += (freq ** 2) * job.task.c_m
            
            # Optional execution if included
            if solution.optional_execution[job_id] == 1:
                energy += (freq ** 2) * job.task.c_o
        
        # Calculate performance penalty (based on skipped optional parts)
        performance_penalty = 0
        for i in range(self.num_jobs):
            job = self.jobs[i]
            job_id = job.id
            if solution.optional_execution[job_id] == 0:
                # Squared penalty for skipping optional parts (as per paper)
                performance_penalty += (job.task.c_o ** 2)
        
        solution.energy = energy
        solution.performance = performance_penalty
    
    def update_velocity(self, particle, global_best):
        """Update particle's velocity"""
        for key in particle.velocity.keys():
            if key == 'processor':
                # For discrete values (processor assignment)
                r1 = np.random.random(self.num_jobs)
                r2 = np.random.random(self.num_jobs)
                
                # Extract arrays using job IDs in consistent order
                current = np.array([particle.solution.processor_assignment[self.jobs[i].id] for i in range(self.num_jobs)])
                personal_best = np.array([particle.best_solution.processor_assignment[self.jobs[i].id] for i in range(self.num_jobs)])
                global_best_val = np.array([global_best.processor_assignment[self.jobs[i].id] for i in range(self.num_jobs)])

                        
                # Update velocity using PSO formula
                particle.velocity[key] = (self.w * particle.velocity[key] + 
                                         self.c1 * r1 * (personal_best - current) + 
                                         self.c2 * r2 * (global_best_val - current))
            
            elif key == 'frequency':
                # For continuous values (frequency assignment)
                r1 = np.random.random(self.num_jobs)
                r2 = np.random.random(self.num_jobs)
                
                # Extract arrays using job IDs in consistent order
                current = np.array([particle.solution.processor_assignment[self.jobs[i].id] for i in range(self.num_jobs)])
                personal_best = np.array([particle.best_solution.processor_assignment[self.jobs[i].id] for i in range(self.num_jobs)])
                global_best_val = np.array([global_best.processor_assignment[self.jobs[i].id] for i in range(self.num_jobs)])

                
                particle.velocity[key] = (self.w * particle.velocity[key] + 
                                         self.c1 * r1 * (personal_best - current) + 
                                         self.c2 * r2 * (global_best_val - current))
            
            elif key == 'optional':
                # For binary values (optional execution)
                r1 = np.random.random(self.num_jobs)
                r2 = np.random.random(self.num_jobs)
                
                # Extract arrays using job IDs in consistent order
                current = np.array([particle.solution.processor_assignment[self.jobs[i].id] for i in range(self.num_jobs)])
                personal_best = np.array([particle.best_solution.processor_assignment[self.jobs[i].id] for i in range(self.num_jobs)])
                global_best_val = np.array([global_best.processor_assignment[self.jobs[i].id] for i in range(self.num_jobs)])
            
                        
                particle.velocity[key] = (self.w * particle.velocity[key] + 
                                         self.c1 * r1 * (personal_best - current) + 
                                         self.c2 * r2 * (global_best_val - current))
    
    def update_position(self, particle):
        """Update particle's position based on its velocity"""
        # Create a new solution
        new_solution = Solution(self.jobs, self.processors)
        
        # Update processor assignment (discrete)
        for i in range(self.num_jobs):
            # Apply sigmoid function to velocity and use probabilistic assignment
            job_id = self.jobs[i].id
            proc_probs = np.zeros(len(self.processors))
            for j in range(len(self.processors)):
                if j == particle.solution.processor_assignment[job_id]:
                    proc_probs[j] = 0.5 + 0.5 * (1 / (1 + np.exp(-particle.velocity['processor'][i])))
                else:
                    proc_probs[j] = (1 - proc_probs.sum()) / (len(self.processors) - j)
                    if proc_probs[j] < 0:
                        proc_probs[j] = 0
            
            # Normalize probabilities
            if proc_probs.sum() > 0:
                proc_probs = proc_probs / proc_probs.sum()
            else:
                proc_probs = np.ones(len(self.processors)) / len(self.processors)
            
            # Select processor based on probability
            new_solution.processor_assignment[job_id] = np.random.choice(len(self.processors), p=proc_probs)
        
        # Update frequency assignment (continuous, but need to select from available frequencies)
        for i in range(self.num_jobs):
            job_id = self.jobs[i].id
            proc_idx = new_solution.processor_assignment[job_id]
            proc = self.processors[proc_idx]

            
            # Current frequency plus velocity
            new_freq = particle.solution.frequency_assignment[job_id] + particle.velocity['frequency'][i]
            
            # Find the closest available frequency
            available_freqs = proc.freq_levels
            closest_idx = np.argmin(np.abs(np.array(available_freqs) - new_freq))
            new_solution.frequency_assignment[job_id] = available_freqs[closest_idx]
        
        # Update optional execution (binary)
        for i in range(self.num_jobs):
            job_id = self.jobs[i].id
            # Apply sigmoid function to get probability
            prob = 1 / (1 + np.exp(-particle.velocity['optional'][i]))
            if np.random.random() < prob:
                new_solution.optional_execution[job_id] = 1
            else:
                new_solution.optional_execution[job_id] = 0
        
        # # Determine start times and check feasibility
        # new_solution = self.determine_start_times(new_solution)
        
        # If solution is not feasible, try to repair it
        if not new_solution.feasible:
            new_solution = self.repair_solution(new_solution)
        
        # Evaluate new solution
        self.evaluate_solution(new_solution)
        
        # Update particle's solution
        particle.solution = new_solution
        
        # Update personal best if new solution is better
        if new_solution.feasible and (not particle.best_solution.feasible or 
            (new_solution.energy <= particle.best_solution.energy and 
             new_solution.performance <= particle.best_solution.performance)):
            particle.best_solution = new_solution.copy()

    def select_leader(self):
        """Select a leader from the repository using crowding distance"""
        if not self.repository:
            return None
        
        # If repository has only one solution, return it
        if len(self.repository) == 1:
            return self.repository[0]
        
        # Calculate crowding distance for each solution in the repository
        crowding_distances = self.calculate_crowding_distances()
        
        # Select leader using binary tournament selection based on crowding distance
        idx1 = random.randint(0, len(self.repository) - 1)
        idx2 = random.randint(0, len(self.repository) - 1)
        
        if crowding_distances[idx1] > crowding_distances[idx2]:
            return self.repository[idx1]
        else:
            return self.repository[idx2]
    
    def calculate_crowding_distances(self):
        """Calculate crowding distance for each solution in the repository"""
        if len(self.repository) <= 2:
            return [float('inf')] * len(self.repository)
        
        # Extract objective values
        energy_values = np.array([sol.energy for sol in self.repository])
        performance_values = np.array([sol.performance for sol in self.repository])
        
        # Initialize crowding distances
        crowding_distances = np.zeros(len(self.repository))
        
        # Sort by energy
        energy_order = np.argsort(energy_values)
        crowding_distances[energy_order[0]] = float('inf')
        crowding_distances[energy_order[-1]] = float('inf')
        
        # Calculate crowding distance contribution from energy
        energy_range = energy_values.max() - energy_values.min()
        if energy_range > 0:  # Avoid division by zero
            for i in range(1, len(self.repository) - 1):
                idx = energy_order[i]
                prev_idx = energy_order[i - 1]
                next_idx = energy_order[i + 1]
                crowding_distances[idx] += (energy_values[next_idx] - energy_values[prev_idx]) / energy_range
        
        # Sort by performance
        performance_order = np.argsort(performance_values)
        crowding_distances[performance_order[0]] = float('inf')
        crowding_distances[performance_order[-1]] = float('inf')
        
        # Calculate crowding distance contribution from performance
        performance_range = performance_values.max() - performance_values.min()
        if performance_range > 0:  # Avoid division by zero
            for i in range(1, len(self.repository) - 1):
                idx = performance_order[i]
                prev_idx = performance_order[i - 1]
                next_idx = performance_order[i + 1]
                crowding_distances[idx] += (performance_values[next_idx] - performance_values[prev_idx]) / performance_range
        
        return crowding_distances
    
    def update_repository(self, solution):
        """Update the repository with a new solution"""
        # Check if the new solution is dominated by any solution in the repository
        is_dominated = False
        for repo_sol in self.repository[:]:  # Create a copy to iterate safely
            if repo_sol.dominates(solution):
                is_dominated = True
                break
        
        if is_dominated:
            return
        
        # Remove solutions that are dominated by the new solution
        self.repository = [sol for sol in self.repository if not solution.dominates(sol)]
        
        # Add the new solution to the repository
        self.repository.append(solution.copy())
        
        # If repository exceeds capacity, remove solutions with lowest crowding distances
        if len(self.repository) > self.repository_capacity:
            crowding_distances = self.calculate_crowding_distances()
            # Remove the solution with the lowest crowding distance
            min_idx = np.argmin(crowding_distances)
            self.repository.pop(min_idx)

    def run(self,debug=False):
        """Run MOPSO algorithm similar to NSGA-II"""
        # Record the start time
        start_time = time.time()
        
        self.debug = debug
        # Initialize particles (population in NSGA-II)
        particles = self.initialize_particles()

        
        feasible_count = sum(1 for p in particles if p.solution.feasible)

        if debug: 
            print(f"Feasible individuals in initial population: {feasible_count}/{len(particles)}")
        
        # Track Pareto front solutions over generations (similar to NSGA-II's history)
        pareto_history = []
        
        # Main loop
        for iteration in range(self.max_iterations):
            if debug:
                print(f"Iteration {iteration + 1}/{self.max_iterations}")
            
            # Create offspring population (particles will "evolve" over time)
            for particle in particles:
                # Select a leader (global best) from the repository, like NSGA-II's tournament selection
                leader = self.select_leader()
                if leader is None:
                    continue
                
                # Update particle's velocity and position (analogous to crossover and mutation)
                self.update_velocity(particle, leader)
                self.update_position(particle)
                
                # Update repository with the new solution (new "offspring")
                if particle.solution.feasible:
                    self.update_repository(particle.solution)
            
            # Store Pareto front for this iteration
            pareto_front = self.repository 
            pareto_history.append(pareto_front)
            
            # Optional: Print progress every few iterations
            avg_energy = sum(particle.solution.energy for particle in particles) / len(particles)
            avg_perf = sum(particle.solution.performance for particle in particles) / len(particles)
            if debug:
                print(f"  Avg Energy: {avg_energy:.2f}, Avg Performance Penalty: {avg_perf:.2f}") 
                print(f"  Pareto front size: {len(pareto_front)}")

        # Record the end time
        end_time = time.time()
        
        # Print total time for MOPSO run
        print(f"MOPSO completed in {end_time - start_time:.2f} seconds")
        
        # Return the final Pareto front and Pareto history
        final_front = self.repository 
        return final_front, pareto_history


