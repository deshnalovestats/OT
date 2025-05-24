import task as tsk
from modules import random, time
from chromosome import Chromosome
class NSGAII:
    def __init__(self, tasks, processors, hyperperiod):
        self.tasks = tasks
        self.processors = processors
        self.hyperperiod = hyperperiod
        
        # Generate all jobs in the hyperperiod
        self.jobs = self.generate_jobs()
        
        # NSGA-II parameters
        self.pop_size = 100
        self.max_generations = 100
        self.crossover_rate = 0.9
        self.mutation_rate = 1.0 / (len(self.jobs) * 3)  # 1/L where L is chromosome length
        
        # Bounds for normalization
        self.e_min = self.calculate_e_min()
        self.e_max = self.calculate_e_max()
        self.j_min = 0  # Best performance = 0 penalty
        self.j_max = self.calculate_j_max()
        self.debug = False
        
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
    
    def dominates(self, chrom1, chrom2):
        """Check if chrom1 dominates chrom2"""
        # If chrom1 is not feasible, it cannot dominate
        if not chrom1.feasible:
            return False
        
        # If chrom1 is feasible and chrom2 is not, chrom1 dominates
        if chrom1.feasible and not chrom2.feasible:
            return True
        
        # If both are feasible, check objectives
        energy1, perf1 = self.normalize_objectives(chrom1.energy, chrom1.performance)
        energy2, perf2 = self.normalize_objectives(chrom2.energy, chrom2.performance)
        
        if (energy1 <= energy2 and perf1 < perf2) or (energy1 < energy2 and perf1 <= perf2):
            return True
        
        return False
    
    def fast_non_dominated_sort(self, population):
        """Perform fast non-dominated sorting"""
        fronts = [[]]
        
        # For each chromosome
        for p in population:
            # Initialize domination counter and dominated set
            p.n_dominated = 0  # Number of solutions that dominate p
            p.dominates = []   # Set of solutions that p dominates
            
            # Compare with every other chromosome
            for q in population:
                if self.dominates(p, q):
                    p.dominates.append(q)
                elif self.dominates(q, p):
                    p.n_dominated += 1
            # if self.debug:
            #     print(f"Chrom {p} dominated by {p.n_dominated} others; feasible={p.feasible}")

            
            # If p is non-dominated, add to first front
            if p.n_dominated == 0:
                p.rank = 0
                fronts[0].append(p)
        
        # Generate subsequent fronts
        i = 0
        while i < len(fronts) and fronts[i]:
            next_front = []

            for p in fronts[i]:
                for q in p.dominates:
                    q.n_dominated -= 1
                    if q.n_dominated == 0:
                        q.rank = i + 1
                        next_front.append(q)

            i += 1
            fronts.append(next_front)  # Allow appending empty list; loop checks length
            
        # if self.debug:
        #     for idx, front in enumerate(fronts):
        #         print(f"Front {idx}: {len(front)} individuals")

        return fronts
    
    def calculate_crowding_distance(self, front):
        """Calculate crowding distance for solutions in a front"""
        if len(front) <= 2:
            for p in front:
                p.crowding_distance = float('inf')
            return
        
        # Initialize crowding distance
        for p in front:
            p.crowding_distance = 0
        
        # Calculate crowding distance for each objective
        for obj_idx, obj_name in enumerate(['energy', 'performance']):
            # Sort by objective value
            front.sort(key=lambda x: getattr(x, obj_name))
            
            # Boundary points have infinite distance
            front[0].crowding_distance = float('inf')
            front[-1].crowding_distance = float('inf')
            
            # Calculate distance for intermediate points
            f_max = getattr(front[-1], obj_name)
            f_min = getattr(front[0], obj_name)
            
            if f_max == f_min:
                continue
                
            for i in range(1, len(front) - 1):
                front[i].crowding_distance += (
                    getattr(front[i+1], obj_name) - getattr(front[i-1], obj_name)
                ) / (f_max - f_min)
    
    def crowded_comparison_operator(self, p, q):
        """Crowded comparison operator for tournament selection"""
        if p.rank < q.rank:
            return p
        elif p.rank > q.rank:
            return q
        elif p.crowding_distance > q.crowding_distance:
            return p
        elif p.crowding_distance < q.crowding_distance:
            return q
        else:
            return random.choice([p, q])
    
    def tournament_selection(self, population):
        """Binary tournament selection"""
        candidates = random.sample(population, 2)
        return self.crowded_comparison_operator(candidates[0], candidates[1])
    
    def crossover(self, parent1, parent2):
        """Crossover operator for creating offspring"""
        if random.random() > self.crossover_rate:
            return parent1.copy(), parent2.copy()
        
        child1 = parent1.copy()
        child2 = parent2.copy()
        
        # One-point crossover for each decision variable
        for decision_var in ['proc_assignment', 'freq_assignment', 'opt_execution']:
            # Convert to list for easier crossover
            var1 = list(getattr(parent1, decision_var).items())
            var2 = list(getattr(parent2, decision_var).items())
            
            # Crossover point
            point = random.randint(1, len(var1) - 1)
            
            # Swap segments
            new_var1 = dict(var1[:point] + var2[point:])
            new_var2 = dict(var2[:point] + var1[point:])
            
            # Update children
            setattr(child1, decision_var, new_var1)
            setattr(child2, decision_var, new_var2)
        
        # Check and repair feasibility
        child1.check_feasibility()
        child2.check_feasibility()
        
        return child1, child2
    
    def mutation(self, chromosome):
        """Mutation operator"""
        mutated = chromosome.copy()
        
        for job in self.jobs:
            # Mutate processor assignment
            if random.random() < self.mutation_rate:
                new_proc = random.choice(self.processors)
                mutated.proc_assignment[job.id] = new_proc.id
            
            # Mutate frequency assignment
            if random.random() < self.mutation_rate:
                proc_id = mutated.proc_assignment[job.id]
                proc = next(p for p in self.processors if p.id == proc_id)
                
                # Choose frequency that can at least handle mandatory part
                valid_freqs = [f for f in proc.freq_levels if (job.task.c_m / f) <= job.task.period]
                if valid_freqs:
                    mutated.freq_assignment[job.id] = random.choice(valid_freqs)
            
            # Mutate optional execution
            if random.random() < self.mutation_rate:
                # Flip but ensure feasibility
                opt_value = 1 - mutated.opt_execution[job.id]
                
                # Check if flipping to 1 is feasible
                if opt_value == 1:
                    freq = mutated.freq_assignment[job.id]
                    exec_time = job.task.c_m / freq + job.task.c_o / freq
                    
                    if exec_time <= job.task.period:
                        mutated.opt_execution[job.id] = opt_value
                else:
                    # Always feasible to skip optional part
                    mutated.opt_execution[job.id] = opt_value
        
        # Check and potentially repair full schedule
        mutated.check_feasibility()
        
        return mutated
    
    def repair(self, chromosome):
        """Repair infeasible solutions"""
        repaired = chromosome.copy()
        
        if repaired.feasible:
            return repaired

        # Strategy 1: Try to increase frequencies
        for job in self.jobs:
            proc_id = repaired.proc_assignment[job.id]
            proc = next(p for p in self.processors if p.id == proc_id)
            freq = repaired.freq_assignment[job.id]

        # ensure current frequency is in the processor's list
        if freq not in proc.freq_levels:
            # Replace with the closest valid frequency
            corrected_freq = min(proc.freq_levels, key=lambda f: abs(f - freq))
            repaired.freq_assignment[job.id] = corrected_freq
            current_freq_idx = proc.freq_levels.index(corrected_freq)
        else:
            current_freq_idx = proc.freq_levels.index(freq)

        # Try increasing frequency if possible
        if current_freq_idx < len(proc.freq_levels) - 1:
            new_freq = proc.freq_levels[current_freq_idx + 1]
            repaired.freq_assignment[job.id] = new_freq

        # Strategy 2: Drop optional executions
        for job in self.jobs:
            if repaired.opt_execution[job.id] == 1:
                repaired.opt_execution[job.id] = 0
        
        # Check if repair worked
        repaired.check_feasibility()
        
        return repaired
    
    def create_initial_population(self):
        """Create initial population with feasible solutions"""
        population = []
        
        while len(population) < self.pop_size:
            chrom = Chromosome(self.jobs, self.processors)
            chrom.initialize_random()
            
            if not chrom.feasible:
                chrom = self.repair(chrom)
            
            if chrom.feasible:
                chrom.calculate_objectives()
                population.append(chrom)
        
        return population
    
    def run(self,debug=False):
        """Run NSGA-II algorithm"""
        start_time = time.time()
        
        self.debug = debug


        # Initialize population
        population = self.create_initial_population()
        
        feasible_count = sum(1 for p in population if p.feasible)

        if debug: 
            print(f"Feasible individuals in initial population: {feasible_count}/{len(population)}")

        fronts = self.fast_non_dominated_sort(population)
        for front in fronts:
            self.calculate_crowding_distance(front)
        
        # Track Pareto fronts over generations
        pareto_history = []
        
        # Main loop
        for gen in range(self.max_generations):
            if debug:
             print(f"Generation {gen+1}/{self.max_generations}")
            
            # Create offspring population
            offspring = []
            
            while len(offspring) < self.pop_size:
                # Selection
                parent1 = self.tournament_selection(population)
                parent2 = self.tournament_selection(population)
                
                # Crossover
                child1, child2 = self.crossover(parent1, parent2)
                
                # Mutation
                child1 = self.mutation(child1)
                child2 = self.mutation(child2)
                
                # Repair if needed
                if not child1.feasible:
                    child1 = self.repair(child1)
                if not child2.feasible:
                    child2 = self.repair(child2)
                
                # Calculate objectives
                if child1.feasible:
                    child1.calculate_objectives()
                    offspring.append(child1)
                if child2.feasible and len(offspring) < self.pop_size:
                    child2.calculate_objectives()
                    offspring.append(child2)
            
            # Combine parent and offspring populations
            combined = population + offspring
            
            # Non-dominated sorting
            fronts = self.fast_non_dominated_sort(combined)
            
            # Calculate crowding distance for each front
            for front in fronts:
                self.calculate_crowding_distance(front)
            
            # Select next generation
            new_population = []
            front_idx = 0
            
            while len(new_population) + len(fronts[front_idx]) <= self.pop_size:
                # Add whole front
                new_population.extend(fronts[front_idx])
                front_idx += 1
                
                # Check if we've used all fronts
                if front_idx == len(fronts):
                    break
            
            # If needed, add solutions from the next front based on crowding distance
            if len(new_population) < self.pop_size and front_idx < len(fronts):
                # Sort by crowding distance
                fronts[front_idx].sort(key=lambda x: x.crowding_distance, reverse=True)
                
                # Add solutions until population is full
                new_population.extend(fronts[front_idx][:self.pop_size - len(new_population)])
            
            # Update population
            population = new_population
            
            # Store pareto front for this generation
            pareto_front = [p for p in population if p.rank == 0]
            pareto_history.append(pareto_front)
            
            # Print progress
            avg_energy = sum(p.energy for p in population) / len(population)
            avg_perf = sum(p.performance for p in population) / len(population)

            if debug: 
                print(f"  Avg Energy: {avg_energy:.2f}, Avg Performance Penalty: {avg_perf:.2f}")
                print(f"  Pareto front size: {len(pareto_front)}")
        
        end_time = time.time()
        print(f"NSGA-II completed in {end_time - start_time:.2f} seconds")
        
        # Return the final Pareto front and history
        final_front = [p for p in population if p.rank == 0]
        return final_front, pareto_history
