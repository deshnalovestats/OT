from modules import np
from solution import Solution

# Particle class - represents a particle in the MOPSO algorithm
class Particle:
    def __init__(self, solution: Solution, velocity=None):
        """Initialize the particle with a solution and an optional velocity."""
        self.solution = solution  # Current solution
        self.best_solution = solution.copy()  # Personal best solution
        
        if velocity is None:
            # Initialize the velocity components for processor, frequency, and optional execution
            velocity_shape = len(solution.processor_assignment)
            self.velocity = {
                'processor': np.zeros(velocity_shape),
                'frequency': np.zeros(velocity_shape),
                'optional': np.zeros(velocity_shape)
            }
        else:
            # If velocity is provided, use the passed velocity
            self.velocity = velocity
    
    def update_velocity(self, global_best, w=0.4, c1=1.5, c2=2.0):
        """Update particle's velocity based on current position, personal best, and global best."""
        for key in self.velocity.keys():
            if key == 'processor':
                # For discrete values (processor assignment)
                r1 = np.random.random(len(self.solution.processor_assignment))
                r2 = np.random.random(len(self.solution.processor_assignment))
                
                current = self.solution.processor_assignment
                personal_best = self.best_solution.processor_assignment
                global_best_val = global_best.processor_assignment
                
                # Update velocity using PSO formula
                self.velocity[key] = (w * self.velocity[key] + 
                                      c1 * r1 * (personal_best - current) + 
                                      c2 * r2 * (global_best_val - current))
            
            elif key == 'frequency':
                # For continuous values (frequency assignment)
                r1 = np.random.random(len(self.solution.processor_assignment))
                r2 = np.random.random(len(self.solution.processor_assignment))
                
                current = self.solution.frequency_assignment
                personal_best = self.best_solution.frequency_assignment
                global_best_val = global_best.frequency_assignment
                
                self.velocity[key] = (w * self.velocity[key] + 
                                      c1 * r1 * (personal_best - current) + 
                                      c2 * r2 * (global_best_val - current))
            
            elif key == 'optional':
                # For binary values (optional execution)
                r1 = np.random.random(len(self.solution.processor_assignment))
                r2 = np.random.random(len(self.solution.processor_assignment))
                
                current = self.solution.optional_execution
                personal_best = self.best_solution.optional_execution
                global_best_val = global_best.optional_execution
                
                self.velocity[key] = (w * self.velocity[key] + 
                                      c1 * r1 * (personal_best - current) + 
                                      c2 * r2 * (global_best_val - current))
    
    def update_position(self):
        """Update the particle's position based on its velocity."""
        new_solution = Solution(len(self.solution.processor_assignment), len(self.solution.processor_assignment))
        
        # Update processor assignment (discrete)
        for i in range(len(self.solution.processor_assignment)):
            # Apply sigmoid function to get probabilities for processor selection
            prob = 1 / (1 + np.exp(-self.velocity['processor'][i]))
            if np.random.random() < prob:
                new_solution.processor_assignment[i] = 1  # Assign to some processor (this is an example)
            else:
                new_solution.processor_assignment[i] = 0  # Assign to a different processor
        
        # Update frequency assignment (continuous)
        for i in range(len(self.solution.processor_assignment)):
            # Update the frequency based on velocity
            new_solution.frequency_assignment[i] = self.solution.frequency_assignment[i] + self.velocity['frequency'][i]
        
        # Update optional execution (binary)
        for i in range(len(self.solution.processor_assignment)):
            # Apply sigmoid function to get probability for optional execution
            prob = 1 / (1 + np.exp(-self.velocity['optional'][i]))
            if np.random.random() < prob:
                new_solution.optional_execution[i] = 1
            else:
                new_solution.optional_execution[i] = 0
        
        # Evaluate the new solution using the jobs and processors from the Solution class
        new_solution.evaluate(self.solution.jobs, self.solution.processors)  # Use jobs and processors from Solution
        
        # Update particle's solution
        self.solution = new_solution
        
        # Update personal best if the new solution is better
        if new_solution.dominates(self.best_solution):
            self.best_solution = new_solution.copy()