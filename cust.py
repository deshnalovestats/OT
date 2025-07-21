from logging_config import LoggingFlags

def customisedLogging():
    # Main execution flags
    LoggingFlags.MAIN_MENU = True
    LoggingFlags.OPTIMIZATION_PROGRESS = True
    LoggingFlags.SOLUTION_ANALYSIS = True
    
    # Problem evaluation flags
    LoggingFlags.EVALUATION_DETAILS = False
    LoggingFlags.GENERATION_STATS = False
    LoggingFlags.FEASIBILITY_CHECKS = False
    LoggingFlags.REPAIR_OPERATIONS = False
    
    # Sampling and genetic operators flags
    LoggingFlags.SAMPLING_PROGRESS = False
    LoggingFlags.CROSSOVER_DETAILS = False
    LoggingFlags.MUTATION_DETAILS = False
    
    # Performance and debugging flags
    LoggingFlags.TIMING_CONSTRAINTS = False
    LoggingFlags.ENERGY_CALCULATION = False
    LoggingFlags.PERFORMANCE_CALCULATION = False

# Configure logging
# LoggingFlags.silent_run()
customisedLogging()

from nsga2_driver import run_nsga2_optimization

# Run optimization
result, problem = run_nsga2_optimization(
    'Tasks_20/instance_fixedproc_001.json',
    n_generations=50,
    pop_size=100
)