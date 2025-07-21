class LoggingFlags:
    """Centralized logging configuration with custom flags"""
    
    # Main execution flags
    MAIN_MENU = True
    OPTIMIZATION_PROGRESS = True
    SOLUTION_ANALYSIS = True
    
    # Problem evaluation flags
    EVALUATION_DETAILS = False
    GENERATION_STATS = False
    FEASIBILITY_CHECKS = False
    REPAIR_OPERATIONS = True
    
    # Sampling and genetic operators flags
    SAMPLING_PROGRESS = False
    CROSSOVER_DETAILS = False
    MUTATION_DETAILS = False
    
    # Performance and debugging flags
    TIMING_CONSTRAINTS = False
    ENERGY_CALCULATION = False
    PERFORMANCE_CALCULATION = False
    
    @classmethod
    def enable_all_debug(cls):
        """Enable all debug flags"""
        for attr in dir(cls):
            if not attr.startswith('_') and attr.isupper():
                setattr(cls, attr, True)
    
    @classmethod
    def disable_all_debug(cls):
        """Disable all debug flags except main execution"""
        for attr in dir(cls):
            if not attr.startswith('_') and attr.isupper():
                if attr not in ['MAIN_MENU', 'OPTIMIZATION_PROGRESS']:
                    setattr(cls, attr, False)
    
    @classmethod
    def set_production_mode(cls):
        """Set flags for production run (minimal logging)"""
        cls.disable_all_debug()
        cls.OPTIMIZATION_PROGRESS = True
        cls.REPAIR_OPERATIONS = True
        cls.SOLUTION_ANALYSIS = True

def log_if(flag: bool, message: str, *args, **kwargs):
    """Print message only if flag is True"""
    if flag:
        print(message, *args, **kwargs)