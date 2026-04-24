import logging
import sys

class Logger:
    """Simple Logger utility for the research assistant."""
    
    def __init__(self):
        self.loggers = {}

    def get_logger(self, name: str) -> logging.Logger:
        if name in self.loggers:
            return self.loggers[name]
        
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        self.loggers[name] = logger
        return logger
