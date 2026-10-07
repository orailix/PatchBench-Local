from rich.console import Console
import os
from datetime import datetime

class LogConfig:
    def __init__(self, enabled=True):
        self.enabled = enabled
        self.file_output = False
        self.file_path = None
        
class GlobalConfig:
    _instance = None
    console = Console()
    log_configs = {
        "global": LogConfig(enabled=True),
        "MalleableModel": LogConfig(enabled=True),
        "SteeringVector": LogConfig(enabled=True),
        "SteeringDataset": LogConfig(enabled=True)
    }
    log_directory = "activation_steering_logs"
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(GlobalConfig, cls).__new__(cls)
            cls._instance.initialize_log_files()
        return cls._instance

    @classmethod
    def initialize_log_files(cls):
        if not cls._initialized:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            os.makedirs(cls.log_directory, exist_ok=True)
            for class_name in cls.log_configs:
                cls.log_configs[class_name].file_path = os.path.join(cls.log_directory, f"{class_name}_{timestamp}.log")
            cls._initialized = True

    @classmethod
    def set_verbose(cls, verbose: bool, class_name: str = "global"):
        if class_name in cls.log_configs:
            cls.log_configs[class_name].enabled = verbose

    @classmethod
    def is_verbose(cls, class_name: str = "global"):
        return cls.log_configs[class_name].enabled and cls.log_configs["global"].enabled

    @classmethod
    def set_file_output(cls, enabled: bool, class_name: str = "global"):
        cls.initialize_log_files()  # Ensure file paths are set
        if class_name in cls.log_configs:
            cls.log_configs[class_name].file_output = enabled

    @classmethod
    def should_log_to_file(cls, class_name: str):
        return cls.log_configs[class_name].file_output or cls.log_configs["global"].file_output

    @classmethod
    def get_file_path(cls, class_name: str):
        cls.initialize_log_files()  # Ensure file paths are set
        return cls.log_configs[class_name].file_path

def log(message: str, style: str = None, class_name: str = "global"):
    if GlobalConfig.is_verbose(class_name):
        GlobalConfig.console.print(message, style=style)
    
    if GlobalConfig.should_log_to_file(class_name):
        file_path = GlobalConfig.get_file_path(class_name)
        with open(file_path, "a") as f:
            f.write(f"{datetime.now().isoformat()} - {message}\n")