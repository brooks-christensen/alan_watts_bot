from dataclasses import dataclass
import yaml

@dataclass
class PathsConfig:
    data_path: str
    model_save_path: str
    model_load_path: str
    log_path: str
    
@dataclass
class TokenizerConfig:
    max_length: int

@dataclass
class AdamWConfig:
    lr: float
    adam_epsilon: float
    weight_decay: float
    beta1: float
    beta2: float
    
@dataclass
class OptimizerConfig:
    type: str
    adamw: AdamWConfig
    
@dataclass
class TrainingConfig:
    logger_level: str
    seed_val: int
    batch_size: int
    train_percent: float
    epochs: int
    warmup_steps: float
    sample_every: int
    max_patience: int
    num_workers: int
    
@dataclass
class TestingConfig:
    logger_level: str
    test_flag: bool
    test_length: int
    
@dataclass
class GenerationConfig:
    do_sample: bool
    temperature: float
    top_k: int
    max_length: int
    top_p: float
    num_return_sequences: int
    num_return_sequences_training: int
    model_version: str

@dataclass
class Config:
    paths: PathsConfig
    tokenizer: TokenizerConfig
    optimizer: OptimizerConfig
    training: TrainingConfig
    testing: TestingConfig
    Generation: GenerationConfig

def load_config(filename: str = './language_model/config/config.yaml') -> Config:
    """
    Load the config file

    Args:
    -----
    filename (str): the path to the config file

    Returns:
    --------
    Dict[str, Any]: the config file as a dictionary
    """

    # load the config file
    with open(filename, 'r') as file:
        config = yaml.safe_load(file)

    return config