from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config, get_linear_schedule_with_warmup
from typing import Tuple, Dict, Any
from torch.optim import AdamW
from loguru import logger
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader


def create_optimizer(config: Dict[str, Any], model: GPT2LMHeadModel) -> AdamW:
    """
    Create the optimizer

    Args:
    -----
    config (Dict[str, Any]): the config file
    model (GPT2LMHeadModel): the GPT2 model

    Returns:
    --------
    AdamW: the optimizer
    """
    if config['optimizer']['type'] == 'AdamW':
        # Note: AdamW is a class from the huggingface library (as opposed to pytorch)
        opt_config = config['optimizer']['adamw']
        logger.debug(opt_config['lr'])
        optimizer = AdamW(
            model.parameters(),
            lr=float(opt_config['lr']),
            eps=float(opt_config['adam_epsilon']),
            weight_decay=opt_config['weight_decay'],
            betas=(opt_config['beta1'], opt_config['beta2'])
        )

    return optimizer


def create_scheduler(
    config: dict, 
    optimizer: AdamW, 
    train_dataloader: DataLoader
) -> LambdaLR:
    """
    Create the learning rate scheduler

    Args:
    -----
    config (dict): the config file
    optimizer (AdamW): the optimizer
    train_dataloader (DataLoader): the training dataloader

    Returns:
    --------
    torch.optim.lr_scheduler.LambdaLR: the learning rate scheduler
    """

    # Total number of training steps is [number of batches] x [number of epochs].
    # (Note that this is not the same as the number of training samples).
    total_steps = len(train_dataloader) * config['training']['epochs']

    # Create the learning rate scheduler.
    # This changes the learning rate as the training loop progresses
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps = float(config['training']['warmup_steps']),
        num_training_steps = total_steps
    )

    return scheduler


def create_model(tokenizer: GPT2Tokenizer) -> Tuple[GPT2LMHeadModel, str]:
    """
    Create the GPT2 model

    Args:
    -----
    tokenizer (GPT2Tokenizer): the GPT2 tokenizer

    Returns:
    --------
    List[GPT2LMHeadModel, str]: the GPT2 model and the device
    """    

    # I'm not really doing anything with the config buheret
    configuration = GPT2Config.from_pretrained('gpt2', output_hidden_states=False)

    # instantiate the model
    model = GPT2LMHeadModel.from_pretrained("gpt2", config=configuration)

    # this step is necessary because I've added some tokens (bos_token, etc) to the embeddings
    # otherwise the tokenizer and model tensors won't match up
    model.resize_token_embeddings(len(tokenizer))

    return model