import argparse
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import nltk
import sys
import time
from loguru import logger
from torch.cuda import is_available
# from torch import load as torchload
from pathlib import Path

from .config.config import load_config
from .data.data import load_training_text, create_dataloaders
from .utils.utils import set_seeds, training_reporting, save_training_data
from .model.model import create_model, create_optimizer, create_scheduler
from .train.train import train
from .train.generate import generate_text
  

def main():

    # parse arguments
    parser = argparse.ArgumentParser(description='Train or generate text using the GPT-2 model.')
    parser.add_argument('mode', choices=['train', 'generate'], help='Mode to run the script in: "train" or "generate"')
    args = parser.parse_args()

    if args.mode not in ['train', 'generate']:
        logger.error("Invalid mode. Choose 'train' or 'generate'.")
        sys.exit(1)

    # load the config file
    logger.info('Loading config...')
    config = load_config()

    # add logging
    logger.info('Adding logging...')
    logger.remove()
    if config['testing']['test_flag']:
        logger.add(sys.stdout, level=config['testing']['logger_level'])
    else:
        logger.add(sys.stdout, level=config['training']['logger_level'])
        log_path = config['paths']['log_path'] + f"training_{str(time.time())}.log"
        logger.add(log_path, level=config['training']['logger_level'])

    # Load the GPT tokenizer
    logger.info("Loading GPT2 tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained(
        pretrained_model_name_or_path='gpt2', #gpt2-medium
        bos_token='<|startoftext|>', 
        eos_token='<|endoftext|>', 
        pad_token='<|pad|>', 
        clean_up_tokenization_spaces=True
    ) 

    # check if CUDA is available
    device = "cuda" if is_available() else "cpu"
    logger.info(f'Device: {device}')
    
    if args.mode == 'train':

        # download the 'punkt' tokenizer models from NLTK
        nltk.download('punkt')

        # load the data
        logger.info("Loading data...")
        training_text = load_training_text(config=config)

        # create dataloaders
        logger.info("Creating dataloaders...")
        train_dataloader, validation_dataloader = create_dataloaders(
            training_text=training_text, 
            tokenizer=tokenizer, 
            batch_size=config['training']['batch_size'], 
            train_percent=config['training']['train_percent'], 
            max_length=config['tokenizer']['max_length'],
            num_workers=config['training']['num_workers']
        )

        # create model
        logger.info("Creating model...")
        model = create_model(tokenizer=tokenizer)
        model = model.to(device)
        logger.info(f"Number of parameters: {model.num_parameters()}")

        # set the random seeds
        logger.info("Setting random seeds...")
        set_seeds(config=config)

        # create the optimizer
        logger.info("Creating optimizer...")
        optimizer = create_optimizer(config=config, model=model)

        # create the scheduler
        logger.info("Creating scheduler...")
        scheduler = create_scheduler(
            config=config, 
            optimizer=optimizer, 
            train_dataloader=train_dataloader
        )

        # Training
        logger.info("Training model...")
        fine_tuned_model, training_stats, total_t0 = train(
            model=model, 
            device=device, 
            train_dataloader=train_dataloader, 
            validation_dataloader=validation_dataloader, 
            tokenizer=tokenizer,
            optimizer=optimizer, 
            scheduler=scheduler, 
            config=config
        )

        # Training reporting
        training_reporting(total_t0=total_t0, training_stats=training_stats)

        # save training data
        save_training_data(config=config, model=fine_tuned_model, tokenizer=tokenizer)

    elif args.mode == 'generate':

        # load fine tuned model
        model_path = Path(config['paths']['model_path']).resolve() / f"{config['generation']['model_version']}.pth"
        logger.info(f"Loading fine-tuned model from {model_path}...")
        model = GPT2LMHeadModel.from_pretrained(model_path)
        model = model.to(device)

        # load tokenizer
        logger.info(f"Loading fine-tuned model from {model_path}....")        
        tokenizer = GPT2Tokenizer.from_pretrained(model_path)

        # Generate Text
        logger.info("Generating text...")
        generate_text(
            model=model, 
            tokenizer=tokenizer, 
            device=device, 
            config=config
        )


if __name__ == '__main__':
    main()