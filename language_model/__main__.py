import argparse
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import nltk
import sys
import time
from loguru import logger
from torch.cuda import is_available
# from torch import load as torchload
from pathlib import Path
import optuna

from .config.config import load_config
from .data.data import load_training_text, create_dataloaders
from .utils.utils import set_seeds, training_reporting, save_training_data
from .model.model import create_model, create_optimizer, create_scheduler
from .train.train import train
from .train.generate import generate_text
  

def main():

    # parse arguments
    parser = argparse.ArgumentParser(description='Train or generate text using the GPT-2 model.')
    parser.add_argument('mode', choices=['train', 'generate', 'optimize'], help='Mode to run the script in: "train" or "generate"')
    args = parser.parse_args()

    if args.mode not in ['train', 'generate', 'optimize']:
        logger.error("Invalid mode. Choose 'train', 'generate', or 'optimize'.")
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
        log_path = config['paths']['log_path'] + time.strftime("%m-%d-%YT%H-%M-%S") + ".log"
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

    elif args.mode == 'optimize':
        def objective(trial):
            # Sample hyperparameters
            learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-3)
            batch_size = trial.suggest_categorical('batch_size', [4, 8, 16, 32])
            # Update your configuration with these sampled hyperparameters.
            config['training']['learning_rate'] = learning_rate
            config['training']['batch_size'] = batch_size
            
            # Re-create your dataloaders, model, optimizer, and scheduler with new parameters
            train_dataloader, validation_dataloader = create_dataloaders(
                training_text=training_text, 
                tokenizer=tokenizer, 
                batch_size=batch_size, 
                train_percent=config['training']['train_percent'], 
                max_length=config['tokenizer']['max_length'],
                num_workers=config['training']['num_workers']
            )
            model = create_model(tokenizer=tokenizer).to(device)
            optimizer = create_optimizer(config=config, model=model)
            scheduler = create_scheduler(
                config=config, 
                optimizer=optimizer, 
                train_dataloader=train_dataloader
            )
            
            # Train for a few epochs (or one epoch) to get a validation score.
            _, training_stats, _ = train(
                model=model, 
                device=device, 
                train_dataloader=train_dataloader, 
                validation_dataloader=validation_dataloader, 
                tokenizer=tokenizer,
                optimizer=optimizer, 
                scheduler=scheduler, 
                config=config,
                start_epoch=0,          # See checkpointing modifications below.
                max_epochs=3            # Short run for tuning.
            )
            # Return the validation loss from the last epoch as the objective.
            val_loss = training_stats[-1]['Valid. Loss']
            return val_loss
        
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=20)
        print("Best hyperparameters:", study.best_params)


if __name__ == '__main__':
    main()