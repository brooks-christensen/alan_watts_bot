from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
from transformers import get_linear_schedule_with_warmup

import nltk

import yaml
import os
import sys
import time
import datetime
# from google.colab import drive
from loguru import logger

import pandas as pd
# import seaborn as sns
import numpy as np
import random

from typing import List, Dict, Any, Tuple

# import matplotlib.pyplot as plt
# %matplotlib inline

import torch
from torch.utils.data import Dataset, DataLoader, SequentialSampler # , random_split, RandomSampler
from torch.optim import AdamW
torch.manual_seed(42)

nltk.download('punkt')


def load_config(filename: str = './config.yaml') -> Dict[str, Any]:
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


def load_training_text(config: Dict[str, Any]) -> pd.Series:
    """
    Load the training text from a file

    Args:
    -----
    config (Dict[str, Any]): the config file

    Returns:
    --------
    pd.Series: the training text as a pandas series
    """

    filename = config['paths']['data_path']

    # load text into dataframe from file
    with open(filename, 'r') as file:
        data = file.read().split('\n')

    # create a dataframe
    df = pd.DataFrame(data, columns=['text'])

    # remove empty lines
    df = df[df.text != '']

    # remove duplicate lines
    df = df.drop_duplicates().reset_index(drop=True)

    if config['testing']['test_flag']:
        df = df.loc[:config['testing']['test_length'], :]

    # # remove lines with more than 768 tokens
    # df['word_count'] = df['text'].apply(lambda x: len(nltk.word_tokenize(x)))
    # df = df[df['word_count'] < 768]

    # extract the text from the dataframe
    training_text = df.text.copy()

    # Add a print statement to verify the loaded data
    logger.debug("Sample loaded text data:")
    logger.debug(f"\n{training_text.head(5)}")  # Print first 5 lines for inspection

    # sys.exit()

    return training_text


def create_dataloaders(
    training_text: pd.Series, 
    tokenizer: GPT2Tokenizer, 
    batch_size: int = 2, 
    train_percent: float = 0.8, 
    max_length: int = 768
) -> List[DataLoader]:
    """
    Create the training and validation dataloaders

    Args:
    -----
    training_text (pd.Series): the training text
    tokenizer (GPT2Tokenizer): the GPT2 tokenizer
    batch_size (int): the batch size
    train_percent (float): the percentage of the data to use for training
    max_length (int): the maximum length of the input text

    Returns:
    --------
    List[DataLoader]: the training and validation dataloaders
    """

    # create a dataset
    dataset = GPT2Dataset(training_text, tokenizer, max_length=max_length)

    # Split into training and validation sets
    train_size = int(train_percent * len(dataset))
    # val_size = len(dataset) - train_size

    # train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    # train_dataset, val_dataset = dataset[:train_size], dataset[train_size:]
    # Perform sequential splitting
    train_dataset = torch.utils.data.Subset(dataset, range(train_size))  # First part is for training
    val_dataset = torch.utils.data.Subset(dataset, range(train_size, len(dataset)))  # Second part is for validation

    # Create the DataLoaders for our training and validation datasets.
    # We'll take training samples in sequential order.
    train_dataloader = DataLoader(
        train_dataset,  # The training samples.
        sampler = SequentialSampler(train_dataset), # Select batches randomly
        batch_size = batch_size # Trains with this batch size.
    )

    # For validation the order doesn't matter, so we'll just read them sequentially.
    validation_dataloader = DataLoader(
        val_dataset, # The validation samples.
        sampler = SequentialSampler(val_dataset), # Pull out batches sequentially.
        batch_size = batch_size # Evaluate with this batch size.
    )

    return train_dataloader, validation_dataloader


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

    # Tell pytorch to run this model on the GPU, if available.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    return model, device


class GPT2Dataset(Dataset):

    def __init__(
        self, 
        txt_list: pd.Series,
        tokenizer: GPT2Tokenizer, 
        max_length=768
    ):
        """
        txt_list: List of text to encode
        tokenizer: GPT-2 tokenizer
        max_length: maximum length at which to truncate the text
        """
        self.tokenizer = tokenizer
        self.input_ids = []
        self.attn_masks = []

        for txt in txt_list:

            encodings_dict = tokenizer(
                '<|startoftext|>'+ txt + '<|endoftext|>', 
                truncation=True, 
                max_length=max_length, 
                padding="max_length"
            )
            
            self.input_ids.append(torch.tensor(encodings_dict['input_ids']))
            self.attn_masks.append(torch.tensor(encodings_dict['attention_mask']))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.attn_masks[idx]
  
  
def format_time(elapsed: float) -> str:
    """
    Takes a time in seconds and returns a string hh:mm:ss

    Args:
    -----
    elapsed (float): the time in seconds

    Returns:
    --------
    str: the time in hh:mm:ss format
    """

    return str(datetime.timedelta(seconds=int(round((elapsed)))))


def set_seeds(config: Dict[str, Any]) -> None:
    """
    Set the random seeds for reproducibility

    Args:
    -----
    config (Dict[str, Any]): the config file
    """

    random.seed(config['training']['seed_val'])
    np.random.seed(config['training']['seed_val'])
    torch.manual_seed(config['training']['seed_val'])
    torch.cuda.manual_seed_all(config['training']['seed_val'])


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
            weight_decay=opt_config['weight_decay']
        )

    return optimizer


def create_scheduler(
    config: dict, 
    optimizer: AdamW, 
    train_dataloader: DataLoader
) -> torch.optim.lr_scheduler.LambdaLR:
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


def train(
    model: GPT2LMHeadModel, 
    device: str, 
    train_dataloader: DataLoader, 
    validation_dataloader: DataLoader, 
    optimizer: AdamW, 
    scheduler: torch.optim.lr_scheduler.LambdaLR, 
    config: Dict[str, Any]
) -> Tuple[GPT2LMHeadModel, List[Dict[str, float]], float]:
    """
    Train the model

    Args:
    -----
    model (GPT2LMHeadModel): the GPT2 model
    device (str): the device to use
    train_dataloader (DataLoader): the training dataloader
    validation_dataloader (DataLoader): the validation dataloader
    optimizer (AdamW): the optimizer
    scheduler: the learning rate scheduler
    config (dict): the config file

    Returns:
    --------
    List[GPT2LMHeadModel, List[dict], float]: the trained model, the training statistics, the total time taken
    """

    total_t0 = time.time()

    training_stats = []

    best_val_loss = float('inf')
    patience = 0
    max_patience = config['training']['max_patience']

    model = model.to(device)
    best_model_state_dict = None

    prompt = "The philosophy of Alan Watts begins with the idea that"

    # Tokenize the prompt and prepare the input tensor
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs['input_ids']  # The token IDs
    attention_mask = inputs['attention_mask']  # The attention mask

    # Print the generated input tensor
    logger.info(input_ids)
    logger.info(attention_mask)

    for epoch_i in range(0, config['training']['epochs']):

        # ========================================
        #               Training
        # ========================================

        logger.info("")
        logger.info('======== Epoch {:} / {:} ========'.format(epoch_i + 1, config['training']['epochs']))
        logger.info('Training...')

        t0 = time.time()

        total_train_loss = 0

        model.train()

        for step, batch in enumerate(train_dataloader):

            b_input_ids = batch[0].to(device)
            b_labels = batch[0].to(device)
            b_masks = batch[1].to(device)

            model.zero_grad()

            outputs = model(  
                b_input_ids,
                labels=b_labels,
                attention_mask = b_masks,
                token_type_ids=None
            )

            loss = outputs[0]

            batch_loss = loss.item()
            total_train_loss += batch_loss

            # Get sample every x batches.
            if step % config['training']['sample_every'] == 0 and not step == 0:

                elapsed = format_time(time.time() - t0)
                logger.info('  Batch {:>5,}  of  {:>5,}. Loss: {:>5,}.   Elapsed: {:}.'.format(step, len(train_dataloader), batch_loss, elapsed))

                model.eval()

                # sample_outputs = model.generate(
                #     bos_token_id=random.randint(1,30000),
                #     do_sample=True,
                #     top_k=50,
                #     max_length = 200,
                #     top_p=0.95,
                #     num_return_sequences=1
                # )

                sample_outputs = model.generate(
                    # generated,
                    input_ids=input_ids,
                    attention_mask=attention_mask,  # Include the attention mask
                    # bos_token_id=random.randint(1,30000),
                    do_sample=config['generation']['do_sample'],
                    temperature=config['generation']['temperature'],
                    top_k=config['generation']['top_k'],
                    max_length=config['generation']['max_length'],
                    top_p=config['generation']['top_p'],
                    num_return_sequences=config['generation']['num_return_sequences_training'],
                    pad_token_id=tokenizer.eos_token_id  # Set pad token ID explicitly
                )

                for i, sample_output in enumerate(sample_outputs):
                    logger.info("{}: {}".format(i, tokenizer.decode(sample_output, skip_special_tokens=True)))

                model.train()

            loss.backward()

            optimizer.step()

            scheduler.step()

        # Calculate the average loss over all of the batches.
        avg_train_loss = total_train_loss / len(train_dataloader)

        # Measure how long this epoch took.
        training_time = format_time(time.time() - t0)

        logger.info("")
        logger.info("  Average training loss: {0:.2f}".format(avg_train_loss))
        logger.info("  Training epoch took: {:}".format(training_time))

        # ========================================
        #               Validation
        # ========================================

        logger.info("")
        logger.info("Running Validation...")

        t0 = time.time()

        model.eval()

        total_eval_loss = 0
        # nb_eval_steps = 0

        # Evaluate data for one epoch
        for batch in validation_dataloader:

            b_input_ids = batch[0].to(device)
            b_labels = batch[0].to(device)
            b_masks = batch[1].to(device)

            with torch.no_grad():

                outputs  = model(
                    b_input_ids,
                    # token_type_ids=None,
                    attention_mask = b_masks,
                    labels=b_labels
                )

                loss = outputs[0]

            batch_loss = loss.item()
            total_eval_loss += batch_loss

        avg_val_loss = total_eval_loss / len(validation_dataloader)

        validation_time = format_time(time.time() - t0)

        logger.info("  Validation Loss: {0:.2f}".format(avg_val_loss))
        logger.info("  Validation took: {:}".format(validation_time))

        # Record all statistics from this epoch.
        training_stats.append(
            {
                'epoch': epoch_i + 1,
                'Training Loss': avg_train_loss,
                'Valid. Loss': avg_val_loss,
                'Training Time': training_time,
                'Validation Time': validation_time
            }
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state_dict = model.state_dict()
            patience = 0
            logger.info("New best model found!\nValidation loss: {:.4f}".format(best_val_loss))
        else:
            patience += 1
            logger.info("No improvement in validation loss. Patience: {}/{}".format(patience, max_patience))
            if patience >= max_patience:
                logger.info("Stopping early!")
                break

    # If early stopping was triggered, load the best model's state
    if best_model_state_dict:
        model.load_state_dict(best_model_state_dict)

    return model, training_stats, total_t0


def training_reporting(
    total_t0: float, 
    training_stats: List[Dict[str, float]]
) -> None:
    """
    Print the training time

    Args:
    -----
    total_t0 (float): the total time taken for training
    training_stats (List[Dict[str, float]]): the training statistics
    """

    logger.info("")
    logger.info("Training complete!")
    logger.info("Total training took {:} (h:mm:ss)".format(format_time(time.time() - total_t0)))

    logger.info("")
    logger.info("Training stats:")
    pd.set_option('display.max_columns', None)  # Show all columns
    pd.set_option('display.width', 1000)  # Set overall width
    pd.set_option('display.max_colwidth', 50) # Set maximum column width
    pd.set_option('display.precision', 2)  # Set decimal precision
    training_stats_df = pd.DataFrame(training_stats)
    logger.info(f"\n{training_stats_df.to_string(index=False)}")


def save_training_data(config: dict, model: GPT2LMHeadModel, tokenizer: GPT2Tokenizer) -> None:
    """
    Save the training data

    Args:
    -----
    config (dict): the config file
    model (GPT2LMHeadModel): the GPT2 model
    tokenizer (GPT2Tokenizer): the GPT2 tokenizer
    """

    # NOTE
    # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()

    # Create output directory
    output_dir = config['paths']['model_path']

    # Create output directory if needed
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    elif not config['testing']['test_flag']:
        output_dir = output_dir[:-1] + '_{}'.format(str(time.time()))
        os.makedirs(output_dir)

    logger.info("Saving model to %s" % output_dir)

    # Save a trained model, configuration and tokenizer using `save_pretrained()`.
    # They can then be reloaded using `from_pretrained()`
    model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
    model_to_save.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Good practice: save your training arguments together with the trained model
    torch.save(config, os.path.join(output_dir, 'training_args.pt'))


def generate_text(model, tokenizer, device, config) -> None:
    """
    Generate text using the trained model

    Args:
    -----
    model (GPT2LMHeadModel): the trained GPT2 model
    tokenizer (GPT2Tokenizer): the GPT2 tokenizer
    device (str): the device to use
    """

    model.eval()

    # prompt = "<|startoftext|>"
    prompt = "The philosophy of Alan Watts begins with the idea that"

    # generated = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0)
    # generated = generated.to(device)

    # print(generated)

    # Tokenize the prompt and prepare the input tensor
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs['input_ids']  # The token IDs
    attention_mask = inputs['attention_mask']  # The attention mask

    # Print the generated input tensor
    logger.info(input_ids)
    logger.info(attention_mask)

    sample_outputs = model.generate(
        # generated,
        input_ids=input_ids,
        attention_mask=attention_mask,  # Include the attention mask
        # bos_token_id=random.randint(1,30000),
        do_sample=config['generation']['do_sample'],
        temperature=config['generation']['temperature'],
        top_k=config['generation']['top_k'],
        max_length=config['generation']['max_length'],
        top_p=config['generation']['top_p'],
        num_return_sequences=config['generation']['num_return_sequences'],
        pad_token_id=tokenizer.eos_token_id  # Set pad token ID explicitly
    )

    for i, sample_output in enumerate(sample_outputs):
        logger.info("{}: {}\n\n".format(i, tokenizer.decode(sample_output, skip_special_tokens=True)))

    # token = tokenizer.decode([36])
    # print(f"Token ID 36 corresponds to: {token}")

  

if __name__ == '__main__':

    # load the config file
    config = load_config()

    # add logging
    logger.remove()
    if config['testing']['test_flag']:
        logger.add(sys.stdout, level=config['testing']['logger_level'])
    else:
        logger.add(sys.stdout, level=config['training']['logger_level'])
        log_path = config['paths']['log_path'] + f"training_{str(time.time())}.log"
        logger.add(log_path, level=config['training']['logger_level'])

    # load the data
    logger.info("Loading data...")
    training_text = load_training_text(config=config)

    # Load the GPT tokenizer
    logger.info("Loading GPT2 tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained(
        pretrained_model_name_or_path='gpt2', #gpt2-medium
        bos_token='<|startoftext|>', 
        eos_token='<|endoftext|>', 
        pad_token='<|pad|>', 
        clean_up_tokenization_spaces=True
    ) 

    # create dataloaders
    logger.info("Creating dataloaders...")
    train_dataloader, validation_dataloader = create_dataloaders(
        training_text=training_text, 
        tokenizer=tokenizer, 
        batch_size=config['training']['batch_size'], 
        train_percent=config['training']['train_percent'], 
        max_length=config['tokenizer']['max_length']
    )

    # create model
    logger.info("Creating model...")
    model, device = create_model(tokenizer=tokenizer)

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
    model, training_stats, total_t0 = train(
        model=model, 
        device=device, 
        train_dataloader=train_dataloader, 
        validation_dataloader=validation_dataloader, 
        optimizer=optimizer, 
        scheduler=scheduler, 
        config=config
    )

    # Training reporting
    training_reporting(total_t0=total_t0, training_stats=training_stats)

    # save training data
    save_training_data(config=config, model=model, tokenizer=tokenizer)

    # Generate Text
    logger.info("Generating text...")
    generate_text(
        model=model, 
        tokenizer=tokenizer, 
        device=device, 
        config=config
    )