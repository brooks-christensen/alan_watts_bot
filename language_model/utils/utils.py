from datetime import datetime, timedelta
from typing import Dict, Any, List, Union
import random
import numpy as np
import torch
from loguru import logger
import pandas as pd
import os
import time
import json
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from pathlib import Path
import requests
import zipfile

from language_model.config.config import load_config


config = load_config()


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

    return str(timedelta(seconds=int(round((elapsed)))))


def set_seeds(seed_val: Union[int, float] = config['general']['seed_val']) -> None:
    """
    Set the random seeds for reproducibility

    Args:
    -----
    config (Dict[str, Any]): the config file
    """

    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)


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
    pd.set_option('display.precision', 5)  # Set decimal precision
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

    # Construct output directory path
    output_dir = Path(config['paths']['model_save_path']).resolve()

    current_time = datetime.now().strftime("%m-%d-%YT%H-%M-%S")
    if config['testing']['test_flag']:
        output_dir = output_dir / 'test' / f"training_{current_time}.log"
    else:
        output_dir = output_dir / current_time

    # Create output directory, if needed
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger.info("Saving model to %s" % output_dir)

    # Save a trained model, configuration and tokenizer using `save_pretrained()`.
    # They can then be reloaded using `from_pretrained()`
    model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
    model_to_save.save_pretrained(output_dir)
    logger.info(f'Saved model to : {output_dir}')
    tokenizer.save_pretrained(output_dir)
    logger.info(f'Tokenizer saved to: {output_dir}')

    # Good practice: save your training arguments together with the trained model
    # torch.save(config, os.path.join(output_dir, 'training_args.pt'))
    # save configurations as json file
    logger.info(f'Saving configurations to {Path(output_dir)}')
    with open(os.path.join(output_dir, 'training_args.json'), 'w') as f:
        json.dump(config, f)


def download_and_extract_model() -> None:
    if os.path.exists(config['paths']['model_dir']):
        logger.info("Model directory already exists.")
        return

    os.makedirs(config['paths']['model_dir'], exist_ok=True)
    zip_path = os.path.join(config['paths']['model_dir'], "model.zip")
    
    logger.info("Downloading model archive...")
    response = requests.get(config['paths']['model_zip_url'], stream=True)
    if response.status_code == 200:
        with open(zip_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        logger.success("Download complete.")
    else:
        raise Exception(f"Failed to download model archive. HTTP status: {response.status_code}")

    logger.info("Extracting model archive...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(config['paths']['model_dir'])
    os.remove(zip_path)
    print("Extraction complete. Model is ready for use.")