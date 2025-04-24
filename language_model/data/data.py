import pandas as pd
from typing import Dict, Any, List
from transformers import GPT2Tokenizer
from loguru import logger
from torch.utils.data import Dataset, DataLoader, SequentialSampler
import torch

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
    max_length: int = 768,
    num_workers: int = 4
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
        batch_size = batch_size, # Trains with this batch size.
        num_workers=num_workers
    )

    # For validation the order doesn't matter, so we'll just read them sequentially.
    validation_dataloader = DataLoader(
        val_dataset, # The validation samples.
        sampler = SequentialSampler(val_dataset), # Pull out batches sequentially.
        batch_size = batch_size, # Evaluate with this batch size.
        num_workers=num_workers
    )

    return train_dataloader, validation_dataloader