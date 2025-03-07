from transformers import GPT2LMHeadModel, GPT2Tokenizer
import time
from loguru import logger
from typing import List, Dict, Any, Tuple
from torch import no_grad
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
import os
from torch import load as torchload

from ..utils.utils import format_time
  

def train(
    model: GPT2LMHeadModel, 
    device: str, 
    train_dataloader: DataLoader, 
    validation_dataloader: DataLoader, 
    tokenizer: GPT2Tokenizer,
    optimizer: AdamW, 
    scheduler: LambdaLR, 
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
    logger.debug(input_ids)
    logger.debug(attention_mask)

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
                logger.info('  Batch {:>5,}  of  {:>5,}. Loss: {:>5,}.   Total Elapsed this Epoch: {:}.'.format(step, len(train_dataloader), batch_loss, elapsed))

                model.eval()

                # sample_outputs = model.generate(
                #     bos_token_id=random.randint(1,30000),
                #     do_sample=True,
                #     top_k=50,
                #     max_length = 200,
                #     top_p=0.95,
                #     num_return_sequences=1
                # )

                with no_grad():
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

            with no_grad():

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

        logger.info("  Validation Loss: {0:.5f}".format(avg_val_loss))
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
            logger.info("New best model found!")
            logger.info("Validation loss: {:.5f}".format(best_val_loss))
        else:
            patience += 1
            logger.info("No improvement in validation loss. Patience: {}/{}".format(patience, max_patience))
            if patience >= max_patience:
                logger.info("Stopping early!")
                break

    # load the best model's state
    if best_model_state_dict:
        model.load_state_dict(best_model_state_dict)

    return model, training_stats, total_t0


def resume_training_if_possible(model, optimizer, scheduler, config):
    checkpoint_path = config['paths']['checkpoint_path']
    start_epoch = 0
    if os.path.exists(checkpoint_path):
        logger.info(f"Found checkpoint at {checkpoint_path}. Resuming training...")
        checkpoint = torchload(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        logger.info(f"Resuming from epoch {start_epoch}")
    return start_epoch

# def train(model, device, train_dataloader, validation_dataloader, tokenizer,
#           optimizer, scheduler, config, start_epoch=0, max_epochs=None):
#     total_t0 = time.time()
#     training_stats = []
#     best_val_loss = float('inf')
#     max_patience = config['training']['max_patience']
#     patience = 0
#     checkpoint_path = config['paths']['checkpoint_path']

#     for epoch in range(start_epoch, config['training']['epochs'] if max_epochs is None else max_epochs):
#         logger.info(f"Epoch {epoch+1}/{config['training']['epochs']}")
#         # (Your existing training loop code goes here.)
#         # ...
        
#         # After validation, compute average validation loss:
#         avg_val_loss = total_eval_loss / len(validation_dataloader)
#         logger.info("Validation Loss: {:.5f}".format(avg_val_loss))
        
#         # Save checkpoint if this is the best model so far:
#         if avg_val_loss < best_val_loss:
#             best_val_loss = avg_val_loss
#             checkpoint = {
#                 'epoch': epoch,
#                 'model_state_dict': model.state_dict(),
#                 'optimizer_state_dict': optimizer.state_dict(),
#                 'scheduler_state_dict': scheduler.state_dict(),
#                 'best_val_loss': best_val_loss,
#             }
#             torch.save(checkpoint, checkpoint_path)
#             logger.info("Checkpoint saved.")
#             patience = 0  # Reset patience
#         else:
#             patience += 1
#             if patience >= max_patience:
#                 logger.info("Early stopping triggered.")
#                 break
        
#         # Log training statistics
#         training_stats.append({
#             'epoch': epoch + 1,
#             'Valid. Loss': avg_val_loss,
#             # other stats...
#         })
#     return model, training_stats, total_t0
