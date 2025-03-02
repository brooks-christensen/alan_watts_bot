from loguru import logger

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