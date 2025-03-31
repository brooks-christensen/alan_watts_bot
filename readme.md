# Alan Watts Chatbot

Brooks Christensen 2025

This package is an attempt to create the likeness of Alan Watts, the English philosopher who lived and taught on the west coast in the 1950s and 60s, for the purpose of understanding what he might have said about a particular subject.

The language model is currently monologic and will therefore only generate text. A prompt-based, dialogic model is currently under development.

## How to Generate Text

Navigate to project folder

```bash
python3 -m language_model generate
```

For reproducible output:

```bash
python3 -m language_model generate --seed 42
```

## How to Train

Navigate to project folder

```bash
python3 -m language_model train
```

Models are output to `./language_model/

## Configurations and Hyperparameters

Configurations and hyperparameters are stored in `./language_model/config/config.yaml`