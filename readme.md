# Alan Watts Chatbot

Brooks Christensen 2025

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