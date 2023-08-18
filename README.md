# simpleGPT GPT-2 Chatbot

This repo contains code to train and interact with a simple GPT-2 based chatbot model.
Model Details

    GPT-2 style transformer model architecture
    10 encoder layers
    300 dimensional embeddings
    Trained on generic conversational text data
    256 max sequence length

Usage

train.py - Trains the model on the conversational data

interact.py - Generates chat responses interactively
Training

    CrossEntropyLoss optimized with Adam
    Learning rate of 1e-4
    Batch size of 4
    200 epochs

Examples
Human: Hello!
AI: Hi there! How can I help you today?
