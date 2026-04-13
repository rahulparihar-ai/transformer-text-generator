# Transformer Text Generator

A complete, hackathon-ready text generation project built from scratch using a custom Transformer architecture.

## Description
This project implements a next-token prediction model using a proprietary Transformer built in PyTorch. The model has been trained on a text corpus and is capable of generating fluent, human-readable sentences based on a given prompt. The project includes a complete pipeline from training scripts to a fully working interactive web interface using Streamlit, making it perfectly ready for demonstration.

## Features
- **Transformer from scratch**: Custom PyTorch implementation featuring multi-head self-attention and positional encoding.
- **Text generation**: Autoregressive sequence generation loop ensuring context retention.
- **Top-k + temperature sampling**: Advanced sampling techniques implemented to allow diversity and control over text generation. Repetition penalties are included to avoid arbitrary loops.
- **Web app (Streamlit)**: Clean, user-friendly UI to interact with the trained model in real-time.

## Demo
**Input**: `to be or not to`  
**Output**: `to be or not to be a great leader` *(example output)*

## How to run
1. Install dependencies:
```bash
pip install -r requirements.txt
