#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Update and upgrade system packages (optional, uncomment if needed)
# sudo apt update && sudo apt upgrade -y

# Install necessary packages if not installed (optional)
# sudo apt install -y python3 python3-pip

# Install Python dependencies
pip install -r requirements.txt

# Pull the llama3.1 model using Ollama
ollama pull llama3.1

# Create the custom model
ollama create custom-llama3.1:8b -f custom-llama3

echo "Setup complete!"
