#!/bin/bash

# Create data directory if it doesn't exist
mkdir -p data

# Generate synthetic data for the application
echo "Generating synthetic data..."
python data_generator.py

echo "Setup completed successfully!"