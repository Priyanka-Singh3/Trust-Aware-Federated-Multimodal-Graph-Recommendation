#!/bin/bash
# Activation script for venv
echo "Activating virtual environment: venv"
source venv/bin/activate
echo "Virtual environment activated!"
echo "Python path: $(which python)"
echo "Pip path: $(which pip)"
