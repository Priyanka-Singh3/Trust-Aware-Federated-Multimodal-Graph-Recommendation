# Virtual Environment Setup Guide

## Overview

This guide helps you set up a clean virtual environment for the Trust-Aware Federated Multimodal Graph Recommendation System.

## Quick Setup

### Option 1: Automated Setup (Recommended)
```bash
# Run the automated setup script
python setup_venv.py

# Activate the environment
source activate_venv.sh
```

### Option 2: Manual Setup
```bash
# Create virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate

# Install dependencies
pip install -r requirements_venv.txt
```

## Detailed Instructions

### Step 1: Create Virtual Environment

```bash
# Using Python 3
python3 -m venv venv

# Or using specific Python version
python3.9 -m venv venv
```

### Step 2: Activate Virtual Environment

**On Linux/Mac:**
```bash
source venv/bin/activate
```

**On Windows:**
```bash
venv\Scripts\activate
```

**Using our scripts:**
```bash
# Activate
source activate_venv.sh

# Deactivate
./deactivate_venv.sh
```

### Step 3: Install Dependencies

```bash
# Install all required packages
pip install -r requirements_venv.txt

# Or install from original requirements
pip install -r requirements.txt
```

### Step 4: Verify Installation

```bash
# Test key imports
python -c "
import torch
import torch_geometric
import flwr
import numpy
import pandas
print('All dependencies installed successfully!')
"
```

## Virtual Environment Scripts

We've created several helper scripts:

### `setup_venv.py`
- Automated virtual environment setup
- Installs all dependencies
- Creates activation scripts
- Tests the installation

### `activate_venv.sh`
- Quick activation script
- Shows environment info

### `deactivate_venv.sh`
- Quick deactivation script

### `requirements_venv.txt`
- Optimized requirements list for virtual environment

## Usage Examples

### Setup and Run
```bash
# 1. Setup virtual environment
python setup_venv.py

# 2. Activate
source activate_venv.sh

# 3. Run the system
python main.py prepare
python main.py test
python main.py demo

# 4. Start web UI
python app.py

# 5. Deactivate when done
./deactivate_venv.sh
```

### Development Workflow
```bash
# Activate environment for development
source venv/bin/activate

# Install additional packages if needed
pip install some-new-package

# Update requirements
pip freeze > requirements_updated.txt

# Run your code
python main.py simulate

# Deactivate when done
deactivate
```

## Troubleshooting

### Common Issues

1. **Python not found**:
   ```bash
   # Use python3 instead of python
   python3 -m venv venv
   ```

2. **Permission denied**:
   ```bash
   # Make scripts executable
   chmod +x activate_venv.sh
   chmod +x deactivate_venv.sh
   ```

3. **Dependencies fail to install**:
   ```bash
   # Upgrade pip first
   pip install --upgrade pip
   
   # Install one by one to identify the issue
   pip install torch
   pip install torch-geometric
   pip install flwr
   ```

4. **CUDA issues**:
   ```bash
   # Install CPU version if GPU not available
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
   ```

5. **Virtual environment location**:
   ```bash
   # Check if venv exists
   ls -la venv/
   
   # Check Python location
   which python
   # Should show: ./venv/bin/python
   ```

### Recreate Environment
```bash
# Remove existing environment
rm -rf venv

# Recreate
python setup_venv.py --recreate
```

## Best Practices

1. **Always use virtual environment** for project dependencies
2. **Keep requirements.txt updated** when adding new packages
3. **Don't commit venv/ folder** to version control
4. **Use specific Python version** for reproducibility
5. **Test imports** after installation

## Environment Information

Once activated, you can check:

```bash
# Python version
python --version

# Package versions
pip list

# Package locations
pip show torch
```

## IDE Integration

### VS Code
1. Open the project folder
2. Press `Ctrl+Shift+P` (or `Cmd+Shift+P` on Mac)
3. Type "Python: Select Interpreter"
4. Choose `./venv/bin/python`

### PyCharm
1. File > Settings > Project > Python Interpreter
2. Click gear icon > Add
3. Select "Existing environment"
4. Choose `./venv/bin/python`

## Project Structure with Virtual Environment

```
BTP-8/
|
# Virtual environment (don't commit to git)
|--- venv/
|    |--- bin/
|    |--- lib/
|    |--- include/
|
# Project files
|--- data/
|--- models/
|--- server/
|--- client/
|--- utils/
|--- notebooks/
|--- main.py
|--- app.py
|--- setup_venv.py
|--- requirements_venv.txt
|--- activate_venv.sh
|--- deactivate_venv.sh
```

## Next Steps

After setting up the virtual environment:

1. **Test the system**: `python main.py test`
2. **Prepare data**: `python main.py prepare`
3. **Run simulation**: `python main.py simulate`
4. **Start web UI**: `python app.py`
5. **Explore notebook**: `jupyter notebook notebooks/demo.ipynb`

Happy coding!
