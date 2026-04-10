#!/usr/bin/env python3
"""
Virtual Environment Setup Script for Trust-Aware Federated Recommendation System
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def create_virtual_environment(venv_name="venv"):
    """Create a virtual environment"""
    print(f"Creating virtual environment: {venv_name}")
    
    try:
        # Create virtual environment
        subprocess.check_call([sys.executable, "-m", "venv", venv_name])
        print(f"Virtual environment '{venv_name}' created successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error creating virtual environment: {e}")
        return False

def get_venv_python(venv_name="venv"):
    """Get the python executable path in the virtual environment"""
    if sys.platform == "win32":
        return os.path.join(venv_name, "Scripts", "python.exe")
    else:
        return os.path.join(venv_name, "bin", "python")

def get_venv_pip(venv_name="venv"):
    """Get the pip executable path in the virtual environment"""
    if sys.platform == "win32":
        return os.path.join(venv_name, "Scripts", "pip")
    else:
        return os.path.join(venv_name, "bin", "pip")

def install_dependencies_in_venv(venv_name="venv"):
    """Install dependencies in the virtual environment"""
    print(f"Installing dependencies in virtual environment: {venv_name}")
    
    venv_pip = get_venv_pip(venv_name)
    
    # Upgrade pip first
    print("Upgrading pip...")
    try:
        subprocess.check_call([venv_pip, "install", "--upgrade", "pip"])
    except subprocess.CalledProcessError as e:
        print(f"Warning: Could not upgrade pip: {e}")
    
    # Install requirements
    print("Installing project dependencies...")
    try:
        subprocess.check_call([venv_pip, "install", "-r", "requirements.txt"])
        print("Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error installing dependencies: {e}")
        return False

def create_activation_scripts(venv_name="venv"):
    """Create activation scripts for the virtual environment"""
    
    # Create activation script for Unix/Linux/Mac
    unix_script = f"""#!/bin/bash
# Activation script for {venv_name}
echo "Activating virtual environment: {venv_name}"
source {venv_name}/bin/activate
echo "Virtual environment activated!"
echo "Python path: $(which python)"
echo "Pip path: $(which pip)"
"""
    
    with open(f"activate_{venv_name}.sh", "w") as f:
        f.write(unix_script)
    
    # Make it executable
    os.chmod(f"activate_{venv_name}.sh", 0o755)
    
    # Create activation script for Windows
    windows_script = f"""@echo off
REM Activation script for {venv_name}
echo Activating virtual environment: {venv_name}
call {venv_name}\\Scripts\\activate.bat
echo Virtual environment activated!
echo Python path: %PYTHONPATH%
echo Pip path: where pip
"""
    
    with open(f"activate_{venv_name}.bat", "w") as f:
        f.write(windows_script)
    
    print(f"Activation scripts created: activate_{venv_name}.sh and activate_{venv_name}.bat")

def create_deactivation_script(venv_name="venv"):
    """Create deactivation script"""
    
    script = """#!/bin/bash
# Deactivation script for virtual environment
echo "Deactivating virtual environment..."
deactivate
echo "Virtual environment deactivated!"
"""
    
    with open("deactivate_venv.sh", "w") as f:
        f.write(script)
    
    os.chmod("deactivate_venv.sh", 0o755)
    print("Deactivation script created: deactivate_venv.sh")

def test_virtual_environment(venv_name="venv"):
    """Test if the virtual environment works correctly"""
    print(f"Testing virtual environment: {venv_name}")
    
    venv_python = get_venv_python(venv_name)
    
    # Test Python
    try:
        result = subprocess.run([venv_python, "--version"], capture_output=True, text=True)
        print(f"Python version in venv: {result.stdout.strip()}")
    except subprocess.CalledProcessError as e:
        print(f"Error testing Python in venv: {e}")
        return False
    
    # Test key imports
    test_script = """
import sys
try:
    import torch
    print("torch: OK")
except ImportError as e:
    print(f"torch: FAILED - {e}")

try:
    import torch_geometric
    print("torch_geometric: OK")
except ImportError as e:
    print(f"torch_geometric: FAILED - {e}")

try:
    import flwr
    print("flwr: OK")
except ImportError as e:
    print(f"flwr: FAILED - {e}")

try:
    import numpy
    print("numpy: OK")
except ImportError as e:
    print(f"numpy: FAILED - {e}")

try:
    import pandas
    print("pandas: OK")
except ImportError as e:
    print(f"pandas: FAILED - {e}")

try:
    import sklearn
    print("sklearn: OK")
except ImportError as e:
    print(f"sklearn: FAILED - {e}")

try:
    import fastapi
    print("fastapi: OK")
except ImportError as e:
    print(f"fastapi: FAILED - {e}")
"""
    
    try:
        result = subprocess.run([venv_python, "-c", test_script], capture_output=True, text=True)
        print("Import test results:")
        print(result.stdout)
        if result.stderr:
            print("Errors:")
            print(result.stderr)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"Error testing imports: {e}")
        return False

def create_venv_requirements():
    """Create a requirements file specifically for virtual environment setup"""
    
    venv_requirements = """# Virtual Environment Requirements
# Core ML/DL libraries
torch>=2.0.0
torchvision>=0.15.0
torch-geometric>=2.3.0

# Federated Learning
flwr>=1.4.0

# Data Processing
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0

# Visualization
matplotlib>=3.5.0
seaborn>=0.11.0

# Utilities
tqdm>=4.62.0
Pillow>=8.3.0

# Web Interface
fastapi>=0.68.0
uvicorn>=0.15.0

# Jupyter (optional)
jupyter>=1.0.0

# Text Processing (optional, for advanced features)
transformers>=4.20.0
"""
    
    with open("requirements_venv.txt", "w") as f:
        f.write(venv_requirements)
    
    print("Virtual environment requirements file created: requirements_venv.txt")

def main():
    """Main setup function"""
    parser = argparse.ArgumentParser(description="Set up virtual environment for Trust-Aware Federated Recommendation System")
    parser.add_argument("--venv-name", default="venv", help="Name of the virtual environment (default: venv)")
    parser.add_argument("--skip-deps", action="store_true", help="Skip dependency installation")
    parser.add_argument("--skip-test", action="store_true", help="Skip testing")
    parser.add_argument("--recreate", action="store_true", help="Recreate existing virtual environment")
    
    args = parser.parse_args()
    
    print("Setting up Virtual Environment for Trust-Aware Federated Recommendation System")
    print("=" * 80)
    
    venv_name = args.venv_name
    
    # Check if virtual environment already exists
    venv_path = Path(venv_name)
    if venv_path.exists():
        if args.recreate:
            print(f"Removing existing virtual environment: {venv_name}")
            import shutil
            shutil.rmtree(venv_name)
        else:
            print(f"Virtual environment '{venv_name}' already exists")
            response = input("Do you want to recreate it? (y/N): ")
            if response.lower() != 'y':
                print("Setup cancelled")
                return
            else:
                import shutil
                shutil.rmtree(venv_name)
    
    # Create virtual environment
    if not create_virtual_environment(venv_name):
        print("Failed to create virtual environment")
        sys.exit(1)
    
    # Install dependencies
    if not args.skip_deps:
        if not install_dependencies_in_venv(venv_name):
            print("Failed to install dependencies")
            sys.exit(1)
    
    # Create activation scripts
    create_activation_scripts(venv_name)
    create_deactivation_script(venv_name)
    
    # Create venv-specific requirements
    create_venv_requirements()
    
    # Test virtual environment
    if not args.skip_test:
        if not test_virtual_environment(venv_name):
            print("Virtual environment test failed")
            sys.exit(1)
    
    print("\nVirtual environment setup completed successfully!")
    print("\nTo use the virtual environment:")
    print(f"1. Activate: source activate_{venv_name}.sh")
    print(f"2. Or manually: source {venv_name}/bin/activate")
    print("3. Deactivate: deactivate")
    print("4. Or use: ./deactivate_venv.sh")
    print(f"\nVirtual environment location: ./{venv_name}/")
    print(f"Python executable: {get_venv_python(venv_name)}")
    print(f"Pip executable: {get_venv_pip(venv_name)}")

if __name__ == "__main__":
    main()
