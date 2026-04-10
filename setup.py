#!/usr/bin/env python3
"""
Setup script for Trust-Aware Federated Multimodal Graph Recommendation System
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Error: Python 3.8 or higher is required")
        sys.exit(1)
    print(f"✅ Python version: {sys.version}")

def install_dependencies():
    """Install required dependencies"""
    print("📦 Installing dependencies...")
    
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ])
        print("✅ Dependencies installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing dependencies: {e}")
        sys.exit(1)

def create_directories():
    """Create necessary directories"""
    print("📁 Creating directories...")
    
    directories = [
        "data/raw",
        "data/processed", 
        "models/encoders",
        "models/gnn",
        "models/trust",
        "server",
        "client",
        "utils",
        "notebooks",
        "templates",
        "static/css",
        "static/js"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    print("✅ Directories created")

def test_imports():
    """Test if all modules can be imported"""
    print("🧪 Testing imports...")
    
    test_modules = [
        "torch",
        "torch_geometric", 
        "flwr",
        "numpy",
        "pandas",
        "sklearn",
        "matplotlib",
        "seaborn",
        "PIL",
        "fastapi",
        "uvicorn"
    ]
    
    failed_imports = []
    
    for module in test_modules:
        try:
            __import__(module)
            print(f"  ✅ {module}")
        except ImportError as e:
            print(f"  ❌ {module}: {e}")
            failed_imports.append(module)
    
    if failed_imports:
        print(f"\n❌ Failed to import: {', '.join(failed_imports)}")
        print("Please install the missing dependencies manually")
        return False
    
    print("✅ All imports successful")
    return True

def run_dataset_preparation():
    """Prepare the dataset"""
    print("📊 Preparing dataset...")
    
    try:
        from data.dataset_preparation import DatasetPreparation
        
        dataset_prep = DatasetPreparation()
        client_data, metadata = dataset_prep.prepare_full_dataset(
            num_users=100,
            num_items=50, 
            num_interactions=500,
            num_clients=5
        )
        
        print("✅ Dataset prepared successfully")
        return True
    except Exception as e:
        print(f"❌ Error preparing dataset: {e}")
        return False

def run_model_tests():
    """Test model components"""
    print("🔧 Testing model components...")
    
    try:
        from models.encoders.multimodal_encoders import test_encoders
        from models.gnn.graph_models import test_gnn_models
        from models.trust.trust_mechanism import test_trust_mechanism
        from utils.recommendation_system import test_recommendation_system
        
        print("  Testing encoders...")
        test_encoders()
        print("  ✅ Encoders working")
        
        print("  Testing GNN models...")
        test_gnn_models()
        print("  ✅ GNN models working")
        
        print("  Testing trust mechanism...")
        test_trust_mechanism()
        print("  ✅ Trust mechanism working")
        
        print("  Testing recommendation system...")
        test_recommendation_system()
        print("  ✅ Recommendation system working")
        
        print("✅ All model tests passed")
        return True
    except Exception as e:
        print(f"❌ Error in model tests: {e}")
        return False

def create_quickstart_script():
    """Create a quickstart script"""
    script_content = """#!/bin/bash
# Quickstart script for Trust-Aware Federated Recommendation System

echo "🚀 Starting Trust-Aware Federated Recommendation System"
echo "=================================================="

# Step 1: Prepare dataset
echo "📊 Step 1: Preparing dataset..."
python main.py prepare --num-users 100 --num-items 50 --num-interactions 500 --num-clients 5

# Step 2: Test models
echo "🧪 Step 2: Testing models..."
python main.py test

# Step 3: Start server (in background)
echo "🖥️ Step 3: Starting server..."
python main.py server --server-address localhost:8080 --num-rounds 5 &
SERVER_PID=$!

# Wait for server to start
sleep 3

# Step 4: Start clients
echo "👥 Step 4: Starting clients..."
for i in {0..4}; do
    python main.py client --client-id $i --server-address localhost:8080 &
done

# Wait for all processes to complete
wait

echo "✅ Quickstart completed!"
"""
    
    with open("quickstart.sh", "w") as f:
        f.write(script_content)
    
    # Make script executable
    os.chmod("quickstart.sh", 0o755)
    print("✅ Quickstart script created")

def main():
    """Main setup function"""
    parser = argparse.ArgumentParser(description="Setup Trust-Aware Federated Recommendation System")
    parser.add_argument("--skip-deps", action="store_true", help="Skip dependency installation")
    parser.add_argument("--skip-tests", action="store_true", help="Skip model tests")
    parser.add_argument("--prepare-data", action="store_true", help="Prepare dataset")
    
    args = parser.parse_args()
    
    print("🔧 Setting up Trust-Aware Federated Multimodal Graph Recommendation System")
    print("=" * 70)
    
    # Check Python version
    check_python_version()
    
    # Create directories
    create_directories()
    
    # Install dependencies
    if not args.skip_deps:
        install_dependencies()
    
    # Test imports
    if not test_imports():
        print("❌ Setup failed due to import errors")
        sys.exit(1)
    
    # Run model tests
    if not args.skip_tests:
        if not run_model_tests():
            print("❌ Setup failed due to model test errors")
            sys.exit(1)
    
    # Prepare dataset
    if args.prepare_data:
        if not run_dataset_preparation():
            print("❌ Setup failed due to dataset preparation errors")
            sys.exit(1)
    
    # Create quickstart script
    create_quickstart_script()
    
    print("\n🎉 Setup completed successfully!")
    print("\nNext steps:")
    print("1. Run './quickstart.sh' for a quick demo")
    print("2. Or use 'python main.py --help' to see available commands")
    print("3. Open 'notebooks/demo.ipynb' for interactive exploration")
    print("4. Run 'python app.py' to start the web UI")

if __name__ == "__main__":
    main()
