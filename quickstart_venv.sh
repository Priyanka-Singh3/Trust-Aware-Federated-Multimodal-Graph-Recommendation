#!/bin/bash
# Quickstart script with Virtual Environment for Trust-Aware Federated Recommendation System

echo "Setting up Virtual Environment and Starting Trust-Aware Federated Recommendation System"
echo "=========================================================================================="

# Step 1: Setup virtual environment
echo "Step 1: Setting up virtual environment..."
python setup_venv.py

# Check if setup was successful
if [ $? -ne 0 ]; then
    echo "Virtual environment setup failed. Please check the errors above."
    exit 1
fi

# Step 2: Activate virtual environment
echo "Step 2: Activating virtual environment..."
source activate_venv.sh

# Step 3: Prepare dataset
echo "Step 3: Preparing dataset..."
python main.py prepare --num-users 100 --num-items 50 --num-interactions 500 --num-clients 5

# Step 4: Test models
echo "Step 4: Testing models..."
python main.py test

# Step 5: Start server (in background)
echo "Step 5: Starting server..."
python main.py server --server-address localhost:8080 --num-rounds 5 &
SERVER_PID=$!

# Wait for server to start
sleep 3

# Step 6: Start clients
echo "Step 6: Starting clients..."
for i in {0..4}; do
    python main.py client --client-id $i --server-address localhost:8080 &
done

# Wait for all processes to complete
wait

echo "Virtual environment setup and quickstart completed!"
echo "To continue using the virtual environment, run: source activate_venv.sh"
