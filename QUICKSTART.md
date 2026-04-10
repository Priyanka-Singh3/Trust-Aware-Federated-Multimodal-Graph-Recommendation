# Quick Start Guide

## 🚀 Trust-Aware Federated Multimodal Graph Recommendation System

This guide will help you get the system running in minutes.

## Prerequisites

- Python 3.8+
- Git
- Basic understanding of federated learning concepts

## Installation

### 1. Clone/Setup the Project
```bash
# If you have the project files, navigate to the directory
cd BTP-8

# Run the setup script
python setup.py

# Or install manually
pip install -r requirements.txt
```

### 2. Quick Demo (One Command)
```bash
# This will prepare data, test models, and run a complete simulation
./quickstart.sh
```

## Step-by-Step Usage

### Step 1: Prepare Dataset
```bash
python main.py prepare --num-users 100 --num-items 50 --num-interactions 500 --num-clients 5
```

### Step 2: Test Components
```bash
python main.py test
```

### Step 3: Run Complete Simulation
```bash
# This starts server and clients automatically
python main.py simulate --num-clients 5 --num-rounds 10 --trust-enabled
```

### Step 4: Interactive Demo
```bash
python main.py demo
```

### Step 5: Web UI
```bash
# Start the web interface
python app.py

# Then open http://localhost:8000 in your browser
```

## Manual Federated Learning

### Start Server
```bash
# Terminal 1
python main.py server --server-address localhost:8080 --num-rounds 10 --trust-enabled
```

### Start Clients
```bash
# Terminal 2-6 (one for each client)
python main.py client --client-id 0 --server-address localhost:8080
python main.py client --client-id 1 --server-address localhost:8080
python main.py client --client-id 2 --server-address localhost:8080
python main.py client --client-id 3 --server-address localhost:8080
python main.py client --client-id 4 --server-address localhost:8080
```

## Jupyter Notebook Demo

For an interactive exploration:
```bash
jupyter notebook notebooks/demo.ipynb
```

## Key Features Demonstrated

1. **Multi-modal Encoders**: Text (TF-IDF) + Image (ResNet18)
2. **Graph Neural Networks**: Bipartite user-item graphs
3. **Trust Mechanism**: Consistency, performance, and similarity scoring
4. **Federated Learning**: Privacy-preserving distributed training
5. **Trust-weighted Aggregation**: Advanced federated averaging
6. **Recommendation System**: Complete end-to-end pipeline
7. **Web UI**: Interactive demonstration interface

## Project Structure

```
BTP-8/
├── data/                   # Dataset and preprocessing
├── models/                 # Neural network models
│   ├── encoders/          # Multi-modal encoders
│   ├── gnn/               # Graph neural networks
│   └── trust/             # Trust mechanisms
├── server/                # Federated learning server
├── client/                # Federated learning clients
├── utils/                 # Recommendation system
├── notebooks/             # Jupyter demos
├── templates/             # Web UI templates
├── static/                # Web UI assets
├── main.py               # Main CLI interface
├── app.py                # Web UI server
└── setup.py              # Setup script
```

## Configuration Options

### Main CLI Commands
- `prepare`: Create sample dataset
- `server`: Start federated server
- `client`: Start federated client
- `simulate`: Run complete simulation
- `test`: Test all components
- `demo`: Interactive command-line demo

### Key Parameters
- `--num-users`: Number of users (default: 100)
- `--num-items`: Number of items (default: 50)
- `--num-clients`: Number of federated clients (default: 5)
- `--num-rounds`: Federated learning rounds (default: 10)
- `--trust-enabled`: Enable trust mechanism (default: True)

## Troubleshooting

### Common Issues

1. **Import Errors**: Make sure all dependencies are installed
   ```bash
   pip install -r requirements.txt
   ```

2. **Port Already in Use**: Change server address
   ```bash
   python main.py server --server-address localhost:8081
   ```

3. **CUDA Errors**: System will fall back to CPU automatically

4. **Memory Issues**: Reduce dataset size
   ```bash
   python main.py prepare --num-users 50 --num-items 25 --num-interactions 200
   ```

### Performance Tips

- Use GPU for faster training (if available)
- Reduce `num-interactions` for quicker testing
- Increase `num-rounds` for better convergence
- Adjust trust weights in `models/trust/trust_mechanism.py`

## Next Steps for Research

1. **Real Dataset Integration**: Replace sample data with Amazon Reviews
2. **Advanced Encoders**: Use BERT for text, better CNNs for images
3. **Sophisticated GNNs**: Implement attention-based graph networks
4. **Advanced Trust**: Add adversarial robustness and explainability
5. **Production Deployment**: Docker containers, cloud deployment

## Academic Citation

If you use this system in your research, please cite:

```
Trust-Aware Federated Multimodal Graph Recommendation System
BTP Project Implementation
[Your Name], [Your Institution], [Year]
```

## Support

For issues and questions:
1. Check the troubleshooting section above
2. Review the code comments and documentation
3. Test individual components using `python main.py test`
4. Explore the interactive notebook demo

Happy researching! 🎓
