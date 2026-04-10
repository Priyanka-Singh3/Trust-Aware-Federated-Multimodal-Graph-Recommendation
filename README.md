# Trust-Aware Federated Multimodal Graph Recommendation System

A BTP-level implementation of a federated learning system with trust mechanisms for multimodal recommendation.

## Architecture

### Layer 1: Client Side (User Devices)
- Private data storage (reviews, images, interactions)
- Local training and modality encoders

### Layer 2: Federated Server
- Trust-weighted aggregation
- Model update collection and distribution

### Layer 3: Graph + Recommendation Engine
- User-Item graph construction
- GNN-based recommendations
- Trust score computation

## Tech Stack
- Python
- PyTorch
- PyTorch Geometric
- Flower (Federated Learning)
- FastAPI (Optional UI)

## Installation
```bash
pip install -r requirements.txt
```

## Usage
```bash
# Start server
python server.py

# Start clients (multiple terminals)
python client.py --client-id 1
python client.py --client-id 2
```

## Project Structure
```
├── data/
│   ├── processed/
│   └── raw/
├── models/
│   ├── encoders/
│   ├── gnn/
│   └── trust/
├── server/
├── client/
├── utils/
└── notebooks/
```
