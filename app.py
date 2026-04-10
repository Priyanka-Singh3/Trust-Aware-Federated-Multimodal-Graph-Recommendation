#!/usr/bin/env python3
"""
Simple FastAPI UI for the Trust-Aware Federated Multimodal Graph Recommendation System
"""

import fastapi
from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import List, Dict, Optional
import uvicorn
import logging
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from utils.recommendation_system import RecommendationSystem, RecommendationAPI
from models.encoders.multimodal_encoders import RecommendationEncoder
from models.gnn.graph_models import BipartiteGraphRecommender
from models.trust.trust_mechanism import TrustMechanism

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Trust-Aware Federated Recommendation System",
    description="A BTP-level implementation of trust-aware federated learning for multimodal recommendations",
    version="1.0.0"
)

# Setup templates and static files
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Pydantic models for API
class RecommendationRequest(BaseModel):
    user_id: int
    top_k: int = 10
    trust_aware: bool = True

class SimilarItemsRequest(BaseModel):
    item_id: int
    top_k: int = 5

class UserInteractionRequest(BaseModel):
    user_id: int
    item_id: int
    rating: float
    review_text: str = ""
    timestamp: int = 0

class RecommendationResponse(BaseModel):
    success: bool
    user_id: int
    recommendations: List[Dict]
    num_recommendations: int
    error: Optional[str] = None

class SimilarItemsResponse(BaseModel):
    success: bool
    item_id: int
    similar_items: List[Dict]
    num_similar_items: int
    error: Optional[str] = None

class InteractionResponse(BaseModel):
    success: bool
    message: Optional[str] = None
    error: Optional[str] = None

# Global variables
recommendation_system: Optional[RecommendationSystem] = None
api: Optional[RecommendationAPI] = None

def initialize_system():
    """Initialize the recommendation system"""
    global recommendation_system, api
    
    try:
        # Try to load existing models and data
        import torch
        
        # Load metadata if available
        metadata_path = "data/processed/metadata.pt"
        if os.path.exists(metadata_path):
            metadata = torch.load(metadata_path, weights_only=False)
            
            # Create models
            encoder = RecommendationEncoder(
                metadata['num_users'],
                metadata['num_items'],
                metadata['text_feature_dim']
            )
            gnn = BipartiteGraphRecommender(
                metadata['num_users'],
                metadata['num_items']
            )
            
            # Create trust mechanism
            trust_mechanism = TrustMechanism()
            
            # Create recommendation system
            recommendation_system = RecommendationSystem(
                encoder, gnn, trust_mechanism,
                metadata['num_users'],
                metadata['num_items']
            )
            
            # Create API
            api = RecommendationAPI(recommendation_system)
            
            # Set some dummy item metadata
            item_metadata = {
                i: {
                    'name': f'Product {i}',
                    'category': f'Category {i % 5}',
                    'price': f'${10.0 + i * 2.5:.2f}',
                    'description': f'This is product {i} from category {i % 5}'
                } for i in range(metadata['num_items'])
            }
            recommendation_system.set_item_metadata(item_metadata)
            
            logger.info("Recommendation system initialized successfully")
        else:
            logger.warning("No metadata found. Using dummy system.")
            create_dummy_system()
            
    except Exception as e:
        logger.error(f"Failed to initialize system: {e}")
        create_dummy_system()

def create_dummy_system():
    """Create a dummy recommendation system for demo purposes"""
    global recommendation_system, api
    
    # Create dummy models
    encoder = RecommendationEncoder(100, 50, 1000)
    gnn = BipartiteGraphRecommender(100, 50)
    trust_mechanism = TrustMechanism()
    
    recommendation_system = RecommendationSystem(encoder, gnn, trust_mechanism, 100, 50)
    api = RecommendationAPI(recommendation_system)
    
    # Set dummy item metadata
    item_metadata = {
        i: {
            'name': f'Product {i}',
            'category': f'Category {i % 5}',
            'price': f'${10.0 + i * 2.5:.2f}',
            'description': f'This is product {i} from category {i % 5}'
        } for i in range(50)
    }
    recommendation_system.set_item_metadata(item_metadata)
    
    logger.info("Dummy recommendation system created")

# Initialize system on startup
@app.on_event("startup")
async def startup_event():
    initialize_system()

# API Routes
@app.get("/", response_class=fastapi.responses.HTMLResponse)
async def home(request: Request):
    """Home page"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/api/recommendations", response_model=RecommendationResponse)
async def get_recommendations(request: RecommendationRequest):
    """Get recommendations for a user"""
    if api is None:
        raise HTTPException(status_code=500, detail="Recommendation system not initialized")
    
    result = api.get_recommendations(request.user_id, request.top_k, request.trust_aware)
    
    if result['success']:
        return RecommendationResponse(**result)
    else:
        raise HTTPException(status_code=400, detail=result.get('error', 'Unknown error'))

@app.post("/api/similar-items", response_model=SimilarItemsResponse)
async def get_similar_items(request: SimilarItemsRequest):
    """Get similar items"""
    if api is None:
        raise HTTPException(status_code=500, detail="Recommendation system not initialized")
    
    result = api.get_similar_items(request.item_id, request.top_k)
    
    if result['success']:
        return SimilarItemsResponse(**result)
    else:
        raise HTTPException(status_code=400, detail=result.get('error', 'Unknown error'))

@app.post("/api/interaction", response_model=InteractionResponse)
async def update_interaction(request: UserInteractionRequest):
    """Update user interaction"""
    if api is None:
        raise HTTPException(status_code=500, detail="Recommendation system not initialized")
    
    result = api.update_user_interaction(
        request.user_id, request.item_id, request.rating, 
        request.review_text, request.timestamp
    )
    
    if result['success']:
        return InteractionResponse(**result)
    else:
        raise HTTPException(status_code=400, detail=result.get('error', 'Unknown error'))

@app.get("/api/system-info")
async def get_system_info():
    """Get system information"""
    if recommendation_system is None:
        raise HTTPException(status_code=500, detail="Recommendation system not initialized")
    
    return {
        "num_users": recommendation_system.num_users,
        "num_items": recommendation_system.num_items,
        "trust_enabled": recommendation_system.trust_mechanism is not None,
        "status": "active"
    }

# Create directories for templates and static files
def create_web_files():
    """Create web UI files"""
    
    # Create directories
    os.makedirs("templates", exist_ok=True)
    os.makedirs("static/css", exist_ok=True)
    os.makedirs("static/js", exist_ok=True)
    
    # Create HTML template
    html_template = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Trust-Aware Federated Recommendation System</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="/static/css/style.css" rel="stylesheet">
</head>
<body>
    <div class="container mt-4">
        <header class="text-center mb-5">
            <h1 class="display-4">Trust-Aware Federated Recommendation System</h1>
            <p class="lead">BTP Project - Multimodal Graph Recommendations with Trust Mechanisms</p>
        </header>

        <div class="row">
            <div class="col-md-6">
                <div class="card">
                    <div class="card-header">
                        <h3>Get Recommendations</h3>
                    </div>
                    <div class="card-body">
                        <form id="recommendForm">
                            <div class="mb-3">
                                <label for="userId" class="form-label">User ID</label>
                                <input type="number" class="form-control" id="userId" value="0" min="0">
                            </div>
                            <div class="mb-3">
                                <label for="topK" class="form-label">Number of Recommendations</label>
                                <input type="number" class="form-control" id="topK" value="5" min="1" max="20">
                            </div>
                            <div class="mb-3 form-check">
                                <input type="checkbox" class="form-check-input" id="trustAware" checked>
                                <label class="form-check-label" for="trustAware">Trust-Aware</label>
                            </div>
                            <button type="submit" class="btn btn-primary">Get Recommendations</button>
                        </form>
                    </div>
                </div>
            </div>

            <div class="col-md-6">
                <div class="card">
                    <div class="card-header">
                        <h3>Find Similar Items</h3>
                    </div>
                    <div class="card-body">
                        <form id="similarForm">
                            <div class="mb-3">
                                <label for="itemId" class="form-label">Item ID</label>
                                <input type="number" class="form-control" id="itemId" value="0" min="0">
                            </div>
                            <div class="mb-3">
                                <label for="similarTopK" class="form-label">Number of Similar Items</label>
                                <input type="number" class="form-control" id="similarTopK" value="5" min="1" max="20">
                            </div>
                            <button type="submit" class="btn btn-secondary">Find Similar Items</button>
                        </form>
                    </div>
                </div>
            </div>
        </div>

        <div class="row mt-4">
            <div class="col-12">
                <div class="card">
                    <div class="card-header">
                        <h3>Update User Interaction</h3>
                    </div>
                    <div class="card-body">
                        <form id="interactionForm">
                            <div class="row">
                                <div class="col-md-3">
                                    <div class="mb-3">
                                        <label for="intUserId" class="form-label">User ID</label>
                                        <input type="number" class="form-control" id="intUserId" value="0" min="0">
                                    </div>
                                </div>
                                <div class="col-md-3">
                                    <div class="mb-3">
                                        <label for="intItemId" class="form-label">Item ID</label>
                                        <input type="number" class="form-control" id="intItemId" value="0" min="0">
                                    </div>
                                </div>
                                <div class="col-md-3">
                                    <div class="mb-3">
                                        <label for="rating" class="form-label">Rating (1-5)</label>
                                        <input type="number" class="form-control" id="rating" value="5" min="1" max="5" step="0.1">
                                    </div>
                                </div>
                                <div class="col-md-3">
                                    <div class="mb-3">
                                        <label for="reviewText" class="form-label">Review</label>
                                        <input type="text" class="form-control" id="reviewText" placeholder="Optional review">
                                    </div>
                                </div>
                            </div>
                            <button type="submit" class="btn btn-success">Update Interaction</button>
                        </form>
                    </div>
                </div>
            </div>
        </div>

        <div class="row mt-4">
            <div class="col-12">
                <div id="results" class="d-none">
                    <div class="card">
                        <div class="card-header">
                            <h3>Results</h3>
                        </div>
                        <div class="card-body">
                            <div id="resultsContent"></div>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <div class="row mt-4">
            <div class="col-12">
                <div class="card">
                    <div class="card-header">
                        <h3>System Information</h3>
                    </div>
                    <div class="card-body">
                        <div id="systemInfo">Loading...</div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
    <script src="/static/js/app.js"></script>
</body>
</html>"""
    
    with open("templates/index.html", "w") as f:
        f.write(html_template)
    
    # Create CSS file
    css_content = """body {
    background-color: #f8f9fa;
}

.card {
    box-shadow: 0 0.125rem 0.25rem rgba(0, 0, 0, 0.075);
    border: 1px solid rgba(0, 0, 0, 0.125);
}

.card-header {
    background-color: #007bff;
    color: white;
    border-bottom: 1px solid rgba(0, 0, 0, 0.125);
}

.btn-primary {
    background-color: #007bff;
    border-color: #007bff;
}

.btn-primary:hover {
    background-color: #0056b3;
    border-color: #0056b3;
}

.trust-badge {
    font-size: 0.8em;
}

.trust-high {
    background-color: #28a745;
}

.trust-medium {
    background-color: #ffc107;
}

.trust-low {
    background-color: #dc3545;
}

.recommendation-item {
    border-left: 4px solid #007bff;
    padding-left: 15px;
    margin-bottom: 15px;
}

.similarity-item {
    border-left: 4px solid #6c757d;
    padding-left: 15px;
    margin-bottom: 10px;
}

.alert {
    margin-top: 15px;
}"""
    
    with open("static/css/style.css", "w") as f:
        f.write(css_content)
    
    # Create JavaScript file
    js_content = """// API Base URL
const API_BASE = '/api';

// Load system information on page load
document.addEventListener('DOMContentLoaded', function() {
    loadSystemInfo();
});

// Load system information
async function loadSystemInfo() {
    try {
        const response = await fetch(`${API_BASE}/system-info`);
        const data = await response.json();
        
        document.getElementById('systemInfo').innerHTML = `
            <div class="row">
                <div class="col-md-3">
                    <strong>Number of Users:</strong> ${data.num_users}
                </div>
                <div class="col-md-3">
                    <strong>Number of Items:</strong> ${data.num_items}
                </div>
                <div class="col-md-3">
                    <strong>Trust Enabled:</strong> ${data.trust_enabled ? 'Yes' : 'No'}
                </div>
                <div class="col-md-3">
                    <strong>Status:</strong> <span class="badge bg-success">${data.status}</span>
                </div>
            </div>
        `;
    } catch (error) {
        document.getElementById('systemInfo').innerHTML = 
            `<div class="alert alert-danger">Failed to load system information: ${error.message}</div>`;
    }
}

// Handle recommendation form submission
document.getElementById('recommendForm').addEventListener('submit', async function(e) {
    e.preventDefault();
    
    const userId = parseInt(document.getElementById('userId').value);
    const topK = parseInt(document.getElementById('topK').value);
    const trustAware = document.getElementById('trustAware').checked;
    
    try {
        const response = await fetch(`${API_BASE}/recommendations`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                user_id: userId,
                top_k: topK,
                trust_aware: trustAware
            })
        });
        
        const data = await response.json();
        
        if (data.success) {
            displayRecommendations(data);
        } else {
            showError(data.error || 'Failed to get recommendations');
        }
    } catch (error) {
        showError(`Error: ${error.message}`);
    }
});

// Handle similar items form submission
document.getElementById('similarForm').addEventListener('submit', async function(e) {
    e.preventDefault();
    
    const itemId = parseInt(document.getElementById('itemId').value);
    const topK = parseInt(document.getElementById('similarTopK').value);
    
    try {
        const response = await fetch(`${API_BASE}/similar-items`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                item_id: itemId,
                top_k: topK
            })
        });
        
        const data = await response.json();
        
        if (data.success) {
            displaySimilarItems(data);
        } else {
            showError(data.error || 'Failed to get similar items');
        }
    } catch (error) {
        showError(`Error: ${error.message}`);
    }
});

// Handle interaction form submission
document.getElementById('interactionForm').addEventListener('submit', async function(e) {
    e.preventDefault();
    
    const userId = parseInt(document.getElementById('intUserId').value);
    const itemId = parseInt(document.getElementById('intItemId').value);
    const rating = parseFloat(document.getElementById('rating').value);
    const reviewText = document.getElementById('reviewText').value;
    
    try {
        const response = await fetch(`${API_BASE}/interaction`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                user_id: userId,
                item_id: itemId,
                rating: rating,
                review_text: reviewText
            })
        });
        
        const data = await response.json();
        
        if (data.success) {
            showSuccess(data.message || 'Interaction updated successfully');
            // Clear form
            document.getElementById('reviewText').value = '';
        } else {
            showError(data.error || 'Failed to update interaction');
        }
    } catch (error) {
        showError(`Error: ${error.message}`);
    }
});

// Display recommendations
function displayRecommendations(data) {
    const resultsDiv = document.getElementById('results');
    const resultsContent = document.getElementById('resultsContent');
    
    let html = `
        <h4>Recommendations for User ${data.user_id}</h4>
        <div class="row">
    `;
    
    data.recommendations.forEach((rec, index) => {
        const trustClass = rec.trust_score > 0.7 ? 'trust-high' : 
                          rec.trust_score > 0.4 ? 'trust-medium' : 'trust-low';
        
        html += `
            <div class="col-md-6 mb-3">
                <div class="recommendation-item">
                    <h5>Item ${rec.item_id}</h5>
                    <p><strong>Score:</strong> ${rec.score.toFixed(3)}</p>
                    <p><strong>Predicted Rating:</strong> ${rec.rating_prediction.toFixed(1)}/5.0</p>
                    <p><strong>Trust Score:</strong> 
                        <span class="badge ${trustClass} trust-badge">${(rec.trust_score * 100).toFixed(1)}%</span>
                    </p>
                    <p><strong>Reason:</strong> ${rec.recommendation_reason}</p>
                    ${rec.metadata.name ? `<p><strong>Product:</strong> ${rec.metadata.name}</p>` : ''}
                    ${rec.metadata.price ? `<p><strong>Price:</strong> ${rec.metadata.price}</p>` : ''}
                </div>
            </div>
        `;
    });
    
    html += '</div>';
    resultsContent.innerHTML = html;
    resultsDiv.classList.remove('d-none');
}

// Display similar items
function displaySimilarItems(data) {
    const resultsDiv = document.getElementById('results');
    const resultsContent = document.getElementById('resultsContent');
    
    let html = `
        <h4>Items Similar to Item ${data.item_id}</h4>
        <div class="row">
    `;
    
    data.similar_items.forEach((item, index) => {
        html += `
            <div class="col-md-6 mb-3">
                <div class="similarity-item">
                    <h5>Item ${item.item_id}</h5>
                    <p><strong>Similarity:</strong> ${item.similarity.toFixed(3)}</p>
                    ${item.metadata.name ? `<p><strong>Product:</strong> ${item.metadata.name}</p>` : ''}
                    ${item.metadata.price ? `<p><strong>Price:</strong> ${item.metadata.price}</p>` : ''}
                </div>
            </div>
        `;
    });
    
    html += '</div>';
    resultsContent.innerHTML = html;
    resultsDiv.classList.remove('d-none');
}

// Show success message
function showSuccess(message) {
    const resultsDiv = document.getElementById('results');
    const resultsContent = document.getElementById('resultsContent');
    
    resultsContent.innerHTML = `
        <div class="alert alert-success" role="alert">
            <strong>Success!</strong> ${message}
        </div>
    `;
    resultsDiv.classList.remove('d-none');
}

// Show error message
function showError(message) {
    const resultsDiv = document.getElementById('results');
    const resultsContent = document.getElementById('resultsContent');
    
    resultsContent.innerHTML = `
        <div class="alert alert-danger" role="alert">
            <strong>Error!</strong> ${message}
        </div>
    `;
    resultsDiv.classList.remove('d-none');
}"""
    
    with open("static/js/app.js", "w") as f:
        f.write(js_content)
    
    logger.info("Web UI files created successfully")

if __name__ == "__main__":
    # Create web files
    create_web_files()
    
    # Run the FastAPI app
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
