// API Base URL
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
}