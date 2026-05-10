// Charts and Interactive Visualizations
let trustChart = null;
let distributionChart = null;

// Initialize charts on page load
document.addEventListener('DOMContentLoaded', function() {
    initCharts();
    setupRangeSliders();
});

// Setup range slider listeners
function setupRangeSliders() {
    // Top K slider for recommendations
    const topKRange = document.getElementById('topKRange');
    const topKValue = document.getElementById('topKValue');
    const topKInput = document.getElementById('topK');
    
    if (topKRange) {
        topKRange.addEventListener('input', function() {
            topKValue.textContent = this.value;
            topKInput.value = this.value;
        });
    }
    
    // Similar items Top K slider
    const similarTopKRange = document.getElementById('similarTopKRange');
    const similarTopKValue = document.getElementById('similarTopKValue');
    const similarTopKInput = document.getElementById('similarTopK');
    
    if (similarTopKRange) {
        similarTopKRange.addEventListener('input', function() {
            similarTopKValue.textContent = this.value;
            similarTopKInput.value = this.value;
        });
    }
    
    // Rating slider with star display
    const ratingRange = document.getElementById('ratingRange');
    const ratingValue = document.getElementById('ratingValue');
    const ratingInput = document.getElementById('rating');
    
    if (ratingRange) {
        ratingRange.addEventListener('input', function() {
            const val = parseFloat(this.value).toFixed(1);
            ratingValue.textContent = val + ' ⭐';
            ratingInput.value = val;
        });
    }
}

// Initialize Chart.js charts
function initCharts() {
    // Trust Score Chart
    const trustCtx = document.getElementById('trustChart');
    if (trustCtx) {
        trustChart = new Chart(trustCtx, {
            type: 'doughnut',
            data: {
                labels: ['High Trust (70-100%)', 'Medium Trust (40-70%)', 'Low Trust (0-40%)'],
                datasets: [{
                    data: [30, 50, 20],
                    backgroundColor: ['#28a745', '#ffc107', '#dc3545'],
                    borderWidth: 2,
                    borderColor: '#fff'
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    title: {
                        display: true,
                        text: 'Trust Score Distribution',
                        font: { size: 14, weight: 'bold' }
                    },
                    legend: {
                        position: 'bottom',
                        labels: { font: { size: 11 } }
                    }
                }
            }
        });
    }
    
    // Distribution Chart
    const distCtx = document.getElementById('distributionChart');
    if (distCtx) {
        distributionChart = new Chart(distCtx, {
            type: 'bar',
            data: {
                labels: ['Users', 'Items', 'Interactions', 'Trust Scores'],
                datasets: [{
                    label: 'System Statistics',
                    data: [99, 50, 473, 5],
                    backgroundColor: ['#007bff', '#17a2b8', '#28a745', '#ffc107'],
                    borderWidth: 1,
                    borderColor: '#fff'
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    title: {
                        display: true,
                        text: 'System Overview',
                        font: { size: 14, weight: 'bold' }
                    },
                    legend: { display: false }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        ticks: { font: { size: 10 } }
                    },
                    x: {
                        ticks: { font: { size: 10 } }
                    }
                }
            }
        });
    }
}

// Update charts with real data from API
function updateChartsWithData(recommendations) {
    if (!trustChart || !recommendations) return;
    
    // Calculate trust distribution from recommendations
    let highTrust = 0, mediumTrust = 0, lowTrust = 0;
    
    recommendations.forEach(rec => {
        if (rec.trust_score > 0.7) highTrust++;
        else if (rec.trust_score > 0.4) mediumTrust++;
        else lowTrust++;
    });
    
    // Update trust chart
    trustChart.data.datasets[0].data = [highTrust, mediumTrust, lowTrust];
    trustChart.update();
}

// Create recommendation score chart
function createRecommendationChart(recommendations, containerId) {
    const container = document.getElementById(containerId);
    if (!container) return;
    
    // Create canvas for chart
    const canvasId = 'recChart_' + Date.now();
    const canvasHtml = `<canvas id="${canvasId}" height="250"></canvas>`;
    
    // Add chart container
    const chartDiv = document.createElement('div');
    chartDiv.className = 'mt-3';
    chartDiv.innerHTML = '<h6 class="text-muted mb-2">Recommendation Scores Visualization</h6>' + canvasHtml;
    container.appendChild(chartDiv);
    
    // Prepare data
    const labels = recommendations.map(r => `Item ${r.item_id}`);
    const scores = recommendations.map(r => r.score);
    const trustScores = recommendations.map(r => r.trust_score * 100);
    
    // Create chart
    setTimeout(() => {
        new Chart(document.getElementById(canvasId), {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [
                    {
                        label: 'Recommendation Score',
                        data: scores,
                        backgroundColor: 'rgba(0, 123, 255, 0.7)',
                        borderColor: 'rgba(0, 123, 255, 1)',
                        borderWidth: 1
                    },
                    {
                        label: 'Trust Score (%)',
                        data: trustScores,
                        backgroundColor: 'rgba(40, 167, 69, 0.7)',
                        borderColor: 'rgba(40, 167, 69, 1)',
                        borderWidth: 1
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    title: {
                        display: true,
                        text: 'Recommendations Analysis',
                        font: { size: 13, weight: 'bold' }
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 1.2
                    }
                }
            }
        });
    }, 100);
}

// Create similarity visualization
function createSimilarityChart(similarItems, containerId) {
    const container = document.getElementById(containerId);
    if (!container) return;
    
    const canvasId = 'simChart_' + Date.now();
    const canvasHtml = `<canvas id="${canvasId}" height="200"></canvas>`;
    
    const chartDiv = document.createElement('div');
    chartDiv.className = 'mt-3';
    chartDiv.innerHTML = '<h6 class="text-muted mb-2">Similarity Scores</h6>' + canvasHtml;
    container.appendChild(chartDiv);
    
    // Prepare data
    const labels = similarItems.map(i => `Item ${i.item_id}`);
    const similarities = similarItems.map(i => i.similarity);
    
    setTimeout(() => {
        new Chart(document.getElementById(canvasId), {
            type: 'radar',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Similarity Score',
                    data: similarities,
                    backgroundColor: 'rgba(23, 162, 184, 0.2)',
                    borderColor: 'rgba(23, 162, 184, 1)',
                    borderWidth: 2,
                    pointBackgroundColor: 'rgba(23, 162, 184, 1)'
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    title: {
                        display: true,
                        text: 'Item Similarity Profile',
                        font: { size: 13, weight: 'bold' }
                    }
                },
                scales: {
                    r: {
                        beginAtZero: true,
                        max: 1
                    }
                }
            }
        });
    }, 100);
}

// Hide results panel
function hideResults() {
    document.getElementById('results').classList.add('d-none');
}

// Export functions for use in app.js
window.updateChartsWithData = updateChartsWithData;
window.createRecommendationChart = createRecommendationChart;
window.createSimilarityChart = createSimilarityChart;
window.hideResults = hideResults;