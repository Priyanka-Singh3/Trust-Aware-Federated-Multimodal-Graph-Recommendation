#!/usr/bin/env python3
"""
Generate all methodology diagrams for the BTP paper
Uses matplotlib to create publication-quality figures
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle
import numpy as np
import os

# Set style
plt.style.use('default')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 1.5

OUTPUT_DIR = "research_output/methodology_diagrams"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def save_fig(name):
    """Save figure with tight layout"""
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/{name}", dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"✓ Generated: {name}")

def draw_box(ax, x, y, width, height, text, color, text_color='white', fontsize=9):
    """Draw a rounded box with text"""
    box = FancyBboxPatch((x - width/2, y - height/2), width, height,
                         boxstyle="round,pad=0.02,rounding_size=0.1",
                         facecolor=color, edgecolor='black', linewidth=2)
    ax.add_patch(box)
    ax.text(x, y, text, ha='center', va='center', fontsize=fontsize,
            color=text_color, fontweight='bold', wrap=True)

def draw_arrow(ax, x1, y1, x2, y2, color='black'):
    """Draw an arrow between two points"""
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=2))

# ============================================================================
# FIG 1: SYSTEM ARCHITECTURE
# ============================================================================
def fig1_system_architecture():
    """Fig 1: Overall system architecture"""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    # Title
    ax.text(7, 7.5, 'Fig. 1: TAFMGR System Architecture', 
            ha='center', va='center', fontsize=14, fontweight='bold')
    
    # Data Layer
    draw_box(ax, 2, 5.5, 2.5, 1, 'Data Layer\n(Yelp Dataset)', '#3498db')
    
    # Multimodal Encoders
    draw_box(ax, 5.5, 5.5, 2.5, 1, 'Multimodal\nEncoders', '#e74c3c')
    
    # GNN Layer
    draw_box(ax, 9, 5.5, 2.5, 1, 'GNN Layer\n(Bipartite Graph)', '#2ecc71')
    
    # Trust Mechanism
    draw_box(ax, 9, 3.5, 2.5, 1, 'Trust\nMechanism', '#f39c12')
    
    # Federated Aggregation
    draw_box(ax, 5.5, 3.5, 2.5, 1, 'Federated\nAggregation', '#9b59b6')
    
    # Recommendations
    draw_box(ax, 2, 3.5, 2.5, 1, 'Top-K\nRecommendations', '#1abc9c')
    
    # Arrows
    draw_arrow(ax, 3.25, 5.5, 4.25, 5.5)  # Data -> Encoders
    draw_arrow(ax, 6.75, 5.5, 7.75, 5.5)  # Encoders -> GNN
    draw_arrow(ax, 9, 5, 9, 4)            # GNN -> Trust
    draw_arrow(ax, 7.75, 3.5, 6.75, 3.5)  # Trust -> Federated
    draw_arrow(ax, 4.25, 3.5, 3.25, 3.5)  # Federated -> Output
    
    # Legend
    legend_y = 1.5
    ax.text(1, legend_y + 0.5, 'Components:', fontsize=10, fontweight='bold')
    components = [
        ('#3498db', 'Raw Data'),
        ('#e74c3c', 'Feature Extraction'),
        ('#2ecc71', 'Graph Learning'),
        ('#f39c12', 'Trust Scoring'),
        ('#9b59b6', 'Privacy-Preserving FL'),
        ('#1abc9c', 'Output')
    ]
    for i, (color, label) in enumerate(components):
        x_pos = 1 + (i % 3) * 4
        y_pos = legend_y - (i // 3) * 0.5
        ax.add_patch(Rectangle((x_pos, y_pos), 0.3, 0.3, facecolor=color))
        ax.text(x_pos + 0.4, y_pos + 0.15, label, fontsize=8, va='center')
    
    save_fig('fig1_system_architecture.png')

# ============================================================================
# FIG 2: DATA SCHEMA
# ============================================================================
def fig2_data_schema():
    """Fig 2: Entity relationship diagram"""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    ax.text(6, 7.5, 'Fig. 2: Data Schema and Relationships', 
            ha='center', fontsize=14, fontweight='bold')
    
    # Users entity
    draw_box(ax, 2, 5, 2.5, 1.5, 'USERS\n• user_id (PK)\n• location\n• review_count', '#3498db')
    
    # Businesses entity
    draw_box(ax, 6, 5, 2.5, 1.5, 'BUSINESSES\n• business_id (PK)\n• categories\n• stars, location', '#e74c3c')
    
    # Reviews entity
    draw_box(ax, 4, 2.5, 2.5, 1.5, 'REVIEWS\n• review_id (PK)\n• user_id (FK)\n• business_id (FK)\n• text, rating', '#2ecc71')
    
    # Photos entity
    draw_box(ax, 8, 2.5, 2.5, 1.5, 'PHOTOS\n• photo_id (PK)\n• business_id (FK)\n• image_features', '#f39c12')
    
    # Relationships
    ax.text(3.5, 5, '1:N', ha='center', fontsize=9, fontweight='bold')
    ax.text(5, 5, 'writes', ha='center', fontsize=9)
    ax.text(6.5, 5, '1:N', ha='center', fontsize=9, fontweight='bold')
    
    ax.text(3.5, 3.7, 'N:1', ha='center', fontsize=9, fontweight='bold')
    ax.text(5, 3.7, 'reviews', ha='center', fontsize=9)
    ax.text(6.5, 3.7, 'N:1', ha='center', fontsize=9, fontweight='bold')
    
    ax.text(7, 3.7, 'N:1', ha='center', fontsize=9, fontweight='bold')
    ax.text(8.5, 3.7, 'contains', ha='center', fontsize=9)
    
    # Bipartite graph indicator
    ax.text(4, 1, 'Bipartite Graph: Users ↔ Items (Businesses)', 
            ha='center', fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightyellow', edgecolor='orange'))
    
    save_fig('fig2_data_schema.png')

# ============================================================================
# FIG 3: MULTIMODAL ENCODER
# ============================================================================
def fig3_multimodal_encoder():
    """Fig 3: Multimodal encoder architecture"""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    ax.text(7, 7.5, 'Fig. 3: Multimodal Encoder Architecture', 
            ha='center', fontsize=14, fontweight='bold')
    
    # Inputs
    draw_box(ax, 2, 5.5, 2, 1, 'User ID\nEmbedding', '#3498db')
    draw_box(ax, 5, 5.5, 2, 1, 'Item ID\nEmbedding', '#e74c3c')
    draw_box(ax, 8, 5.5, 2, 1, 'Text\n(TF-IDF)', '#2ecc71')
    draw_box(ax, 11, 5.5, 2, 1, 'Image\n(ResNet)', '#f39c12')
    
    # Projection layers
    draw_box(ax, 2, 4, 2, 0.7, 'Linear\n64-dim', '#95a5a6')
    draw_box(ax, 5, 4, 2, 0.7, 'Linear\n64-dim', '#95a5a6')
    draw_box(ax, 8, 4, 2, 0.7, 'MLP\n128→64', '#95a5a6')
    draw_box(ax, 11, 4, 2, 0.7, 'MLP\n512→64', '#95a5a6')
    
    # Fusion layer
    draw_box(ax, 6.5, 2.5, 3, 1, 'Fusion Layer\nConcat + MLP\n(256 → 128 → 64)', '#9b59b6')
    
    # Output
    draw_box(ax, 6.5, 1, 3, 0.8, 'Score Prediction\nSigmoid → Rating', '#1abc9c')
    
    # Arrows
    for x in [2, 5, 8, 11]:
        draw_arrow(ax, x, 5, x, 4.35)
        draw_arrow(ax, x, 3.65, 6.5, 3)
    
    draw_arrow(ax, 6.5, 2, 6.5, 1.4)
    
    # Feature dimensions
    ax.text(2, 6.7, '(1, 827)', ha='center', fontsize=8, style='italic')
    ax.text(5, 6.7, '(1, 760)', ha='center', fontsize=8, style='italic')
    ax.text(8, 6.7, '(1, 1000)', ha='center', fontsize=8, style='italic')
    ax.text(11, 6.7, '(1, 512)', ha='center', fontsize=8, style='italic')
    
    save_fig('fig3_multimodal_encoder.png')

# ============================================================================
# FIG 4: GNN MESSAGE PASSING
# ============================================================================
def fig4_gnn_message_passing():
    """Fig 4: GNN architecture and message passing"""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    ax.text(7, 7.5, 'Fig. 4: GNN Message Passing Architecture', 
            ha='center', fontsize=14, fontweight='bold')
    
    # Left side - Users
    ax.text(2, 6.5, 'USERS', ha='center', fontsize=12, fontweight='bold', color='#3498db')
    user_nodes = [(1.5, 5.5), (2.5, 5.5), (1.5, 4.5), (2.5, 4.5)]
    for i, (x, y) in enumerate(user_nodes):
        circle = Circle((x, y), 0.25, facecolor='#3498db', edgecolor='black', linewidth=2)
        ax.add_patch(circle)
        ax.text(x, y, f'U{i+1}', ha='center', va='center', fontsize=8, color='white')
    
    # Right side - Items
    ax.text(10, 6.5, 'ITEMS', ha='center', fontsize=12, fontweight='bold', color='#e74c3c')
    item_nodes = [(9.5, 5.5), (10.5, 5.5), (9.5, 4.5), (10.5, 4.5)]
    for i, (x, y) in enumerate(item_nodes):
        circle = Circle((x, y), 0.25, facecolor='#e74c3c', edgecolor='black', linewidth=2)
        ax.add_patch(circle)
        ax.text(x, y, f'I{i+1}', ha='center', va='center', fontsize=8, color='white')
    
    # Edges (bipartite connections)
    edges = [
        (user_nodes[0], item_nodes[0]),
        (user_nodes[0], item_nodes[1]),
        (user_nodes[1], item_nodes[1]),
        (user_nodes[1], item_nodes[2]),
        (user_nodes[2], item_nodes[2]),
        (user_nodes[2], item_nodes[3]),
        (user_nodes[3], item_nodes[0]),
        (user_nodes[3], item_nodes[3]),
    ]
    for (u, i) in edges:
        ax.plot([u[0], i[0]], [u[1], i[1]], 'k-', linewidth=1.5, alpha=0.6)
    
    # Message passing arrows
    ax.annotate('', xy=(4, 5), xytext=(3.5, 5),
                arrowprops=dict(arrowstyle='->', color='#2ecc71', lw=3))
    ax.text(3.75, 5.3, 'Message', ha='center', fontsize=9, color='#2ecc71')
    
    ax.annotate('', xy=(8.5, 5), xytext=(8, 5),
                arrowprops=dict(arrowstyle='->', color='#2ecc71', lw=3))
    ax.text(8.25, 5.3, 'Update', ha='center', fontsize=9, color='#2ecc71')
    
    # GNN Layers box
    draw_box(ax, 6, 2.5, 4, 1.5, 
             'GCN Layers\nLayer 1: Aggregation + ReLU\nLayer 2: Final embeddings', 
             '#9b59b6')
    
    # Equation
    ax.text(6, 1, r'$h_u^{(l+1)} = \sigma\left(\sum_{i \in \mathcal{N}(u)} \frac{1}{c_{ui}} h_i^{(l)} W^{(l)}\right)$',
            ha='center', fontsize=10, 
            bbox=dict(boxstyle='round', facecolor='lightyellow'))
    
    save_fig('fig4_gnn_architecture.png')

# ============================================================================
# FIG 5: TRUST MECHANISM
# ============================================================================
def fig5_trust_mechanism():
    """Fig 5: Trust score computation pipeline"""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    ax.text(7, 7.5, 'Fig. 5: Trust Score Computation Pipeline', 
            ha='center', fontsize=14, fontweight='bold')
    
    # Input
    draw_box(ax, 2, 5.5, 2, 1, 'User History\n(Ratings, Interactions)', '#3498db')
    
    # Four components
    components = [
        (4.5, 6.5, 'Rating\nConsistency', '#e74c3c', 'w_c=0.35'),
        (7, 6.5, 'Item\nPopularity', '#2ecc71', 'w_p=0.25'),
        (4.5, 4.5, 'Interaction\nRecency', '#f39c12', 'w_r=0.25'),
        (7, 4.5, 'User\nActivity', '#9b59b6', 'w_a=0.15'),
    ]
    
    for x, y, text, color, weight in components:
        draw_box(ax, x, y, 1.8, 0.9, text, color)
        ax.text(x + 1.1, y, weight, fontsize=8, style='italic')
    
    # Weighted combination
    draw_box(ax, 9.5, 5.5, 2.5, 1, 'Weighted\nCombination\nΣ w_i × s_i', '#1abc9c')
    
    # Output
    draw_box(ax, 12.5, 5.5, 1.5, 1, 'Trust\nScore\n[0,1]', '#34495e')
    
    # Arrows from input to components
    for x, y, _, _, _ in components:
        draw_arrow(ax, 3, 5.5, x - 0.9, y)
    
    # Arrows from components to combination
    for x, y, _, _, _ in components:
        draw_arrow(ax, x + 0.9, y, 8.25, 5.5)
    
    draw_arrow(ax, 10.75, 5.5, 11.75, 5.5)
    
    # Formula at bottom
    ax.text(7, 2, r'$T(u,i) = w_c \cdot C(u,i) + w_p \cdot P(i) + w_r \cdot R(u,i) + w_a \cdot A(u)$',
            ha='center', fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightyellow', edgecolor='orange'))
    
    ax.text(7, 1, 'Weights: w_c=0.35, w_p=0.25, w_r=0.25, w_a=0.15',
            ha='center', fontsize=9, style='italic')
    
    save_fig('fig5_trust_mechanism.png')

# ============================================================================
# FIG 6: FEDERATED LEARNING
# ============================================================================
def fig6_federated_learning():
    """Fig 6: Federated learning architecture"""
    fig, ax = plt.subplots(figsize=(14, 9))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 9)
    ax.axis('off')
    
    ax.text(7, 8.5, 'Fig. 6: Federated Learning Architecture', 
            ha='center', fontsize=14, fontweight='bold')
    
    # Central Server
    draw_box(ax, 6, 7, 3, 1.2, 'CENTRAL SERVER\nModel Aggregation\n(FedAvg)', '#e74c3c')
    
    # Clients
    clients = [
        (2, 4.5, 'Client 1\n(Local Data)\nTrain + Noise'),
        (6, 4.5, 'Client 2\n(Local Data)\nTrain + Noise'),
        (10, 4.5, 'Client 3\n(Local Data)\nTrain + Noise'),
    ]
    
    for i, (x, y, text) in enumerate(clients):
        draw_box(ax, x, y, 2.5, 1.2, text, '#3498db')
        # Arrows to/from server
        ax.annotate('', xy=(x, 5.1), xytext=(x, 6.4),
                    arrowprops=dict(arrowstyle='->', color='green', lw=2))
        ax.text(x - 0.3, 5.75, '↓', fontsize=14, color='green', fontweight='bold')
        
        ax.annotate('', xy=(6.5 + (x-6)*0.3, 6.4), xytext=(x, 5.1),
                    arrowprops=dict(arrowstyle='->', color='blue', lw=2))
        ax.text(x - 0.3, 5.5, '↑', fontsize=14, color='blue', fontweight='bold')
    
    # Privacy box
    draw_box(ax, 6, 2.5, 4, 1, 'Privacy Protection\n• Local Differential Privacy (ε=1.2)\n• Secure Aggregation', '#9b59b6')
    
    # Communication round box
    ax.text(7, 1, 'Communication Round: Server sends global model → Clients train locally → Clients send updates → Server aggregates',
            ha='center', fontsize=9, style='italic',
            bbox=dict(boxstyle='round', facecolor='lightyellow'))
    
    save_fig('fig6_federated_architecture.png')

# ============================================================================
# FIG 7: PRIVACY MECHANISMS
# ============================================================================
def fig7_privacy_mechanisms():
    """Fig 7: Privacy protection mechanisms"""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    ax.text(7, 7.5, 'Fig. 7: Privacy Protection Mechanisms', 
            ha='center', fontsize=14, fontweight='bold')
    
    # Input
    draw_box(ax, 1.5, 5.5, 2, 1, 'Raw\nGradients', '#e74c3c')
    
    # Step 1: Differential Privacy
    draw_box(ax, 4.5, 5.5, 2.5, 1, 'Differential Privacy\nAdd Gaussian Noise\nN(0, σ²S²)', '#3498db')
    
    # Step 2: Secure Aggregation
    draw_box(ax, 8, 5.5, 2.5, 1, 'Secure Aggregation\nEncrypt + Mask\nHomomorphic', '#2ecc71')
    
    # Output
    draw_box(ax, 11.5, 5.5, 2, 1, 'Private\nUpdates', '#9b59b6')
    
    # Arrows
    draw_arrow(ax, 2.5, 5.5, 3.25, 5.5)
    draw_arrow(ax, 5.75, 5.5, 6.75, 5.5)
    draw_arrow(ax, 9.25, 5.5, 10.5, 5.5)
    
    # Privacy budget
    ax.text(4.5, 4, r'Privacy Budget: $\epsilon = 1.2$ (Strong Privacy)',
            ha='center', fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightyellow', edgecolor='orange'))
    
    # Noise formula
    ax.text(4.5, 3, r'Noise Scale: $\sigma = \frac{\sqrt{2\ln(1.25/\delta)}}{\epsilon}$',
            ha='center', fontsize=10)
    
    # Privacy guarantee box
    draw_box(ax, 7, 1.5, 5, 1, 'Privacy Guarantee\n• Data never leaves client\n• Only model updates shared\n• Noise prevents reconstruction', '#1abc9c')
    
    save_fig('fig7_privacy_mechanisms.png')

# ============================================================================
# FIG 8: RECOMMENDATION WORKFLOW
# ============================================================================
def fig8_recommendation_workflow():
    """Fig 8: Recommendation generation workflow"""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    ax.text(7, 7.5, 'Fig. 8: Recommendation Generation Workflow', 
            ha='center', fontsize=14, fontweight='bold')
    
    # Steps in a flowchart
    steps = [
        (1.5, 5.5, 'User ID\nInput', '#3498db'),
        (4, 5.5, 'Feature\nRetrieval', '#e74c3c'),
        (6.5, 5.5, 'Multimodal\nEncoder', '#2ecc71'),
        (9, 5.5, 'GNN\nPropagation', '#f39c12'),
        (11.5, 5.5, 'Trust\nScoring', '#9b59b6'),
        (9, 3, 'Ranking\n& Sort', '#1abc9c'),
        (6.5, 3, 'Top-K\nOutput', '#34495e'),
    ]
    
    for i, (x, y, text, color) in enumerate(steps):
        if i < 5:  # Top row
            draw_box(ax, x, y, 1.8, 1, text, color)
            if i < 4:
                draw_arrow(ax, x + 0.9, y, steps[i+1][0] - 0.9, y)
        else:  # Bottom row
            draw_box(ax, x, y, 1.8, 1, text, color)
    
    # Connect rows
    draw_arrow(ax, 11.5, 5, 9, 3.5)
    draw_arrow(ax, 9 - 0.9, 3, 6.5 + 0.9, 3)
    
    # Candidate items box
    draw_box(ax, 11.5, 3, 2, 1, 'Candidate\nItems (760)', '#95a5a6')
    draw_arrow(ax, 11.5, 4, 11.5, 5)
    
    # Final output details
    ax.text(6.5, 1.8, 'K = 10 recommendations', ha='center', fontsize=10, style='italic')
    
    # Latency indicator
    ax.text(7, 1, 'Latency: < 100ms (real-time)', 
            ha='center', fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightgreen', edgecolor='green'))
    
    save_fig('fig8_recommendation_workflow.png')

# ============================================================================
# FIG 9: END-TO-END WORKFLOW
# ============================================================================
def fig9_end_to_end_workflow():
    """Fig 9: Complete end-to-end workflow"""
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    ax.text(7, 9.5, 'Fig. 9: End-to-End System Workflow', 
            ha='center', fontsize=14, fontweight='bold')
    
    # Phase 1: Data Preprocessing
    draw_box(ax, 2, 8, 3, 1, 'Phase 1\nData Preprocessing\n• Clean reviews\n• Extract features\n• Build graph', '#3498db')
    
    # Phase 2: Local Training
    draw_box(ax, 6, 8, 3, 1, 'Phase 2\nLocal Training (FL)\n• Client-side updates\n• Privacy noise\n• Gradient masking', '#e74c3c')
    
    # Phase 3: Aggregation
    draw_box(ax, 10, 8, 3, 1, 'Phase 3\nGlobal Aggregation\n• FedAvg algorithm\n• Secure combining\n• Model distribution', '#2ecc71')
    
    # Phase 4: Inference
    draw_box(ax, 6, 5.5, 3, 1.5, 'Phase 4\nRecommendation Inference\n• User query\n• Encoder forward\n• GNN propagation\n• Trust scoring', '#f39c12')
    
    # Phase 5: Output
    draw_box(ax, 6, 3, 3, 1, 'Phase 5\nTop-K Results\n• Ranked items\n• Scores & explanations\n• User feedback', '#9b59b6')
    
    # Arrows
    draw_arrow(ax, 3.5, 8, 4.5, 8)
    draw_arrow(ax, 7.5, 8, 8.5, 8)
    ax.annotate('', xy=(8.5, 8), xytext=(8.5, 6.25),
                arrowprops=dict(arrowstyle='->', color='black', lw=2, 
                               connectionstyle="arc3,rad=.3"))
    draw_arrow(ax, 6, 4.75, 6, 3.5)
    
    # Feedback loop
    ax.annotate('', xy=(4.5, 3.5), xytext=(4.5, 8),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.5, 
                               linestyle='dashed', connectionstyle="arc3,rad=-.5"))
    ax.text(3.5, 5.5, 'Continuous\nLearning', fontsize=8, style='italic', color='gray')
    
    # Timeline
    timeline_y = 1.5
    phases = ['Data Prep', 'Local Train', 'Aggregate', 'Inference', 'Output']
    for i, phase in enumerate(phases):
        x_pos = 1.5 + i * 2.5
        ax.add_patch(Circle((x_pos, timeline_y), 0.15, facecolor='#34495e'))
        ax.text(x_pos, timeline_y - 0.5, phase, ha='center', fontsize=8)
        if i < 4:
            ax.plot([x_pos + 0.15, x_pos + 2.35], [timeline_y, timeline_y], 'k-', linewidth=2)
    
    save_fig('fig9_end_to_end_workflow.png')

# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    print("Generating Methodology Diagrams for BTP Paper...")
    print("=" * 60)
    
    fig1_system_architecture()
    fig2_data_schema()
    fig3_multimodal_encoder()
    fig4_gnn_message_passing()
    fig5_trust_mechanism()
    fig6_federated_learning()
    fig7_privacy_mechanisms()
    fig8_recommendation_workflow()
    fig9_end_to_end_workflow()
    
    print("=" * 60)
    print(f"✅ All diagrams saved to: {OUTPUT_DIR}/")
    print("\nGenerated files:")
    for i in range(1, 10):
        print(f"  • fig{i}_*.png")
