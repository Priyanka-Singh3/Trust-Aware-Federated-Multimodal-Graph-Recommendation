"""
GNN Embedding Transformation Demonstration for Recommendation Systems

This script demonstrates how Graph Neural Networks transform node embeddings
in a bipartite recommendation graph using PyTorch Geometric.

Key Components:
1. Bipartite graph with user-item interaction clusters
2. GraphSAGE model for embedding transformation
3. Visualizations showing embedding evolution
4. TensorBoard export for 3D interactive viewing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GCNConv
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import pandas as pd
import os
import time
from tqdm import tqdm

# Set up interactive matplotlib
plt.ion()  # Turn on interactive mode

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class GraphSAGERecommender(nn.Module):
    """
    GraphSAGE model for recommendation systems.
    Takes user/item IDs and outputs transformed embeddings.
    """
    def __init__(self, num_nodes, embedding_dim=64, hidden_dim=32):
        super(GraphSAGERecommender, self).__init__()
        
        # Initial embedding layer - maps node IDs to embedding space
        self.embedding = nn.Embedding(num_nodes, embedding_dim)
        
        # GraphSAGE layers for message passing
        self.conv1 = SAGEConv(embedding_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        
        # Initialize embeddings with Xavier uniform
        nn.init.xavier_uniform_(self.embedding.weight)
    
    def forward(self, edge_index):
        """
        Forward pass through the GNN
        Args:
            edge_index: Graph connectivity in COO format [2, num_edges]
        Returns:
            Transformed node embeddings
        """
        # Get initial embeddings for all nodes
        x = self.embedding.weight
        
        # First GraphSAGE layer with ReLU activation
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        
        # Second GraphSAGE layer
        x = self.conv2(x, edge_index)
        
        return x
    
    def get_initial_embeddings(self):
        """Return the initial random embeddings before message passing"""
        return self.embedding.weight.detach().cpu().numpy()

def create_bipartite_recommendation_graph():
    """
    Create a smaller, clearer bipartite graph with interaction clusters:
    - Users 0-2 like Items 0-2 (Cluster 1) - Clear pattern
    - Users 3-5 like Items 3-5 (Cluster 2) - Clear pattern
    - Few cross-cluster connections for realism
    """
    num_users = 6  # Reduced from 10
    num_items = 6  # Reduced from 10
    total_nodes = num_users + num_items
    
    # Create edge list for bipartite graph with clearer patterns
    edges = []
    
    # Cluster 1: Users 0-2 -> Items 0-2 (very clear pattern)
    cluster1_connections = [
        [0, 6], [0, 7], [0, 8],  # User 0 likes all cluster 1 items
        [1, 6], [1, 7],          # User 1 likes first 2 cluster 1 items
        [2, 7], [2, 8]           # User 2 likes last 2 cluster 1 items
    ]
    edges.extend(cluster1_connections)
    
    # Cluster 2: Users 3-5 -> Items 3-5 (very clear pattern)
    cluster2_connections = [
        [3, 9], [3, 10], [3, 11],  # User 3 likes all cluster 2 items
        [4, 9], [4, 10],           # User 4 likes first 2 cluster 2 items
        [5, 10], [5, 11]           # User 5 likes last 2 cluster 2 items
    ]
    edges.extend(cluster2_connections)
    
    # Add just 2 cross-cluster connections for realism
    edges.extend([[1, 9], [4, 7]])  # User 1 likes Item 3, User 4 likes Item 1
    
    # Convert to PyTorch tensor
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    
    # Create node labels for visualization
    node_labels = []
    for i in range(num_users):
        node_labels.append(f'U{i}')
    for i in range(num_items):
        node_labels.append(f'I{i}')
    
    # Create node types (0 for users, 1 for items)
    node_types = [0] * num_users + [1] * num_items
    
    return Data(edge_index=edge_index, num_nodes=total_nodes), node_labels, node_types

def visualize_bipartite_graph(data, node_labels, node_types):
    """
    Visualization 1: Interactive bipartite graph with clear layout
    Users and items are colored differently for clarity
    """
    print("   🔄 Converting to NetworkX graph...")
    plt.figure(figsize=(10, 6))
    
    # Convert to NetworkX graph
    G = to_networkx(data, to_undirected=True)
    print(f"   📊 Graph converted: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    # Create a custom bipartite layout for better clarity
    print("   🔄 Computing bipartite layout...")
    start_time = time.time()
    
    # Manual bipartite layout: users on left, items on right
    pos = {}
    user_nodes = [i for i, t in enumerate(node_types) if t == 0]
    item_nodes = [i for i, t in enumerate(node_types) if t == 1]
    
    # Place users on the left
    for i, node in enumerate(user_nodes):
        pos[node] = (0, i * 1.5 - len(user_nodes) * 0.75)
    
    # Place items on the right
    for i, node in enumerate(item_nodes):
        pos[node] = (3, i * 1.5 - len(item_nodes) * 0.75)
    
    elapsed = time.time() - start_time
    print(f"   ✅ Layout computed in {elapsed:.2f} seconds")
    
    # Draw the graph with interactive features
    print("   🎨 Drawing interactive graph...")
    
    # Draw edges first (so they appear behind nodes)
    nx.draw_networkx_edges(G, pos, alpha=0.4, edge_color='gray', width=2)
    
    # Draw users with hover effect
    user_scatter = nx.draw_networkx_nodes(G, pos, nodelist=user_nodes, 
                                          node_color='lightblue', node_size=800, 
                                          label='Users', alpha=0.9, edgecolors='navy', linewidths=2)
    
    # Draw items with hover effect
    item_scatter = nx.draw_networkx_nodes(G, pos, nodelist=item_nodes, 
                                          node_color='lightcoral', node_size=800, 
                                          label='Items', alpha=0.9, edgecolors='darkred', linewidths=2)
    
    # Add all labels (smaller graph, so we can show all)
    nx.draw_networkx_labels(G, pos, dict(zip(range(len(node_labels)), node_labels)), 
                           font_size=10, font_weight='bold')
    
    plt.title('Interactive Bipartite Recommendation Graph\n(Blue: Users U0-U5, Red: Items I0-I5)', 
              fontsize=14, fontweight='bold')
    plt.legend(loc='upper left')
    plt.axis('off')
    plt.tight_layout()
    
    # Add grid for better visual reference
    plt.grid(True, alpha=0.1)
    
    print("   💾 Saving and displaying plot...")
    plt.savefig('bipartite_graph.png', dpi=300, bbox_inches='tight')
    print("   ✅ Plot saved as bipartite_graph.png")
    
    # Interactive display - will show the plot
    plt.show(block=False)  # Non-blocking show for interactivity
    plt.pause(2)  # Pause to allow interaction
    print("   🖱️  Graph displayed interactively - you can zoom and pan!")
    # Don't close - keep plot open

def plot_embeddings_tsne(embeddings, node_labels, node_types, title, save_name):
    """
    Interactive plot embeddings using t-SNE for 2D visualization
    """
    print(f"   🔄 Applying t-SNE to reduce {embeddings.shape[1]}D to 2D...")
    start_time = time.time()
    
    # Apply t-SNE to reduce to 2D
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(5, len(embeddings)-1))
    embeddings_2d = tsne.fit_transform(embeddings)
    
    elapsed = time.time() - start_time
    print(f"   ✅ t-SNE completed in {elapsed:.2f} seconds")
    
    plt.figure(figsize=(10, 8))
    
    # Separate users and items
    user_mask = np.array(node_types) == 0
    item_mask = np.array(node_types) == 1
    
    # Plot users with larger, clearer points
    user_scatter = plt.scatter(embeddings_2d[user_mask, 0], embeddings_2d[user_mask, 1], 
                              c='lightblue', s=200, alpha=0.8, label='Users', 
                              edgecolors='navy', linewidths=2)
    
    # Plot items with larger, clearer points
    item_scatter = plt.scatter(embeddings_2d[item_mask, 0], embeddings_2d[item_mask, 1], 
                              c='lightcoral', s=200, alpha=0.8, label='Items', 
                              edgecolors='darkred', linewidths=2)
    
    # Add all labels (smaller graph = clearer)
    for i, label in enumerate(node_labels):
        plt.annotate(label, (embeddings_2d[i, 0], embeddings_2d[i, 1]), 
                    fontsize=10, fontweight='bold', alpha=0.9,
                    xytext=(5, 5), textcoords='offset points')
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    print("   💾 Saving and displaying plot...")
    plt.savefig(save_name, dpi=300, bbox_inches='tight')
    print(f"   ✅ Plot saved as {save_name}")
    
    # Interactive display
    plt.show(block=False)
    plt.pause(2)
    print("   🖱️  Embedding plot displayed interactively!")
    # Don't close - keep plot open

def plot_embeddings_pca(embeddings, node_labels, node_types, title, save_name):
    """
    Interactive plot embeddings using PCA for 2D visualization
    """
    print(f"   🔄 Applying PCA to reduce {embeddings.shape[1]}D to 2D...")
    start_time = time.time()
    
    # Apply PCA to reduce to 2D
    pca = PCA(n_components=2, random_state=42)
    embeddings_2d = pca.fit_transform(embeddings)
    
    elapsed = time.time() - start_time
    print(f"   ✅ PCA completed in {elapsed:.2f} seconds")
    
    plt.figure(figsize=(10, 8))
    
    # Separate users and items
    user_mask = np.array(node_types) == 0
    item_mask = np.array(node_types) == 1
    
    # Plot users with larger, clearer points
    user_scatter = plt.scatter(embeddings_2d[user_mask, 0], embeddings_2d[user_mask, 1], 
                              c='lightblue', s=200, alpha=0.8, label='Users', 
                              edgecolors='navy', linewidths=2)
    
    # Plot items with larger, clearer points
    item_scatter = plt.scatter(embeddings_2d[item_mask, 0], embeddings_2d[item_mask, 1], 
                              c='lightcoral', s=200, alpha=0.8, label='Items', 
                              edgecolors='darkred', linewidths=2)
    
    # Add all labels (smaller graph = clearer)
    for i, label in enumerate(node_labels):
        plt.annotate(label, (embeddings_2d[i, 0], embeddings_2d[i, 1]), 
                    fontsize=10, fontweight='bold', alpha=0.9,
                    xytext=(5, 5), textcoords='offset points')
    
    plt.title(f'{title} (PCA)', fontsize=14, fontweight='bold')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    print("   💾 Saving and displaying plot...")
    plt.savefig(save_name, dpi=300, bbox_inches='tight')
    print(f"   ✅ Plot saved as {save_name}")
    
    # Interactive display
    plt.show(block=False)
    plt.pause(2)
    print("   🖱️  PCA plot displayed interactively!")
    # Don't close - keep plot open

def export_for_tensorboard(embeddings, node_labels, output_dir='tensorboard_data'):
    """
    Export embeddings and metadata for TensorBoard Projector
    Creates .tsv files that can be uploaded to projector.tensorflow.org
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Export embeddings as TSV
    embeddings_df = pd.DataFrame(embeddings)
    embeddings_df.to_csv(f'{output_dir}/embeddings.tsv', sep='\t', 
                        header=False, index=False)
    
    # Export metadata as TSV
    metadata_df = pd.DataFrame({
        'label': node_labels,
        'type': ['User' if 'User' in label else 'Item' for label in node_labels]
    })
    metadata_df.to_csv(f'{output_dir}/metadata.tsv', sep='\t', 
                      header=False, index=False)
    
    print(f"TensorBoard data exported to '{output_dir}/' directory")
    print("Upload embeddings.tsv and metadata.tsv to projector.tensorflow.org")

def display_all_plots():
    """
    Display all saved plots in a grid for final viewing
    """
    print("\n🎯 Displaying all plots for final review...")
    
    # Create a figure with subplots for all plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('GNN Embedding Transformation - Complete Results', fontsize=16, fontweight='bold')
    
    # Load and display each plot
    plot_files = [
        ('bipartite_graph.png', axes[0, 0], '1. Bipartite Graph'),
        ('initial_embeddings_tsne.png', axes[0, 1], '2. Initial Embeddings (t-SNE)'),
        ('initial_embeddings_pca.png', axes[0, 2], '3. Initial Embeddings (PCA)'),
        ('transformed_embeddings_tsne.png', axes[1, 0], '4. Transformed Embeddings (t-SNE)'),
        ('transformed_embeddings_pca.png', axes[1, 1], '5. Transformed Embeddings (PCA)')
    ]
    
    for i, (filename, ax, title) in enumerate(plot_files):
        if os.path.exists(filename):
            img = plt.imread(filename)
            ax.imshow(img)
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.axis('off')
        else:
            ax.text(0.5, 0.5, f'{title}\n(Not found)', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.axis('off')
    
    # Use the last subplot for summary text
    axes[1, 2].axis('off')
    summary_text = """
    📚 Key Observations:
    
    • Graph: Clear bipartite structure
    • Initial: Random embedding distribution
    • Transformed: Clear clustering based on
      user-item interactions
    
    🎯 Nodes cluster together based on
    their neighborhood relationships!
    """
    axes[1, 2].text(0.1, 0.9, summary_text, transform=axes[1, 2].transAxes,
                   fontsize=10, verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    
    # Keep this plot open permanently
    print("   🖼️  Final summary plot displayed - All plots will remain open!")
    print("   ⌨️  Press Ctrl+C in terminal to close all plots")
    
    plt.show(block=True)  # Blocking show to keep plots open
    
    return fig

def main():
    """
    Main function to run the complete demonstration
    """
    total_start_time = time.time()
    
    print("🚀 Starting GNN Embedding Transformation Demonstration")
    print("=" * 60)
    
    # Progress tracking
    steps = [
        "Creating bipartite recommendation graph",
        "Initializing GraphSAGE model", 
        "Plotting bipartite graph",
        "Extracting initial embeddings",
        "Plotting initial embeddings (t-SNE)",
        "Plotting initial embeddings (PCA)",
        "Applying GNN transformation",
        "Plotting transformed embeddings (t-SNE)",
        "Plotting transformed embeddings (PCA)",
        "Exporting data for TensorBoard"
    ]
    
    current_step = 0
    
    # Step 1: Create the bipartite recommendation graph
    current_step += 1
    print(f"\n[{current_step}/{len(steps)}] 📊 Creating bipartite recommendation graph...")
    data, node_labels, node_types = create_bipartite_recommendation_graph()
    print(f"   ✅ Created graph with {data.num_nodes} nodes and {data.edge_index.shape[1]} edges")
    
    # Step 2: Initialize the GraphSAGE model
    current_step += 1
    print(f"\n[{current_step}/{len(steps)}] 🧠 Initializing GraphSAGE model...")
    model = GraphSAGERecommender(num_nodes=data.num_nodes, 
                                embedding_dim=64, 
                                hidden_dim=32)
    print(f"   ✅ Model initialized with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Visualization 1: Plot the bipartite graph
    current_step += 1
    print(f"\n[{current_step}/{len(steps)}] 🎨 Plotting bipartite graph...")
    start_time = time.time()
    visualize_bipartite_graph(data, node_labels, node_types)
    elapsed = time.time() - start_time
    print(f"   ✅ Graph visualization completed in {elapsed:.2f} seconds")
    
    # Get initial embeddings (before message passing)
    current_step += 1
    print(f"\n[{current_step}/{len(steps)}] 📈 Extracting initial embeddings...")
    initial_embeddings = model.get_initial_embeddings()
    print(f"   ✅ Initial embeddings extracted: {initial_embeddings.shape}")
    
    # Visualization 2a: Plot initial embeddings (t-SNE)
    current_step += 1
    print(f"\n[{current_step}/{len(steps)}] 🎨 Plotting initial embeddings (t-SNE)...")
    start_time = time.time()
    plot_embeddings_tsne(initial_embeddings, node_labels, node_types, 
                        'Initial Random Embeddings (Before GNN)', 
                        'initial_embeddings_tsne.png')
    elapsed = time.time() - start_time
    print(f"   ✅ t-SNE plot saved in {elapsed:.2f} seconds")
    
    # Visualization 2b: Plot initial embeddings (PCA)
    current_step += 1
    print(f"\n[{current_step}/{len(steps)}] 🎨 Plotting initial embeddings (PCA)...")
    start_time = time.time()
    plot_embeddings_pca(initial_embeddings, node_labels, node_types, 
                       'Initial Random Embeddings (Before GNN)', 
                       'initial_embeddings_pca.png')
    elapsed = time.time() - start_time
    print(f"   ✅ PCA plot saved in {elapsed:.2f} seconds")
    
    # Step 3: Pass through GNN (single forward pass)
    current_step += 1
    print(f"\n[{current_step}/{len(steps)}] 🔄 Applying GNN transformation...")
    start_time = time.time()
    model.eval()
    with torch.no_grad():
        transformed_embeddings = model(data.edge_index)
    transformed_embeddings = transformed_embeddings.detach().cpu().numpy()
    elapsed = time.time() - start_time
    print(f"   ✅ GNN transformation completed in {elapsed:.2f} seconds: {transformed_embeddings.shape}")
    
    # Visualization 3a: Plot transformed embeddings (t-SNE)
    current_step += 1
    print(f"\n[{current_step}/{len(steps)}] 🎨 Plotting transformed embeddings (t-SNE)...")
    start_time = time.time()
    plot_embeddings_tsne(transformed_embeddings, node_labels, node_types, 
                        'Transformed Embeddings (After GNN)', 
                        'transformed_embeddings_tsne.png')
    elapsed = time.time() - start_time
    print(f"   ✅ Transformed t-SNE plot saved in {elapsed:.2f} seconds")
    
    # Visualization 3b: Plot transformed embeddings (PCA)
    current_step += 1
    print(f"\n[{current_step}/{len(steps)}] 🎨 Plotting transformed embeddings (PCA)...")
    start_time = time.time()
    plot_embeddings_pca(transformed_embeddings, node_labels, node_types, 
                       'Transformed Embeddings (After GNN)', 
                       'transformed_embeddings_pca.png')
    elapsed = time.time() - start_time
    print(f"   ✅ Transformed PCA plot saved in {elapsed:.2f} seconds")
    
    # Step 4: Export for TensorBoard
    current_step += 1
    print(f"\n[{current_step}/{len(steps)}] 💾 Exporting data for TensorBoard Projector...")
    start_time = time.time()
    export_for_tensorboard(transformed_embeddings, node_labels)
    elapsed = time.time() - start_time
    print(f"   ✅ Data exported in {elapsed:.2f} seconds")
    
    # Final summary
    total_elapsed = time.time() - total_start_time
    print(f"\n🎉 All {len(steps)} steps completed successfully!")
    print(f"⏱️  Total time: {total_elapsed:.2f} seconds")
    print("=" * 60)
    
    # Explanation of the transformation
    print("\n📚 Why do nodes move closer together?")
    print("=" * 50)
    print("""
    The Graph Neural Network transforms embeddings through message passing:
    
    1. **Initial State**: Each node starts with a random embedding vector,
       representing no knowledge about its neighbors.
    
    2. **Message Passing**: During GNN forward pass:
       - Each node aggregates information from its neighbors
       - Users gather signals from items they interacted with
       - Items gather signals from users who liked them
       
    3. **Cluster Formation**: 
       - Users U0-U2 and Items I0-I2 move closer because they're densely connected
       - Users U3-U5 and Items I3-I5 form another cluster
       - Nodes with similar interaction patterns develop similar embeddings
    
    4. **Semantic Meaning**: The transformed embeddings capture:
       - User preferences based on their interaction history
       - Item characteristics based on who interacts with them
       - Latent similarity between users and items
    
    This is why GNNs are powerful for recommendations - they learn
    meaningful representations that reflect the graph structure!
    """)
    
    print("\n✅ Demonstration complete!")
    print("📁 Generated files:")
    print("   - bipartite_graph.png")
    print("   - initial_embeddings_tsne.png")
    print("   - initial_embeddings_pca.png") 
    print("   - transformed_embeddings_tsne.png")
    print("   - transformed_embeddings_pca.png")
    print("   - tensorboard_data/embeddings.tsv")
    print("   - tensorboard_data/metadata.tsv")
    
    # Display all plots together and keep them open
    try:
        display_all_plots()
    except KeyboardInterrupt:
        print("\n👋 Goodbye! All plots closed.")
    except Exception as e:
        print(f"\n⚠️  Could not display summary plot: {e}")
        print("📁 Individual plot files are still available in the directory.")

if __name__ == "__main__":
    main()
