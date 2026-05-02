#!/usr/bin/env python3
"""
Generate Component Interaction Matrix as a publication-quality table/image
"""

import matplotlib.pyplot as plt
import numpy as np

# Data for the component interaction matrix
components = [
    'Data Layer',
    'Text Encoder', 
    'Image Encoder',
    'User Embedding',
    'Item Embedding',
    'Fusion Layer',
    'GNN Layer',
    'Trust Mechanism',
    'Scoring Module',
    'Ranking Module',
    'FL Server',
    'FL Client'
]

inputs = [
    'Raw Yelp files',
    'TF-IDF vectors',
    'ResNet features',
    'User IDs',
    'Item IDs',
    '4×64-dim concat',
    '64-dim + graph',
    'History, metadata',
    'Base + Trust scores',
    'All item scores',
    'Client updates',
    'Global model'
]

outputs = [
    'Clean tensors, graphs',
    '64-dim features',
    '64-dim features',
    '64-dim embeddings',
    '64-dim embeddings',
    '64-dim unified',
    '64-dim refined',
    'Trust score [0,1]',
    'Final score [0,1]',
    'Top-K sorted list',
    'Global model',
    'Local updates'
]

next_steps = [
    'Encoder',
    'Fusion',
    'Fusion',
    'Fusion',
    'Fusion',
    'GNN',
    'Trust',
    'Scoring',
    'Ranking',
    'Output',
    'Clients',
    'Server'
]

# Create figure
fig, ax = plt.subplots(figsize=(14, 8))
ax.axis('tight')
ax.axis('off')

# Title
fig.suptitle('Fig. X: Component Data Flow Matrix', fontsize=16, fontweight='bold', y=0.98)

# Create table data
table_data = []
for i in range(len(components)):
    table_data.append([components[i], inputs[i], outputs[i], next_steps[i]])

# Create table
table = ax.table(
    cellText=table_data,
    colLabels=['Component', 'Input', 'Output', 'Next Step'],
    loc='center',
    cellLoc='left',
    colColours=['#2c3e50'] * 4,
    colWidths=[0.25, 0.25, 0.25, 0.15]
)

# Style the table
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 2.5)

# Color header
for i in range(4):
    table[(0, i)].set_text_props(color='white', fontweight='bold')
    table[(0, i)].set_facecolor('#2c3e50')

# Alternate row colors
colors = ['#ecf0f1', '#ffffff']
for i in range(1, len(components) + 1):
    for j in range(4):
        table[(i, j)].set_facecolor(colors[i % 2])
        if j == 0:  # Component column
            table[(i, j)].set_text_props(fontweight='bold', color='#2c3e50')

# Save
plt.savefig('research_output/component_interaction_matrix.png', 
            dpi=300, bbox_inches='tight', facecolor='white')
print("✅ Generated: component_interaction_matrix.png")

# Also create a second version with color coding by category
fig2, ax2 = plt.subplots(figsize=(14, 9))
ax2.axis('tight')
ax2.axis('off')

fig2.suptitle('Fig. X: Component Interaction Matrix (by Category)', fontsize=16, fontweight='bold', y=0.98)

# Category colors
categories = [
    ('Data', '#3498db'),
    ('Encoder', '#e74c3c'),
    ('Encoder', '#e74c3c'),
    ('Encoder', '#e74c3c'),
    ('Encoder', '#e74c3c'),
    ('Fusion', '#2ecc71'),
    ('GNN', '#f39c12'),
    ('Trust', '#9b59b6'),
    ('Scoring', '#1abc9c'),
    ('Output', '#34495e'),
    ('FL Server', '#e67e22'),
    ('FL Client', '#16a085')
]

# Create table data with 5 columns including category
table_data2 = []
for i in range(len(components)):
    table_data2.append([components[i], inputs[i], outputs[i], next_steps[i], categories[i][0]])

table2 = ax2.table(
    cellText=table_data2,
    colLabels=['Component', 'Input', 'Output', 'Next Step', 'Category'],
    loc='center',
    cellLoc='left',
    colWidths=[0.20, 0.22, 0.22, 0.13, 0.13]
)

table2.auto_set_font_size(False)
table2.set_fontsize(9)
table2.scale(1, 2.5)

# Color header - 5 columns for second table
for i in range(5):
    table2[(0, i)].set_text_props(color='white', fontweight='bold')
    table2[(0, i)].set_facecolor('#2c3e50')

# Color rows by category
for i, (cat_name, cat_color) in enumerate(categories, 1):
    for j in range(5):
        table2[(i, j)].set_facecolor(cat_color)
        table2[(i, j)].set_alpha(0.3)
        if j == 0:
            table2[(i, j)].set_text_props(fontweight='bold', color='#2c3e50')
    # Highlight category column
    table2[(i, 4)].set_text_props(fontweight='bold', color='#2c3e50')
    table2[(i, 4)].set_facecolor(cat_color)
    table2[(i, 4)].set_alpha(0.5)

plt.savefig('research_output/component_matrix_colored.png', 
            dpi=300, bbox_inches='tight', facecolor='white')
print("✅ Generated: component_matrix_colored.png")

print("\nBoth versions saved to research_output/")
