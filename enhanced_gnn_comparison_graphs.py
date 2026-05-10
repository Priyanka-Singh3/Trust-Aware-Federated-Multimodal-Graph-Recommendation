#!/usr/bin/env python3
"""
Generate comparison graphs for Enhanced GNN vs Original GNN
========================================================
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Results data
models = ['Random Baseline', 'Original GNN', 'Enhanced GNN']
hr10 = [0.1000, 0.1342, 0.1848]
ndcg10 = [0.0465, 0.0626, 0.0970]
coverage = [13.16, 8.82, 6.97]

# Create figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Enhanced GNN vs Original GNN Performance Comparison', fontsize=16, fontweight='bold')

# HR@10 Comparison
ax1 = axes[0, 0]
bars = ax1.bar(models, hr10, color=['gray', 'lightblue', 'darkblue'])
ax1.set_title('Hit Rate @10 Comparison', fontweight='bold')
ax1.set_ylabel('HR@10')
ax1.set_ylim(0, 0.2)
# Add value labels on bars
for bar, value in zip(bars, hr10):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
             f'{value:.4f}', ha='center', va='bottom', fontweight='bold')

# NDCG@10 Comparison
ax2 = axes[0, 1]
bars = ax2.bar(models, ndcg10, color=['gray', 'lightcoral', 'darkred'])
ax2.set_title('NDCG @10 Comparison', fontweight='bold')
ax2.set_ylabel('NDCG@10')
ax2.set_ylim(0, 0.12)
# Add value labels on bars
for bar, value in zip(bars, ndcg10):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003, 
             f'{value:.4f}', ha='center', va='bottom', fontweight='bold')

# Improvement over Random
ax3 = axes[1, 0]
improvement = [1.0, 1.34, 1.85]
bars = ax3.bar(models, improvement, color=['gray', 'lightgreen', 'darkgreen'])
ax3.set_title('Improvement vs Random Baseline', fontweight='bold')
ax3.set_ylabel('Improvement Factor (×)')
ax3.set_ylim(0, 2.0)
# Add value labels on bars
for bar, value in zip(bars, improvement):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, 
             f'{value:.2f}×', ha='center', va='bottom', fontweight='bold')

# Coverage Comparison
ax4 = axes[1, 1]
bars = ax4.bar(models, coverage, color=['gray', 'lightyellow', 'orange'])
ax4.set_title('Catalog Coverage Comparison', fontweight='bold')
ax4.set_ylabel('Coverage (%)')
ax4.set_ylim(0, 16)
# Add value labels on bars
for bar, value in zip(bars, coverage):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, 
             f'{value:.2f}%', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('/Users/priyankasingh/Documents/BTP-8/enhanced_gnn_comparison.png', 
           dpi=300, bbox_inches='tight')
plt.close()

print("✅ Enhanced GNN comparison graphs generated!")
print("📊 Saved as: enhanced_gnn_comparison.png")

# Create performance improvement summary
fig, ax = plt.subplots(figsize=(10, 6))

# Performance metrics
metrics = ['HR@10', 'NDCG@10', 'vs Random']
original_values = [0.1342, 0.0626, 1.34]
enhanced_values = [0.1848, 0.0970, 1.85]

x = np.arange(len(metrics))
width = 0.35

bars1 = ax.bar(x - width/2, original_values, width, label='Original GNN', color='lightblue', alpha=0.8)
bars2 = ax.bar(x + width/2, enhanced_values, width, label='Enhanced GNN', color='darkblue', alpha=0.8)

ax.set_title('Enhanced GNN Performance Improvements', fontsize=14, fontweight='bold')
ax.set_ylabel('Performance Metric')
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.legend()
ax.grid(axis='y', alpha=0.3)

# Add improvement percentages
for i, (orig, enh) in enumerate(zip(original_values, enhanced_values)):
    improvement_pct = ((enh - orig) / orig) * 100
    ax.text(i, max(orig, enh) + 0.02, f'+{improvement_pct:.1f}%', 
             ha='center', va='bottom', fontweight='bold', color='green')

plt.tight_layout()
plt.savefig('/Users/priyankasingh/Documents/BTP-8/enhanced_gnn_improvements.png', 
           dpi=300, bbox_inches='tight')
plt.close()

print("📈 Performance improvement graph generated!")
print("📊 Saved as: enhanced_gnn_improvements.png")
