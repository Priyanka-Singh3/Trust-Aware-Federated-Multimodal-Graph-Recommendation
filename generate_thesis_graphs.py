import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.5)
colors = sns.color_palette("Set2")

def save_fig(name):
    plt.tight_layout()
    plt.savefig(name, dpi=300, bbox_inches='tight')
    plt.close()

def plot_dataset_scaling_comparison():
    labels = ['Proof-of-Concept (2.7k)', 'Scaled Dataset (41k)']
    hr10 = [0.1848, 0.2473]
    ndcg10 = [0.0970, 0.1271]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 6))
    rects1 = ax.bar(x - width/2, hr10, width, label='HR@10', color='#4c72b0')
    rects2 = ax.bar(x + width/2, ndcg10, width, label='NDCG@10', color='#dd8452')

    ax.set_ylabel('Score')
    ax.set_title('Recommendation Performance: Proof-of-Concept vs Scaled Dataset')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    
    # Add values on top
    for rects in [rects1, rects2]:
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.4f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=11, fontweight='bold')

    save_fig('fig5_dataset_scaling_comparison.png')

def plot_catalog_coverage_comparison():
    labels = ['Proof-of-Concept', 'Scaled Dataset']
    coverage = [6.97, 11.87] # percentages
    
    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(labels, coverage, color=['#c44e52', '#55a868'], width=0.5)
    
    ax.set_ylabel('Catalog Coverage (%)')
    ax.set_title('Impact of Dataset Scaling on Diversity')
    ax.set_ylim(0, 15)
    
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=12, fontweight='bold')
                    
    save_fig('fig6_catalog_coverage_comparison.png')

if __name__ == "__main__":
    print("Generating thesis comparison graphs...")
    plot_dataset_scaling_comparison()
    plot_catalog_coverage_comparison()
    print("Done!")
