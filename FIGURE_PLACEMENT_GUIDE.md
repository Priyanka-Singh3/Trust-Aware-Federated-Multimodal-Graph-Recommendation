# FIGURE PLACEMENT GUIDE FOR BTP PAPER
## Complete Mapping of Which Images Go Where

---

## 📊 SECTION-WISE FIGURE PLACEMENT

### **SECTION V: EXPERIMENTAL EVALUATION**

| Figure | Filename | Placement | Caption |
|--------|----------|-----------|---------|
| **Fig. 17** | *(Create in draw.io)* | Dataset Section | "Yelp Dataset Distribution: (a) User interaction count, (b) Item popularity, (c) Rating distribution" |

**What to create:** Simple histograms showing:
- X-axis: Number of interactions | Y-axis: Count of users
- X-axis: Number of ratings | Y-axis: Count of items  
- X-axis: Rating (1-5) | Y-axis: Frequency

---

### **SECTION VI: RESULTS AND ANALYSIS**

#### **Subsection 6.1: Overall Accuracy Results**

| Figure | Filename | Placement | Caption |
|--------|----------|-----------|---------|
| **Fig. 18** | *(Use Fig. 31)* | After Table 9 | "Comparison of proposed TAFMGR with baseline methods (MF, NCF, GNN variants) across all metrics" |
| **Fig. 10** | fig1_accuracy_metrics.png | After Table 10 | "Recommendation accuracy metrics (Precision, Recall, NDCG) at different K values" |

**Note:** Use Fig. 31 (comprehensive_baselines.png) as Fig. 18 for baseline comparison.

---

#### **Subsection 6.2: Trust Mechanism Analysis**

| Figure | Filename | Placement | Caption |
|--------|----------|-----------|---------|
| **Fig. 19** | fig2_trust_impact.png | After Table 11 | "Impact of trust mechanism: Non-trust-aware vs Trust-aware recommendations showing 12.36% improvement" |
| **Fig. 20** | *(Use new chart)* | After Table 12 | "Ablation study of individual trust factors showing cumulative contribution" |

**Alternative for Fig. 20:** You can create a simple stacked bar in PowerPoint showing:
- Baseline: 0.2978
+ Consistency: +0.0111
+ Popularity: +0.0056
+ Recency: +0.0078
+ Activity: +0.0118
= Final: 0.3341

---

#### **Subsection 6.3: Cold-Start Performance**

| Figure | Filename | Placement | Caption |
|--------|----------|-----------|---------|
| **Fig. 21** | fig3_cold_start.png | After Table 13 | "Cold-start performance: Recommendation quality improves with user history size (0 to 10+ interactions)" |
| **Fig. 22** | *(Use bar chart)* | After Table 14 | "Comparison of cold-start handling: TAFMGR vs baselines at different history sizes" |

**Alternative for Fig. 22:** Create grouped bar chart in Excel/PowerPoint with:
- X-axis: 0, 1, 5 interactions
- Groups: MF, NCF, GNN, TAFMGR (4 bars per group)
- Y-axis: Score (0-0.6)

---

#### **Subsection 6.4: Diversity and Coverage**

| Figure | Filename | Placement | Caption |
|--------|----------|-----------|---------|
| **Fig. 23** | fig4_diversity_coverage.png | After Table 15 | "Diversity and coverage analysis: (a) Catalog coverage pie chart (72.37%), (b) Diversity metrics bar chart" |

**Note:** This figure already has pie + bar combined.

---

#### **Subsection 6.5: Federated Learning Convergence**

| Figure | Filename | Placement | Caption |
|--------|----------|-----------|---------|
| **Fig. 25** | *(Create line chart)* | After Table 17 | "Training convergence: Validation NDCG@10 improvement over federated rounds" |

**How to create:** Simple line graph in Excel:
- X-axis: Round 1, 3, 5, 7, 10
- Y-axis: NDCG@10: 0.1876 → 0.2567 → 0.2987 → 0.3212 → 0.3341

---

#### **Subsection 6.6: Comprehensive Baseline Comparison** ← NEW

| Figure | Filename | Placement | Caption |
|--------|----------|-----------|---------|
| **Fig. 31** | fig31_comprehensive_baselines.png | After Table 25 | "Comprehensive baseline comparison: TAFMGR vs 9 baseline methods including MF, NCF, GNN variants, and federated versions" |

**Key Message:** Shows TAFMGR beats ALL baselines including federated and centralized variants.

---

#### **Subsection 6.7: Multimodal Ablation Study** ← NEW

| Figure | Filename | Placement | Caption |
|--------|----------|-----------|---------|
| **Fig. 32** | fig32_multimodal_ablation.png | After Table 26 | "Multimodal ablation study: (a) Performance of different fusion strategies, (b) Single-modal vs multimodal comparison proving multimodal > single-modal (p<0.01)" |

**Key Message:** Answers reviewer: "Multimodal significantly outperforms single-modal."

---

#### **Subsection 6.8: Federated vs Centralized** ← NEW

| Figure | Filename | Placement | Caption |
|--------|----------|-----------|---------|
| **Fig. 33** | fig33_privacy_accuracy_tradeoff.png | After Table 28 | "Privacy-accuracy trade-off: Federated learning achieves 0.3341 NDCG@10 with strong privacy (ε=1.2) vs 0.3423 centralized, showing only 2.4% accuracy cost" |

**Key Message:** Federated has minimal impact (2.4%) while preserving privacy.

---

#### **Subsection 6.9: Component Contribution** ← NEW

| Figure | Filename | Placement | Caption |
|--------|----------|-----------|---------|
| **Fig. 34** | fig34_component_waterfall.png | After Table 30 | "Incremental component contribution waterfall: MF baseline → GNN (+27.2%) → Text (+2.1%) → Image (+2.2%) → Fusion (+1.3%) → Trust (+1.3%) = Total +36.0%" |

**Key Message:** Shows each component's contribution to final performance.

---

#### **Subsection 6.10: Privacy Budget Analysis** ← NEW

| Figure | Filename | Placement | Caption |
|--------|----------|-----------|---------|
| **Fig. 35** | fig35_privacy_budget_impact.png | After Table 32 | "Privacy budget (ε) impact on accuracy: System maintains >0.3341 NDCG@10 at ε=3 (recommended setting), balancing privacy and utility" |

**Key Message:** User can tune privacy-accuracy trade-off.

---

### **SECTION VII: PERFORMANCE EVALUATION**

#### **Subsection 7.1: System Latency**

| Figure | Filename | Placement | Caption |
|--------|----------|-----------|---------|
| **Fig. 26** | fig5_latency.png | After Table 18 | "System latency performance: All operations complete in <100ms (real-time capable)" |
| **Fig. 27** | *(Create line chart)* | After Table 19 | "Scalability analysis: Throughput and latency under increasing concurrent users" |

**How to create Fig. 27:** Two line graphs in Excel:
- Left: Users (1,10,50,100,500) vs Throughput (22, 210, 957, 1618, 5079 req/s)
- Right: Users vs Latency (45, 48, 52, 62, 98 ms)

---

#### **Subsection 7.2: Communication Efficiency**

| Figure | Filename | Placement | Caption |
|--------|----------|-----------|---------|
| **Fig. 28** | *(Area chart)* | After Table 20 | "Cumulative communication cost over federated rounds with different compression strategies" |

**How to create:** Stacked area chart in Excel:
- X-axis: Rounds 1-30
- Y-axis: Cumulative MB
- Lines: No Comp (351 MB), FP16 (175.5 MB), Top-50% (175.5 MB), Quantized (87.9 MB)

---

### **SECTION VIII: CONCLUSION**

| Figure | Filename | Placement | Caption |
|--------|----------|-----------|---------|
| **Fig. 29** | *(Radar chart)* | After Table 23 | "Overall performance radar: TAFMGR achieves excellent ratings across all dimensions" |
| **Fig. 30** | *(Timeline graphic)* | After Section 8.3 | "Future research roadmap: Short-term (6-12 months), medium-term (1-2 years), and long-term (2+ years) directions" |

**How to create Fig. 29:** Use Excel/PowerPoint radar chart with 6 axes:
- Accuracy: 5/5
- Latency: 5/5
- Diversity: 4/5
- Privacy: 5/5
- Scalability: 5/5
- Cold-start: 5/5

---

## 📁 COMPLETE FILE INVENTORY

### **Existing Images (MUST USE):**

| # | Filename | Section | Use As |
|---|----------|---------|--------|
| 1 | fig1_accuracy_metrics.png | VI.6.1 | Fig. 10 |
| 2 | fig2_trust_impact.png | VI.6.2 | Fig. 19 |
| 3 | fig3_cold_start.png | VI.6.3 | Fig. 21 |
| 4 | fig4_diversity_coverage.png | VI.6.4 | Fig. 23 |
| 5 | fig5_latency.png | VII.7.1 | Fig. 26 |
| 6 | fig6_heatmap.png | VI.6.1 | Fig. 15 (optional) |

### **New Images (BASELINE COMPARISON):**

| # | Filename | Section | Use As |
|---|----------|---------|--------|
| 7 | fig31_comprehensive_baselines.png | VI.6.6 | Fig. 18 or Fig. 31 |
| 8 | fig32_multimodal_ablation.png | VI.6.7 | Fig. 32 |
| 9 | fig33_privacy_accuracy_tradeoff.png | VI.6.8 | Fig. 33 |
| 10 | fig34_component_waterfall.png | VI.6.9 | Fig. 34 |
| 11 | fig35_privacy_budget_impact.png | VI.6.10 | Fig. 35 |

### **Images You Need to Create (Simple Charts):**

| # | Description | Tool | Section |
|---|-------------|------|---------|
| A | Dataset distribution histograms | Excel/draw.io | V.5.1.2 (Fig. 17) |
| B | Trust factor ablation stacked | PowerPoint | VI.6.2 (Fig. 20) |
| C | Cold-start comparison bars | Excel | VI.6.3 (Fig. 22) |
| D | Convergence line chart | Excel | VI.6.5 (Fig. 25) |
| E | Scalability dual charts | Excel | VII.7.1 (Fig. 27) |
| F | Communication area chart | Excel | VII.7.2 (Fig. 28) |
| G | Performance radar | Excel | VIII.8.4 (Fig. 29) |
| H | Future roadmap timeline | PowerPoint | VIII.8.4 (Fig. 30) |

---

## 🎯 RECOMMENDED FIGURE SELECTION (FOR 25-PAGE PAPER)

### **ESSENTIAL Figures (Must Include):**

1. **Fig. 31** - Comprehensive baselines (PROVES your method is best)
2. **Fig. 32** - Multimodal ablation (ANSWERS reviewer: "Why not just text?")
3. **Fig. 33** - Privacy trade-off (ANSWERS reviewer: "Where is federated impact?")
4. **Fig. 1** - Accuracy at K values (shows standard metrics)
5. **Fig. 2** - Trust impact (your key contribution)
6. **Fig. 3** - Cold-start (shows practical value)
7. **Fig. 5** - Latency (shows real-time capability)

### **OPTIONAL Figures (If Space Permits):**

8. Fig. 34 - Component waterfall (nice-to-have breakdown)
9. Fig. 35 - Privacy budget (detailed analysis)
10. Fig. 4 - Diversity (if emphasizing coverage)
11. Fig. 6 - Heatmap (redundant with Fig. 1)

---

## 📝 LATEX FIGURE CODE TEMPLATE

```latex
\begin{figure}[h]
\centering
\includegraphics[width=0.85\textwidth]{research_output/fig31_comprehensive_baselines.png}
\caption{Comprehensive baseline comparison: TAFMGR vs 9 baseline methods including MF, NCF, GNN variants, and federated versions. Our method achieves 0.3341 NDCG@10, outperforming all baselines.}
\label{fig:baselines}
\end{figure}
```

---

## ✅ CHECKLIST FOR PAPER SUBMISSION

- [ ] Fig. 31 - Baselines included
- [ ] Fig. 32 - Multimodal ablation included  
- [ ] Fig. 33 - Privacy trade-off included
- [ ] Fig. 1-3, 5 - Original results included
- [ ] All figures have captions below them
- [ ] All figures referenced in text (e.g., "As shown in Fig. 31...")
- [ ] Figure numbers are consecutive (Fig. 1, 2, 3... not Fig. 1, 31, 32)
- [ ] All 11 PNG files in research_output/ folder

---

## 📊 SUMMARY: USE THESE 7 IMAGES FOR STRONGEST PAPER

1. **fig31_comprehensive_baselines.png** → Baselines comparison
2. **fig32_multimodal_ablation.png** → Proves multimodal > single
3. **fig33_privacy_accuracy_tradeoff.png** → FL vs centralized
4. **fig1_accuracy_metrics.png** → Standard accuracy metrics
5. **fig2_trust_impact.png** → Trust contribution
6. **fig3_cold_start.png** → Cold-start handling
7. **fig5_latency.png** → Real-time performance

**Total: 7 high-impact figures that answer all reviewer concerns!**
