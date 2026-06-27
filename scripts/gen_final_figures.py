import sys
sys.path.insert(0, "/home/jilin/DeepTCR/clusTCR")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
from clustcr.input.vdjdb import parse_vdjdb
from clustcr.clustering.clustering import Clustering

def get_chain_data(q):
    vdjdb = parse_vdjdb("./clustcr/input/vdjdb/vdjdb_full.txt", q=q)
    epitopes = vdjdb.drop(columns=["cdr3.alpha", "v.alpha"]).dropna().drop_duplicates()
    epitopes = epitopes.rename(columns={"cdr3.beta":"CDR3","v.beta":"V","antigen.epitope":"Epitope"})
    chain = epitopes.drop(columns="Epitope").drop_duplicates().reset_index(drop=True)
    return chain, epitopes

published = {
    0: {"retention": 0.2517, "purity": 0.5871, "purity_90": 0.4089, "consistency": 0.1300},
    1: {"retention": 0.2363, "purity": 0.8581, "purity_90": 0.7265, "consistency": 0.3614},
    2: {"retention": 0.2623, "purity": 0.9286, "purity_90": 0.9038, "consistency": 0.4796},
}

# Collect reproduced results
reproduced = {}
for q in [0, 1, 2]:
    chain_data, epitope_data = get_chain_data(q)
    cdr3_input = chain_data.CDR3.drop_duplicates()
    epi_metrics = epitope_data.drop(columns=["V","subject","count"]).drop_duplicates()
    epi_renamed = epi_metrics.rename(columns={"CDR3":"junction_aa","Epitope":"epitope"})
    result = Clustering(n_cpus=8).fit(cdr3_input)
    metrics = result.metrics(epi_renamed).summary()
    reproduced[q] = {
        "retention": metrics[metrics["metrics"]=="retention"]["actual"].values[0],
        "purity": metrics[metrics["metrics"]=="purity"]["actual"].values[0],
        "purity_90": metrics[metrics["metrics"]=="purity_90"]["actual"].values[0],
        "consistency": metrics[metrics["metrics"]=="consistency"]["actual"].values[0],
    }

# Figure 1: Published vs Reproduced comparison
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
metrics_names = ["retention", "purity", "purity_90", "consistency"]
labels = ["Retention", "Purity", "Purity$_{90}$", "Consistency"]
x = np.arange(len(metrics_names))
width = 0.3
q_labels = ["q=0\n(All)", "q>=1\n(Score>=1)", "q>=2\n(Score>=2)"]

for idx, q in enumerate([0, 1, 2]):
    ax = axes[idx]
    pub_vals = [published[q][m] for m in metrics_names]
    rep_vals = [reproduced[q][m] for m in metrics_names]

    bars1 = ax.bar(x - width/2, pub_vals, width, label="Published", color="#4C72B0", alpha=0.85)
    bars2 = ax.bar(x + width/2, rep_vals, width, label="Current Code", color="#DD8452", alpha=0.85)

    ax.set_ylabel("Score")
    ax.set_title(q_labels[idx], fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Add deviation annotations
    for j, (p, r) in enumerate(zip(pub_vals, rep_vals)):
        dev = abs(p - r) / p * 100
        if dev > 20:
            ax.annotate(f"{dev:.0f}%\ndev", xy=(j + width/2, r), fontsize=8,
                       ha="center", va="bottom", color="red", fontweight="bold")

fig.suptitle("clusTCR Reproduction: Published (2021) vs Current Code (2024)", fontsize=14, fontweight="bold", y=1.02)
fig.tight_layout()
fig.savefig("/home/jilin/DeepTCR/figures/fig_clustcr_root_cause_code_version.png", dpi=150, bbox_inches="tight")
print("Saved: fig_clustcr_root_cause_code_version.png")

# Figure 2: Relative deviation heatmap
fig2, ax2 = plt.subplots(figsize=(8, 4))
dev_matrix = np.zeros((3, 4))
for i, q in enumerate([0, 1, 2]):
    for j, m in enumerate(metrics_names):
        dev_matrix[i, j] = abs(published[q][m] - reproduced[q][m]) / published[q][m] * 100

im = ax2.imshow(dev_matrix, cmap="RdYlGn_r", aspect="auto", vmin=0, vmax=100)
ax2.set_xticks(range(4))
ax2.set_xticklabels(["Retention", "Purity", "Purity$_{90}$", "Consistency"], fontsize=11)
ax2.set_yticks(range(3))
ax2.set_yticklabels(["q=0 (All)", "q>=1", "q>=2"], fontsize=11)

for i in range(3):
    for j in range(4):
        val = dev_matrix[i, j]
        color = "white" if val > 50 else "black"
        ax2.text(j, i, f"{val:.1f}%", ha="center", va="center", fontsize=12, fontweight="bold", color=color)

plt.colorbar(im, label="Relative Deviation (%)")
ax2.set_title("Reproduction Deviation: Published vs Current Code (%)", fontsize=13, fontweight="bold")
fig2.tight_layout()
fig2.savefig("/home/jilin/DeepTCR/figures/fig_clustcr_deviation_heatmap.png", dpi=150, bbox_inches="tight")
print("Saved: fig_clustcr_deviation_heatmap.png")

# Figure 3: Timeline of code changes
fig3, ax3 = plt.subplots(figsize=(14, 5))
changes = [
    ("2021-04-29", "c774ffd", "Retention bug fix", "#F39C12"),
    ("2021-05", "Paper", "Paper published\n(Bioinformatics)", "#2ECC71"),
    ("2021-xx", "Results", "Results generated\n(published version)", "#3498DB"),
    ("2022-08-01", "4c35a5c", "Clustering refactoring", "#E74C3C"),
    ("2022-08-02", "0606804", "AIRR standard rename", "#E74C3C"),
    ("2022-08-03", "e6ce0f8", "CDR3->junction_aa\nrename", "#E74C3C"),
    ("2022-08-03", "266980a", "Metrics column rename\nCDR3->junction_aa", "#E74C3C"),
    ("2024-03-27", "935dfeb", "Chain parameter added", "#9B59B6"),
    ("2024-03-28", "b50d759", "Bugfixes", "#9B59B6"),
    ("2024-04-03", "5fa6b46", "Empty cluster bug fix", "#9B59B6"),
]

for i, (date, commit, desc, color) in enumerate(changes):
    y = 0.8 if i % 2 == 0 else 0.3
    ax3.plot(i, y, "o", color=color, markersize=12, zorder=5)
    ax3.annotate(f"{date}\n{commit}\n{desc}", xy=(i, y), xytext=(0, 20 if y > 0.5 else -60),
                textcoords="offset points", ha="center", fontsize=8, va="center",
                bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.2))

ax3.set_xlim(-0.5, len(changes) - 0.5)
ax3.set_ylim(-0.2, 1.2)
ax3.axhline(y=0.55, color="gray", linestyle="--", alpha=0.3)
ax3.set_title("clusTCR Code Evolution Timeline: Paper vs Current Version", fontsize=13, fontweight="bold")
ax3.axis("off")

# Legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor="#2ECC71", alpha=0.5, label="Paper milestone"),
    Patch(facecolor="#F39C12", alpha=0.5, label="Bug fix (behavior change)"),
    Patch(facecolor="#E74C3C", alpha=0.5, label="Major refactor (2022)"),
    Patch(facecolor="#9B59B6", alpha=0.5, label="Later updates (2024)"),
]
ax3.legend(handles=legend_elements, loc="lower right", fontsize=9)
fig3.tight_layout()
fig3.savefig("/home/jilin/DeepTCR/figures/fig_clustcr_code_evolution_timeline.png", dpi=150, bbox_inches="tight")
print("Saved: fig_clustcr_code_evolution_timeline.png")

print("\nDone! All figures saved.")
