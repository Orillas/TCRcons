#!/usr/bin/env python3
"""
Reproduce clusTCR paper benchmarking results (Valkiers et al., Bioinformatics, 2021)
DOI: 10.1093/bioinformatics/btab446

Reproduces:
  - Figure 2: Method comparison (ClusTCR vs GLIPH2 vs iSMART vs TCRDist)
  - MCL hyperparameter grid search (Inflation x Expansion)

Key methodology:
  1. Data: VDJdb beta chain, Homo sapiens, quality score filtering (q=0,1,2)
  2. Metrics: Retention, Purity, Purity_90, Consistency (with random baseline)
  3. clusTCR default: MCL inflation=1.2, expansion=2

Usage:
  python reproduce_clustcr_paper.py                  # Full reproduction
  python reproduce_clustcr_paper.py --quick           # q=1 only, skip grid
  python reproduce_clustcr_paper.py --skip-mcl-grid   # Skip 30-min grid search
"""

import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

# ─── Paths ───
FIGURE_DIR = "/home/jilin/DeepTCR/figures"
RESULTS_DIR = "/home/jilin/DeepTCR/tcrconsensus/results/clustcr_reproduction"
CLUSTCR_REPO = "/home/jilin/DeepTCR/clusTCR"

os.makedirs(FIGURE_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ─── Plot style (matching paper) ───
try:
    plt.style.use(['seaborn-v0_8-white', 'seaborn-v0_8-paper'])
except OSError:
    plt.style.use(['seaborn-white', 'seaborn-paper'])
plt.rc('font', family='serif')
sns.set_palette('Set1')
sns.set_context('paper', font_scale=1.3)

# ─── Prepend clusTCR source to sys.path so GLIPH2/iSMART find their lib/ ───
# The pip-installed clustcr lacks the lib/ binaries; source repo has them.
if CLUSTCR_REPO not in sys.path:
    sys.path.insert(0, CLUSTCR_REPO)
os.chdir(CLUSTCR_REPO)


# ═══════════════════════════════════════════════════════════════
# DATA LOADING (exact match to paper's pipeline)
# ═══════════════════════════════════════════════════════════════

def load_paper_data(q=0):
    """
    Load VDJdb beta chain with quality filtering.
    Mirrors clusTCR's parse_vdjdb + get_chain_data('beta').
    """
    from clustcr.input.vdjdb import parse_vdjdb

    vdjdb = parse_vdjdb(
        os.path.join(CLUSTCR_REPO, 'clustcr/input/vdjdb/vdjdb_full.txt'),
        q=q
    )

    # Paper: extract beta chain
    epitopes = vdjdb[['cdr3.beta', 'v.beta', 'antigen.epitope']].dropna().drop_duplicates()
    epitopes = epitopes.rename(columns={
        'cdr3.beta': 'junction_aa',
        'v.beta': 'v_call',
        'antigen.epitope': 'epitope'
    }).reset_index(drop=True)

    cdr3 = epitopes[['junction_aa', 'v_call']].drop_duplicates().reset_index(drop=True)
    return cdr3, epitopes


# ═══════════════════════════════════════════════════════════════
# METRIC HELPERS (paper's exact definitions)
# ═══════════════════════════════════════════════════════════════

def compute_retention(nodelist, epitopes):
    """Fraction of sequences assigned to clusters of size >= 2."""
    clustered = nodelist[nodelist.duplicated(subset='cluster', keep=False)]
    return len(clustered['junction_aa'].unique()) / len(epitopes['junction_aa'].unique())


def compute_purity(nodelist, epitopes):
    """
    Weighted purity: sum of max-epitope counts / total clustered.
    Returns (true_purity, baseline_purity).
    """
    gt = pd.merge(epitopes, nodelist, on='junction_aa')

    # True
    conf = gt.groupby(['cluster', 'epitope']).size().reset_index(name='count')
    cluster_max = conf.groupby('cluster')['count'].max()
    purity = cluster_max.sum() / conf['count'].sum()

    # Baseline (random permutation)
    gt_b = gt.copy()
    gt_b['cluster'] = np.random.permutation(gt_b['cluster'])
    conf_b = gt_b.groupby(['cluster', 'epitope']).size().reset_index(name='count')
    cluster_max_b = conf_b.groupby('cluster')['count'].max()
    purity_b = cluster_max_b.sum() / conf_b['count'].sum()

    return purity, purity_b


def compute_purity_90(nodelist, epitopes):
    """Fraction of clusters with purity >= 0.90."""
    gt = pd.merge(epitopes, nodelist, on='junction_aa')

    def frac_90(df):
        conf = df.groupby(['cluster', 'epitope']).size().reset_index(name='count')
        totals = conf.groupby('cluster')['count'].sum()
        maxes = conf.groupby('cluster')['count'].max()
        purity_per = maxes / totals
        return (purity_per >= 0.9).sum() / len(purity_per)

    return frac_90(gt), frac_90(gt.assign(cluster=np.random.permutation(gt['cluster'])))


def compute_consistency(nodelist, epitopes):
    """
    Optimal diagonal matching of clusters to epitopes (recursive).
    Returns (true, baseline).
    """
    gt = pd.merge(epitopes, nodelist, on='junction_aa')

    def _consist(df):
        conf = df.groupby(['cluster', 'epitope']).size().unstack(fill_value=0)
        mat = conf.values
        def rec_max(m):
            if m.size == 0:
                return 0
            idx = np.unravel_index(m.argmax(), m.shape)
            val = m[idx]
            m2 = np.delete(np.delete(m, idx[0], axis=0), idx[1], axis=1)
            return val + rec_max(m2)
        return rec_max(mat.copy()) / len(df)

    return _consist(gt), _consist(gt.assign(cluster=np.random.permutation(gt['cluster'])))


def full_metrics(nodelist, epitopes, method_name, q):
    """Compute all paper metrics for a clustering result."""
    ret = compute_retention(nodelist, epitopes)
    pur_t, pur_b = compute_purity(nodelist, epitopes)
    p90_t, p90_b = compute_purity_90(nodelist, epitopes)
    con_t, con_b = compute_consistency(nodelist, epitopes)

    rows = [
        {'metrics': 'retention',   'actual': ret,   'baseline': ret,   'method': method_name, 'q': q},
        {'metrics': 'purity',      'actual': pur_t, 'baseline': pur_b, 'method': method_name, 'q': q},
        {'metrics': 'purity_90',   'actual': p90_t, 'baseline': p90_b, 'method': method_name, 'q': q},
        {'metrics': 'consistency', 'actual': con_t, 'baseline': con_b, 'method': method_name, 'q': q},
    ]
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════
# METHOD RUNNERS
# ═══════════════════════════════════════════════════════════════

def run_clustcr(cdr3_series, n_cpus=8):
    """Run clusTCR two-step clustering."""
    from clustcr.clustering.clustering import Clustering
    os.chdir(CLUSTCR_REPO)
    return Clustering(n_cpus=n_cpus).fit(cdr3_series)


def run_gliph2(chain_data):
    """Run GLIPH2 directly from source repo lib/ directory."""
    lib_dir = os.path.join(CLUSTCR_REPO, 'clustcr/modules/gliph2/lib')
    os.chdir(lib_dir)

    # Write input file
    chain_data.to_csv('tcrconsensus_input.txt', index=False, header=False, sep='\t')

    print('    Running GLIPH2 binary...')
    t0 = time.time()
    ret = os.system('./irtools.centos -c parameters_tcrconsensus')
    dt = time.time() - t0

    if ret != 0:
        raise RuntimeError(f"GLIPH2 exited with code {ret}")

    # Parse output
    clusters = pd.DataFrame()
    c = 0
    with open('tcrconsensus_output_cluster.txt', 'r') as f:
        for line in f.read().splitlines():
            cols = line.split(' ')
            motif = cols[3]
            cluster = cols[4:]
            if len(cluster) >= 2:
                nodes = pd.DataFrame({'junction_aa': cluster})
                nodes['cluster'] = c
                nodes['motif'] = motif
                clusters = pd.concat([clusters, nodes], ignore_index=True)
                c += 1

    clusters = clusters.drop(columns=['motif'], errors='ignore').drop_duplicates()
    return clusters, dt


def run_ismart(chain_data):
    """Run iSMART directly from source repo, using venv python."""
    lib_dir = os.path.join(CLUSTCR_REPO, 'clustcr/modules/ismart/lib')
    venv_python = '/home/jilin/DeepTCR/.venv/bin/python3'
    os.chdir(lib_dir)

    data = chain_data.drop(columns=['subject', 'count'], errors='ignore')
    data.to_csv('input.txt', index=False, header=False, sep='\t')

    has_v = 'V' in data.columns
    v_flag = 'True' if has_v else 'False'

    print(f'    Running iSMARTf3.py...')
    t0 = time.time()
    ret = os.system(f'{venv_python} iSMARTf3.py -f input.txt -v {v_flag}')
    dt = time.time() - t0

    if ret != 0:
        raise RuntimeError(f"iSMART exited with code {ret}")

    # Parse output (skip header lines starting with #)
    with open('input_clustered_v3.txt', 'r') as f:
        lines = [l for l in f.read().splitlines() if not l.startswith('#')]

    # Always 3 columns: CDR3, V, cluster (even with -v False)
    parsed = pd.DataFrame([x.split('\t') for x in lines if x.strip()],
                          columns=['CDR3', 'V', 'cluster'])

    # Paper's join_cdr3_v: combine CDR3 + V as unique identifier
    parsed = parsed.drop_duplicates().copy()
    parsed['junction_aa'] = parsed['CDR3'] + '_' + parsed['V']
    parsed = parsed[['junction_aa', 'cluster']].drop_duplicates()
    # Store original CDR3 for epitope matching
    parsed['_cdr3'] = parsed['junction_aa'].str.split('_').str[0]

    return parsed, dt


def run_tcrdist(q):
    """Run TCRDist from source repo (q >= 1 only)."""
    from clustcr.modules.tcrdist.pw_tcrdist import TCRDist, cluster_TCRDist_matrix
    from clustcr.input.vdjdb import parse_vdjdb
    os.chdir(CLUSTCR_REPO)
    vdjdb = parse_vdjdb(
        os.path.join(CLUSTCR_REPO, 'clustcr/input/vdjdb/vdjdb_full.txt'), q=q
    )
    d, seq, gt = TCRDist(vdjdb)
    output = cluster_TCRDist_matrix(d, seq, gt)
    # Reformat to match our standard metrics format
    return output


# ═══════════════════════════════════════════════════════════════
# PUBLISHED RESULTS (from repo results/method_comparison_accuracy_beta.tsv)
# ═══════════════════════════════════════════════════════════════

PUBLISHED = pd.read_csv(
    os.path.join(CLUSTCR_REPO, 'results/method_comparison_accuracy_beta.tsv'),
    sep='\t'
)


# ═══════════════════════════════════════════════════════════════
# BENCHMARK RUNNER
# ═══════════════════════════════════════════════════════════════

def run_benchmark(q_values=None, skip_gliph2=False, skip_ismart=False, skip_tcrdist=False):
    """
    Run the full method comparison across quality scores.
    """
    if q_values is None:
        q_values = [0, 1, 2]

    all_results = []

    for q in q_values:
        print(f"\n{'='*60}")
        print(f"  Quality Score q >= {q}")
        print(f"{'='*60}")

        cdr3, epi = load_paper_data(q=q)
        print(f"  CDR3 sequences: {len(cdr3)}")
        print(f"  Epitope pairs:  {len(epi)}")
        print(f"  Unique epitopes: {epi['epitope'].nunique()}")

        epi_for_metrics = epi[['junction_aa', 'epitope']].drop_duplicates()

        # ─── ClusTCR ───
        print(f"  [1/4] ClusTCR (two-step)...")
        t0 = time.time()
        result = run_clustcr(cdr3.junction_aa.drop_duplicates())
        dt = time.time() - t0
        nodelist = result.clusters_df[['junction_aa', 'cluster']]
        res = full_metrics(nodelist, epi_for_metrics, 'ClusTCR', q)
        res['runtime'] = dt
        all_results.append(res)
        print(f"    Done in {dt:.2f}s  n_clusters={nodelist['cluster'].nunique()}")

        # ─── GLIPH2 ───
        if not skip_gliph2:
            print(f"  [2/4] GLIPH2...")
            try:
                t0 = time.time()
                nodes, _ = run_gliph2(cdr3)
                dt = time.time() - t0
                res = full_metrics(nodes, epi_for_metrics, 'GLIPH2', q)
                res['runtime'] = dt
                all_results.append(res)
                print(f"    Done in {dt:.2f}s  n_clusters={nodes['cluster'].nunique()}")
            except Exception as e:
                print(f"    FAILED: {e}")

        # ─── iSMART ───
        if not skip_ismart:
            print(f"  [3/4] iSMART...")
            try:
                t0 = time.time()
                nodes, _ = run_ismart(cdr3)
                dt = time.time() - t0
                # iSMART uses CDR3_V as junction_aa; prepare matching epitope data
                epi_ismart = epi.copy()
                epi_ismart['junction_aa'] = epi_ismart['junction_aa'] + '_' + epi_ismart['v_call']
                epi_ismart = epi_ismart[['junction_aa', 'epitope']].drop_duplicates()
                res = full_metrics(nodes, epi_ismart, 'iSMART', q)
                res['runtime'] = dt
                all_results.append(res)
                print(f"    Done in {dt:.2f}s  n_clusters={nodes['cluster'].nunique()}")
            except Exception as e:
                print(f"    FAILED: {e}")

        # ─── TCRDist (q >= 1 only) ───
        if q >= 1 and not skip_tcrdist:
            print(f"  [4/4] TCRDist...")
            try:
                t0 = time.time()
                output = run_tcrdist(q)
                dt = time.time() - t0
                output['runtime'] = dt
                all_results.append(output)
                print(f"    Done in {dt:.2f}s")
            except Exception as e:
                print(f"    FAILED: {e}")
        else:
            print(f"  [4/4] TCRDist skipped (q={q} or --skip-tcrdist)")

    benchmark = pd.concat(all_results, ignore_index=True)
    return benchmark


# ═══════════════════════════════════════════════════════════════
# COMPARISON WITH PUBLISHED RESULTS
# ═══════════════════════════════════════════════════════════════

def compare_with_published(benchmark_df):
    """Compare reproduced results with clusTCR repo's published values."""
    print("\n" + "="*80)
    print("  COMPARISON: Reproduced vs Published (Valkiers et al., 2021)")
    print("="*80)

    rows = []
    for _, pub_row in PUBLISHED.iterrows():
        method = pub_row['method']
        q = pub_row['q']
        metric = pub_row['metrics']
        pub_val = pub_row['actual']

        mask = (benchmark_df['method'] == method) & \
               (benchmark_df['q'] == q) & \
               (benchmark_df['metrics'] == metric)
        rep_vals = benchmark_df[mask]['actual'].values
        rep_val = rep_vals[0] if len(rep_vals) > 0 else None

        diff_pct = None
        if rep_val is not None and pub_val > 0:
            diff_pct = abs(rep_val - pub_val) / pub_val * 100

        rows.append({
            'Method': method, 'Q': q, 'Metric': metric,
            'Published': round(pub_val, 4),
            'Reproduced': round(rep_val, 4) if rep_val is not None else None,
            'Abs_Diff': round(abs(rep_val - pub_val), 4) if rep_val is not None else None,
            'Rel_Diff_%': round(diff_pct, 2) if diff_pct is not None else None,
        })

    comp = pd.DataFrame(rows)
    print(comp.to_string(index=False))

    # Summary stats
    diffs = comp['Rel_Diff_%'].dropna()
    if len(diffs) > 0:
        print(f"\n  Mean relative difference: {diffs.mean():.2f}%")
        print(f"  Max  relative difference: {diffs.max():.2f}%")
        print(f"  Within 5%:  {(diffs < 5).sum()}/{len(diffs)}")
        print(f"  Within 10%: {(diffs < 10).sum()}/{len(diffs)}")
        print(f"  Within 20%: {(diffs < 20).sum()}/{len(diffs)}")

    return comp


# ═══════════════════════════════════════════════════════════════
# MCL HYPERPARAMETER GRID SEARCH
# ═══════════════════════════════════════════════════════════════

def run_mcl_grid(q=1):
    """
    Grid search over MCL inflation (1.1-3.0) x expansion (2-5).
    Paper uses q=1 with ~4000 CDR3s.
    """
    from clustcr.clustering.clustering import Clustering

    cdr3, epi = load_paper_data(q=q)
    epi_for_metrics = epi[['junction_aa', 'epitope']].drop_duplicates()
    cdr3_input = cdr3.junction_aa.drop_duplicates()

    inflations = [round(x, 1) for x in np.arange(1.1, 3.05, 0.1)]
    expansions = [2.0, 3.0, 4.0, 5.0]

    results = []
    total = len(inflations) * len(expansions)
    done = 0

    for inf in inflations:
        for exp in expansions:
            done += 1
            print(f"  [{done}/{total}] inflation={inf}, expansion={exp}", end='')
            try:
                os.chdir(CLUSTCR_REPO)
                result = Clustering(n_cpus=4, mcl_params=[inf, exp]).fit(cdr3_input)
                nodelist = result.clusters_df[['junction_aa', 'cluster']]
                metrics_df = full_metrics(nodelist, epi_for_metrics, 'ClusTCR_grid', q)

                row = {
                    'Inflation': inf, 'Expansion': int(exp),
                    'Retention': metrics_df[metrics_df['metrics']=='retention']['actual'].values[0],
                    'Purity_r': metrics_df[metrics_df['metrics']=='purity']['actual'].values[0],
                    'Purity_b': metrics_df[metrics_df['metrics']=='purity']['baseline'].values[0],
                    'Consistency_r': metrics_df[metrics_df['metrics']=='consistency']['actual'].values[0],
                    'Consistency_b': metrics_df[metrics_df['metrics']=='consistency']['baseline'].values[0],
                }
                results.append(row)
                print(f"  Purity={row['Purity_r']:.3f}")
            except Exception as e:
                print(f"  FAILED: {e}")

    return pd.DataFrame(results)


# ═══════════════════════════════════════════════════════════════
# VISUALIZATION
# ═══════════════════════════════════════════════════════════════

def plot_figure2(benchmark_df):
    """Reproduce paper's Figure 2: 4-panel method comparison."""
    metric_names = ['retention', 'purity', 'purity_90', 'consistency']
    ylabels = ['Retention', 'Purity', r'$f_{purity > 0.90}$', 'Consistency']
    panel_ids = ['A', 'B', 'C', 'D']
    markers = {'ClusTCR': 's', 'GLIPH2': 'o', 'iSMART': 'v', 'TCRDist': 'X'}

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for idx, (metric, ylabel, panel) in enumerate(zip(metric_names, ylabels, panel_ids)):
        ax = axes[idx]
        for method, marker in markers.items():
            sub = benchmark_df[
                (benchmark_df['method'] == method) &
                (benchmark_df['metrics'] == metric)
            ].sort_values('q')

            if len(sub) == 0:
                continue

            ax.plot(sub['q'], sub['actual'], marker=marker, ms=8, lw=2, label=method)

        ax.set_ylabel(ylabel)
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(['All (q≥0)', 'Q ≥ 1', 'Q ≥ 2'])
        ax.text(-0.15, 1.08, panel, transform=ax.transAxes,
                fontweight='bold', fontsize=14)

    handles, labels = axes[3].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=4, loc='lower right',
               bbox_to_anchor=(0.94, -0.05), fontsize=12)

    fig.suptitle('Reproduction: clusTCR Paper Figure 2\n'
                 '(Valkiers et al., Bioinformatics, 2021)',
                 fontsize=14, fontweight='bold')
    fig.tight_layout()
    path = os.path.join(FIGURE_DIR, 'fig_clustcr_paper_figure2_reproduction.png')
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_published_table():
    """Reference table of paper's published values."""
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.axis('off')

    rows = []
    for _, r in PUBLISHED.iterrows():
        rows.append([
            r['method'], str(r['q']),
            f"{r['actual']:.4f}", f"{r['baseline']:.4f}",
        ])

    col_labels = ['Method', 'Q', 'Actual', 'Baseline']

    table = ax.table(cellText=rows, colLabels=col_labels,
                     loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.3)

    for j in range(len(col_labels)):
        table[0, j].set_facecolor('#4472C4')
        table[0, j].set_text_props(color='white', fontweight='bold')

    ax.set_title('Published clusTCR Results (from repo results/)\n'
                 'Valkiers et al., Bioinformatics, 2021',
                 fontsize=13, fontweight='bold', pad=20)

    path = os.path.join(FIGURE_DIR, 'fig_clustcr_published_results_table.png')
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_comparison_bars(comp_df):
    """Bar chart comparing published vs reproduced."""
    metrics = comp_df['Metric'].unique()
    n = len(metrics)
    fig, axes = plt.subplots(1, n, figsize=(5*n, 5))

    if n == 1:
        axes = [axes]

    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        sub = comp_df[comp_df['Metric'] == metric].dropna(subset=['Reproduced'])
        if len(sub) == 0:
            continue

        x = np.arange(len(sub))
        w = 0.35
        ax.bar(x - w/2, sub['Published'], w, label='Published', color='#4C72B0', alpha=0.8)
        ax.bar(x + w/2, sub['Reproduced'], w, label='Reproduced', color='#DD8452', alpha=0.8)

        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_xticks(x)
        ax.set_xticklabels(
            [f"{r['Method']}\nq={r['Q']}" for _, r in sub.iterrows()],
            fontsize=7, rotation=45, ha='right'
        )
        ax.set_title(metric.replace('_', ' ').title())
        ax.legend(fontsize=8)

        for i, (_, row) in enumerate(sub.iterrows()):
            if row['Rel_Diff_%'] is not None:
                c = 'green' if row['Rel_Diff_%'] < 5 else ('orange' if row['Rel_Diff_%'] < 15 else 'red')
                ax.annotate(f"±{row['Rel_Diff_%']:.1f}%",
                           xy=(i, max(row['Published'], row['Reproduced'])),
                           fontsize=6, ha='center', va='bottom', color=c)

    fig.suptitle('Reproduction Accuracy', fontsize=14, fontweight='bold')
    fig.tight_layout()
    path = os.path.join(FIGURE_DIR, 'fig_clustcr_reproduction_accuracy.png')
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_mcl_heatmap(mcl_df):
    """Heatmap of MCL grid search results."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for idx, col in enumerate(['Purity_r', 'Retention', 'Consistency_r']):
        pivot = mcl_df.pivot_table(index='Inflation', columns='Expansion', values=col)
        sns.heatmap(pivot, ax=axes[idx], cmap='YlOrRd', annot=True, fmt='.3f',
                    annot_kws={'size': 7})
        axes[idx].set_title(col)

    fig.suptitle('MCL Hyperparameter Grid (q=1)', fontsize=14, fontweight='bold')
    fig.tight_layout()
    path = os.path.join(FIGURE_DIR, 'fig_clustcr_mcl_hyperparameter_grid.png')
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════════════════
# METRIC EXPLANATION
# ═══════════════════════════════════════════════════════════════

def print_metric_explanation():
    print("""
    ═══════════════════════════════════════════════════════════════
    clusTCR Paper Metrics (Valkiers et al., 2021)
    ═══════════════════════════════════════════════════════════════

    RETENTION  - Fraction of sequences placed in clusters (size > 1)
                 Higher = more sequences clustered (fewer singletons)

    PURITY     - Weighted avg of per-cluster majority-epitope fraction
                 Each cluster assigned to its most common epitope

    PURITY_90  - Fraction of clusters with purity >= 0.90
                 Measures "high-quality cluster" proportion

    CONSISTENCY- Optimal matching accuracy (recursive diagonal of confusion matrix)
                 Treats clustering as supervised classification

    BASELINE   - Same metrics on randomly permuted cluster labels

    ═══════════════════════════════════════════════════════════════
    clusTCR vs tcrconsensus Metric Comparison
    ═══════════════════════════════════════════════════════════════

    clusTCR          tcrconsensus     Notes
    ─────────────    ─────────────    ──────────────────────────
    Retention        Retention        Same definition
    Purity           Purity           clusTCR: weighted; ours: per-cluster avg
    Purity_90        —                clusTCR specific
    Consistency      —                clusTCR specific (optimal diagonal)
    —                ARI              Standard metric, clusTCR unused
    —                AMI              Standard metric, clusTCR unused
    —                F1               tcrconsensus specific
    —                Sensitivity      tcrconsensus specific
    —                Precision        tcrconsensus specific
    """)


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Reproduce clusTCR paper benchmarks')
    parser.add_argument('--quick', action='store_true',
                       help='Only q=1, skip grid search')
    parser.add_argument('--skip-mcl-grid', action='store_true',
                       help='Skip MCL grid search (~30 min)')
    parser.add_argument('--skip-gliph2', action='store_true')
    parser.add_argument('--skip-ismart', action='store_true')
    parser.add_argument('--skip-tcrdist', action='store_true')
    args = parser.parse_args()

    print_metric_explanation()

    # Step 0: Published reference table
    print("[Step 0] Generating published results reference table...")
    plot_published_table()

    # Step 1: Run benchmark
    print("\n[Step 1] Running method comparison benchmark...")
    q_values = [1] if args.quick else [0, 1, 2]
    benchmark = run_benchmark(
        q_values=q_values,
        skip_gliph2=args.skip_gliph2,
        skip_ismart=args.skip_ismart,
        skip_tcrdist=args.skip_tcrdist,
    )
    benchmark.to_csv(os.path.join(RESULTS_DIR, 'reproduced_benchmark.tsv'),
                     sep='\t', index=False)

    # Step 2: Compare with published
    print("\n[Step 2] Comparing with published results...")
    comp_df = compare_with_published(benchmark)
    comp_df.to_csv(os.path.join(RESULTS_DIR, 'reproduction_comparison.tsv'),
                   sep='\t', index=False)

    # Step 3: Figures
    print("\n[Step 3] Generating figures...")
    plot_figure2(benchmark)
    plot_comparison_bars(comp_df)

    # Step 4: MCL grid search
    if not args.skip_mcl_grid and not args.quick:
        print("\n[Step 4] Running MCL hyperparameter grid search...")
        print("  (This may take 20-30 minutes...)")
        mcl_df = run_mcl_grid(q=1)
        mcl_df.to_csv(os.path.join(RESULTS_DIR, 'mcl_hyperparameter_grid.tsv'),
                      sep='\t', index=False)
        plot_mcl_heatmap(mcl_df)
    else:
        print("\n[Step 4] Skipping MCL grid search.")

    print("\n" + "="*60)
    print("  Reproduction complete!")
    print(f"  Figures: {FIGURE_DIR}/")
    print(f"  Results: {RESULTS_DIR}/")
    print("="*60)
