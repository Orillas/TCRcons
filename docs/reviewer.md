# tcrconsensus Review Guidelines

> BMC Bioinformatics reviewer perspective: evidence required to show consensus clustering is a real methodological contribution, not a wrapper around existing tools.

---

## 1. Scope and Claims

Use this document to keep manuscript claims testable.

Allowed claims:

```text
Consensus improves retention-purity-sensitivity tradeoffs on multiple labeled benchmarks.
Consensus is more stable than individual methods across dataset and epitope blocks.
Consensus degrades gracefully when one base method fails or is unavailable.
Scenario-aware defaults reduce inappropriate method choice.
```

Avoid overclaims:

```text
Consensus is universally best.
Consensus never fails.
Consensus discovers specificity without uncertainty.
Consensus finds motifs missed by all individual methods.
```

Better wording:

```text
Single methods often capture fragmented or complementary subsets; consensus unifies supported subclusters and assigns confidence.
```

---

## 2. Metric Definitions

Define every metric before showing results. Reviewer must not infer metric semantics.

### Retention

```text
retention = number of TCRs assigned to any non-singleton cluster / total number of input TCRs
```

Report both:

```text
overall retention
retention among labeled antigen-specific TCRs
```

### Purity

Cluster-level purity:

```text
purity(cluster) = max_epitope_count(cluster) / cluster_size
```

Dataset-level purity:

```text
weighted mean cluster purity, weighted by cluster size
```

Also report unweighted mean purity as supplementary because large clusters can dominate weighted purity.

### Sensitivity

Use pairwise sensitivity:

```text
sensitivity = same-epitope TCR pairs placed in same cluster / all same-epitope TCR pairs
```

This directly measures whether antigen-specific TCRs are recovered together.

### Per-Epitope F1

Use pairwise one-vs-rest F1 per epitope:

```text
positive pairs = TCR pairs sharing the target epitope
predicted positive pairs = TCR pairs co-clustered by the method
precision = correct positive co-clustered pairs / all co-clustered pairs involving target epitope
recall = correct positive co-clustered pairs / all true same-epitope pairs
F1 = 2 * precision * recall / (precision + recall)
```

Report distribution across epitopes, not only global mean.

### ARI, AMI, V-Measure

Report:

```text
ARI
AMI
V-measure
homogeneity
completeness
```

Reason:

```text
ARI is sensitive to pairwise agreement.
AMI is information-theoretic and chance-corrected.
V-measure decomposes cluster purity and epitope completeness.
```

---

## 3. Primary Evidence

### Delta vs Strongest Observed Single-Method Baseline

Do not call best single method an upper bound. It is the strongest observed baseline.

Required table format:

```text
Dataset      Strongest single method   Consensus mode   Delta ARI   95% CI for Delta
VDJdb        <method>                  balanced         <value>     [<low>, <high>]
McPAS-TCR    <method>                  balanced         <value>     [<low>, <high>]
IEDB         <method>                  balanced         <value>     [<low>, <high>]
```

Use real results only. Do not include placeholder numeric values in manuscript tables.

### Per-Epitope Performance Distribution

Required figure:

```text
boxplot or violin plot of per-epitope F1
methods = consensus + strongest 3-5 single methods + simple baselines
```

Acceptable claim:

```text
Consensus ranks top-2 or top-3 for most epitopes.
```

Avoid claim:

```text
Consensus is never worst on any epitope.
```

### Matched-Operating-Point Comparison

Show improvement at matched tradeoff points:

```text
sensitivity at fixed purity
purity at fixed retention
F1 at matched retention band
```

This matters because retention and purity are inherently opposed.

### Method Availability Robustness

This is engineering robustness, not biological accuracy.

Run leave-one-method-out consensus:

```text
full consensus
remove clusTCR
remove GLIPH2
remove tcrdist3
remove HD baseline
randomly remove 1 method
randomly remove 2 methods
```

Report:

```text
metric drop relative to full consensus
number of clusters preserved
runtime impact
failure reason if method unavailable
```

Do not equate single-method wrapper failure with biological ARI = 0 in main results.

---

## 4. Statistical Testing

Avoid relying on non-overlapping confidence intervals alone.

Required tests:

```text
paired bootstrap confidence intervals for delta metrics
Wilcoxon signed-rank test over epitope-level paired results
permutation test for paired metric deltas when sample size permits
```

Recommended resampling unit:

```text
epitope blocks
dataset x epitope blocks
subject-disjoint blocks when subject_id exists
```

Report:

```text
effect size
95% CI for paired delta
p-value only as secondary evidence
multiple-testing correction if many pairwise method comparisons are shown
```

Cross-validation note:

```text
For unsupervised clustering, k-fold CV is not primary evidence unless weights or recommender policies are trained.
Use leave-one-dataset-out or leave-one-epitope-out only for learned/tuned components.
```

---

## 5. Leakage Controls

Reviewer will look for label leakage.

Required controls:

```text
epitope labels are used only for evaluation unless explicitly tuning weights
when tuning exists, tune on separate datasets or held-out epitopes
deduplicate identical CDR3/V/J records across train/test splits
use subject-disjoint splits when subject_id is available
report whether alpha, beta, or paired alpha-beta chains are used
report whether V/J genes are available and used by each method
```

For any adaptive recommendation result:

```text
train selector/weights on source datasets
evaluate on held-out target dataset
never optimize selector on the same dataset used for final claim
```

---

## 6. Secondary Evidence

### Rank Stability

Spearman rank across only 3 datasets is weak.

Preferred:

```text
bootstrap rank probability across dataset x epitope blocks
fraction of blocks where each method ranks 1st, top-2, top-3
rank variance across benchmarks
```

Report example format:

```text
Method       P(rank=1)   P(top-2)   Mean rank   Rank SD
Consensus    <value>     <value>    <value>     <value>
clusTCR      <value>     <value>    <value>     <value>
tcrdist3     <value>     <value>    <value>     <value>
GLIPH2       <value>     <value>    <value>     <value>
```

### Confidence Calibration

Question:

```text
Do high-confidence clusters have high purity?
```

Required plot:

```text
x-axis = confidence bin
y-axis = mean purity
error bars = bootstrap CI
```

Report:

```text
monotonicity
calibration slope
number of clusters per bin
```

### Runtime and Scalability

Reviewer will ask whether consensus adds prohibitive O(N^2) cost.

Required:

```text
runtime vs number of TCRs
memory vs number of TCRs
pairwise graph density after thresholding
maximum tested N
failure point or timeout point
```

Implementation detail to document:

```text
sparse co-association graph
length/V-gene blocking before pair expansion
optional cap on maximum cluster expansion
```

---

## 7. Noise and Bulk Robustness

Do not use label noise as main bulk-noise benchmark. Real bulk noise is irrelevant background TCRs.

Required stress tests:

```text
add 10k unrelated background TCRs
add 100k unrelated background TCRs
add 1M unrelated background TCRs if computationally feasible
```

Also report mixing ratios:

```text
1:10 antigen-specific to background
1:100 antigen-specific to background
1:1000 antigen-specific to background
```

Metrics:

```text
ARI on labeled subset
purity among retained labeled clusters
retention of labeled antigen-specific TCRs
false recruitment rate of background TCRs
runtime and memory
```

Optional supplementary test:

```text
label noise at 10%, 25%, 50%
```

Label noise is supplementary because it tests annotation corruption, not bulk repertoire contamination.

---

## 8. Weights and Tuning

Weights are a major reviewer concern.

Main result should use:

```text
pre-registered weights from external benchmark priors
or fixed objective presets declared before evaluation
```

Ablations:

```text
equal weights
pre-registered benchmark-prior weights
per-dataset tuned weights as optimistic upper bound
random weights sanity check
```

Report:

```text
exact weight formula
source of priors
which datasets were used to set weights
whether weights changed after seeing test data
```

Interpretation:

```text
If equal weights perform close to prior weights, consensus signal is robust.
If tuned weights dominate, main claim must be reframed as tunable ensemble, not default robust method.
```

---

## 9. Mandatory Baselines

Beyond individual methods, include:

```text
HD baseline
majority vote consensus
intersection-only consensus
union-only consensus
random clustering sanity check
strongest observed single method
```

Purpose:

```text
HD baseline tests whether simple sequence similarity is enough.
Majority vote tests whether weighting adds value.
Intersection-only tests high-purity conservative extreme.
Union-only tests high-retention extreme.
Random clustering checks metric implementation.
Strongest observed single method anchors the main delta claim.
```

---

## 10. Biological Case Study

Use 1-2 well-characterized epitopes.

Recommended examples:

```text
Flu-M1 GILGFVFTL
CMV-pp65 NLVPMVATV
MART-1 ELAGIGILTV
```

Show:

```text
cluster membership table
which base methods support each member
core vs peripheral labels
sequence motif/logo
V/J gene enrichment
confidence score
literature reference
```

Safe claim:

```text
Consensus unified method-supported subclusters into a biologically coherent cluster.
```

Unsafe claim:

```text
Consensus discovered a cluster missed by all individual methods.
```

Consensus can add value by unifying and scoring complementary evidence, not by inventing unsupported edges.

---

## 11. Failure Modes

Include specific negative results. This prevents overclaiming.

Expected failure cases:

```text
very rare epitopes with fewer than 5 TCRs
short alpha-chain CDR3-only inputs
low V/J completeness
datasets dominated by public motifs shared across epitopes
very high background contamination
low agreement across all base methods
base methods with incompatible chain assumptions
```

For each failure mode, report:

```text
how often it occurs
which metric degrades
whether confidence scores flag the problem
recommended user action
```

---

## 12. External Tool Transparency

For every base method, report:

```text
method version
install source
exact command or API call
exact parameters
input fields used
chain mode used
runtime environment
failure handling
number of successful runs
number of failed runs
```

Recommended table:

```text
Method     Version     Params     Chain     V/J used     Runs OK     Runs failed
clusTCR    <ver>       <params>   beta      yes/no       <n>         <n>
GLIPH2     <ver>       <params>   beta      yes          <n>         <n>
tcrdist3   <ver>       <params>   beta      yes          <n>         <n>
HD         internal    <params>   beta      optional     <n>         <n>
```

---

## 13. Fatal Errors

| Error | Reviewer conclusion |
|-------|---------------------|
| Only synthetic data validation | Not proven useful for real biology |
| Placeholder numbers shown as results | Results may be fabricated |
| Global average only, no per-epitope breakdown | May fail for most epitopes |
| No variance or paired confidence intervals | Could be luck |
| No weight ablation | Weights may be overfitted |
| Only ARI, no AMI or V-measure | Metric selection may be biased |
| No comparison to simple baselines | Trivial solution might work |
| No leakage control | Labels may contaminate clustering |
| No scalability evidence | Not practical for repertoire-scale data |
| No failure mode discussion | Overclaimed |

---

## 14. Recommended Experiment Suite

### Experiment 1: Cross-Benchmark Comparison

```text
Data: VDJdb + McPAS-TCR + IEDB or equivalent independent sources
Methods: single methods + consensus modes + simple baselines
Metric: ARI, AMI, V-measure, retention, purity, sensitivity, F1
Statistics: paired bootstrap over dataset x epitope blocks
Report: per-dataset and per-epitope results
```

### Experiment 2: Background Robustness Stress Test

```text
Inject: 10k / 100k / 1M unrelated background TCRs
Metric: labeled-subset ARI, labeled retention, false background recruitment
Claim: consensus maintains higher labeled-cluster quality or degrades more slowly
Plot: background size or ratio vs metric
```

### Experiment 3: Component Ablation

```text
Conditions:
  full consensus
  equal weights
  majority vote
  intersection only
  union only
  no refinement
  leave-one-method-out
  alternative graph clustering backend
```

Report:

```text
marginal contribution of weighting
marginal contribution of refinement
robustness to method removal
```

### Experiment 4: Generalization of Adaptive Recommendation

```text
Train/tune: source datasets
Evaluate: held-out dataset
Compare: auto-selected mode vs fixed balanced mode vs user-selected single methods
Report: whether recommendation improves or at least avoids bad choices
```

### Experiment 5: Biological Case Study

```text
Select: 1-2 known epitopes
Show: method support, cluster motif, V/J enrichment, confidence, literature context
Claim: consensus creates coherent, confidence-scored cluster from complementary method evidence
```

---

## 15. Reporting Checklist

- [ ] All metric definitions stated before results
- [ ] Per-dataset per-method ARI/AMI/V-measure table
- [ ] Per-epitope F1 distribution with paired uncertainty
- [ ] Delta vs strongest observed single method, per dataset
- [ ] Matched purity/retention comparison
- [ ] Paired bootstrap CI for delta metrics
- [ ] Wilcoxon or permutation test over paired epitope blocks
- [ ] Subject-level and duplicate leakage controls
- [ ] Background TCR stress test
- [ ] Method availability robustness
- [ ] Rank stability across dataset x epitope blocks
- [ ] Weight ablation: equal, prior, tuned, random
- [ ] Refinement ablation: split/merge/filter on vs off
- [ ] Confidence calibration curve
- [ ] Runtime and memory scaling
- [ ] External tool version and parameter table
- [ ] At least 1 biological case study with literature validation
- [ ] Failure mode discussion
- [ ] Code, config, seeds, and data manifests for reproducibility

---

## 16. Example Figure Layout

```text
Figure 1: Architecture overview and evidence flow
Figure 2: Per-dataset ARI/AMI/V-measure comparison with paired CI
Figure 3: Per-epitope F1 distribution, consensus vs strongest single methods
Figure 4: Background TCR robustness and false recruitment rate
Figure 5: Ablation: weights, voting baselines, refinement, method removal
Figure 6: Biological case study with method support and motif evidence
Figure S1: Confidence calibration
Figure S2: Runtime and memory scaling
Figure S3: Rank stability bootstrap
Figure S4: Failure mode examples
```
