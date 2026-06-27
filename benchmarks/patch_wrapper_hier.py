"""Insert hierarchical_optimization patch into deeptcr_wrapper.py.
All patch lines at 8-space indent (same level as `from DeepTCR.DeepTCR import DeepTCR_U`)."""
with open('/home/jilin/DeepTCR/tcrconsensus/src/tcrconsensus/clusterers/deeptcr_wrapper.py', 'r') as f:
    lines = f.readlines()

new_lines = []
for i, line in enumerate(lines):
    new_lines.append(line)
    if line.strip() == 'from DeepTCR.DeepTCR import DeepTCR_U':
        new_lines.extend([
            '\n',
            '        # ---- Patch hierarchical_optimization for silhouette edge case ----\n',
            '        from DeepTCR.functions import utils_u as _utils_u\n',
            '        import numpy as _np\n',
            '        import sklearn.metrics as _skm\n',
            '        _orig_hier = _utils_u.hierarchical_optimization\n',
            '        def _patched_hierarchical(distances, features, method="ward", criterion="distance"):\n',
            '            from scipy.cluster.hierarchy import linkage, fcluster\n',
            '            d = distances.copy() if hasattr(distances, "copy") else distances\n',
            '            Z = linkage(d, method=method)\n',
            '            t_list = _np.arange(0, 100, 1)\n',
            '            sil = []\n',
            '            for t in t_list:\n',
            '                IDX = fcluster(Z, t, criterion=criterion)\n',
            '                sel = IDX > 0\n',
            '                n_labels = len(_np.unique(IDX[sel]))\n',
            '                n_samples = _np.sum(sel)\n',
            '                if n_labels <= 1 or n_labels >= n_samples:\n',
            '                    sil.append(-1.0)\n',
            '                else:\n',
            '                    sil.append(_skm.silhouette_score(features[sel, :], IDX[sel]))\n',
            '            sil = _np.array(sil)\n',
            '            t_opt = t_list[_np.argmax(sil)]\n',
            '            logger.info("Hierarchical opt: t_opt=%d, sil_max=%.4f" % (t_opt, _np.max(sil)))\n',
            '            return fcluster(Z, t_opt, criterion=criterion)\n',
            '        import DeepTCR.DeepTCR as _dtcr_mod\n',
            '        _dtcr_mod.hierarchical_optimization = _patched_hierarchical\n',
            '        _utils_u.hierarchical_optimization = _patched_hierarchical\n',
            '        logger.info("Applied hierarchical_optimization patch for silhouette edge case")\n',
        ])

with open('/home/jilin/DeepTCR/tcrconsensus/src/tcrconsensus/clusterers/deeptcr_wrapper.py', 'w') as f:
    f.writelines(new_lines)

print("Patch inserted successfully")
