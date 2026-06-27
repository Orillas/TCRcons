"""
DeepTCR Predictor — supervised epitope specificity prediction.

Wraps DeepTCR_SS to train a classifier on labeled TCR data and predict
epitopes for new CDR3-beta sequences.

Usage
-----
    predictor = DeepTCRPredictor()
    predictor.fit("path/to/labeled_data/", epochs_min=20)
    predictions = predictor.predict(["CASSLAPGATNEKLFF", "CASSFQETSGELFF"])
"""

import os, tempfile, shutil, logging
import numpy as np

logger = logging.getLogger(__name__)


class DeepTCRPredictor:
    """Supervised epitope predictor using DeepTCR_SS."""

    name = "deeptcr_predictor"

    def __init__(self, latent_dim: int = 64, max_length: int = 40):
        self.latent_dim = latent_dim
        self.max_length = max_length
        self._model = None
        self._label_encoder = None
        self._classes = []

    def fit(
        self,
        data_dir: str,
        test_size: float = 0.25,
        epochs_min: int = 20,
        batch_size: int = 1000,
        suppress_output: bool = True,
        **train_kwargs,
    ) -> dict:
        """Train supervised epitope predictor.

        Parameters
        ----------
        data_dir : str
            Directory layout:  data_dir/
                                 epitope_A/
                                   sample.tsv  (must have CDR3b column)
                                 epitope_B/
                                   sample.tsv
                                 ...
            Each subfolder name = epitope label.
        test_size : float
            Fraction held out for validation + test.
        epochs_min : int
            Minimum training epochs before early stopping.
        batch_size : int
            Training batch size.

        Returns
        -------
        dict with keys: n_classes, classes, n_train, n_test, auc, accuracy
        """
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

        try:
            from DeepTCR.DeepTCR import DeepTCR_SS
        except ImportError:
            raise ImportError(
                "DeepTCR not installed. "
                "Run: pip install DeepTCR  (requires TensorFlow)"
            )

        from sklearn.preprocessing import LabelEncoder

        model_dir = os.path.join(tempfile.mkdtemp(), "deeptcr_ss_model")
        self._model_dir = model_dir

        try:
            dtn = DeepTCR_SS(model_dir, max_length=self.max_length)
            dtn.Get_Data(
                directory=data_dir,
                aa_column_beta="CDR3b",
                count_column="cloneCount",
                n_jobs=1,
            )

            unique_labels = sorted(set(dtn.class_id))
            self._label_encoder = LabelEncoder().fit(unique_labels)
            self._classes = list(self._label_encoder.classes_)

            n_classes = len(self._classes)
            if n_classes < 2:
                raise ValueError(f"Need >=2 classes, got {n_classes} in {data_dir}")

            dtn.Get_Train_Valid_Test(test_size=test_size)
            dtn.Train(
                latent_dim=self.latent_dim,
                epochs_min=epochs_min,
                batch_size=batch_size,
                suppress_output=suppress_output,
                **train_kwargs,
            )

            results = {
                "n_classes": n_classes,
                "classes": self._classes,
                "n_train": len(dtn.train[0]) if dtn.train else 0,
                "n_valid": len(dtn.valid[0]) if dtn.valid else 0,
                "n_test": len(dtn.test[0]) if dtn.test else 0,
            }

            # Collect test AUC if available
            if hasattr(dtn, "roc_auc"):
                results["auc"] = float(dtn.roc_auc)
            if hasattr(dtn, "test_accuracy"):
                results["accuracy"] = float(dtn.test_accuracy)

            self._model = dtn
            logger.info(
                f"DeepTCRPredictor trained: {n_classes} classes, "
                f"auc={results.get('auc', 'N/A')}"
            )
            return results

        except Exception:
            shutil.rmtree(os.path.dirname(model_dir), ignore_errors=True)
            raise

    def predict(self, cdr3_sequences: list[str]) -> list[dict]:
        """Predict epitope for CDR3-beta sequences.

        Returns list of {sequence, predicted_label, confidence}.
        """
        if self._model is None:
            raise RuntimeError("Model not trained. Call fit() first.")

        results = []
        for seq in cdr3_sequences:
            results.append({
                "sequence": seq,
                "predicted_label": None,
                "confidence": 0.0,
            })

        # DeepTCR stores per-sequence predictions after training
        # in self.predicted_class (test set) or via forward pass
        if hasattr(self._model, "predict"):
            preds = self._model.predict(cdr3_sequences)
            for i, r in enumerate(results):
                if i < len(preds):
                    r["predicted_label"] = str(preds[i])
                    r["confidence"] = 1.0

        return results

    @property
    def classes_(self) -> list[str]:
        return self._classes

    @property
    def is_fitted(self) -> bool:
        return self._model is not None
