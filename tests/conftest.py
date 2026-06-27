import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile


@pytest.fixture
def sample_tcr_df():
    return pd.DataFrame({
        "tcr_id": [f"tcr_{i:04d}" for i in range(20)],
        "chain_mode": ["beta_only"] * 20,
        "cdr3_alpha": [None] * 20,
        "cdr3_beta": [
            "CASSLAPGATNEKLFF", "CASSLAPGATNEKLFF",
            "CASSLAGGTSGELFF", "CASSLAGGTSGELFF", "CASSLAGGTSGELFF",
            "CATSNEQFF", "CATSNEQFF",
            "CASSQETQYF", "CASSQETQYF", "CASSQETQYF",
            "CAVRDSNYQLIW", "CAVRDSNYQLIW",
            "CASSFQETQYF", "CASSFQDTQYF",
            "CASSLAPGATNEKLFF",
            "CASSPDRGQPQHF", "CASSPDRGQPQHF", "CASSPDRGQPQHF",
            "CASSLAGGTSGELFF",
            "CATSNEQFF",
        ],
        "v_alpha": [None] * 20,
        "j_alpha": [None] * 20,
        "v_beta": ["TRBV1"]*5 + ["TRBV2"]*5 + ["TRBV3"]*5 + ["TRBV4"]*5,
        "j_beta": ["TRBJ1-1"]*10 + ["TRBJ2-1"]*10,
        "subject_id": ["S1"]*10 + ["S2"]*10,
        "sample_id": ["sample_A"] * 20,
        "epitope": ["A","A","B","B","B","C","C","A","A","A","D","D","A","E","E","E","A","B","C","A"],
        "hla": ["HLA-A*02:01"] * 20,
        "count": [5,3,10,8,7,2,2,15,12,9,1,1,6,4,4,4,3,8,2,5],
        "frequency": [None] * 20,
        "source_dataset": ["test"] * 20,
    })


@pytest.fixture
def sample_tsv(tmp_path):
    df = pd.DataFrame({
        "tcr_id": [f"tcr_{i:03d}" for i in range(10)],
        "cdr3_beta": [
            "CASSLAPGATNEKLFF", "CASSLAPGATNEKLFF",
            "CASSLAGGTSGELFF", "CASSLAGGTSGELFF",
            "CATSNEQFF", "CATSNEQFF",
            "CASSQETQYF", "CASSQETQYF",
            "CASSPDRGQPQHF", "CASSPDRGQPQHF",
        ],
        "v_beta": ["TRBV1"] * 4 + ["TRBV2"] * 4 + ["TRBV3"] * 2,
        "j_beta": ["TRBJ1-1"] * 10,
        "epitope": ["A","A","B","B","C","C","A","A","D","D"],
        "count": [5,3,10,8,2,2,15,12,4,4],
    })
    path = tmp_path / "sample.tsv"
    df.to_csv(path, sep="\t", index=False)
    return str(path)


@pytest.fixture
def sample_labels():
    return np.array([0,0,1,1,2,2,0,0,3,3])


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)
