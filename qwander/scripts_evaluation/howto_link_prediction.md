# Quantum Walk Link Prediction: Quickstart

This pipeline benchmarks DTQW-based node embeddings vs classical methods (DeepWalk/Node2Vec) for link prediction, 
using split graphs (Karate/Facebook).

## 1. Prepare Your Config

Your `linkpred_config.json` file controls all experiments.

**Edit these fields as needed:**

```json
{
    "dataset": "karate",           // or "facebook"
    "paths": {
        "splits_root": "./datasets_linkpred",              // Where splits/experiments are saved
        "results_root": "./results",                       // Where result CSV/plots go
        "facebook_edgelist": "../../datasets/Facebook/facebook_combined.txt"  // Facebook data file
    },
    "split": {
        "test_ratios": [0.1, 0.25, 0.5],  // Train/test fractions
        "do_reduced": true,               // For Facebook, use a random subgraph (recommended)
        "seed": 42                        // Random seed for reproducibility
    },
    "simulation": {
        "coin_types": ["grover"],         // Quantum walk coin types
        "potentials": [null],             // Quantum walk potential (null = none)
        "steps": 4,                       // Number of time steps per walk
        "random_seed": 42                 // Random seed for DTQW
    },
    "dtqw_embedding": {
      "do_final": true,                   // Save final-step embeddings
      "do_average": true,                 // Save time-averaged embeddings
      "do_svd": true,                     // Save SVD embeddings
      "do_kernel_pca": true,              // Save Kernel PCA embeddings
      "svd_dim": 32,                      // SVD embedding dim
      "kpca_dim": 32,                     // KPCA embedding dim
      "kpca_metric": "bhattacharyya",     // KPCA kernel
      "random_state": 42
    },
    "classical_embedding": {
      "do_deepwalk": true,                // Run DeepWalk
      "do_node2vec": true,                // Run Node2Vec
      "dw_dim": 32,                       // DeepWalk dim
      "n2v_dim": 32,                      // Node2Vec dim
      "walk_length": 20,                  // Walk length
      "num_walks": 100,                   // Walks per node
      "window_size": 5,                   // Context window
      "epochs": 5,                        // Training epochs
      "p_param": 0.25,                    // Node2Vec parameter p
      "q_param": 0.75                     // Node2Vec parameter q
    },
    "evaluation": {
      "score": "dot"                      // Scoring method for link prediction
    }
}
```

## 2. Pipeline Steps

Run these scripts in order:

### 1. **Split the Graph**

    python build_linkpred_splits.py

* Creates train/test splits in the `splits_root` directory, with all needed files.

### 2. **Simulate Quantum Walks**

    python run_dtqw_simulations.py

* Runs DTQW node-wise simulations for each split.
* Saves probability histories for each node.

### 3. **Extract Embeddings**

    python extract_embeddings.py

* Converts DTQW outputs to various embedding types.
* Also runs DeepWalk and Node2Vec if enabled.

### 4. **Evaluate Link Prediction**

    python evaluate_link_prediction.py

* Runs link prediction (AUC, precision, recall, hits\@K, etc) for all embeddings.
* Results are saved as a CSV in `results_root`, with summary plots.

---

## Why is this useful?

* Automates the whole quantum walk vs. classical link prediction benchmark.
* **Easily configurable**: set parameters in the config, and rerun experiments.
* Output is fully reproducible and ready for further analysis or publication.

---

**That’s it!**
Just edit the config, run the four scripts above in order, and analyze the results in your results folder.
