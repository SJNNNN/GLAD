# GLAD: Global-Local Attention Difference Learning for Sequence-Based Prediction of Protein Stability Changes

## Overview

**GLAD** is a sequence-based framework for predicting mutation-induced changes in protein stability. Built upon pretrained ESM-2 protein language model representations, GLAD integrates global sequence difference learning with local attention-based modeling to capture complementary information associated with amino acid mutations.

Accurately predicting protein stability changes requires understanding both the global structural and evolutionary context of a protein sequence and the local sequence environment surrounding a mutation. GLAD addresses this challenge through a global-local learning architecture that models mutation-associated differences at complementary scales.

The framework takes wild-type and mutant protein sequences as inputs and learns mutation-induced representations for protein stability change prediction.

## Key Features

- **ESM-2-based protein representations:** Leverages pretrained protein language model embeddings to capture contextual sequence information.
- **Global difference learning:** Models mutation-induced changes at the whole-sequence level.
- **Local attention mechanism:** Focuses on mutation-associated local sequence contexts to capture localized effects.
- **Complementary feature integration:** Combines global and local representations to improve the modeling of protein stability changes.
- **Sequence-based prediction:** Does not require experimentally determined three-dimensional protein structures.

## Model Architecture

GLAD consists of two complementary learning branches:

1. **Global Branch:** Extracts and models sequence-level differences between wild-type and mutant proteins, capturing mutation-induced changes in the global sequence representation.

2. **Local Attention Branch:** Focuses on the mutation-associated local sequence context and uses attention-based feature learning to characterize the local effects of amino acid substitutions.

The learned global and local representations are integrated to generate the final prediction of protein stability changes.

## Project Structure

```text
GLAD/
│
├── train.py                  # Model training entry point
├── test.py                   # Model evaluation entry point
├── zero_multi_predict.py     # Zero-shot multi-mutation prediction
│
├── datasets/                 # Dataset and dataloader utilities
├── model/                    # GLAD model architecture
├── processor/                # Training and evaluation procedures
├── utils/                    # Utility functions, random seeds and logging
│
├── ddgdata/                  # Training and evaluation datasets
│   ├── S8754_parsed.csv
│   └── S669_with_name.csv
│
├── checkpoint/               # Model checkpoints
│
└── README.md
```

## Environment and Dependencies

The model was developed and tested in the following environment:

| Package | Version |
|---|---|
| Python | 3.8.19 |
| PyTorch | 2.2.2 |
| DGL | 2.3.0+cu121 |
| NetworkX | 3.1 |
| NumPy | 1.24.4 |
| scikit-learn | 1.3.2 |
| tqdm | 4.67.1 |

Additional dependencies:

- `esm`
- `pandas`
- `scipy`

Please ensure that the installed PyTorch and DGL versions are compatible with your CUDA environment.

## Data Preparation

The current training and evaluation scripts use the following CSV files:

- `ddgdata/S8754_parsed.csv`
- `ddgdata/S669_with_name.csv`

The datasets contain paired wild-type and mutant protein sequences, together with their corresponding stability change measurements.

Typical data columns include:

| Column | Description |
|---|---|
| `name` | Protein or mutation identifier |
| `wt_seq` | Wild-type protein sequence |
| `mut_seq` | Mutant protein sequence |
| `ddg` / `ddG` | Experimental stability change label |
| `pos` | Mutation position |

Please ensure that the column names and mutation-position indexing convention are consistent with the dataset loader.

### Input Format

Each sample consists of a paired sequence representation:

- Wild-type sequence (`wt_seq`)
- Mutant sequence (`mut_seq`)
- Mutation position (`pos`)
- Experimental stability change label (`ddg` or `ddG`)

For zero-shot multi-mutation prediction, the prediction script expects:

- `wt_seq`
- `mut_seq`
- `ddG`

## Pretrained Model Weights

Pretrained GLAD model weights are available through Google Drive.

**Download:**

[GLAD Pretrained Weights and Checkpoints](https://drive.google.com/drive/folders/17R9kfl0uGjOY1zuEaZu8x-zKuIX_X77O?hl=zh-cn)

After downloading, place the checkpoint files in the local `checkpoint/` directory, or update the checkpoint paths in the corresponding scripts.

Please ensure that the selected checkpoint matches the model configuration and evaluation task.

## Usage

### 1. Model Training

Run the following command to train GLAD:

```bash
python train.py
```

### 2. Model Evaluation

To evaluate a trained GLAD model:

```bash
python test.py
```

Make sure that the checkpoint paths and dataset configurations are correctly specified before evaluation.

### 3. Zero-Shot Multi-Mutation Prediction

GLAD also provides a script for zero-shot prediction on multi-mutation protein variants:

```bash
python zero_multi_predict.py
```

The input CSV file should contain the required `wt_seq`, `mut_seq`, and `ddG` columns.

## Implementation Notes

1. **ESM-2 initialization**

   GLAD uses the pretrained ESM-2 model:

   ```python
   esm.pretrained.esm2_t33_650M_UR50D()
   ```

   The pretrained model weights may be downloaded automatically during the first initialization. An internet connection and sufficient storage are recommended.

2. **Solver dependency**

   The current implementation references `solver.py`. Ensure that this file is available in the project environment before running the scripts.

3. **Processor initialization**

   `processor/__init__.py` imports `do_train1`. If this function or its corresponding module is unavailable in your local version, update the import statement accordingly.

4. **Checkpoint configuration**

   Checkpoint paths may need to be adjusted according to the location of your downloaded model weights.

5. **Hardware requirements**

   GPU execution is recommended for ESM-2 feature extraction and model training. Actual memory requirements depend on sequence lengths, batch sizes, and the selected computational configuration.

## Citation

If you use GLAD in your research, please cite the following work:

**GLAD: Global-Local Attention Difference Learning for Sequence-Based Prediction of Protein Stability Changes.**

Please refer to the associated publication for the complete bibliographic information.

## Contact

For questions regarding the implementation, model weights, or research applications, please refer to the repository or contact the authors.
