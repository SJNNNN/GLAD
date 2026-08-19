# GLAD

**GLAD: Global-Local Attention Difference Learning for Sequence-Based Prediction of Protein Stability Changes**

## Overview

GLAD is a protein stability change prediction framework based on ESM-2 embeddings, combining a global branch and a local attention branch to model mutation-induced sequence differences.

## Environment

- Python 3.8.19
- PyTorch 2.2.2
- DGL 2.3.0+cu121
- NetworkX 3.1
- NumPy 1.24.4
- scikit-learn 1.3.2
- tqdm 4.67.1

Additional runtime dependencies used in the code:

- `esm`
- `pandas`
- `scipy`

## Project Structure

- `train.py`: training entry
- `test.py`: evaluation entry
- `zero_multi_predict.py`: zero-shot multi-mutation prediction
- `datasets/`: dataset and dataloader utilities
- `model/`: GLAD model definition
- `processor/`: training / testing logic
- `utils/`: seed and logging helpers
- `ddgdata/`: training and test CSV files
- `checkpoint/`: saved model weights

## Data

The main training/evaluation CSV files used by the current scripts are:

- `ddgdata/S8754_parsed.csv`
- `ddgdata/S669_with_name.csv`

Typical columns include:

- `name`
- `wt_seq`
- `mut_seq`
- `ddg` / `ddG`
- `pos`

For zero-shot multi-mutation prediction, `zero_multi_predict.py` expects:

- `wt_seq`
- `mut_seq`
- `ddG`

## Usage

### Train

```bash
python train.py
```

### Test

```bash
python test.py
```

### Zero-shot multi-mutation prediction

```bash
python zero_multi_predict.py
```

## Notes

- The model uses `esm.pretrained.esm2_t33_650M_UR50D()`, so first-time download may take time.
- Current scripts reference `solver.py`; make sure it exists in your local environment.
- `processor/__init__.py` imports `do_train1`; if that file is not present, adjust the import before running.
- Checkpoint paths may need to be updated to match your local saved weights.

## Citation

If you use this code, please cite the paper:

**GLAD: Global-Local Attention Difference Learning for Sequence-Based Prediction of Protein Stability Changes**
