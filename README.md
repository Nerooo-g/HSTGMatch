# HSTGMatch

**Hierarchical Spatial-Temporal Graph-Enhanced Model for Map-Matching**

> Anjun Gao, Zhenglin Wan, Pingfu Chao, and Shunyu Yao
> *Australasian Database Conference 2024*

**This is the official implementation**.

HSTGMatch is a deep learning model for map-matching that aligns GPS trajectories to road segment sequences. It addresses three key challenges: scarce labeled data, ineffective spatial-temporal modeling, and train/test distribution mismatch — through a two-stage pipeline of hierarchical self-supervised pre-training followed by spatial-temporal supervised fine-tuning.

## Architecture

The model comprises three main components:

- **Hierarchical Self-supervised Learning (HSL)** — Pre-trains on unlabeled trajectory data by masking and reconstructing both coarse grid IDs and fine-grained (longitude, latitude) tuples simultaneously, using a shared sentinel token strategy for better generalization.
- **Adaptive Trajectory Adjacency Graph (ATA-Graph) + opt-GATs** — Constructs a proximity-based graph over grid cells and applies optimized Graph Attention Networks (dot-product attention, no concatenation) weighted by local trajectory density.
- **Spatial-Temporal Factors (STF)** — Explicitly encodes distance and time intervals between trajectory points, with a logarithmic decay coefficient that down-weights long-range interval contributions.

These feed into a Transformer-based Seq2Seq model (encoder + decoder with teacher forcing) that outputs a segment-based route.

![HSTGMatch Architecture](main_fig.pdf)

## Results

Evaluated on three Beijing vehicle trajectory datasets (small ~5×5 km, medium ~8×8 km, large ~12×12 km):

| Model       | Beijing-S F1 | Beijing-M F1 | Beijing-L F1 |
|-------------|:------------:|:------------:|:------------:|
| LSTM        | 61.66        | 60.04        | 49.71        |
| ST-RNN      | 63.04        | 61.72        | 52.99        |
| HST-LSTM    | 70.84        | 74.41        | 56.74        |
| DeepMM      | 73.79        | 80.06        | 59.72        |
| TransMM     | 79.82        | 85.67        | 78.17        |
| GraphMM     | 84.82        | 88.30        | 84.84        |
| **HSTGMatch** | **88.59**  | **90.42**    | **89.22**    |

## Repository Structure

```
HSTGMatch/
├── configs/
│   └── default.yaml          # Model and training hyperparameters
├── data/
│   └── format.md             # Data format specification (bring your own data)
├── scripts/
│   ├── pretrain.py           # Stage 1: hierarchical self-supervised pre-training
│   └── train.py              # Stage 2: supervised fine-tuning
├── src/
│   ├── data/
│   │   ├── dataset.py        # Dataset and DataLoader
│   │   └── preprocess.py     # Grid mapping, graph construction, normalization
│   ├── models/
│   │   ├── graph_embedding.py  # ATA-Graph construction + opt-GATs
│   │   ├── encoder.py          # Transformer encoder
│   │   ├── decoder.py          # Attention decoder
│   │   ├── spatial_temporal.py # Spatial-Temporal Factor module
│   │   └── hstgmatch.py        # Full model assembly
│   └── utils/
│       └── metrics.py        # Precision, Recall, F1 computation
├── checkpoints/              # Saved model weights (not tracked by git)
├── requirements.txt
└── LICENSE
```

## Setup

```bash
pip install -r requirements.txt
```

PyTorch and PyTorch Geometric must be installed separately according to your CUDA version. See the [PyG installation guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html).

## Data

The datasets used in the paper are private Beijing vehicle trajectory datasets and cannot be redistributed. To use this code, you will need to supply your own trajectory data or a compatible public dataset (e.g., [T-Drive](https://www.microsoft.com/en-us/research/publication/t-drive-trajectory-data-sample/) or [GeoLife](https://www.microsoft.com/en-us/research/publication/geolife-gps-trajectory-dataset-user-guide/)). See [data/format.md](data/format.md) for the expected input format.

The map is divided into 100×100 meter grids; coordinates and time intervals are normalized with Z-score.

## Usage

### Stage 1 — Pre-training

```bash
python scripts/pretrain.py --config configs/default.yaml --data_dir data/your_dataset
```

Trains the encoder via masked trajectory reconstruction. Checkpoints are saved to `checkpoints/pretrain/`.

### Stage 2 — Supervised fine-tuning

```bash
python scripts/train.py --config configs/default.yaml --data_dir data/your_dataset
```

Loads the pre-trained encoder, attaches the Spatial-Temporal Factor module and Seq2Seq decoder, and fine-tunes end-to-end. Checkpoints are saved to `checkpoints/train/`.

## Configuration

Key hyperparameters in `configs/default.yaml`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `data.grid_size_meters` | 100 | Grid cell size |
| `data.distance_threshold` | 200 | ATA-Graph edge threshold (m) |
| `model.d_model` | 128 | Model hidden dimension |
| `model.n_heads` | 8 | Attention heads |
| `pretrain.k` | 0.5 | Grid CE vs. coord RMSE balance |
| `pretrain.mask_ratio` | 0.15 | Fraction of tokens masked |

## Citation

```bibtex
@inproceedings{gao2024h,
  title={H ierarchical S patial-T emporal G raph-Enhanced Model for Map-Match ing},
  author={Gao, Anjun and Wan, Zhenglin and Chao, Pingfu and Yao, Shunyu},
  booktitle={Australasian Database Conference},
  pages={44--57},
  year={2024},
  organization={Springer}
}
```

## License

See [LICENSE](LICENSE).
