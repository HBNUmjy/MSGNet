# MSGNet

Official implementation of **MSGNet**, a multi-scale dynamic graph network for longitudinal multimodal depression-risk detection from short-video histories.

## Architecture

MSGNet preserves the frame-video-user hierarchy:

1. Each 161-dimensional acoustic-visual frame feature is projected to 16 latent graph nodes.
2. A global graph and a video-specific dynamic graph are fused with a learned scalar coefficient.
3. Parallel zero-, first-, and second-order graph propagation is followed by gated depthwise temporal convolution.
4. Node attention and temporal attention produce a 128-dimensional video embedding.
5. A bidirectional GRU with 64 hidden units per direction models the chronological video sequence.
6. Masked video attention and a linear layer produce one user-level logit.

## Data

The MUD3 dataset is not distributed with this repository. Prepare a directory with:

```text
MUD3_final/
├── dep_feat.pkl
├── nondep_feat.pkl
└── labels.csv
```


Preprocessing follows the manuscript: all-zero frames are removed, videos with at most five valid frames are discarded, short videos are polynomially resampled to 60 frames, long videos are truncated to the first 60 frames, and each user history is limited to 360 videos.

## Installation

Python 3.11 is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

On Windows PowerShell, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

Install a CUDA-enabled PyTorch build separately when required by the target system.

## Training

```bash
python train.py --data_dir ./MUD3_final --device cuda
```

## Tests

```bash
python -m unittest discover -s tests -v
```

## Citation

If this code is useful in your research, please cite the accompanying manuscript:

```bibtex
@article{ma2026msgnet,
  title   = {Longitudinal Multimodal Depression-Risk Detection via Sample-Adaptive Graph Learning and Cross-Video Sequence Aggregation},
  author  = {Ma, Junyi and Cao, Yue and Wan, Meng and Liu, Shang},
  year    = {2026}
}
```
