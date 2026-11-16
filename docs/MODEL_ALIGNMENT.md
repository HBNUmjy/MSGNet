# Model Alignment with the Manuscript

This document records the correspondence between the manuscript description and the released implementation.

| Manuscript component | Implementation |
| --- | --- |
| Frame feature dimension: 161 | `MSGNet(input_dim=161)` |
| Latent graph nodes: 16 | `MSGNet(graph_nodes=16)` |
| Global graph: sigmoid, symmetrization, zero diagonal | `MultiScaleGraph._global_adjacency` |
| Dynamic graph: channel mean, temporal centering, L2 normalization, positive Gram matrix, zero diagonal | `MultiScaleGraph._dynamic_adjacency` |
| Learned scalar graph mixing coefficient | `MultiScaleGraph.mix_logit` |
| One self-loop followed by symmetric degree normalization | `MultiScaleGraph._normalize_adjacency` |
| Parallel 0-, 1-, and 2-hop propagation with channel-wise scales | `MultiScaleGraph.forward` |
| 1x1 fusion, batch normalization, ReLU | `MultiScaleGraph.fusion` |
| Gated depthwise temporal convolution, kernel 1x9 | `GatedTemporalConv` |
| Three STBlocks with widths 1 -> 32 -> 64 -> 128 | `MSGNet.video_encoder` |
| Node attention followed by temporal attention | `HierarchicalAttentionPool` |
| Fully connected layer, ReLU, dropout before user modeling | `MSGNet.video_projection` |
| BiGRU, 64 hidden units per direction | `UserBiGRU` |
| Layer normalization and dropout | `UserBiGRU.norm`, `UserBiGRU.dropout` |
| Masked additive attention over videos | `MaskedAttentionPool` |
| Binary user-level logit | `MSGNet.classifier` |
| Focal loss, alpha 0.75, gamma 2.0 | `FocalLoss` and `train.py` |
| Adam, constant learning rate 1e-5 | `train.py`, `config.yaml` |
| Gradient clipping, maximum norm 1.0 | `train.py` |
| 100 epochs, minimum validation-loss checkpoint | `config.yaml`, `train.py` |
| Validation-selected F1 threshold | `select_threshold` in `train.py` |
| Prefix-only early detection with fixed model and threshold | `early_detection.py` |

## Corrections Applied Before Release

The pre-release folder differed from the manuscript in several material ways. The release version corrects the following:

- The global adjacency is now symmetrized and has its diagonal removed before self-loops are added.
- Dynamic adjacency also removes its diagonal before graph fusion.
- Graph propagation now contracts the source-node dimension and writes to the destination-node dimension.
- The gated temporal module no longer adds an internal residual in addition to the STBlock residual.
- Packed BiGRU sequences prevent padded videos from affecting backward recurrent states.
- Training uses Adam with a constant learning rate instead of AdamW with cosine scheduling.
- Default batch size and training duration are changed to 6 and 100 epochs.
- Early detection uses the validation-selected threshold for every chronological prefix.
- Dataset preprocessing now implements the frame filtering and polynomial resampling described in the manuscript.
