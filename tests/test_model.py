import unittest

import torch
import numpy as np

from datasets.dvlog import FRAME_COUNT, preprocess_video
from models.msgnet import MSGNet, MultiScaleGraph


class MSGNetTest(unittest.TestCase):
    def test_video_preprocessing_matches_fixed_length_protocol(self):
        video = np.arange(10 * 161, dtype=np.float32).reshape(10, 161)
        video[:2] = 0.0
        processed = preprocess_video(video)
        self.assertEqual(processed.shape, (FRAME_COUNT, 161))
        self.assertEqual(processed.dtype, np.float32)

        invalid_video = np.zeros((5, 161), dtype=np.float32)
        self.assertIsNone(preprocess_video(invalid_video))

    def test_forward_shape_and_padding_invariance(self):
        torch.manual_seed(7)
        model = MSGNet(dropout=0.0).eval()
        features = torch.randn(2, 4, 60, 161)
        lengths = torch.tensor([4, 2])
        features[1, 2:] = 0.0

        with torch.no_grad():
            logits = model((features, lengths, ["a", "b"]))
            padded = torch.cat(
                [features, torch.zeros(2, 2, 60, 161)], dim=1
            )
            padded_logits = model((padded, lengths, ["a", "b"]))

        self.assertEqual(tuple(logits.shape), (2, 1))
        self.assertTrue(torch.allclose(logits, padded_logits, atol=1e-5))

    def test_global_adjacency_is_symmetric_without_self_loops(self):
        graph = MultiScaleGraph(1, 8, 16)
        adjacency = graph._global_adjacency().detach()
        self.assertTrue(torch.allclose(adjacency, adjacency.transpose(0, 1)))
        self.assertEqual(torch.count_nonzero(torch.diag(adjacency)).item(), 0)


if __name__ == "__main__":
    unittest.main()
