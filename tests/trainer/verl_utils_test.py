import unittest
from unittest import mock

import torch

from trinity.common.experience import EID, Experience
from trinity.trainer.verl.utils import to_data_proto


class TestToDataProtoRoutedExperts(unittest.TestCase):
    def test_to_data_proto_pads_routed_experts_to_full_sequence(self):
        exp1_routed_experts = torch.tensor(
            [
                [[1, 2], [3, 4]],
                [[5, 6], [7, 8]],
                [[9, 10], [11, 12]],
                [[13, 14], [15, 16]],
            ],
            dtype=torch.uint8,
        )
        exp2_routed_experts = torch.tensor(
            [
                [[21, 22], [23, 24]],
                [[25, 26], [27, 28]],
            ],
            dtype=torch.uint8,
        )
        experiences = [
            Experience(
                eid=EID(batch=1, task=1, run=1, step=1),
                tokens=torch.tensor([10, 11, 12, 13, 14], dtype=torch.int32),
                prompt_length=2,
                routed_experts=exp1_routed_experts,
            ),
            Experience(
                eid=EID(batch=1, task=2, run=1, step=1),
                tokens=torch.tensor([20, 21, 22], dtype=torch.int32),
                prompt_length=1,
                routed_experts=exp2_routed_experts,
            ),
        ]

        batch = to_data_proto(experiences, pad_token_id=0, model=object(), logger=mock.Mock())

        self.assertIn("routed_experts", batch.batch)
        routed_experts = batch.batch["routed_experts"]
        self.assertEqual(routed_experts.dtype, torch.uint8)
        self.assertEqual(tuple(routed_experts.shape), (2, 5, 2, 2))

        expected_exp1 = torch.tensor(
            [
                [[1, 2], [3, 4]],
                [[5, 6], [7, 8]],
                [[9, 10], [11, 12]],
                [[13, 14], [15, 16]],
                [[0, 0], [0, 0]],
            ],
            dtype=torch.uint8,
        )
        expected_exp2 = torch.tensor(
            [
                [[0, 0], [0, 0]],
                [[21, 22], [23, 24]],
                [[25, 26], [27, 28]],
                [[0, 0], [0, 0]],
                [[0, 0], [0, 0]],
            ],
            dtype=torch.uint8,
        )
        self.assertTrue(torch.equal(routed_experts[0], expected_exp1))
        self.assertTrue(torch.equal(routed_experts[1], expected_exp2))
