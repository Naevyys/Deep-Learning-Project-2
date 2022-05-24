from unittest import TestCase
from src.Layers.relu import ReLU
import torch


class TestReLU(TestCase):
    def test_forward(self):
        x = torch.tensor([-10, -1, -0.5, 0, 0.5, 1, 10])
        expected = torch.relu(x)
        actual = ReLU().forward(x)
        self.assertTrue(torch.allclose(expected, actual))
