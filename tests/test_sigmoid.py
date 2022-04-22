from unittest import TestCase
from src.Layers.sigmoid import Sigmoid
import torch


class TestSigmoid(TestCase):
    def test_forward(self):
        x = torch.tensor([-10, -1, -0.5, 0, 0.5, 1, 10])
        expected = torch.sigmoid(x)
        actual = Sigmoid().forward(x)
        self.assertTrue(torch.allclose(expected, actual))
