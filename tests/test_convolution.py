from unittest import TestCase
from mock import patch
import torch
from src.Layers.convolution import Conv2d


class TestConv2d(TestCase):

    @patch('torch.randn')
    def test_forward(self, torch_randn_patch):
        x = torch.tensor([  # (2, 2, 3, 4) -> batch size 2, 2 channels, height 3, width 4
            [
                [
                    [-1, 0, 1, 2],  # Pattern 1
                    [0, 1, 2, -1],
                    [1, 1, 1, 1],
                ],
                [
                    [-2, 1, 0, 1],  # Pattern 2
                    [0, 1, 2, -1],
                    [1, -1, 1, 0],
                ],
            ],
            [
                [
                    [-2, 1, 0, 1],  # Pattern 2
                    [0, 1, 2, -1],
                    [1, -1, 1, 0],
                ],
                [
                    [0, 1, 2, 3],  # Pattern 3
                    [0, 1, 2, -1],
                    [-1, -1, -1, -1],
                ],
            ]
        ])

        kernel_size = 2
        out_channels = 3
        in_channels = x.size()[1]
        bias = False
        torch_randn_patch.side_effect = torch.tensor([
            # Patching random initialisation of weights w when instantiating a Conv2d object
            # I.e. we manually set the kernel values such that we can compute the expected outcome of the convolution
            # (3, 2, 2) -> 3 out_channels, kernel_size 2 by 2
            [
                [1, 1],  # 2
                [1, 1],
            ],
            [
                [0, 0],
                [0, 0],
            ],
            [
                [-2, -1],
                [0, 3],
            ],
        ])

        expected = torch.tensor([  # (2, 3, 2, 3) -> batch size 2,  3 out channels, height 2, width 3
            [
                [
                    [0, 8, 6],
                    [4, 8, 5],
                ],
                [
                    [0, 0, 0],
                    [0, 0, 0],
                ],
                [
                    [11, 9, -7],  # [5+6, 5+4, -3-4]
                    [-2, 2, -3],  # [2-4, 3-1, 0-3]
                ],
            ],
            [
                [
                    [2, 10, 8],
                    [0, 4, 1],
                ],
                [
                    [0, 0, 0],
                    [0, 0, 0],
                ],
                [
                    [4, 6, -14],  # [6-2, 4+2, -4-10]
                    [-8, -8, -9],  # [-4-4, -1-7, -3-6]
                ],
            ],
        ])

        conv2d = Conv2d(in_channels, out_channels, kernel_size, bias=bias)
        actual = conv2d.forward(x)

        self.assertTrue(expected == actual)
