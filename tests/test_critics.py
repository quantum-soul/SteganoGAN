# -*- coding: utf-8 -*-

from unittest import TestCase
from unittest.mock import Mock, call, patch

import torch

from steganogan import critics
from tests.utils import assert_called_with_tensors


class TestBasicCritic(TestCase):

    class TestCritic(critics.BasicCritic):
        def __init__(self):
            pass

    def setUp(self):
        self.test_critic = self.TestCritic()

    @patch('steganogan.critics.nn.Conv2d', autospec=True)
    def test__covn2d(self, conv2d_mock):
        """Conv2d must be called with given args and kernel_size=3 and padding=1"""

        # run
        result = self.test_critic._conv2d(2, 4)

        # asserts
        assert result == conv2d_mock.return_value
        conv2d_mock.assert_called_once_with(
            in_channels=2,
            out_channels=4,
            kernel_size=3,
        )

    @patch('steganogan.critics.nn.Sequential')
    @patch('steganogan.critics.nn.BatchNorm2d')
    @patch('steganogan.critics.nn.Conv2d')
    def test___init__(self, conv2d_mock, batchnorm2d_mock, sequential_mock):
        """Test that conv2d and batchnorm are called when creating a new critic with hidden_size"""

        # setup
        hidden_size = 2

        expected_conv2d_calls = [
            call(in_channels=3, out_channels=2, kernel_size=3),
            call(in_channels=2, out_channels=2, kernel_size=3),
            call(in_channels=2, out_channels=2, kernel_size=3),
            call(in_channels=2, out_channels=1, kernel_size=3)
        ]

        expected_batch_calls = [call(2), call(2), call(2)]

        # run
        critic = critics.BasicCritic(hidden_size)

        # assert
        assert conv2d_mock.call_args_list == expected_conv2d_calls
        assert batchnorm2d_mock.call_args_list == expected_batch_calls

    def test_forward(self):

        """Test the return value of method forward"""
        # setup
        test_critic = self.TestCritic()

        layer1 = Mock(return_value=torch.Tensor([[5, 6], [7, 8]]))
        test_critic._models = layer1

        image = torch.Tensor([[1, 2], [3, 4]])

        call_1 = call(torch.Tensor([[1, 2], [3, 4]]))

        expected = torch.Tensor([[5, 6], [7, 8]])

        expected = torch.mean(expected.view(expected.size(0), -1), dim=1)

        # run
        result = test_critic.forward(image)

        # assert
        assert (result == expected).all()
        assert_called_with_tensors(layer1, [call_1])

    def test_upgrade_legacy(self):
        """Test that upgrade legacy works, must set _models to layers"""
        # setup
        self.test_critic.layers = Mock(return_value=torch.Tensor([[5, 6], [7, 8]]))
        expected_version = '1'

        # run
        self.test_critic.upgrade_legacy()

        # assert
        assert self.test_critic.VERSION == expected_version
        assert self.test_critic._models == self.test_critic.layers

