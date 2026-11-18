import unittest

import torch

from vsgmfss_fortuna.MetricNet import backwarp_tenGrid, prepare_backwarp_grid


class BackwarpGridCacheTests(unittest.TestCase):
    def setUp(self):
        backwarp_tenGrid.clear()

    def test_reuses_grid_for_same_shape_device_and_dtype(self):
        flow = torch.zeros((1, 2, 8, 12), dtype=torch.float32)
        first = prepare_backwarp_grid(flow)
        second = prepare_backwarp_grid(flow)

        self.assertIs(first, second)
        self.assertEqual(first.device, flow.device)
        self.assertEqual(first.dtype, flow.dtype)

    def test_separates_dtypes_for_same_shape(self):
        flow32 = torch.zeros((1, 2, 8, 12), dtype=torch.float32)
        flow16 = torch.zeros((1, 2, 8, 12), dtype=torch.float16)

        grid32 = prepare_backwarp_grid(flow32)
        grid16 = prepare_backwarp_grid(flow16)

        self.assertIsNot(grid32, grid16)
        self.assertEqual(grid32.dtype, torch.float32)
        self.assertEqual(grid16.dtype, torch.float16)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_separates_devices_for_same_shape_and_dtype(self):
        flow_cpu = torch.zeros((1, 2, 8, 12), dtype=torch.float32)
        flow_cuda = flow_cpu.cuda()

        grid_cpu = prepare_backwarp_grid(flow_cpu)
        grid_cuda = prepare_backwarp_grid(flow_cuda)

        self.assertIsNot(grid_cpu, grid_cuda)
        self.assertEqual(grid_cpu.device.type, "cpu")
        self.assertEqual(grid_cuda.device.type, "cuda")


if __name__ == "__main__":
    unittest.main()
