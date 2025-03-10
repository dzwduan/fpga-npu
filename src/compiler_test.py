import unittest
import numpy as np
from compiler import npu

class TestAllocationSpace(unittest.TestCase):
	
	def setUp(self):
		# Create a minimal NPU instance for testing
		arch_params = {
			'tiles': 2,
			'dpes': 4,
			'lanes': 4,
			'vrf_depth': 16,
			'mrf_depth': 32,
			'max_tag': 16
		}
		flow_opts = {
			'checkpoint_name': 'test',
			'pac': 0,
			'rtl_sim': 0,
			'perf_sim': 0,
			'verbose': 0,
			'mif_gen': 0,
			'freq': 300,
			'pcie_gen': 0,
			'program_loops': 1
		}
		self.npu_instance = npu(arch_params, flow_opts)
	
	def test_successful_allocation(self):
		# Test allocating space in 'mvu_vrf'
		addr = self.npu_instance.alloc_space('mvu_vrf', 3)
		self.assertEqual(addr, 0)  # First allocation should start at address 0
		# Check that memory space was properly marked
		self.assertEqual(self.npu_instance.mem_space['mvu_vrf'][0], 3)
		self.assertEqual(self.npu_instance.mem_space['mvu_vrf'][1], -1)
		self.assertEqual(self.npu_instance.mem_space['mvu_vrf'][2], -1)

	def test_multiple_allocations(self):
		# First allocation
		addr1 = self.npu_instance.alloc_space('evrf', 2)
		self.assertEqual(addr1, 0)
		
		# Second allocation should start after the first one
		addr2 = self.npu_instance.alloc_space('evrf', 3)
		self.assertEqual(addr2, 2)
		
		# Verify memory space markings
		self.assertEqual(self.npu_instance.mem_space['evrf'][0], 2)
		self.assertEqual(self.npu_instance.mem_space['evrf'][1], -1)
		self.assertEqual(self.npu_instance.mem_space['evrf'][2], 3)
		self.assertEqual(self.npu_instance.mem_space['evrf'][3], -1)
		self.assertEqual(self.npu_instance.mem_space['evrf'][4], -1)

	def test_allocation_failure(self):
		# Fill up the memory space
		addr1 = self.npu_instance.alloc_space('mfu0_add', self.npu_instance.mem_space['mfu0_add'].size - 1)
		self.assertEqual(addr1, 0)
		
		# Try to allocate more space than available
		addr2 = self.npu_instance.alloc_space('mfu0_add', 2)
		self.assertEqual(addr2, -1)  # Should fail and return -1

	def test_mvu_mrf_allocation(self):
		# Test allocation in 'mvu_mrf' which should update mrf_filled_depth
		initial_depth = self.npu_instance.mrf_filled_depth
		addr = self.npu_instance.alloc_space('mvu_mrf', 5)
		
		self.assertEqual(addr, 0)
		self.assertEqual(self.npu_instance.mrf_filled_depth, initial_depth + 5)

	def test_invalid_memory_space(self):
		# Test allocation in a non-existent memory space
		with self.assertRaises(AssertionError):
			self.npu_instance.alloc_space('invalid_space', 1)

	def test_non_contiguous_allocation(self):
		# Create a gap by allocating non-contiguous blocks
		addr1 = self.npu_instance.alloc_space('mfu1_mul', 2)
		self.assertEqual(addr1, 0)
		
		# Manually modify the memory space to create a gap
		self.npu_instance.mem_space['mfu1_mul'][2] = 0  # Skip index 2
		
		addr2 = self.npu_instance.alloc_space('mfu1_mul', 3)
		self.assertEqual(addr2, 3)  # Should allocate after the gap
		
		# Now allocate a size 1 block, should fit in the gap
		addr3 = self.npu_instance.alloc_space('mfu1_mul', 1)
		self.assertEqual(addr3, 2)  # Should use the gap

if __name__ == '__main__':
	unittest.main()