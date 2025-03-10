import math
import numpy as np
import warnings
import pickle
import sys
import copy
import subprocess
import os
import threading
import time
import re

# from fsim import chain
# from fsim import npu_isa_sim


class NPU:
    def __init__(self, arch_params, flow_opts):
        self.arch_params = arch_params
        self.flow_opts = flow_opts
        self.in_data_type = np.int8
        # accumulator data type
        self.ac_data_type = np.int32

        self.mem_space = {
            "mvu_vrf": np.zeros(arch_params["vrf_depth"], dtype=int),
            "mvu_mrf": np.zeros(arch_params["mrf_depth"], dtype=int),
            "evrf": np.zeros(arch_params["vrf_depth"], dtype=int),
            "mfu0_add": np.zeros(arch_params["vrf_depth"], dtype=int),
            "mfu0_mul": np.zeros(arch_params["vrf_depth"], dtype=int),
            "mfu1_add": np.zeros(arch_params["vrf_depth"], dtype=int),
            "mfu1_mul": np.zeros(arch_params["vrf_depth"], dtype=int),
        }

        # TODO:
        self.highest_tag_so_far = 0
        # TODO:
        self.mrf_filled_depth = 0

        # Architecture states

        # mrf = tile x dpes x mrf_depth x lane
        self.mrfs = np.zeros(
            (
                self.arch_params["tiles"],
                self.arch_params["dpes"],
                self.arch_params["mrf_depth"],
                self.arch_params["lanes"],
            ),
            dtype=self.in_data_type,
        )
        # tiles x vrf_depth x lane
        self.vrfs = np.zeros(
            (
                self.arch_params["tiles"],
                self.arch_params["vrf_depth"],
                self.arch_params["lanes"],
            ),
            dtype=self.in_data_type,
        )
        # vrf_depth x lanes
        self.ext_vrf = np.zeros(
            (self.arch_params["vrf_depth"], self.arch_params["lanes"]),
            dtype=self.in_data_type,
        )
        # mfu0_vrf0 : vrf_depth x lanes
        self.mfu0_vrf0 = np.zeros(
            (self.arch_params["vrf_depth"], self.arch_params["lanes"]),
            dtype=self.in_data_type,
        )
        # mfu0_vrf1 : vrf_depth x lanes
        self.mfu0_vrf1 = np.zeros(
            (self.arch_params["vrf_depth"], self.arch_params["lanes"]),
            dtype=self.in_data_type,
        )
        # mfu1_vrf0 : vrf_depth x lanes
        self.mfu1_vrf0 = np.zeros(
            (self.arch_params["vrf_depth"], self.arch_params["lanes"]),
            dtype=self.in_data_type,
        )
        # mfu1_vrf1 : vrf_depth x lanes
        self.mfu1_vrf1 = np.zeros(
            (self.arch_params["vrf_depth"], self.arch_params["lanes"]),
            dtype=self.in_data_type,
        )

		# Instruction, input and golden output queues
        self.inst_q = []
        self.ibuf_q = []     
        self.golden_obuf_q = []
        self.fsim = None

		# Instruction field width parameters
        self.NTAGW = 0  # tag width
        self.NSIZEW = 0 # size width
        self.MRFAW = 0  # mrf address width
        self.VRFAW = 0  # vrf address width
        self.NTILE = 0  # tile num
        self.NVRF = 0   # vrf num
        self.MIW_LD = 0     # macro instruction width for load
        self.MIW_MFU = 0    # macro instruction width for mfu
        self.MIW_EVRF = 0   # macro instruction width for evrf
        self.MIW_MVU = 0    # macro instruction width for mvu
        self.MICW = 0       # micro instruction control width
        self.mvu_minst = 0  # muv micro instruction
        self.evrf_minst = 0 # evrf micro instruction
        self.mfu0_minst = 0 # mfu0 micro instruction
        self.mfu1_minst = 0 # mfu1 micro instruction
        self.ld_minst = 0   # load micro instruction
        self.minst_chain = 0 # micro instruction chain
        self.operands = [] 
        self.unsupported_layers = [] 
        self.ops = 0 

	# This function is used to allocate memory of a specific number of words (size) in a specific memory space.
	# It returns the start address of the allocated memory or -1 if allocation failed.
    def alloc_space(self, space, size):
        """
        idx : 当前搜索的起始位置
        i : 从idx位置开始搜索的偏移值
        mem_space[space][idx+i]: 该非空闲位置存储的值，即已分配区域的大小
            0 : 未分配，即空闲区域
            n : 已分配出去长度为n的区域
            -1: 该位置不可用
        """
        alloc_addr = -1 # return -1 if allocation failed
        idx = 0
        assert space in self.mem_space.keys(), "Invalid memory space"
        space_depth = self.mem_space[space].size # 应该是各个size累加和

		# Start from address 0 in the specified memory space and look for the first empty contiguous location of the required size
        # 该设计是为了简化，实际上应该是一个更复杂的内存管理系统
        while idx < space_depth and alloc_addr == -1:
            assert self.mem_space[space][idx] != -1, "Something wrong with allocation"
            if self.mem_space[space][idx] == 0:
                can_allocate = True
                for i in range(1, size):
                    # 检查现有的空间能否满足分配
                    if idx + i >= space_depth:
                        can_allocate = False
                        idx = space_depth  #TODO:
                        break
                    # 上面检查是否满足分配条件，来这说明没满足，所以要跳过一个已分配的区域
                    if self.mem_space[space][idx + i] != 0:
                        can_allocate = False
                        idx = idx + i +  self.mem_space[space][idx + i]
                        break
                if can_allocate:
                    alloc_addr = idx
                    self.mem_space[space][idx] = size
                    for i in range(1, size):
                        self.mem_space[space][idx + i] = -1
                    idx = space_depth

            else:
                idx = idx + self.mem_space[space][idx]
        #TODO：为什么mrf要额外更新计数器？
        if (space == 'mvu_mrf'):
            self.mrf_filled_depth += size

        return alloc_addr
