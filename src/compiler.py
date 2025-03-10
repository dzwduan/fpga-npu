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
from vector import Vector
from matrix import Matrix

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
        self.NSIZEW = 0  # size width
        self.MRFAW = 0  # mrf address width
        self.VRFAW = 0  # vrf address width
        self.NTILE = 0  # tile num
        self.NVRF = 0  # vrf num
        self.MIW_LD = 0  # macro instruction width for load
        self.MIW_MFU = 0  # macro instruction width for mfu
        self.MIW_EVRF = 0  # macro instruction width for evrf
        self.MIW_MVU = 0  # sets the fields of the MVU macro-instruction
        self.MICW = 0  # micro instruction control width
        self.mvu_minst = 0  # muv micro instruction
        self.evrf_minst = 0  # evrf micro instruction
        self.mfu0_minst = 0  # mfu0 micro instruction
        self.mfu1_minst = 0  # mfu1 micro instruction
        self.ld_minst = 0  # load micro instruction
        self.minst_chain = 0  # micro instruction chain
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
        alloc_addr = -1  # return -1 if allocation failed
        idx = 0
        assert space in self.mem_space.keys(), "Invalid memory space"
        space_depth = self.mem_space[space].size
        # idx 遍历对应space的所有位置
        while idx < space_depth and alloc_addr == -1:
            # 遍历到的位置都是每个block的第一个位置，会记录所分配的大小，不可能是-1
            assert self.mem_space[space][idx] != -1, "Something wrong with allocation"
            # idx位置还没分配
            if self.mem_space[space][idx] == 0:
                can_allocate = True
                # 从idx开始，尝试分配size大小的空间
                for i in range(1, size):
                    # 检查是否越界
                    if idx + i >= space_depth:
                        can_allocate = False
                        idx = space_depth
                        break
                    # 跳过已分配的空间
                    if self.mem_space[space][idx + i] != 0:
                        can_allocate = False
                        idx = idx + i + self.mem_space[space][idx + i]
                        break
                # 可以分配
                if can_allocate:
                    alloc_addr = idx
                    self.mem_space[space][idx] = size
                    # 已分配的第一个位置记录了分配的大小，后面的位置都标记为-1
                    for i in range(1, size):
                        self.mem_space[space][idx + i] = -1
                    # idx = space_depth 仅用于退出while循环
                    idx = space_depth
            # idx位置已分配，mem_space[space][idx]记录了已分配的大小，所以要跳过这个区域
            else:
                idx = idx + self.mem_space[space][idx]
        # TODO：为什么mrf要额外更新计数器？
        if space == "mvu_mrf":
            self.mrf_filled_depth += size

        return alloc_addr

    """
    This function is used for allocating memory for vectors and matrices depending on the dimensions
    and the memory space specified by the user. It is optional to specify data values for the vector.
    """

    def malloc(self, name, dimension_x, dimension_y, space_name, values=[]):
        assert dimension_x > 0, "Invalid dimension"
        assert dimension_y > 0, "Invalid dimension"
        # set number of tiles DPEs lanes
        tiles = self.arch_params["tiles"]
        dpes = self.arch_params["dpes"]
        lanes = self.arch_params["lanes"]

        # make sure the memory space exists
        # 防御性编程，在代码继续执行前，确保space_name是合法的
        warnings.simplefilter(action="ignore", category=FutureWarning)
        assert self.mem_space.get(space_name, "invalid") != "invalid", (
            "Specified memory space does not exist"
        )
        # TODO: dimension_y == None 代表什么？
        if dimension_y == None:
            allocated_mem = Vector(
                name,
                dimension_x,
                space_name,
                tiles,
                dpes,
                lanes,
                self.in_data_type,
                self.ac_data_type,
                values,
            )
            # TODO: temp 仅仅分配一个word_count的空间
            if space_name != "temp":
                allocated_mem.alloc_addr = self.alloc_space(
                    space_name, allocated_mem.word_count
                )
                assert allocated_mem.alloc_addr != -1, "Failed allocate vector " + name
        else:
            assert values != [], "You have to specify matrix data"
            allocated_mem = Matrix(
                name,
                dimension_x,
                dimension_y,
                space_name,
                tiles,
                dpes,
                lanes,
                self.in_data_type,
                values,
            )
            allocated_mem.alloc_addr = self.alloc_space(
                space_name, allocated_mem.word_count
            )
            assert allocated_mem.alloc_addr != -1, "Failed allocate matrix " + name
            # 矩阵寄存器的4维更新 tiles x dpes x mrf_depth x lanes
            # TODO: 如何理解tile_row tile_col
            tile_rows = dimension_y // dpes
            tile_cols = dimension_x // tiles
            #TODO: 为什么主要工作都是在x方向？
            for y in range(allocated_mem.dimension_y_padded):
                for x in range(allocated_mem.dimension_x_padded):
                    # x / tile_cols : 第几个tile
                    # y % dpes : 第几个dpe
                    # allocated_mem.alloc_addr : 起始位置
                    # (x % tile_cols) / lanes : 第几个lane
                    # (y / dpes) * (tile_cols / lanes)当前行在其对应 DPE 上的位置。通过乘法将 DPE 和每个 DPE 处理的列数相结合，确定了在整个内存空间中的偏移量
                    # x % tile_cols : 第几个lane
                    self.mrfs[int(x / tile_cols)][int(y % dpes)][
                        allocated_mem.alloc_addr
                        + int((x % tile_cols) / lanes)
                        + (int(y / dpes) * int(tile_cols / lanes))
                    ][int(x % lanes)] = allocated_mem.data[y][x]

        return allocated_mem

    # This function sets all the parameters for instruction field widths based on those specified by the user
    def set_inst_params(self):
        pass

    # This function puts together the macro-instructions into one VLIW instruction (chain)
    def set_inst(self, inst):
        pass

    # This function sets the fields of the MVU macro-instruction
    def set_mvu_minst(self, inst):
        pass

    # This function sets the fields of the eVRF macro-instruction
    def set_evrf_minst(self, inst):
        pass

    # This function sets the fields of the MFU0 macro-instruction
    def set_mfu0_minst(self, inst):
        pass
