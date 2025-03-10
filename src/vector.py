import numpy as np
import math

"""
The vector class defines the main operands of any NPU program. Each vector has the following parameters:
- name: used for figuring out dependencies, detecting hazards and handling tags
- dimension_x: the real vector length specified by user (without padding)
- space_name: the memory space this vector belongs to (e.g. mvu_vrf)
- alloc_addr: the starting address of this vector
- word_count: the number of memory words this vector occupies in the specified space
- useful_data: the vector data without padding
- data: the vector data with padding to match native dimensions
"""


class Vector:
    def __init__(
        self,
        name,
        dimension_x,
        space_name,
        tiles,
        dpes,
        lanes,
        in_data_type,
        ac_data_type,
        data=[],
    ):
        self.name = name
        self.dimension_x = dimension_x
        self.space_name = space_name
        self.alloc_addr = -1
        # mvu_vrf需要填充后的维度是tiles * landes的整数倍
        if space_name == "mvu_vrf":
            dimension_x_padded = int(
                math.ceil(1.0 * dimension_x / (tiles * lanes)) * tiles * lanes
            )
            self.word_count = int(dimension_x_padded / tiles / lanes)
            self.useful_data = data
            self.data = np.zeros(dimension_x_padded, dtype=in_data_type)
            self.data[: len(self.useful_data)] = self.useful_data
        # eVRF mfu0_vrf0 mfu0_vrf1 mfu1_vrf0 mfu1_vrf1需要根据dimension_x的大小设计不同的填充策略
        # 因为这里只需要进行activation操作
        else:
            # TODO: 根据什么来估的？需要看硬件的代码
            if dimension_x < (dpes * lanes / 10 * 3):
                dimension_x_padded = int(
                    math.ceil(1.0 * dimension_x / (dpes * lanes / 10))
                    * dpes
                    * lanes
                    / 10
                )
            else:
                dimension_x_padded = int(math.ceil(1.0 * dimension_x / dpes) * dpes)

    # This function is used to change the data of a vector. Useful when loading new inputs to the same vector location.
    def change_data(self, data):
        assert len(data) == len(self.useful_data), (
            "Vector length is incompatible with new data"
        )
        self.useful_data = data
        self.data[: len(self.useful_data)] = self.useful_data

    def info(self):
        print(
            "Vector "
            + self.name
            + " , Size: "
            + str(self.dimension_x)
            + " element(s), Mem Space: "
            + self.space_name
            + ", Base Address: "
            + str(self.alloc_addr)
            + ", Word Count: "
            + str(self.word_count)
        )


