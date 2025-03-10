import math
import numpy as np

"""
The matrix class defines the persistent weight matrices. It has same parameters as the vector class in addition to:
- dimension_y: the M dimension of the matrix as specified by the user (vertical dimension -- without padding)
"""


class Matrix:
    def __init__(
        self,
        name,
        dimension_x,
        dimension_y,
        space_name,
        tiles,
        dpes,
        lanes,
        in_data_type,
        data,
    ):
        self.name = name
        self.dimension_x = dimension_x
        self.dimension_y = dimension_y
        self.space_name = space_name
        if dimension_y < (dpes * (3 * lanes / 10)):
            self.dimension_y_padded = int(
                math.ceil(1.0 * dimension_y / dpes / (3 * lanes / 10))
                * dpes
                * (3 * lanes / 10)
            )
        else:
            self.dimension_y_padded = int(math.ceil(1.0 * dimension_y / dpes) * dpes)
        self.dimension_x_padded = int(
            math.ceil(1.0 * dimension_x / (tiles * lanes)) * tiles * lanes
        )
        # TODO: 如何理解y_padded
        self.word_count = int(self.dimension_x_padded / tiles / lanes) * int(
            self.dimension_y_padded / dpes
        )
        self.useful_data = data
        self.alloc_addr = -1
        # TODO: why y * x matrix ? for transose?
        self.data = np.zeros(
            self.dimension_y_padded * self.dimension_x_padded, dtype=in_data_type
        )
        self.data[: self.useful_data.shpae[0], : self.useful_data.shape[1]] = (
            self.useful_data
        )

        # Print some information about the matrix

    def info(self):
        print(
            "Matrix "
            + self.name
            + " , Size: "
            + str(self.dimension_x)
            + "x"
            + str(self.dimension_y)
            + " element(s), Mem Space: "
            + self.space_name
            + ", Base Address: "
            + str(self.alloc_addr)
            + ", Word Count: "
            + str(self.word_count)
        )
