import triton
import triton.language as tl

import torch
import torch.nn as nn


@triton.jit
def inplace_softmax_kernel(
    ptr,
    row_stride: int,
    col_stride: int,
    n_cols: int,
    BLOCK_SIZE: tl.constexpr,
):
    # Get row pointers
    row_id = tl.program_id(0)
    row_ptr = ptr + row_id * row_stride
    offset_col = tl.arange(0, BLOCK_SIZE) * col_stride
    ptrs = row_ptr + offset_col
    # Load
    mask = offset_col < n_cols
    row = tl.load(ptrs, mask=mask, other=-float("inf"))
    row = row - tl.max(row, axis=0)
    log_sum_exp = tl.log(tl.sum(tl.exp(row)))
    log_softmax = row - log_sum_exp
    softmax_output = tl.exp(log_softmax)
    # Store
    tl.store(ptrs, softmax_output, mask=mask)


def inplace_softmax_core(x: torch.Tensor):
    shape = x.shape
    x = x.flatten(0, -2)
    BLOCK_SIZE = triton.next_power_of_2(x.shape[1])
    inplace_softmax_kernel[(x.shape[0],)](
        x,
        x.stride(0),
        x.stride(1),
        x.shape[1],
        BLOCK_SIZE,
    )
    x = x.reshape(shape)
