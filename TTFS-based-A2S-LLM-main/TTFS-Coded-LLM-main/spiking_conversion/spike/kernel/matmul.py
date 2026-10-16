import math

import torch
import triton
import triton.language as tl
from torch import Tensor


@triton.jit
def spiking_matmul_kernel(
        # Matrix pointers
        A_ptr,
        B_ptr,
        C_ptr,
        # Matrix dimensions
        B: int,
        M: int,
        N: int,
        K: int,
        # Others
        max_repr: float,
        # Matrix stride
        stride_ab: int,
        stride_am: int,
        stride_ak: int,
        stride_bb: int,
        stride_bk: int,
        stride_bn: int,
        stride_cb: int,
        stride_cm: int,
        stride_cn: int,
        # Meta-parameters
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
        BLOCK_SIZE_K: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr,
        precision: tl.constexpr,
):
    # Generally, we divide input matrix A into groups of rows and B into groups of columns.
    # The size of group is decided by BLOCK_SIZE_M and BLOCK_SIZE_N respectively.

    batch_id = tl.program_id(0)
    A_ptr += batch_id * stride_ab
    B_ptr += batch_id * stride_bb
    C_ptr += batch_id * stride_cb

    pid = tl.program_id(1)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    m_id = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    n_id = (pid % num_pid_in_group) // group_size_m

    offs_am = (m_id * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (n_id * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = A_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the K dimension.
        # If it is out of bounds, set it to 0.
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        buffer = tl.clamp(tl.log2(tl.abs(a) + 1)[:, :, None] + tl.log2(tl.abs(b) + 1)[None, :, :], 0, max_repr)
        if precision > 0:
            buffer /= max_repr
            buffer = tl.floor(buffer * precision + 0.5) / precision
            buffer *= max_repr
        buffer = tl.exp2(buffer)
        buffer -= tl.abs(a)[:, :, None]
        buffer -= tl.abs(b)[None, :, :]
        buffer -= 1
        buffer = tl.where((a > 0)[:, :, None] ^ (b > 0)[None, :, :], -buffer, buffer)
        # We accumulate along the K dimension.
        accumulator += tl.sum(buffer, axis=1)
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    offs_cm = m_id * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = n_id * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = C_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


def spiking_matmul_core(A: Tensor,
                        B: Tensor,
                        max_abs: float,
                        precision: int,
                        ) -> Tensor:
    r"""
    
    :param A:
    :type A: Tensor
    :param B:
    :type B: Tensor
    :param max_abs:
    :type max_abs: float
    :param precision:
    :type precision: int
    :return:
    :rtype: Tensor
    """
    batch_shape = list(A.shape[:-2])
    M, K, N = A.shape[-2], A.shape[-1], B.shape[-1]
    A = A.reshape((-1, M, K))
    B = B.reshape((-1, K, N))
    batch_size = A.shape[0]
    C = torch.empty((batch_size, M, N), device="cuda")
    # Hyper-parameters
    BLOCK_SIZE_M = 32
    BLOCK_SIZE_N = 32
    BLOCK_SIZE_K = 16
    max_repr = 2 * math.log2(max_abs + 1)
    # Kernel
    grid = lambda META: (
        batch_size,
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )
    spiking_matmul_kernel[grid](
        A,
        B,
        C,
        batch_size,
        M,
        N,
        K,
        max_repr,
        A.stride(0),
        A.stride(1),
        A.stride(2),
        B.stride(0),
        B.stride(1),
        B.stride(2),
        C.stride(0),
        C.stride(1),
        C.stride(2),
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
        BLOCK_SIZE_K,
        8,
        precision,
    )
    return C.reshape(batch_shape + [M, N])
