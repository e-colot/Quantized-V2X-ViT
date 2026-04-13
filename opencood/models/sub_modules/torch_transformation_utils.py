"""
torch_transformation_utils.py
"""
import os

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

def get_roi_and_cav_mask(shape, cav_mask, spatial_correction_matrix,
                         discrete_ratio, downsample_rate):
    """
    Get mask for the combination of cav_mask and rorated ROI mask.
    Parameters
    ----------
    shape : tuple
        Shape of (B, L, H, W, C).
    cav_mask : torch.Tensor
        Shape of (B, L).
    spatial_correction_matrix : torch.Tensor
        Shape of (B, L, 4, 4)
    discrete_ratio : float
        Discrete ratio.
    downsample_rate : float
        Downsample rate.

    Returns
    -------
    com_mask : torch.Tensor
        Combined mask with shape (B, H, W, L, 1).

    """
    B, L, H, W, C = shape
    C = 1
    # (B,L,4,4)
    dist_correction_matrix = get_discretized_transformation_matrix(
        spatial_correction_matrix, discrete_ratio,
        downsample_rate)
    # (B*L,2,3)
    T = get_transformation_matrix(
        dist_correction_matrix.reshape(-1, 2, 3), (H, W))
    # (B,L,1,H,W)
    roi_mask = get_rotated_roi((B, L, C, H, W), T)
    # (B,L,1,H,W)
    com_mask = combine_roi_and_cav_mask(roi_mask, cav_mask)
    # (B,H,W,1,L)
    com_mask = com_mask.permute(0,3,4,2,1)
    return com_mask


def combine_roi_and_cav_mask(roi_mask, cav_mask):
    """
    Combine ROI mask and CAV mask

    Parameters
    ----------
    roi_mask : torch.Tensor
        Mask for ROI region after considering the spatial transformation/correction.
    cav_mask : torch.Tensor
        Mask for CAV to remove padded 0.

    Returns
    -------
    com_mask : torch.Tensor
        Combined mask.
    """
    # (B, L, 1, 1, 1)
    cav_mask = cav_mask.unsqueeze(2).unsqueeze(3).unsqueeze(4)
    # (B, L, C, H, W)
    cav_mask = cav_mask.expand(roi_mask.shape)
    # (B, L, C, H, W)
    com_mask = roi_mask * cav_mask
    return com_mask


def get_rotated_roi(shape, correction_matrix):
    """
    Get rorated ROI mask.

    Parameters
    ----------
    shape : tuple
        Shape of (B,L,C,H,W).
    correction_matrix : torch.Tensor
        Correction matrix with shape (N,2,3).

    Returns
    -------
    roi_mask : torch.Tensor
        Roated ROI mask with shape (N,2,3).

    """
    B, L, C, H, W = shape
    # To reduce the computation, we only need to calculate the
    # mask for the first channel.
    # (B,L,1,H,W)
    x = torch.ones((B, L, 1, H, W), dtype=correction_matrix.dtype, device=correction_matrix.device)
    # (B*L,1,H,W)
    roi_mask = warp_affine(x.reshape(-1, 1, H, W), correction_matrix,
                           dsize=(H, W), mode="bilinear") # was mode='nearest'
    # going from 'nearest' to 'bilinear' lost 0.02% AP @ IOU 0.7 (0% elesewhere)
    # -> acceptable loss

    # (B,L,C,H,W)
    roi_mask = torch.repeat_interleave(roi_mask, C, dim=1).reshape(B, L, C, H,
                                                                   W)
    return roi_mask


def get_discretized_transformation_matrix(matrix, discrete_ratio,
                                          downsample_rate):
    """
    Get disretized transformation matrix.
    Parameters
    ----------
    matrix : torch.Tensor
        Shape -- (B, L, 4, 4) where B is the batch size, L is the max cav
        number.
    discrete_ratio : float
        Discrete ratio.
    downsample_rate : float/int
        downsample_rate

    Returns
    -------
    matrix : torch.Tensor
        Output transformation matrix in 2D with shape (B, L, 2, 3),
        including 2D transformation and 2D rotation.

    """
    matrix = matrix[:, :, [0, 1], :][:, :, :, [0, 1, 3]]
    # normalize the x,y transformation
    matrix[:, :, :, -1] = matrix[:, :, :, -1] \
                          / (discrete_ratio * downsample_rate)

    return matrix.type(dtype=torch.float)

def _3x3_cramer_inverse(input):
    r"""
    Helper function to compute the inverse of a Bx3x3 matrix using Cramer's
    rule. 
    Args:
        input : torch.Tensor
            Tensor to be inversed.

    Returns:
        out : torch.Tensor
            Inversed Tensor.
    """
    a11, a12, a13 = input[..., 0, 0], input[..., 0, 1], input[..., 0, 2]
    a21, a22, a23 = input[..., 1, 0], input[..., 1, 1], input[..., 1, 2]
    a31, a32, a33 = input[..., 2, 0], input[..., 2, 1], input[..., 2, 2]

    det = (a11 * (a22 * a33 - a23 * a32) -
           a12 * (a21 * a33 - a23 * a31) +
           a13 * (a21 * a32 - a22 * a31))

    res_row1 = torch.stack([
        (a22 * a33 - a23 * a32),
        -(a12 * a33 - a13 * a32),
        (a12 * a23 - a13 * a22)
    ], dim=-1)
    res_row2 = torch.stack([
        -(a21 * a33 - a23 * a31),
        (a11 * a33 - a13 * a31),
        -(a11 * a23 - a13 * a21)
    ], dim=-1)
    res_row3 = torch.stack([
        (a21 * a32 - a22 * a31),
        -(a11 * a32 - a12 * a31),
        (a11 * a22 - a12 * a21)
    ], dim=-1)

    adjugate = torch.stack([res_row1, res_row2, res_row3], dim=-2)
    
    return adjugate / (det + 1e-12 * torch.sign(det)).unsqueeze(-1).unsqueeze(-1)


def _torch_inverse_cast(input):
    r"""
    Helper function to make torch.inverse work with other than fp32/64.
    The function torch.inverse is only implemented for fp32/64 which makes
    impossible to be used by fp16 or others. What this function does,
    is cast input data type to fp32, apply torch.inverse,
    and cast back to the input dtype.
    Args:
        input : torch.Tensor
            Tensor to be inversed.

    Returns:
        out : torch.Tensor
            Inversed Tensor.

    """
    Warning('Behavior changed, only computes Nx3x3 matrices. If it corresponds ' \
    'to your usecase, use _3x3_cramer_inverse() instead')
    infp32 = input.float()
    outfp32 = _3x3_cramer_inverse(infp32)
    return outfp32.type_as(input)


def normal_transform_pixel(
        height, width, device, dtype, eps=1e-14):
    r"""
    Compute the normalization matrix from image size in pixels to [-1, 1].
    Args:
        height : int
            Image height.
        width : int
            Image width.
        device : torch.device
            Output tensor devices.
        dtype : torch.dtype
            Output tensor data type.
        eps : float
            Epsilon to prevent divide-by-zero errors.

    Returns:
        tr_mat : torch.Tensor
            Normalized transform with shape :math:`(1, 3, 3)`.
    """
    # prevent divide by zero bugs
    # width_denom = eps if width == 1 else width - 1.0
    # height_denom = eps if height == 1 else height - 1.0

    # slight difference, allows to remove conditional statement
    width_denom = width - 1.0 + eps
    height_denom = height - 1.0 + eps

    el_0_0 = torch.tensor(2.0/width_denom, device=device, dtype=dtype)
    el_1_1 = torch.tensor(2.0/height_denom, device=device, dtype=dtype)
    
    row0 = torch.stack([el_0_0, torch.tensor(0.0, device=device, dtype=dtype), torch.tensor(-1.0, device=device, dtype=dtype)])
    row1 = torch.stack([torch.tensor(0.0, device=device, dtype=dtype), el_1_1, torch.tensor(-1.0, device=device, dtype=dtype)])
    row2 = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=dtype)

    # Stack into a 3x3
    tr_mat = torch.stack([row0, row1, row2], dim=0)

    return tr_mat.unsqueeze(0)  # unsqueeze to 1x3x3


def eye_like(n, B, device, dtype):
    r"""
    Return a 2-D tensor with ones on the diagonal and
    zeros elsewhere with the same batch size as the input.
    Args:
        n : int
            The number of rows :math:`(n)`.
        B : int
            Btach size.
        device : torch.device
            Devices of the output tensor.
        dtype : torch.dtype
            Data type of the output tensor.

    Returns:
       The identity matrix with the shape :math:`(B, n, n)`.
    """

    identity = torch.eye(n, device=device, dtype=dtype)
    return identity[None].repeat(B, 1, 1)


def normalize_homography(dst_pix_trans_src_pix, dsize_src, dsize_dst=None):
    r"""
    Normalize a given homography in pixels to [-1, 1].
    Args:
        dst_pix_trans_src_pix : torch.Tensor
            Homography/ies from source to destination to be normalized with
            shape :math:`(B, 3, 3)`.
        dsize_src : Tuple[int, int]
            Size of the source image (height, width).
        dsize_dst : Tuple[int, int]
            Size of the destination image (height, width).

    Returns:
        dst_norm_trans_src_norm : torch.Tensor
            The normalized homography of shape :math:`(B, 3, 3)`.
    """
    if dsize_dst is None:
        dsize_dst = dsize_src
    # source and destination sizes
    src_h, src_w = dsize_src
    dst_h, dst_w = dsize_dst
    device = dst_pix_trans_src_pix.device
    dtype = dst_pix_trans_src_pix.dtype
    # compute the transformation pixel/norm for src/dst
    src_norm_trans_src_pix = normal_transform_pixel(src_h, src_w, device,
                                                    dtype).to(
        dst_pix_trans_src_pix)

    src_pix_trans_src_norm = _3x3_cramer_inverse(src_norm_trans_src_pix)
    dst_norm_trans_dst_pix = normal_transform_pixel(dst_h, dst_w, device,
                                                    dtype).to(
        dst_pix_trans_src_pix)
    # compute chain transformations
    dst_norm_trans_src_norm: torch.Tensor = dst_norm_trans_dst_pix @ (
            dst_pix_trans_src_pix @ src_pix_trans_src_norm)
    return dst_norm_trans_src_norm


def get_rotation_matrix2d(M, dsize):
    r"""
    Return rotation matrix for torch.affine_grid based on transformation matrix.
    Args:
        M : torch.Tensor
            Transformation matrix with shape :math:`(B, 2, 3)`.
        dsize : Tuple[int, int]
            Size of the source image (height, width).

    Returns:
        R : torch.Tensor
            Rotation matrix with shape :math:`(B, 2, 3)`.
    """
    H, W = dsize
    B = M.shape[0]
    center = torch.Tensor([W / 2, H / 2]).to(M.dtype).to(M.device).unsqueeze(0)
    shift_m = eye_like(3, B, M.device, M.dtype)
    shift_m[:, :2, 2] = center

    shift_m_inv = eye_like(3, B, M.device, M.dtype)
    shift_m_inv[:, :2, 2] = -center

    rotat_m = eye_like(3, B, M.device, M.dtype)
    rotat_m[:, :2, :2] = M[:, :2, :2]
    affine_m = shift_m @ rotat_m @ shift_m_inv
    return affine_m[:, :2, :]  # Bx2x3


def get_transformation_matrix(M, dsize):
    r"""
    Return transformation matrix for torch.affine_grid.
    Args:
        M : torch.Tensor
            Transformation matrix with shape :math:`(N, 2, 3)`.
        dsize : Tuple[int, int]
            Size of the source image (height, width).

    Returns:
        T : torch.Tensor
            Transformation matrix with shape :math:`(N, 2, 3)`.
    """
    T = get_rotation_matrix2d(M, dsize)
    T[..., 2] += M[..., 2]
    return T


def convert_affinematrix_to_homography(A):
    r"""
    Convert to homography coordinates
    Args:
        A : torch.Tensor
            The affine matrix with shape :math:`(B,2,3)`.

    Returns:
        H : torch.Tensor
            The homography matrix with shape of :math:`(B,3,3)`.
    """
    H: torch.Tensor = torch.nn.functional.pad(A, [0, 0, 0, 1], "constant",
                                              value=0.0)
    H[..., -1, -1] += 1.0
    return H


def _reflect_coordinates(coord, low, high):
    """Reflect coordinates into [low, high] using mirror boundary rules."""
    if high <= low:
        return torch.full_like(coord, low)
    span = high - low
    coord = torch.abs(coord - low)
    extra = torch.remainder(coord, 2 * span)
    reflected = torch.where(extra > span, 2 * span - extra, extra)
    return reflected + low


def _gather_from_hw(src, x_idx, y_idx):
    """Gather src[:, :, y_idx, x_idx] for dense index maps."""
    B, C, H, W = src.shape
    H_out, W_out = x_idx.shape[1], x_idx.shape[2]
    linear_idx = (y_idx * W + x_idx).view(B, 1, -1).expand(-1, C, -1)
    src_flat = src.reshape(B, C, H * W)

    # tensorRT struggles with:
    # out = torch.gather(src_flat, 2, linear_idx)
    # -> split it in different steps

    batch_size, seq_len, _ = src_flat.shape
    device = src_flat.device
    
    # Create 1D indices
    # We use .view() to place them in the correct dimensions for broadcasting
    i = torch.arange(batch_size, device=device).view(batch_size, 1, 1) # (B, 1, 1)
    j = torch.arange(seq_len, device=device).view(1, seq_len, 1)    # (1, L, 1)

    # linear_idx is likely (B, L, K)
    # When we index with i and j, PyTorch broadcasts them to match linear_idx
    out = src_flat[i, j, linear_idx]

    return out.view(B, C, H_out, W_out)


def affine_grid_sample_approx(src, theta, dsize,
                              mode='bilinear',
                              padding_mode='zeros',
                              align_corners=True):
    """
    Approximate affine_grid + grid_sample without calling either function.
    """
    if mode not in ('bilinear', 'nearest'):
        raise ValueError(f"Unsupported mode: {mode}")
    if padding_mode not in ('zeros', 'border', 'reflection'):
        raise ValueError(f"Unsupported padding_mode: {padding_mode}")

    B, C, H, W = src.shape
    H_out, W_out = dsize
    device = src.device
    grid_dtype = theta.dtype

    if align_corners:
        # linspace struggles in the tensorRT conversion (due to its dynamic range)
        # ys = torch.linspace(-1.0, 1.0, H_out, device=device, dtype=grid_dtype)
        # xs = torch.linspace(-1.0, 1.0, W_out, device=device, dtype=grid_dtype)
        # torch.arrange lies closer to the HW computations

        # if H_out > 1:
        #     step_y = 2.0 / (H_out - 1)
        #     ys = torch.arange(H_out, device=device, dtype=grid_dtype) * step_y - 1.0
        # else:
        #     ys = torch.zeros(1, device=device, dtype=grid_dtype) # Handle edge case

        # if W_out > 1:
        #     step_x = 2.0 / (W_out - 1)
        #     xs = torch.arange(W_out, device=device, dtype=grid_dtype) * step_x - 1.0
        # else:
        #     xs = torch.zeros(1, device=device, dtype=grid_dtype)

        eps = 1e-14

        denom_y = (H_out - 1.0) + eps
        step_y = 2.0 / denom_y
        ys = torch.arange(H_out, device=device, dtype=grid_dtype) * step_y - 1.0

        denom_x = (W_out - 1.0) + eps
        step_x = 2.0 / denom_x
        xs = torch.arange(W_out, device=device, dtype=grid_dtype) * step_x - 1.0

    else:
        ys = (2.0 * (torch.arange(H_out, device=device, dtype=grid_dtype) + 0.5)
              / H_out - 1.0)
        xs = (2.0 * (torch.arange(W_out, device=device, dtype=grid_dtype) + 0.5)
              / W_out - 1.0)

    # tensorRT struggles with meshgrid
    # grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
    # Use explicit broadcasting shapes:
    grid_y = ys.view(-1, 1)  # Shape: (H_out, 1)
    grid_x = xs.view(1, -1)  # Shape: (1, W_out)
    # expand to 2D grids
    grid_y = grid_y.expand(H_out, W_out)
    grid_x = grid_x.expand(H_out, W_out)

    ones = torch.ones_like(grid_x)
    base_grid = torch.stack((grid_x, grid_y, ones), dim=-1)
    base_grid = base_grid.unsqueeze(0).expand(B, -1, -1, -1)

    norm_grid = torch.einsum('bij,bhwj->bhwi', theta, base_grid)
    x = norm_grid[..., 0]
    y = norm_grid[..., 1]

    if align_corners:
        x = (x + 1) * (W - 1) / 2
        y = (y + 1) * (H - 1) / 2
        reflect_x_low, reflect_x_high = 0.0, W - 1.0
        reflect_y_low, reflect_y_high = 0.0, H - 1.0
    else:
        x = ((x + 1) * W - 1) / 2
        y = ((y + 1) * H - 1) / 2
        reflect_x_low, reflect_x_high = -0.5, W - 0.5
        reflect_y_low, reflect_y_high = -0.5, H - 0.5

    if padding_mode == 'border':
        x = x.clamp(0, W - 1)
        y = y.clamp(0, H - 1)
    elif padding_mode == 'reflection':
        x = _reflect_coordinates(x, reflect_x_low, reflect_x_high).clamp(0, W - 1)
        y = _reflect_coordinates(y, reflect_y_low, reflect_y_high).clamp(0, H - 1)

    src_work = src
    if src_work.dtype != x.dtype:
        src_work = src_work.to(x.dtype)

    # unused (changed get_rotated_roi to use mode='bilinear')
    if mode == 'nearest':
        xn = torch.round(x)
        yn = torch.round(y)
        valid = (xn >= 0) & (xn <= W - 1) & (yn >= 0) & (yn <= H - 1)
        x_idx = xn.clamp(0, W - 1).long()
        y_idx = yn.clamp(0, H - 1).long()
        out = _gather_from_hw(src_work, x_idx, y_idx)
        if padding_mode == 'zeros':
            out = out * valid.unsqueeze(1).to(out.dtype)
        return out

    x0 = torch.floor(x)
    y0 = torch.floor(y)
    x1 = x0 + 1
    y1 = y0 + 1

    wa = (x1 - x) * (y1 - y)
    wb = (x1 - x) * (y - y0)
    wc = (x - x0) * (y1 - y)
    wd = (x - x0) * (y - y0)

    # tensorRT struggles with bitwise operations
    # x0_valid = (x0 >= 0) & (x0 <= W - 1)
    # x1_valid = (x1 >= 0) & (x1 <= W - 1)
    # y0_valid = (y0 >= 0) & (y0 <= H - 1)
    # y1_valid = (y1 >= 0) & (y1 <= H - 1)
    # multiplying by a mask is equivalent
    x0_valid = (x0 >= 0) * (x0 <= W - 1)
    x1_valid = (x1 >= 0) * (x1 <= W - 1)
    y0_valid = (y0 >= 0) * (y0 <= H - 1)
    y1_valid = (y1 >= 0) * (y1 <= H - 1)

    if padding_mode == 'zeros':
        # tensorRT struggles with bitwise operations
        # wa = wa * (x0_valid & y0_valid).to(wa.dtype)
        # wb = wb * (x0_valid & y1_valid).to(wb.dtype)
        # wc = wc * (x1_valid & y0_valid).to(wc.dtype)
        # wd = wd * (x1_valid & y1_valid).to(wd.dtype)
        # multiplying by a mask is equivalent

        wa = wa * (x0_valid.to(wa.dtype) * y0_valid.to(wa.dtype))
        wb = wb * (x0_valid.to(wb.dtype) * y1_valid.to(wb.dtype))
        wc = wc * (x1_valid.to(wc.dtype) * y0_valid.to(wc.dtype))
        wd = wd * (x1_valid.to(wd.dtype) * y1_valid.to(wd.dtype))

    x0_idx = x0.clamp(0, W - 1).long()
    y0_idx = y0.clamp(0, H - 1).long()
    x1_idx = x1.clamp(0, W - 1).long()
    y1_idx = y1.clamp(0, H - 1).long()

    Ia = _gather_from_hw(src_work, x0_idx, y0_idx)
    Ib = _gather_from_hw(src_work, x0_idx, y1_idx)
    Ic = _gather_from_hw(src_work, x1_idx, y0_idx)
    Id = _gather_from_hw(src_work, x1_idx, y1_idx)

    out = (Ia * wa.unsqueeze(1) +
           Ib * wb.unsqueeze(1) +
           Ic * wc.unsqueeze(1) +
           Id * wd.unsqueeze(1))
    return out


def warp_affine(
        src, M, dsize,
        mode='bilinear',
        padding_mode='zeros',
        align_corners=True):
    r"""
    Transform the src based on transformation matrix M.
    Args:
        src : torch.Tensor
            Input feature map with shape :math:`(B,C,H,W)`.
        M : torch.Tensor
            Transformation matrix with shape :math:`(B,2,3)`.
        dsize : tuple
            Tuple of output image H_out and W_out.
        mode : str
            Interpolation methods for F.grid_sample.
        padding_mode : str
            Padding methods for F.grid_sample.
        align_corners : boolean
            Parameter of F.affine_grid.

    Returns:
        Transformed features with shape :math:`(B,C,H,W)`.
    """

    B, C, H, W = src.size()

    # we generate a 3x3 transformation matrix from 2x3 affine
    M_3x3 = convert_affinematrix_to_homography(M)
    dst_norm_trans_src_norm = normalize_homography(M_3x3, (H, W), dsize)

    # src_norm_trans_dst_norm = torch.inverse(dst_norm_trans_src_norm)
    src_norm_trans_dst_norm = _3x3_cramer_inverse(dst_norm_trans_src_norm)
    src_for_sample = src.half() if src_norm_trans_dst_norm.dtype == torch.half else src
    return affine_grid_sample_approx(src_for_sample,
                                     src_norm_trans_dst_norm[:, :2, :],
                                     dsize,
                                     mode=mode,
                                     padding_mode=padding_mode,
                                     align_corners=align_corners)


class Test:
    """
    Test the transformation in this file.
    The methods in this class are not supposed to be used outside of this file.
    """

    def __init__(self):
        pass

    @staticmethod
    def load_img():
        torch.manual_seed(0)
        x = torch.randn(1, 5, 16, 400, 200) * 100
        # x = torch.ones(1, 5, 16, 400, 200)
        return x

    @staticmethod
    def load_raw_transformation_matrix(N):
        a = 90 / 180 * np.pi
        matrix = torch.Tensor([[np.cos(a), -np.sin(a), 10],
                               [np.sin(a), np.cos(a), 10]])
        matrix = torch.repeat_interleave(matrix.unsqueeze(0).unsqueeze(0), N,
                                         dim=1)
        return matrix

    @staticmethod
    def load_raw_transformation_matrix2(N, alpha):
        a = alpha / 180 * np.pi
        matrix = torch.Tensor([[np.cos(a), -np.sin(a), 0, 0],
                               [np.sin(a), np.cos(a), 0, 0]])
        matrix = torch.repeat_interleave(matrix.unsqueeze(0).unsqueeze(0), N,
                                         dim=1)
        return matrix

    @staticmethod
    def test():
        img = Test.load_img()
        B, L, C, H, W = img.shape
        raw_T = Test.load_raw_transformation_matrix(5)
        T = get_transformation_matrix(raw_T.reshape(-1, 2, 3), (H, W))
        img_rot = warp_affine(img.reshape(-1, C, H, W), T, (H, W))
        print(img_rot[0, 0, :, :])
        plt.matshow(img_rot[0, 0, :, :])
        plt.show()

    @staticmethod
    def test_combine_roi_and_cav_mask():
        B = 2
        L = 5
        C = 16
        H = 300
        W = 400
        # 2, 5
        cav_mask = torch.Tensor([[1, 1, 1, 0, 0], [1, 0, 0, 0, 0]])
        x = torch.zeros(B, L, C, H, W)
        correction_matrix = Test.load_raw_transformation_matrix2(5, 10)
        correction_matrix = torch.cat([correction_matrix, correction_matrix],
                                      dim=0)
        mask = get_roi_and_cav_mask((B, L, H, W, C), cav_mask, 
                                    correction_matrix, 0.4, 4)
        plt.matshow(mask[0, :, :, 0, 0])
        plt.show()


if __name__ == "__main__":
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    Test.test_combine_roi_and_cav_mask()
