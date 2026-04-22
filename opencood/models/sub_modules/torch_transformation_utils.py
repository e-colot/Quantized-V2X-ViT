"""
torch_transformation_utils.py
"""
import torch

def get_roi_and_cav_mask(input: torch.Tensor, cav_mask: torch.Tensor, 
                         spatial_correction_matrix: torch.Tensor,
                         discrete_ratio: torch.Tensor, downsample_rate: torch.Tensor):
    """
    Get mask for the combination of cav_mask and rotated ROI mask.
    Parameters
    ----------
    input: torch.Tensor
        input of shape (L, H, W, C).
        only its shape is used
    cav_mask: torch.Tensor
        Shape of (L).
    spatial_correction_matrix: torch.Tensor
        Shape of (M, 4, 4) where M is the max cav
    discrete_ratio: torch.Tensor (scalar float)
        Discrete ratio.
    downsample_rate: torch.Tensor (scalar float)
        Downsample rate.

    Returns
    -------
    com_mask: torch.Tensor
        Combined mask with shape (H, W, 1, L).

    """

    spatial_size = torch.tensor([input.shape[1], input.shape[2]], device='cuda', dtype=torch.int32)
    # spatial_size = [H, W]

    dist_correction_matrix = get_discretized_transformation_matrix(spatial_correction_matrix, 
                                                                   discrete_ratio,downsample_rate)
    # dist_correction_matrix: (M, 2, 3)
    
    transformation_matrix = get_transformation_matrix(dist_correction_matrix, spatial_size)
    # transformation_matrix: (M, 2, 3)
    
    input_slice = torch.narrow(input, 3, 0, 1)
    # input_slice: (L, H, W, 1)
    
    input_reordered = input_slice.permute(0, 3, 1, 2)
    # input_reordered: (L, 1, H, W)

    roi_mask = get_rotated_roi(input_reordered, spatial_size, transformation_matrix)
    # roi_mask: (L, 1, H, W)

    com_mask = combine_roi_and_cav_mask(roi_mask, cav_mask)
    # com_mask: (L, 1, H, W)
    
    com_mask = com_mask.permute(2, 3, 1, 0)
    # (H, W, 1, L)
    return com_mask


def combine_roi_and_cav_mask(roi_mask, cav_mask):
    """
    Combine ROI mask and CAV mask

    Parameters
    ----------
    roi_mask: torch.Tensor
        Mask for ROI region after considering the spatial transformation/correction.
        shape: (L, C, H, W)
    cav_mask: torch.Tensor
        Mask for CAV to remove padded 0.
        shape: (L)

    Returns
    -------
    com_mask: torch.Tensor
        Combined mask, shape (L, C, H, W)
    """
    L, C, H, W = roi_mask.shape[0], roi_mask.shape[1], roi_mask.shape[2], roi_mask.shape[3]

    cav_mask = cav_mask.unsqueeze(1).unsqueeze(2).unsqueeze(3)
    # (L, 1, 1, 1)
    
    cav_mask = cav_mask.expand(L, C, H, W)
    # (L, C, H, W)
    
    com_mask = roi_mask * cav_mask
    # (L, C, H, W)
    return com_mask


def get_rotated_roi(input: torch.Tensor, spatial_size: torch.Tensor, transformation_matrix: torch.Tensor) -> torch.Tensor:
    """
    Get rotated ROI mask.
    Parameters
    ----------
    input: torch.Tensor
        input of shape (L, 1, H, W).
        only its shape is used
    spatial_size: torch.Tensor
        [H, W], same as from input
    transformation_matrix: torch.Tensor
        Transformation matrix with shape (M, 2, 3).

    Returns
    -------
    roi_mask: torch.Tensor
        Rotated ROI mask with shape (L, 1, H, W).
    """

    # To reduce the computation, we only need to calculate the
    # mask for the first channel.
    x = torch.ones(input.shape, device='cuda', dtype=input.dtype)
    # (L, 1, H, W)

    roi_mask = warp_affine(x, transformation_matrix, dsize=spatial_size, mode="bilinear")
    # (L, 1, H, W)

    roi_mask = torch.repeat_interleave(roi_mask, 1, dim=0).reshape(input.shape)
    # (L, 1, H, W)
    return roi_mask


def get_discretized_transformation_matrix(spatial_correction_matrix: torch.Tensor, discrete_ratio: torch.Tensor,
                                          downsample_rate: torch.Tensor):
    """
    Get discretized transformation matrix.
    Parameters
    ----------
    spatial_correction_matrix: torch.Tensor
        Shape -- (M, 4, 4) where M is the max cav number.
    discrete_ratio: torch.Tensor (discrete scalar)
        Discrete ratio.
    downsample_rate: torch.Tensor (discrete scalar)
        downsample_rate
        Warning: previously supported both int and float

    Returns
    -------
    matrix: torch.Tensor
        Output transformation matrix in 2D with shape (M, 2, 3),
        including 2D transformation and 2D rotation.

    """
    scale = (discrete_ratio * downsample_rate).to(torch.float32)

    spatial_correction_matrix = torch.narrow(spatial_correction_matrix, 1, 0, 2).to(torch.float32)
    # spatial_correction_matrix: (M, 2, 4)

    col_0 = torch.narrow(spatial_correction_matrix, 2, 0, 2)
    # col_0: (M, 2, 2)
    
    col_3 = torch.narrow(spatial_correction_matrix, 2, 3, 1) / scale
    # col_3: (M, 2, 1)

    dist_correction_matrix = torch.cat([col_0, col_3], dim=2)
    # dist_correction_matrix: (M, 2, 3)
    return dist_correction_matrix

def _3x3_cramer_inverse(input: torch.Tensor):
    r"""
    Helper function to compute the inverse of a (M, 3, 3) matrix using Cramer's
    rule. 
    Args:
        input: torch.Tensor
            Tensor to be inversed.

    Returns:
        out: torch.Tensor
            Inversed Tensor.
    """
    # Row 1
    a11 = input.select(-2, 0).select(-1, 0)
    a12 = input.select(-2, 0).select(-1, 1)
    a13 = input.select(-2, 0).select(-1, 2)

    # Row 2
    a21 = input.select(-2, 1).select(-1, 0)
    a22 = input.select(-2, 1).select(-1, 1)
    a23 = input.select(-2, 1).select(-1, 2)

    # Row 3
    a31 = input.select(-2, 2).select(-1, 0)
    a32 = input.select(-2, 2).select(-1, 1)
    a33 = input.select(-2, 2).select(-1, 2)

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
    
    eps = torch.tensor(1e-12, dtype=input.dtype, device=input.device)
    return adjugate / (det + eps * torch.sign(det)).unsqueeze(-1).unsqueeze(-1)


def normal_transform_pixel(
        height: torch.tensor, width: torch.tensor, device: torch.device, dtype: torch.dtype):
    r"""
    Compute the normalization matrix from image size in pixels to [-1, 1].
    Args:
        height: torch.tensor (scalar)
            Image height.
        width: torch.tensor (scalar)
            Image width.
        device: torch.device
            Output tensor devices.
        dtype: torch.dtype
            Output tensor data type.
        eps: float
            Epsilon to prevent divide-by-zero errors.

    Returns:
        tr_mat: torch.Tensor
            Normalized transform with shape (1, 3, 3).
    """
    eps = torch.tensor(1e-14, dtype=dtype, device=device)
    zero = torch.tensor(0.0, dtype=dtype, device=device)
    neg_one = torch.tensor(-1.0, dtype=dtype, device=device)
    one = torch.tensor(1.0, dtype=dtype, device=device)
    two = torch.tensor(2.0, dtype=dtype, device=device)

    width_denom = width - one + eps
    height_denom = height - one + eps

    el_0_0 = two / width_denom
    el_1_1 = two / height_denom
    
    row0 = torch.stack([el_0_0.squeeze(), zero, neg_one])
    row1 = torch.stack([zero, el_1_1.squeeze(), neg_one])
    row2 = torch.stack([zero, zero, one])

    # Stack into a 3x3
    tr_mat = torch.stack([row0, row1, row2], dim=0)

    return tr_mat.unsqueeze(0)  # unsqueeze to 1x3x3


def normalize_homography(dst_pix_trans_src_pix: torch.Tensor, dsize_src: torch.Tensor, dsize_dst: torch.Tensor):
    r"""
    Normalize a given homography in pixels to [-1, 1].
    Args:
        dst_pix_trans_src_pix: torch.Tensor
            Homography/ies from source to destination to be normalized with
            shape (M, 3, 3).
        dsize_src: torch.Tensor
            Size of the source image (height, width).
        dsize_dst: torch.Tensor
            Size of the destination image (height, width).

    Returns:
        dst_norm_trans_src_norm: torch.Tensor
            The normalized homography of shape (M, 3, 3).
    """
    # source and destination sizes
    src_h, src_w = dsize_src[0], dsize_src[1]
    dst_h, dst_w = dsize_dst[0], dsize_dst[1]
    device = 'cuda'
    dtype = dst_pix_trans_src_pix.dtype

    src_norm_trans_src_pix = normal_transform_pixel(src_h, src_w, device,
                                        dtype).to(dst_pix_trans_src_pix)
    # src_norm_trans_src_pix: (1, 3, 3)

    src_pix_trans_src_norm = _3x3_cramer_inverse(src_norm_trans_src_pix)

    dst_norm_trans_dst_pix = normal_transform_pixel(dst_h, dst_w, device,
                                                    dtype).to(dst_pix_trans_src_pix)
    # dst_norm_trans_dst_pix: (1, 3, 3)
    
    # compute chain transformations
    dst_norm_trans_src_norm: torch.Tensor = dst_norm_trans_dst_pix @ (
            dst_pix_trans_src_pix @ src_pix_trans_src_norm)
    return dst_norm_trans_src_norm


def get_rotation_matrix2d(dist_correction_matrix: torch.Tensor, spatial_size: torch.Tensor):
    r"""
    Return rotation matrix for torch.affine_grid based on transformation matrix.
    Args:
        dist_correction_matrix: torch.Tensor
            Transformation matrix with shape :math:`(M, 2, 3)`.
        dsize: torch.Tensor
            Size of the source image [H, W].

    Returns:
        R: torch.Tensor
            Rotation matrix with shape :math:`(M, 2, 3)`.
    """
    device = 'cuda'
    dtype = dist_correction_matrix.dtype
    B = dist_correction_matrix.shape[0]

    H = spatial_size[0].to(dtype)
    W = spatial_size[1].to(dtype)
    two = torch.tensor(2.0, dtype=dtype, device=device)

    cx = W / two
    cy = H / two
    cx_b = cx.expand(B)
    cy_b = cy.expand(B)

    ones = torch.ones_like(cx_b)
    zeros = torch.zeros_like(cx_b)

    r0 = torch.stack([ones, zeros, cx_b], dim=1)
    r1 = torch.stack([zeros, ones, cy_b], dim=1)
    r2 = torch.stack([zeros, zeros, ones], dim=1)
    shift_m = torch.stack([r0, r1, r2], dim=1)

    # 3. Construct shift_m_inv functionally
    r0_inv = torch.stack([ones, zeros, -cx_b], dim=1)
    r1_inv = torch.stack([zeros, ones, -cy_b], dim=1)
    shift_m_inv = torch.stack([r0_inv, r1_inv, r2], dim=1)

    # 4. Construct rotat_m functionally
    # Extract the 2x2 rotation from M
    rot_22 = torch.narrow(torch.narrow(dist_correction_matrix, 1, 0, 2), 2, 0, 2)
    # Row 0: [m00, m01, 0]
    # Row 1: [m10, m11, 0]
    # Row 2: [0,   0,   1]
    r0_rot = torch.cat([torch.select(rot_22, 1, 0), zeros.unsqueeze(1)], dim=1)
    r1_rot = torch.cat([torch.select(rot_22, 1, 1), zeros.unsqueeze(1)], dim=1)
    rotat_m = torch.stack([r0_rot, r1_rot, r2], dim=1)

    # 5. Matrix Multiply
    affine_m = torch.bmm(torch.bmm(shift_m, rotat_m), shift_m_inv)
    
    return torch.narrow(affine_m, 1, 0, 2)


def get_transformation_matrix(dist_correction_matrix, spatial_size: torch.Tensor):
    r"""
    Return transformation matrix for torch.affine_grid.
    Args:
        dist_correction_matrix: torch.Tensor
            Transformation matrix with shape :math:`(M, 2, 3)`.
        spatial_size: torch.Tensor
            Size of the source image [H, W].

    Returns:
        transformation_matrix: torch.Tensor
            Transformation matrix with shape :math:`(M, 2, 3)`.
    """
    transformation_matrix = get_rotation_matrix2d(dist_correction_matrix, spatial_size)
    # transformation_matrix: (M, 2, 3)

    T_3 = transformation_matrix.select(2, 2)
    M_3 = dist_correction_matrix.select(2, 2)
    T_3.add_(M_3)          # In-place addition modifies the original T
    return transformation_matrix


def convert_affine_matrix_to_homography(input: torch.Tensor):
    r"""
    Convert to homography coordinates
    Args:
        input: torch.Tensor
            The affine matrix with shape (M, 2, 3).

    Returns:
        H: torch.Tensor
            The homography matrix with shape (M, 3, 3).
    """
    M = input.shape[0]

    # Create the row [0.0, 0.0, 1.0]: (M, 1, 3)
    new_row = torch.tensor([[[0.0, 0.0, 1.0]]], device='cuda', dtype=input.dtype)
    new_row = new_row.expand(M, -1, -1)
    
    # Concatenate along the height dimension (dim=1)
    H = torch.cat([input, new_row], dim=1)
    return H


def _reflect_coordinates(coord: torch.Tensor, low: torch.Tensor, high: torch.Tensor):
    r"""
    Reflect coordinates into [low, high] using mirror boundary rules.
    Args:
        coord: torch.Tensor
            input coordinates
        low: torch.Tensor (scalar)
            lower boundary
        high: torch.Tensor (scalar)
            upper rboundary
    Returns:
        R: torch.Tensor
            the reflected coordinates using mirror boundaries.
    """
    if high <= low:
        return torch.full_like(coord, low)
    span = high - low
    coord = torch.abs(coord - low)
    extra = torch.remainder(coord, 2 * span)
    reflected = torch.where(extra > span, 2 * span - extra, extra)
    R = reflected + low
    return R


def _gather_from_hw(src: torch.Tensor, x_idx: torch.Tensor, y_idx: torch.Tensor):
    r"""
    Gather src[:, :, y_idx, x_idx] for dense index maps.
    Args:
        src: torch.Tensor
            source matrix, shape (B, C, H, W)
        x_idx: torch.Tensor
            index on 4th dimension
        y_idx: torch.Tensor
            index on the 3rd dimension
    Returns:
        out: torch.Tensor
            slice of src
    """
    W = torch.tensor(src.shape[3], dtype=torch.int32, device=src.device)
    B = int(src.shape[0])
    C = int(src.shape[1])
    linear_idx = (y_idx * W + x_idx).clamp(min=0).view(B, 1, -1).expand(-1, C, -1)

    src_flat = src.flatten(2)
    # (B, C, H, W) -> (B, C, H*W)
    device = 'cuda'
    
    # Create 1D indices
    # We use .view() to place them in the correct dimensions for broadcasting
    i = torch.arange(B, device=device, dtype=torch.int32).view(B, 1, 1)
    j = torch.arange(C, device=device, dtype=torch.int32).view(1, C, 1)

    # linear_idx is likely (B, L, K)
    # When we index with i and j, PyTorch broadcasts them to match linear_idx
    out = src_flat[i, j, linear_idx]

    return out.reshape(src.shape)


def _affine_grid_sample_approx_prepare_norm_grid(theta: torch.Tensor, dsize: torch.Tensor, align_corners: bool = True):
    r"""
    Build a normalized sampling grid for affine resampling.
    Args:
        theta: torch.Tensor
            Affine transformation matrix with shape (B, 2, 3).
        dsize: torch.Tensor
            Output spatial size tensor [H, W].
        align_corners: bool
            Whether to align grid endpoints to image corners.

    Returns:
        norm_grid: torch.Tensor
            Normalized grid with shape (B, H, W, 2).
    """
    device = 'cuda'
    grid_dtype = theta.dtype

    H_int = dsize[0]   # stays int32
    W_int = dsize[1]   # stays int32
    H_out = H_int.to(grid_dtype)   # float32, for arithmetic only
    W_out = W_int.to(grid_dtype)

    eps = torch.tensor(1e-14, device=device, dtype=grid_dtype)
    one = torch.tensor(1.0, device=device, dtype=grid_dtype)
    two = torch.tensor(2.0, device=device, dtype=grid_dtype)
    half = torch.tensor(0.5, device=device, dtype=grid_dtype)

    if align_corners:
        step_y = two / (H_out - one + eps)
        ys = torch.arange(H_int, device=device, dtype=grid_dtype) * step_y - one

        step_x = two / (W_out - one + eps)
        xs = torch.arange(W_int, device=device, dtype=grid_dtype) * step_x - one
    else:
        ys = (two * (torch.arange(H_int, device=device, dtype=grid_dtype) + half)
              / H_out - one)
        xs = (two * (torch.arange(W_int, device=device, dtype=grid_dtype) + half)
              / W_out - one)

    grid_y = ys.view(-1, 1)
    grid_x = xs.view(1, -1)

    ones = torch.ones(dsize[0], dsize[1], device=device, dtype=grid_dtype)
    base_grid = torch.stack((
        grid_x.expand(dsize[0], dsize[1]).contiguous(),
        grid_y.expand(dsize[0], dsize[1]).contiguous(),
        ones), dim=-1)

    flat_grid = base_grid.reshape(-1, 3)
    flat_grid = flat_grid.unsqueeze(0).expand(theta.shape[0], -1, -1).contiguous()
    norm_grid = torch.bmm(flat_grid, theta.transpose(1, 2))
    norm_grid = norm_grid.reshape(theta.shape[0], base_grid.size(0), base_grid.size(1), 2)
    return norm_grid


def clamp_tensor(src:torch.Tensor, min:torch.Tensor, max:torch.Tensor):
    r"""
    Clamp tensor values to an inclusive numeric range.
    Args:
        src: torch.Tensor
            Input tensor.
        min: torch.Tensor
            Lower bound scalar/tensor broadcastable to src.
        max: torch.Tensor
            Upper bound scalar/tensor broadcastable to src.

    Returns:
        out: torch.Tensor
            Clamped tensor with same shape as src.
    """
    out = torch.where(src < min, min, src)
    return torch.where(out > max, max, out)


def _affine_grid_sample_approx_bilinear_sample(
        src: torch.Tensor,
        norm_grid: torch.Tensor,
        dsize: torch.Tensor,
        dtype: torch.dtype,
        padding_mode: str = 'zeros',
        align_corners: bool = True):
    r"""
    Sample source features from a normalized grid using bilinear interpolation.
    Args:
        src: torch.Tensor
            Source feature map with shape (L, C, H, W).
        norm_grid: torch.Tensor
            Normalized sampling grid with shape (B, H, W, 2).
        dsize: torch.Tensor 
            Output spatial dimensions as (H, W).
        dtype: torch.dtype
            Computation dtype for coordinate transforms and weights.
        padding_mode: str
            Out-of-bound handling mode: ``zeros``, ``border`` or ``reflection``.
        align_corners: bool
            Whether normalization assumes aligned corner points.

    Returns:
        out: torch.Tensor
            Resampled tensor with shape (L, C, H, W).
    """
    
    device = src.device # Best practice: use the source tensor's device
    
    x = torch.select(norm_grid, -1, 0)
    y = torch.select(norm_grid, -1, 1)

    H = dsize[0].to(dtype)
    W = dsize[1].to(dtype)
    
    zero = torch.tensor(0.0, device=device, dtype=dtype)
    one = torch.tensor(1.0, device=device, dtype=dtype)
    two = torch.tensor(2.0, device=device, dtype=dtype)
    half = torch.tensor(0.5, device=device, dtype=dtype)

    if align_corners:
        x = (x + one) * (W - one) / two
        y = (y + one) * (H - one) / two
        reflect_x_low, reflect_x_high = zero, W - one
        reflect_y_low, reflect_y_high = zero, H - one
    else:
        x = ((x + one) * W - one) / two
        y = ((y + one) * H - one) / two
        reflect_x_low, reflect_x_high = -half, W - half
        reflect_y_low, reflect_y_high = -half, H - half

    W_minus_one = W - one
    H_minus_one = H - one

    if padding_mode == 'border':
        x = clamp_tensor(x, zero, W_minus_one)
        y = clamp_tensor(y, zero, H_minus_one)
    elif padding_mode == 'reflection':
        # Not taken
        x = clamp_tensor(_reflect_coordinates(x, reflect_x_low, reflect_x_high), zero, W_minus_one)
        y = clamp_tensor(_reflect_coordinates(y, reflect_y_low, reflect_y_high), zero, H_minus_one)

    x0 = torch.floor(x)
    y0 = torch.floor(y)
    x1 = x0 + one
    y1 = y0 + one

    wa = (x1 - x) * (y1 - y)
    wb = (x1 - x) * (y - y0)
    wc = (x - x0) * (y1 - y)
    wd = (x - x0) * (y - y0)
    
    # x0_valid = torch.where(x0 >= zero, torch.where(x0 <= W_minus_one, one, zero), zero)
    # x1_valid = torch.where(x1 >= zero, torch.where(x1 <= W_minus_one, one, zero), zero)
    # y0_valid = torch.where(y0 >= zero, torch.where(y0 <= H_minus_one, one, zero), zero)
    # y1_valid = torch.where(y1 >= zero, torch.where(y1 <= H_minus_one, one, zero), zero)
    # To avoid duplicate nodes in graph
    x0_valid = torch.where(x0 >= zero, torch.where(x0 <= W_minus_one, torch.ones_like(x0), torch.zeros_like(x0)), torch.zeros_like(x0))
    x1_valid = torch.where(x1 >= zero, torch.where(x1 <= W_minus_one, torch.ones_like(x1), torch.zeros_like(x1)), torch.zeros_like(x1))
    y0_valid = torch.where(y0 >= zero, torch.where(y0 <= H_minus_one, torch.ones_like(y0), torch.zeros_like(y0)), torch.zeros_like(y0))
    y1_valid = torch.where(y1 >= zero, torch.where(y1 <= H_minus_one, torch.ones_like(y1), torch.zeros_like(y1)), torch.zeros_like(y1))

    if padding_mode == 'zeros':
        # Multiplication is safe as the valid are floats
        wa = wa * x0_valid * y0_valid
        wb = wb * x0_valid * y1_valid
        wc = wc * x1_valid * y0_valid
        wd = wd * x1_valid * y1_valid

    x0_idx = clamp_tensor(x0, zero, W_minus_one).to(torch.int32)
    y0_idx = clamp_tensor(y0, zero, H_minus_one).to(torch.int32)
    x1_idx = clamp_tensor(x1, zero, W_minus_one).to(torch.int32)
    y1_idx = clamp_tensor(y1, zero, H_minus_one).to(torch.int32)

    Ia = _gather_from_hw(src, x0_idx, y0_idx)
    Ib = _gather_from_hw(src, x0_idx, y1_idx)
    Ic = _gather_from_hw(src, x1_idx, y0_idx)
    Id = _gather_from_hw(src, x1_idx, y1_idx)

    out = (Ia * wa.unsqueeze(1) +
           Ib * wb.unsqueeze(1) +
           Ic * wc.unsqueeze(1) +
           Id * wd.unsqueeze(1))
    return out


def affine_grid_sample_approx(src: torch.Tensor, theta: torch.Tensor, dsize: torch.Tensor,
                              mode: str = 'bilinear',
                              padding_mode: str = 'zeros',
                              align_corners: bool = True):
    """
    Approximate affine_grid + grid_sample without calling either function to improve TensorRT compatibility.
    Args:
        src: torch.Tensor 
            Input tensor of shape (L, C, H, W) to be transformed.
        theta: torch.Tensor 
            Affine transformation matrix of shape (B, 2, 3) containing
            the 2x3 affine transformation parameters.
        dsize: torch.Tensor 
            Output spatial dimensions as [H, W].
        [mode]: str
            Interpolation mode. Either 'bilinear' or 'nearest'.
            Default: 'bilinear'.
        [padding_mode]: str 
            Padding mode for out-of-bounds coordinates.
            Either 'zeros', 'border', or 'reflection'. Default: 'zeros'.
        [align_corners]: bool 
            If True, align corners of input and output tensors.
            Default: True.
    Returns:
        torch.Tensor: Transformed tensor of shape (L, C, H, W).
    """
    if mode not in ('bilinear', 'nearest'):
        raise ValueError(f"Unsupported mode: {mode}")
    if padding_mode not in ('zeros', 'border', 'reflection'):
        raise ValueError(f"Unsupported padding_mode: {padding_mode}")

    norm_grid = _affine_grid_sample_approx_prepare_norm_grid(
        theta,
        dsize,
        align_corners=align_corners,
    )
    # norm_grid: (B, H, W, 2)

    return _affine_grid_sample_approx_bilinear_sample(
        src,
        norm_grid,
        dsize,
        theta.dtype,
        padding_mode=padding_mode,
        align_corners=align_corners,
    )
    # (L, C, H, W)


def warp_affine(
        src:torch.Tensor, M:torch.Tensor, dsize: torch.Tensor,
        mode:str='bilinear',
        padding_mode:str='zeros',
        align_corners:bool=True):
    r"""
    Transform the src based on transformation matrix M.
    Args:
        src: torch.Tensor
            Input feature map with shape (L, C, H, W).
        M: torch.Tensor
            Transformation matrix with shape (M, 2, 3).
        dsize: torch.Tensor
            contains [H, W].
        mode: str
            Interpolation methods for F.grid_sample.
        padding_mode: str
            Padding methods for F.grid_sample.
        align_corners: bool
            Parameter of F.affine_grid.

    Returns:
        Transformed features with shape :math:`(B,C,H,W)`.
    """

    M_3x3 = convert_affine_matrix_to_homography(M)
    # M_3x3: (M, 3, 3)

    dst_norm_trans_src_norm = normalize_homography(M_3x3, dsize, dsize)
    # dst_norm_trans_src_norm: (M, 3, 3)

    src_norm_trans_dst_norm = _3x3_cramer_inverse(dst_norm_trans_src_norm)
    src_for_sample = src.half() if src_norm_trans_dst_norm.dtype == torch.half else src

    sliced_src_norm_trans_dst_norm = torch.narrow(src_norm_trans_dst_norm, 1, 0, 2)
    return affine_grid_sample_approx(src_for_sample, sliced_src_norm_trans_dst_norm,
                    dsize, mode=mode, padding_mode=padding_mode, align_corners=align_corners)
    # sliced_src_norm_trans_dst_norm: (M, 2, 3)
