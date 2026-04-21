"""
torch_transformation_utils.py
"""
import torch

def get_roi_and_cav_mask(input: torch.Tensor, cav_mask: torch.Tensor, 
                         spatial_correction_matrix: torch.Tensor,
                         discrete_ratio: torch.Tensor, downsample_rate: torch.Tensor):
    """
    Get mask for the combination of cav_mask and rorated ROI mask.
    Parameters
    ----------
    input: torch.Tensor
        input of shape (B, L, H, W, C).
        only its shape is used
    cav_mask: torch.Tensor
        Shape of (B, L).
    spatial_correction_matrix: torch.Tensor
        Shape of (B, L, 4, 4)
    discrete_ratio: torch.Tensor (scalar float)
        Discrete ratio.
    downsample_rate: torch.Tensor (scalar float)
        Downsample rate.

    Returns
    -------
    com_mask: torch.Tensor
        Combined mask with shape (B, H, W, L, 1).

    """

    spatial_size = torch.tensor([input.shape[2], input.shape[3]], device='cuda', dtype=torch.int32)

    # (B,L,4,4)
    dist_correction_matrix = get_discretized_transformation_matrix(
        spatial_correction_matrix, discrete_ratio,
        downsample_rate)
    # (B*L,2,3)
    T = get_transformation_matrix(
        dist_correction_matrix.reshape(-1, 2, 3), spatial_size)
    
    # (B,L,1,H,W)
    input_slice = input[..., 0:1]
    input_reordered = input_slice.permute(0, 1, 4, 2, 3)

    roi_mask = get_rotated_roi(input_reordered, spatial_size, T)
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
    roi_mask: torch.Tensor
        Mask for ROI region after considering the spatial transformation/correction.
    cav_mask: torch.Tensor
        Mask for CAV to remove padded 0.

    Returns
    -------
    com_mask: torch.Tensor
        Combined mask.
    """
    B, L, C, H, W = roi_mask.shape[0], roi_mask.shape[1], roi_mask.shape[2], roi_mask.shape[3], roi_mask.shape[4]

    # (B, L, 1, 1, 1)
    cav_mask = cav_mask.unsqueeze(2).unsqueeze(3).unsqueeze(4)
    # (B, L, C, H, W)
    cav_mask = cav_mask.expand(B, L, C, H, W)
    # (B, L, C, H, W)
    com_mask = roi_mask * cav_mask
    return com_mask


def get_rotated_roi(input: torch.Tensor, spatial_size: torch.Tensor, correction_matrix: torch.Tensor) -> torch.Tensor:
    """
    Get rotated ROI mask.

    Parameters
    ----------
    input: torch.Tensor
        input of shape (B,L,C,H,W).
        only its shape is used
    spatial_size: torch.Tensor
        (H, W), same as from input
    correction_matrix: torch.Tensor
        Correction matrix with shape (N,2,3).

    Returns
    -------
    roi_mask: torch.Tensor
        Rotated ROI mask with shape (N,2,3).
    """

    # To reduce the computation, we only need to calculate the
    # mask for the first channel.
    # (B,L,1,H,W)
    x = torch.ones(input.shape, device='cuda', dtype=input.dtype)
    # (B*L,1,H,W)
    roi_mask = warp_affine(x.reshape(-1, 1, input.shape[3], input.shape[4]), correction_matrix,
                           dsize=spatial_size, mode="bilinear") # was mode='nearest'
    # going from 'nearest' to 'bilinear' lost 0.02% AP @ IOU 0.7 (0% elesewhere)
    # -> acceptable loss

    # (B,L,C,H,W)
    roi_mask = torch.repeat_interleave(roi_mask, 1, dim=1).reshape(input.shape)
    return roi_mask


def get_discretized_transformation_matrix(matrix: torch.Tensor, discrete_ratio: torch.Tensor,
                                          downsample_rate: torch.Tensor):
    """
    Get discretized transformation matrix.
    Parameters
    ----------
    matrix: torch.Tensor
        Shape -- (B, L, 4, 4) where B is the batch size, L is the max cav
        number.
    discrete_ratio: torch.Tensor (discrete scalar)
        Discrete ratio.
    downsample_rate: torch.Tensor (discrete scalar)
        downsample_rate
        Warning: previously supported both int and float

    Returns
    -------
    matrix: torch.Tensor
        Output transformation matrix in 2D with shape (B, L, 2, 3),
        including 2D transformation and 2D rotation.

    """
    idx1 = torch.tensor([0, 1], dtype=torch.int32, device='cuda')
    idx2 = torch.tensor([0, 1, 3], dtype=torch.int32, device='cuda')

    matrix = matrix[:, :, idx1, :][:, :, :, idx2]
    # normalize the x,y transformation
    scale = (discrete_ratio * downsample_rate).to(torch.float32)
    matrix = matrix.to(torch.float32)
    matrix[:, :, :, -1] = matrix[:, :, :, -1] / scale
    return matrix

def _3x3_cramer_inverse(input: torch.Tensor):
    r"""
    Helper function to compute the inverse of a Bx3x3 matrix using Cramer's
    rule. 
    Args:
        input: torch.Tensor
            Tensor to be inversed.

    Returns:
        out: torch.Tensor
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
            Normalized transform with shape :math:`(1, 3, 3)`.
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
            shape :math:`(B, 3, 3)`.
        dsize_src: torch.Tensor
            Size of the source image (height, width).
        dsize_dst: torch.Tensor
            Size of the destination image (height, width).

    Returns:
        dst_norm_trans_src_norm: torch.Tensor
            The normalized homography of shape :math:`(B, 3, 3)`.
    """
    # source and destination sizes
    src_h, src_w = dsize_src[0], dsize_src[1]
    dst_h, dst_w = dsize_dst[0], dsize_dst[1]
    device = 'cuda'
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


def get_rotation_matrix2d(M: torch.Tensor, dsize: torch.Tensor):
    r"""
    Return rotation matrix for torch.affine_grid based on transformation matrix.
    Args:
        M: torch.Tensor
            Transformation matrix with shape :math:`(B, 2, 3)`.
        dsize: torch.Tensor
            Size of the source image (height, width).

    Returns:
        R: torch.Tensor
            Rotation matrix with shape :math:`(B, 2, 3)`.
    """
    device = 'cuda'
    dtype = M.dtype
    B = M.shape[0]

    H = dsize[0].to(dtype)
    W = dsize[1].to(dtype)
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
    rot_22 = M[:, :2, :2] 
    # Row 0: [m00, m01, 0]
    # Row 1: [m10, m11, 0]
    # Row 2: [0,   0,   1]
    r0_rot = torch.cat([rot_22[:, 0, :], zeros.unsqueeze(1)], dim=1)
    r1_rot = torch.cat([rot_22[:, 1, :], zeros.unsqueeze(1)], dim=1)
    rotat_m = torch.stack([r0_rot, r1_rot, r2], dim=1)

    # 5. Matrix Multiply
    affine_m = torch.bmm(torch.bmm(shift_m, rotat_m), shift_m_inv)
    
    return affine_m[:, :2, :]


def get_transformation_matrix(M, spatial_size: torch.Tensor):
    r"""
    Return transformation matrix for torch.affine_grid.
    Args:
        M: torch.Tensor
            Transformation matrix with shape :math:`(N, 2, 3)`.
        spatial_size: torch.Tensor
            Size of the source image (height, width).

    Returns:
        T: torch.Tensor
            Transformation matrix with shape :math:`(N, 2, 3)`.
    """
    T = get_rotation_matrix2d(M, spatial_size)
    T[..., 2] = T[..., 2] + M[..., 2]
    return T


def convert_affinematrix_to_homography(A: torch.Tensor):
    r"""
    Convert to homography coordinates
    Args:
        A: torch.Tensor
            The affine matrix with shape :math:`(B,2,3)`.

    Returns:
        H: torch.Tensor
            The homography matrix with shape of :math:`(B,3,3)`.
    """
    B = A.shape[0]

    # Create the row [0.0, 0.0, 1.0]: (B, 1, 3)
    new_row = torch.tensor([[[0.0, 0.0, 1.0]]], device='cuda', dtype=A.dtype)
    new_row = new_row.expand(B, -1, -1)
    
    # Concatenate along the height dimension (dim=1)
    H = torch.cat([A, new_row], dim=1)
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
            source matrix
        x_idx: torch.Tensor
            index on 4th dimension
        y_idx: torch.Tensor
            index on the 3rd dimension
    Returns:
        out: torch.Tensor
            slice of src
    """
    # src shape: (B, C, H, W)
    linear_idx = (y_idx * src.shape[3] + x_idx).view(src.shape[0], 1, -1).expand(-1, src.shape[1], -1)

    # (B, C, H, W) -> (B, C, H*W)
    src_flat = src.flatten(2)
    device = 'cuda'
    
    # Create 1D indices
    # We use .view() to place them in the correct dimensions for broadcasting
    i = torch.arange(src.shape[0], device=device, dtype=torch.int32).view(src.shape[0], 1, 1) # (B, 1, 1)
    j = torch.arange(src.shape[1], device=device, dtype=torch.int32).view(1, src.shape[1], 1)    # (1, L, 1)

    # linear_idx is likely (B, L, K)
    # When we index with i and j, PyTorch broadcasts them to match linear_idx
    out = src_flat[i, j, linear_idx]

    return out.reshape(src.shape)


def _affine_grid_sample_approx_prepare_norm_grid(theta: torch.Tensor, dsize: torch.Tensor, align_corners: bool = True):
    r"""
    Build a normalized sampling grid for affine resampling.
    Args:
        theta: torch.Tensor
            Affine transformation matrix with shape :math:`(B, 2, 3)`.
        dsize: torch.Tensor
            Output spatial size tensor :math:`(H_out, W_out)`.
        align_corners: bool
            Whether to align grid endpoints to image corners.

    Returns:
        norm_grid: torch.Tensor
            Normalized grid with shape :math:`(B, H_out, W_out, 2)`.
    """
    device = 'cuda'
    grid_dtype = theta.dtype

    H_out = dsize[0].to(grid_dtype)
    W_out = dsize[1].to(grid_dtype)
    eps = torch.tensor(1e-14, device=device, dtype=grid_dtype)
    one = torch.tensor(1.0, device=device, dtype=grid_dtype)
    two = torch.tensor(2.0, device=device, dtype=grid_dtype)
    half = torch.tensor(0.5, device=device, dtype=grid_dtype)

    if align_corners:
        step_y = two / (H_out - one + eps)
        ys = torch.arange(H_out, device=device, dtype=grid_dtype) * step_y - one

        step_x = two / (W_out - one + eps)
        xs = torch.arange(W_out, device=device, dtype=grid_dtype) * step_x - one
    else:
        ys = (two * (torch.arange(H_out, device=device, dtype=grid_dtype) + half)
              / H_out - one)
        xs = (two * (torch.arange(W_out, device=device, dtype=grid_dtype) + half)
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
        dtype: torch.dtype,
        padding_mode: str = 'zeros',
        align_corners: bool = True):
    r"""
    Sample source features from a normalized grid using bilinear interpolation.
    Args:
        src: torch.Tensor
            Source feature map with shape :math:`(B, C, H, W)`.
        norm_grid: torch.Tensor
            Normalized sampling grid with shape :math:`(B, H_out, W_out, 2)`.
        dtype: torch.dtype
            Computation dtype for coordinate transforms and weights.
        padding_mode: str
            Out-of-bound handling mode: ``zeros``, ``border`` or ``reflection``.
        align_corners: bool
            Whether normalization assumes aligned corner points.

    Returns:
        out: torch.Tensor
            Resampled tensor with shape :math:`(B, C, H_out, W_out)`.
    """
    
    device = src.device # Best practice: use the source tensor's device
    
    x = norm_grid[..., 0]
    y = norm_grid[..., 1]

    # FIX 1: Use .size() and cast to Tensor immediately to stay in the graph
    # This prevents aten::item during the math operations below
    H = torch.tensor(src.shape[2], device=device, dtype=dtype)
    W = torch.tensor(src.shape[3], device=device, dtype=dtype)
    
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

    # FIX 2: Use Tensors for clamping boundaries
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
            Input tensor of shape (B, C, H, W) to be transformed.
        theta: torch.Tensor 
            Affine transformation matrix of shape (B, 2, 3) containing
            the 2x3 affine transformation parameters.
        dsize: torch.Tensor 
            Output spatial dimensions as (H_out, W_out).
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
        torch.Tensor: Transformed tensor of shape (B, C, H_out, W_out).
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

    # unused (changed get_rotated_roi to use mode='bilinear')
    # if mode == 'nearest':
    #     ...

    return _affine_grid_sample_approx_bilinear_sample(
        src,
        norm_grid,
        theta.dtype,
        padding_mode=padding_mode,
        align_corners=align_corners,
    )


def warp_affine(
        src:torch.Tensor, M:torch.Tensor, dsize: torch.Tensor,
        mode:str='bilinear',
        padding_mode:str='zeros',
        align_corners:bool=True):
    r"""
    Transform the src based on transformation matrix M.
    Args:
        src: torch.Tensor
            Input feature map with shape :math:`(B,C,H,W)`.
        M: torch.Tensor
            Transformation matrix with shape :math:`(B,2,3)`.
        dsize: torch.Tensor
            Tuple of output image H_out and W_out.
        mode: str
            Interpolation methods for F.grid_sample.
        padding_mode: str
            Padding methods for F.grid_sample.
        align_corners: bool
            Parameter of F.affine_grid.

    Returns:
        Transformed features with shape :math:`(B,C,H,W)`.
    """

    in_size = torch.tensor([src.shape[2], src.shape[3]], device='cuda', dtype=torch.int32)

    # we generate a 3x3 transformation matrix from 2x3 affine
    M_3x3 = convert_affinematrix_to_homography(M)
    dst_norm_trans_src_norm = normalize_homography(M_3x3, in_size, dsize)

    # src_norm_trans_dst_norm = torch.inverse(dst_norm_trans_src_norm)
    src_norm_trans_dst_norm = _3x3_cramer_inverse(dst_norm_trans_src_norm)
    src_for_sample = src.half() if src_norm_trans_dst_norm.dtype == torch.half else src
    return affine_grid_sample_approx(src_for_sample,
                                     src_norm_trans_dst_norm[:, :2, :],
                                     dsize,
                                     mode=mode,
                                     padding_mode=padding_mode,
                                     align_corners=align_corners)
