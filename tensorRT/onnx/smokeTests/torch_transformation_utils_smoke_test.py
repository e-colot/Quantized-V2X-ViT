import torch

from opencood.models.sub_modules.torch_transformation_utils import (
    _torch_inverse_cast,
    normal_transform_pixel,
)


def make_invertible_batch(batch_size):
    base = torch.tensor(
        [
            [1.4, 0.2, 0.1],
            [0.1, 1.3, -0.2],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    perturb = torch.randn(batch_size, 3, 3, dtype=torch.float32) * 0.01
    return base.unsqueeze(0).repeat(batch_size, 1, 1) + perturb


def assert_matches_legacy(case_name, matrix):
    new_inverse = _torch_inverse_cast(matrix)
    legacy_inverse = torch.inverse(matrix.to(torch.float32)).to(matrix.dtype)
    torch.testing.assert_close(new_inverse, legacy_inverse, atol=1e-5, rtol=1e-5)
    print(f'{case_name}: OK')


def main():
    torch.manual_seed(0)

    assert_matches_legacy('single-matrix path', make_invertible_batch(1))
    assert_matches_legacy('batched path', make_invertible_batch(4))

    pixel_norm = normal_transform_pixel(48, 176, torch.device('cpu'),
                                        torch.float32)
    assert_matches_legacy('pixel-normal transform', pixel_norm)

    print('torch_transformation_utils smoke test passed')


if __name__ == '__main__':
    main()