# License: see [LICENSE, LICENSES/legged_gym/LICENSE]

import torch


def to_torch(x, dtype=torch.float, device="cuda:0", requires_grad=False):
    if isinstance(x, torch.Tensor):
        tensor = x.to(device=device, dtype=dtype)
    else:
        tensor = torch.tensor(x, dtype=dtype, device=device)
    tensor.requires_grad_(requires_grad)
    return tensor


def get_axis_params(value, axis_idx, x_value=0.0, n_dims=3):
    axis_params = [x_value] * n_dims
    axis_params[axis_idx] = value
    return axis_params


def torch_rand_float(lower, upper, shape, device):
    return (upper - lower) * torch.rand(*shape, device=device) + lower


def tensor_clamp(t, min_t, max_t):
    return torch.max(torch.min(t, max_t), min_t)


def normalize(x, eps=1e-9):
    return x / x.norm(p=2, dim=-1, keepdim=True).clamp(min=eps)


def quat_conjugate(q):
    result = q.clone()
    result[..., :3] = -result[..., :3]
    return result


def quat_mul(a, b):
    original_shape = a.shape
    a = a.reshape(-1, 4)
    b = b.reshape(-1, 4)

    ax, ay, az, aw = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    bx, by, bz, bw = b[:, 0], b[:, 1], b[:, 2], b[:, 3]

    return torch.stack(
        (
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ),
        dim=-1,
    ).reshape(original_shape)


def quat_apply(q, v):
    original_shape = v.shape
    q = q.reshape(-1, 4)
    v = v.reshape(-1, 3)
    q_xyz = q[:, :3]
    t = 2.0 * torch.cross(q_xyz, v, dim=-1)
    return (v + q[:, 3:4] * t + torch.cross(q_xyz, t, dim=-1)).reshape(original_shape)


def quat_rotate(q, v):
    return quat_apply(q, v)


def quat_rotate_inverse(q, v):
    original_shape = v.shape
    q = q.reshape(-1, 4)
    v = v.reshape(-1, 3)
    q_xyz = q[:, :3]
    q_w = q[:, 3:4]

    a = v * (2.0 * q_w.square() - 1.0)
    b = 2.0 * q_w * torch.cross(q_xyz, v, dim=-1)
    c = 2.0 * q_xyz * torch.sum(q_xyz * v, dim=-1, keepdim=True)
    return (a - b + c).reshape(original_shape)
