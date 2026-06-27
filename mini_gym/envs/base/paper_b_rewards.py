"""Paper B reward helpers that are independent from Isaac Gym."""

PAPER_B_POSITIVE_REWARDS = (
    "tracking_lin_vel",
    "tracking_ang_vel",
    "feet_air_time",
)

PAPER_B_NEGATIVE_REWARDS = (
    "feet_slip",
    "feet_clearance",
    "orientation",
    "torques",
    "dof_pos",
    "dof_vel",
    "dof_acc",
    "action_smoothness_1",
    "action_smoothness_2",
    "base_motion",
)


def paper_b_total_reward(positive_reward, negative_reward, exponential_scale=0.2):
    exponent = exponential_scale * negative_reward
    if hasattr(exponent, "exp"):
        return positive_reward * exponent.exp()

    import numpy as np

    return positive_reward * np.exp(exponent)


def paper_b_airtime_piecewise(
    feet_air_time,
    first_contact,
    contact_filter,
    command_norm,
    stance_command_threshold=0.1,
    airtime_clip=0.3,
    airtime_max=0.25,
    airtime_cap=0.2,
    dt=0.01,
):
    if hasattr(feet_air_time, "clamp"):
        swing_bonus = (feet_air_time - airtime_clip).clamp(min=0.0, max=airtime_max)
        swing_bonus = swing_bonus.clamp(max=airtime_cap) * first_contact.float()
        stance_bonus = contact_filter.float() * dt
        moving = (command_norm > stance_command_threshold).unsqueeze(1)
        return swing_bonus * moving.float() + stance_bonus * (~moving).float()

    import numpy as np

    swing_bonus = np.clip(feet_air_time - airtime_clip, 0.0, airtime_max)
    swing_bonus = np.minimum(swing_bonus, airtime_cap) * first_contact.astype(float)
    stance_bonus = contact_filter.astype(float) * dt
    moving = np.expand_dims(command_norm > stance_command_threshold, axis=1)
    return np.where(moving, swing_bonus, stance_bonus)
