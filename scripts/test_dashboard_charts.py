import sys
import types
import unittest
from types import SimpleNamespace


sys.modules.setdefault("isaacgym", types.SimpleNamespace())

from scripts import train


def fake_dashboard_env():
    cfg = SimpleNamespace(
        commands=SimpleNamespace(
            command_curriculum=True,
            yaw_command_curriculum=False,
            paper_b_command_curriculum=True,
            paper_b_vx_final=[-1.75, 3.5],
            lin_vel_y=[-1.0, 1.0],
            ang_vel_yaw=[-1.0, 1.0],
            max_yaw_curriculum=1.0,
        ),
        terrain=SimpleNamespace(curriculum=False, num_rows=1),
        rewards=SimpleNamespace(
            paper_b_reward_gate_floor=0.05,
            paper_b_airtime_cap=0.2,
            paper_b_termination_penalty=-10.0,
        ),
        normalization=SimpleNamespace(clip_actions=1.0),
        control=SimpleNamespace(decimation=4),
        sim=SimpleNamespace(dt=0.0025),
        env=SimpleNamespace(episode_length_s=20),
    )
    reward_scales = {
        "tracking_lin_vel": 3.0,
        "tracking_ang_vel": 3.0,
        "feet_air_time": 0.3,
        "feet_slip": -0.08,
        "feet_clearance": -15.0,
        "orientation": -3.0,
        "torques": -6e-4,
        "dof_pos": -0.75,
        "dof_vel": -6e-4,
        "dof_acc": -0.02,
        "action_smoothness_1": -2.5,
        "action_smoothness_2": -1.2,
        "base_motion": -1.5,
        "moving_stand_still": -10.0,
        "termination": -10.0,
    }
    return SimpleNamespace(
        cfg=cfg,
        reward_scales=reward_scales,
        max_episode_length=2000,
        dt=0.01,
        feet_indices=[0, 1, 2, 3],
        torque_limits=[18, 18, 26] * 4,
    )


REWARD_KEYS = [
    "rew_total",
    "rew_paper_b_nontermination_reward",
    "rew_termination",
    "rew_paper_b_gate",
    "rew_paper_b_positive_reward",
    "rew_paper_b_negative_reward",
    "rew_tracking_lin_vel",
    "rew_tracking_ang_vel",
    "rew_moving_stand_still",
    "rew_feet_air_time",
    "rew_feet_clearance",
    "rew_feet_slip",
    "rew_orientation",
    "rew_base_motion",
    "rew_dof_pos",
    "rew_dof_vel",
    "rew_dof_acc",
    "rew_torques",
    "rew_action_smoothness_1",
    "rew_action_smoothness_2",
]

DIAGNOSTIC_KEYS = [
    "mean_cmd_vx",
    "mean_base_vx",
    "mean_abs_vx_error",
    "mean_cmd_vy",
    "mean_base_vy",
    "mean_abs_vy_error",
    "mean_cmd_yaw",
    "mean_base_yaw_rate",
    "mean_abs_yaw_error",
    "moving_cmd_fraction",
    "moving_standstill_fraction",
]


def ykeys(chart_text):
    return [
        line.strip().split(": ", 1)[1]
        for line in chart_text.splitlines()
        if line.strip().startswith("yKey: ")
    ]


class DashboardChartTest(unittest.TestCase):
    def test_chart_order_is_logical_not_alphabetical(self):
        text = train._render_dashboard_charts(REWARD_KEYS, DIAGNOSTIC_KEYS, fake_dashboard_env())
        keys = ykeys(text)

        self.assertEqual(keys[0], "train/episode/rew_total/mean")
        self.assertEqual(keys[1], "train/episode/rew_paper_b_nontermination_reward/mean")
        self.assertNotEqual(keys[1], "train/episode/command_area/mean")
        self.assertGreater(
            keys.index("train/episode/command_area/mean"),
            keys.index("train/episode/rew_action_smoothness_2/mean"),
        )

    def test_feet_clearance_is_labeled_as_negative_penalty(self):
        text = train._render_dashboard_charts(REWARD_KEYS, DIAGNOSTIC_KEYS, fake_dashboard_env())

        self.assertIn("足端抬脚高度误差惩罚 | 范围: (-inf, 0]，0最好\\n备注：越接近0越好", text)
        self.assertNotIn("足端抬脚奖励", text)

    def test_theoretical_ranges_are_rendered_for_key_metrics(self):
        text = train._render_dashboard_charts(REWARD_KEYS, DIAGNOSTIC_KEYS, fake_dashboard_env())

        self.assertIn("总奖励 | 范围: [-inf, 12480]", text)
        self.assertIn("Paper-B非终止奖励 | 范围: [-inf, 12480]", text)
        self.assertIn("线速度跟踪 | 范围: [0, 6000]", text)
        self.assertIn("角速度跟踪 | 范围: [0, 6000]", text)
        self.assertIn("Paper-B指数门控 | 范围: [0.05, 1]", text)
        self.assertIn("移动指令下速度缺口/反向惩罚 | 范围: (-inf, 0]，0最好", text)
        self.assertIn("移动指令下速度不足/反向占比 | 范围: [0, 1]", text)
        self.assertIn("备注：Paper-B路径下仅作兼容显示，不用于判断课程扩展", text)

    def test_unknown_metrics_are_appended_instead_of_dropped(self):
        text = train._render_dashboard_charts(
            REWARD_KEYS + ["rew_new_penalty"],
            DIAGNOSTIC_KEYS + ["new_diagnostic"],
            fake_dashboard_env(),
        )
        keys = ykeys(text)

        self.assertIn("train/episode/rew_new_penalty/mean", keys)
        self.assertIn("train/episode/new_diagnostic/mean", keys)
        self.assertGreater(
            keys.index("train/episode/new_diagnostic/mean"),
            keys.index("train/episode/command_area/mean"),
        )


if __name__ == "__main__":
    unittest.main()
