#!/usr/bin/env python3
"""Open Isaac Gym with Mini Cheetah at the default standing joint angles."""

from joint_pose_viewer_common import run_pose_viewer


if __name__ == "__main__":
    run_pose_viewer("default standing joint angles", zero_joints=False)
