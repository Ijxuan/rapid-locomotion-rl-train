import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from mini_gym.envs.base.paper_b_assets import (
    foot_radius_buckets,
    foot_sphere_radius_from_urdf,
    generate_sphere_foot_urdf_variants,
)
from mini_gym.envs.base.paper_b_commands import paper_b_vx_range


class PaperBCommandAssetTest(unittest.TestCase):
    def test_vx_curriculum_expands_toward_paper_b_range(self):
        initial = paper_b_vx_range(0)
        final = paper_b_vx_range(100000)

        self.assertGreaterEqual(initial[0], -1.75)
        self.assertLessEqual(initial[1], 3.5)
        self.assertAlmostEqual(final[0], -1.75, places=3)
        self.assertAlmostEqual(final[1], 3.5, places=3)

    def test_foot_radius_buckets_cover_six_to_ten_mm(self):
        buckets = foot_radius_buckets([0.006, 0.010])

        self.assertEqual(len(buckets), 5)
        self.assertAlmostEqual(buckets[0], 0.006)
        self.assertAlmostEqual(buckets[-1], 0.010)
        self.assertAlmostEqual(buckets[2], 0.008)

    def test_default_sphere_foot_radius_can_be_read_from_urdf(self):
        root = Path(__file__).resolve().parents[1]
        urdf = root / "resources/robots/mini_cheetah/urdf/mini_cheetah_simple.urdf"

        self.assertAlmostEqual(foot_sphere_radius_from_urdf(urdf), 0.0175)

    def test_sphere_foot_urdf_variants_are_written_with_requested_radii(self):
        root = Path(__file__).resolve().parents[1]
        urdf = root / "resources/robots/mini_cheetah/urdf/mini_cheetah_simple.urdf"
        radii = foot_radius_buckets([0.006, 0.010])

        with TemporaryDirectory() as tmp:
            variants = generate_sphere_foot_urdf_variants(urdf, tmp, radii)

            self.assertEqual(len(variants), len(radii))
            for variant, radius in zip(variants, radii):
                self.assertTrue(Path(variant).exists())
                self.assertAlmostEqual(foot_sphere_radius_from_urdf(variant), radius)

    def test_mini_cheetah_config_source_uses_sphere_foot_asset(self):
        source = (Path(__file__).resolve().parents[1]
                  / "mini_gym/envs/mini_cheetah/mini_cheetah_config.py").read_text(encoding="utf-8")

        self.assertIn("mini_cheetah_simple.urdf", source)
        self.assertIn('_.foot_name = "_foot"', source)
        self.assertIn("_.collapse_fixed_joints = False", source)

    def test_legged_robot_source_assigns_radius_asset_buckets_per_env(self):
        source = (Path(__file__).resolve().parents[1]
                  / "mini_gym/envs/base/legged_robot.py").read_text(encoding="utf-8")

        self.assertIn("asset_paths = self._resolve_robot_asset_paths(asset_path)", source)
        self.assertIn("asset_id = self._paper_b_asset_bucket_id(i)", source)
        self.assertIn("robot_asset = self.robot_assets[asset_id]", source)


if __name__ == "__main__":
    unittest.main()
