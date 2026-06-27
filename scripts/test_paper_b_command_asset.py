import unittest

from mini_gym.envs.base.paper_b_assets import foot_radius_buckets
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

    def test_mini_cheetah_config_source_uses_sphere_foot_asset(self):
        source = ((__import__("pathlib").Path(__file__).resolve().parents[1])
                  / "mini_gym/envs/mini_cheetah/mini_cheetah_config.py").read_text(encoding="utf-8")

        self.assertIn("mini_cheetah_simple.urdf", source)
        self.assertIn('_.foot_name = "_foot"', source)
        self.assertIn("_.collapse_fixed_joints = False", source)


if __name__ == "__main__":
    unittest.main()
