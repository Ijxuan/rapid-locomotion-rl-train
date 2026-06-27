import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class PaperBPpoSourceTest(unittest.TestCase):
    def read(self, relative_path):
        return (ROOT / relative_path).read_text(encoding="utf-8")

    def test_actor_critic_uses_explicit_estimator_architecture(self):
        source = self.read("mini_gym_learn/ppo/actor_critic.py")

        self.assertIn("class Estimator", source)
        self.assertIn("estimator_hidden_dims = [256, 128]", source)
        self.assertIn("actor_hidden_dims = [512, 256, 64]", source)
        self.assertIn("critic_hidden_dims = [512, 256, 64]", source)
        self.assertNotIn("adaptation_module", source)
        self.assertNotIn("env_factor_encoder", source)

    def test_ppo_trains_estimator_from_privileged_targets(self):
        source = self.read("mini_gym_learn/ppo/ppo.py")

        self.assertIn("estimator_optimizer", source)
        self.assertIn("num_estimator_substeps", source)
        self.assertIn("F.mse_loss(state_pred, state_target)", source)
        self.assertIn("F.binary_cross_entropy(contact_pred, contact_target)", source)
        self.assertNotIn("adaptation_module", source)
        self.assertNotIn("env_factor_encoder", source)

    def test_runner_exports_estimator_and_body_jit(self):
        source = self.read("mini_gym_learn/ppo/__init__.py")

        self.assertIn("estimator_latest.jit", source)
        self.assertIn("body_latest.jit", source)
        self.assertNotIn("adaptation_module_latest.jit", source)


if __name__ == "__main__":
    unittest.main()
