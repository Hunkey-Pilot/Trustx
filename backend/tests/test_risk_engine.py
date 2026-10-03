import unittest

import numpy as np

from app.services.risk_engine import RiskEngine, RiskEngineError


class RiskEngineTests(unittest.TestCase):
    def setUp(self):
        self.engine = RiskEngine()

    def test_risk_score_uses_fraud_probability_and_is_bounded(self):
        result = self.engine.calculate(0.73, -0.04)
        self.assertEqual(result.risk_score, 73.0)
        self.assertGreaterEqual(result.risk_score, 0)
        self.assertLessEqual(result.risk_score, 100)

    def test_boundary_risk_levels(self):
        cases = [
            (0, "LOW"),
            (29, "LOW"),
            (30, "MEDIUM"),
            (59, "MEDIUM"),
            (60, "HIGH"),
            (84, "HIGH"),
            (85, "CRITICAL"),
            (100, "CRITICAL"),
        ]
        for score, expected_level in cases:
            with self.subTest(score=score):
                result = self.engine.calculate(score / 100, 0.0)
                self.assertEqual(result.risk_score, score)
                self.assertEqual(result.risk_level, expected_level)

    def test_recommended_actions_match_risk_levels(self):
        cases = [
            (0.29, "ALLOW / MONITOR"),
            (0.30, "ADDITIONAL_VERIFICATION / MONITOR"),
            (0.60, "STEP_UP_VERIFICATION / MANUAL_REVIEW"),
            (0.85, "TEMPORARY_HOLD / ESCALATE_FOR_REVIEW"),
        ]
        for probability, expected_action in cases:
            with self.subTest(probability=probability):
                result = self.engine.calculate(probability, 0.0)
                self.assertEqual(result.recommended_action, expected_action)

    def test_anomaly_signal_is_validated_but_not_normalized_or_combined(self):
        negative = self.engine.calculate(0.40, -0.25)
        positive = self.engine.calculate(0.40, 0.25)
        self.assertEqual(negative, positive)

    def test_output_is_deterministic(self):
        first = self.engine.calculate(0.61, -0.03)
        second = self.engine.calculate(0.61, -0.03)
        self.assertEqual(first, second)

    def test_invalid_model_outputs_are_rejected(self):
        with self.assertRaises(RiskEngineError):
            self.engine.calculate(-0.01, 0.0)
        with self.assertRaises(RiskEngineError):
            self.engine.calculate(1.01, 0.0)
        with self.assertRaises(RiskEngineError):
            self.engine.calculate(0.5, np.inf)
        with self.assertRaises(RiskEngineError):
            self.engine.calculate(0.5, np.nan)


if __name__ == "__main__":
    unittest.main()