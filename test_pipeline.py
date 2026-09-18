"""Regression checks for standalone CNN prediction."""
import contextlib
import io
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch
import predict_leaf


class PipelineTests(unittest.TestCase):
    def test_health_mapping(self):
        for label, expected in [
            ("Apple___healthy", "not diseased"),
            ("Apple___Apple_scab", "diseased"),
            ("uncertain", "uncertain"),
        ]:
            self.assertEqual(predict_leaf.health_from_class(label), expected)

    def test_cli_uses_only_cnn_without_network(self):
        predictor = Mock()
        predictor.predict.return_value = {
            "predicted_class": "Apple___healthy",
            "health_status": "not diseased",
            "confidence": 0.9,
            "top_predictions": [{"class": "Apple___healthy", "confidence": 0.9}],
        }
        output = io.StringIO()
        with patch.object(predict_leaf, "LeafPredictor", return_value=predictor), \
             patch.object(sys, "argv", ["predict_leaf.py", str(Path(__file__)), "--no-show-image"]), \
             patch("urllib.request.urlopen", side_effect=AssertionError("Network call forbidden")), \
             contextlib.redirect_stdout(output):
            predict_leaf.main()
        predictor.predict.assert_called_once()
        self.assertIn("Apple___healthy", output.getvalue())
        self.assertIn("90.00%", output.getvalue())
        self.assertNotIn("Ollama", output.getvalue())


if __name__ == "__main__":
    unittest.main()
