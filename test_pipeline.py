import unittest
from unittest.mock import patch
from pathlib import Path
from predict_leaf import combine_results, analyze_leaf
from test_ollama import _chat
import io
import json


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.cnn = {"predicted_class": "healthy", "confidence": 0.92}
        self.vision = {"predicted_class": "healthy", "observations": "Green leaf"}
        self.review = {"verdict": "supports", "reason": "No visible damage"}

    def test_review_cannot_override_class_or_score(self):
        review = dict(self.review, final_class="disease", confidence=1.0)
        result = combine_results(self.cnn, self.vision, review)
        self.assertEqual(result["final_class"], "healthy")
        self.assertEqual(result["confidence"], 0.92)

    def test_disagreement_cannot_be_erased_by_review(self):
        vision = dict(self.vision, predicted_class="disease")
        result = combine_results(self.cnn, vision, self.review)
        self.assertEqual(result["status"], "needs_review")
        self.assertIn("models_disagree", result["review_reasons"])

    def test_low_confidence_and_uncertain_review(self):
        result = combine_results(dict(self.cnn, confidence=0.4), self.vision,
                                 dict(self.review, verdict="uncertain"))
        self.assertIn("low_cnn_confidence", result["review_reasons"])
        self.assertIn("review_uncertain", result["review_reasons"])

    def test_agreement_does_not_increase_confidence(self):
        result = combine_results(self.cnn, self.vision, self.review)
        self.assertEqual(result["status"], "models_agree")
        self.assertEqual(result["confidence"], 0.92)

    @patch("test_ollama.review_leaf")
    @patch("test_ollama.assess_leaf", side_effect=OSError("offline"))
    def test_offline_retains_cnn(self, assess, review):
        predictor = unittest.mock.Mock()
        predictor.predict.return_value = self.cnn
        result = analyze_leaf(Path(__file__), predictor)
        self.assertEqual(result["final_class"], "healthy")
        self.assertEqual(result["status"], "needs_review")
        review.assert_not_called()

    @patch("test_ollama.review_leaf")
    @patch("test_ollama.assess_leaf")
    def test_both_passes_use_same_image(self, assess, review):
        assess.return_value = self.vision
        review.return_value = self.review
        predictor = unittest.mock.Mock()
        predictor.predict.return_value = self.cnn
        path = Path(__file__).resolve()
        analyze_leaf(path, predictor)
        self.assertEqual(assess.call_args.args[0], path)
        self.assertEqual(review.call_args.args[0], path)
        self.assertEqual(review.call_args.args[1:3], (self.cnn, self.vision))

    @patch("test_ollama.review_leaf")
    @patch("test_ollama.assess_leaf")
    def test_dataset_image_skips_both_ollama_calls(self, assess, review):
        predictor = unittest.mock.Mock()
        predictor.predict.return_value = self.cnn
        path = Path(__file__).resolve()
        result = analyze_leaf(path, predictor, dataset_path=path.parent)
        self.assertEqual(result["final_class"], "healthy")
        self.assertEqual(result["status"], "dataset_cnn_only")
        assess.assert_not_called()
        review.assert_not_called()

    @patch("test_ollama.review_leaf")
    @patch("test_ollama.assess_leaf")
    def test_dataset_confidence_threshold_routes_review(self, assess, review):
        path = Path(__file__).resolve()
        for confidence in (0.6548, 0.7999, 0.8, 0.8336):
            with self.subTest(confidence=confidence):
                assess.reset_mock()
                review.reset_mock()
                assess.return_value = self.vision
                review.return_value = self.review
                predictor = unittest.mock.Mock()
                predictor.predict.return_value = dict(self.cnn, confidence=confidence)
                result = analyze_leaf(path, predictor, dataset_path=path.parent)
                self.assertEqual(result["confidence"], confidence)
                if confidence < 0.8:
                    assess.assert_called_once()
                    review.assert_called_once()
                    self.assertIn("low_cnn_confidence", result["review_reasons"])
                else:
                    assess.assert_not_called()
                    review.assert_not_called()
                    self.assertEqual(result["status"], "dataset_cnn_only")

    def test_dataset_membership_does_not_match_sibling_or_missing_file(self):
        from dataset_config import is_dataset_image
        path = Path(__file__).resolve()
        self.assertFalse(is_dataset_image(path, path.parent / "color"))
        self.assertFalse(is_dataset_image(path.parent / "missing.jpg", path.parent))
        self.assertTrue(is_dataset_image(path, path.parent / "unused" / ".."))

    @patch("test_ollama.urlopen")
    def test_unexpected_response_fields_rejected(self, urlopen):
        content = json.dumps({"verdict": "supports", "final_class": "disease"})
        urlopen.return_value = io.BytesIO(json.dumps(
            {"message": {"content": content}}).encode())
        with self.assertRaises(ValueError):
            _chat(__file__, "review", {"verdict": {"type": "string"}},
                  "test", 1)


if __name__ == "__main__":
    unittest.main()
