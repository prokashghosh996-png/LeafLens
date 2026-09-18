"""Checks for deterministic, disjoint training splits."""
import unittest
from train_plant_disease import split_groups


class TrainingSplitTests(unittest.TestCase):
    def test_every_image_used_once_and_duplicates_stay_together(self):
        groups = [[f"image-{i}"] for i in range(20)]
        groups[0].append("duplicate-0")
        result = split_groups(groups, 123)
        flat = [item for items in result.values() for item in items]
        self.assertEqual(len(flat), 21)
        self.assertEqual(len(set(flat)), 21)
        for items in result.values():
            self.assertEqual("image-0" in items, "duplicate-0" in items)
        self.assertEqual(result, split_groups(groups, 123))
        self.assertTrue(all(result.values()))

    def test_small_class_rejected(self):
        with self.assertRaises(ValueError):
            split_groups([["one"], ["two"]], 123)


if __name__ == "__main__":
    unittest.main()
