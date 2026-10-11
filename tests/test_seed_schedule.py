import unittest

from mopso_drowsiness.seeding import seed_schedule


class SeedScheduleTests(unittest.TestCase):
    def test_primary_algorithm_seed_schedule(self):
        self.assertEqual(seed_schedule(100, 10), list(range(100, 110)))

    def test_nsga2_seed_schedule(self):
        self.assertEqual(seed_schedule(109, 10), list(range(109, 119)))

    def test_seed_schedule_requires_positive_run_count(self):
        with self.assertRaises(ValueError):
            seed_schedule(100, 0)


if __name__ == "__main__":
    unittest.main()
