import unittest
import sys
from pathlib import Path

# Add the project root directory to PYTHONPATH
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def run_tests():
    # Discover and run all tests
    loader = unittest.TestLoader()
    start_dir = project_root / 'tests'
    suite = loader.discover(start_dir, pattern='test_*.py')

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(not success)  # Exit with code 0 if tests pass, 1 if they fail
