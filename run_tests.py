"""
VelocityLLM - Comprehensive Test Runner
Executes the Phase 2 test suite. Can be run directly with:
    python run_tests.py
"""

import sys
import unittest

def main():
    try:
        import pytest
        print("Running test suite with pytest...")
        ret = pytest.main(["tests", "-v", "-s"])
        sys.exit(ret)
    except ImportError:
        print("pytest not found; running tests with standard library unittest...")
        # Discover and run standard tests
        loader = unittest.TestLoader()
        suite = loader.discover("tests")
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        sys.exit(0 if result.wasSuccessful() else 1)

if __name__ == "__main__":
    main()
