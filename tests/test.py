"""
Main test entry point for the hybrid memory architecture.
Runs the comprehensive test suite with all optimizations.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import the main test runner
from run_all_tests import main

if __name__ == "__main__":
    print("=€ Starting Hybrid Memory Architecture Test Suite...")
    print("   This will test all components including LoRA compatibility")
    print("   and performance optimizations.\n")
    
    success = main()
    
    if success:
        print("\n<‰ All tests completed successfully!")
        print("   The hybrid memory architecture is ready for use.")
    else:
        print("\nL Some tests failed.")
        print("   Please check the output above for details.")
    
    sys.exit(0 if success else 1)