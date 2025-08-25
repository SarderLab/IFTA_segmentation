#!/usr/bin/env python3
"""
Comprehensive test runner for all TF2.x migration tests
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

# Add the parent directory to path so we can import the Codes module
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'Codes'))

def run_test(test_file, verbose=False):
    """Run a single test file and return the result."""
    test_path = Path(__file__).parent / test_file
    
    print(f"\n{'='*60}")
    print(f"Running: {test_file}")
    print(f"{'='*60}")
    
    try:
        if verbose:
            result = subprocess.run([sys.executable, str(test_path)], 
                                  check=True, capture_output=False)
        else:
            result = subprocess.run([sys.executable, str(test_path)], 
                                  check=True, capture_output=True, text=True)
            print(result.stdout)
            if result.stderr:
                print("STDERR:", result.stderr)
        
        print(f"✅ {test_file} PASSED")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ {test_file} FAILED")
        if not verbose:
            print("STDOUT:", e.stdout)
            print("STDERR:", e.stderr)
        return False
    except Exception as e:
        print(f"❌ {test_file} ERROR: {e}")
        return False

def main():
    """Run all tests or specific test files."""
    parser = argparse.ArgumentParser(description='Run TF2.x migration tests')
    parser.add_argument('--test', '-t', help='Run specific test file')
    parser.add_argument('--verbose', '-v', action='store_true', 
                       help='Verbose output (don\'t capture stdout/stderr)')
    parser.add_argument('--list', '-l', action='store_true', 
                       help='List available tests')
    
    args = parser.parse_args()
    
    # Available test files
    test_files = [
        'test_tf2_networks.py',
        'test_utils_tf2.py', 
        'test_integration.py'
    ]
    
    if args.list:
        print("Available tests:")
        for i, test in enumerate(test_files, 1):
            print(f"  {i}. {test}")
        return
    
    print("🚀 IFTA Segmentation TF2.x Migration Test Suite")
    print(f"Python: {sys.version}")
    print(f"Working Directory: {os.getcwd()}")
    
    # Import TensorFlow to show version
    try:
        import tensorflow as tf
        print(f"TensorFlow: {tf.__version__}")
    except ImportError:
        print("❌ TensorFlow not available")
        return
    
    results = []
    
    if args.test:
        # Run specific test
        if args.test in test_files:
            success = run_test(args.test, args.verbose)
            results.append((args.test, success))
        else:
            print(f"❌ Test file '{args.test}' not found")
            print("Available tests:", test_files)
            return
    else:
        # Run all tests
        for test_file in test_files:
            success = run_test(test_file, args.verbose)
            results.append((test_file, success))
    
    # Summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_file, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{test_file:<30} {status}")
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED - TF2.x MIGRATION SUCCESSFUL!")
        sys.exit(0)
    else:
        print("⚠️  SOME TESTS FAILED - Please check the output above")
        sys.exit(1)

if __name__ == "__main__":
    main()
