#!/usr/bin/env python3
"""
Quick test script to verify the Advanced Log Reader application functions correctly.
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

def test_imports():
    """Test that all main modules can be imported"""
    try:
        from utils.log_analyzer import analyze_logs_refactored, filter_issues_by_relevance, enhance_log_analysis, extract_key_utterance_components
        print("✓ log_analyzer imports successful")
        
        from utils.utterance_tracer import UtteranceTracer
        print("✓ utterance_tracer imports successful")
        
        from utils.ai_analyzer import AutoLogAI
        print("✓ ai_analyzer imports successful")
        
        from utils.report_generator import generate_qa_summary
        print("✓ report_generator imports successful")
        
        from utils.config_manager import ConfigManager, get_config_manager
        print("✓ config_manager imports successful")
        
        return True
    except Exception as e:
        print(f"✗ Import error: {e}")
        return False

def test_log_analyzer():
    """Test basic log analyzer functionality"""
    try:
        from utils.log_analyzer import analyze_logs_refactored
        
        # Test with the existing log file
        log_file = "logs/car_fail.txt"
        if os.path.exists(log_file):
            df, stats = analyze_logs_refactored(log_file)
            print(f"✓ Log analysis successful: {len(df)} entries processed")
            print(f"  Stats: {stats.get('total_entries', 0)} total entries")
            return True
        else:
            print(f"✗ Log file not found: {log_file}")
            return False
    except Exception as e:
        print(f"✗ Log analyzer error: {e}")
        return False

def test_utterance_tracer():
    """Test utterance tracer functionality"""
    try:
        from utils.utterance_tracer import UtteranceTracer
        import pandas as pd
        
        tracer = UtteranceTracer()
        
        # Create sample data
        sample_data = [
            {'timestamp': '2024-01-01 10:00:00', 'message': 'WakewordDetected payload wakeword ALEXA', 'pid': '123', 'tid': '456'},
            {'timestamp': '2024-01-01 10:00:01', 'message': 'DialogState LISTENING', 'pid': '123', 'tid': '456'},
            {'timestamp': '2024-01-01 10:00:02', 'message': 'DialogState IDLE', 'pid': '123', 'tid': '456'}
        ]
        
        df = pd.DataFrame(sample_data)
        sessions = tracer.identify_utterance_sessions(df)
        print(f"✓ Utterance tracer successful: {len(sessions)} sessions found")
        return True
    except Exception as e:
        print(f"✗ Utterance tracer error: {e}")
        return False

def test_config_manager():
    """Test configuration manager"""
    try:
        from utils.config_manager import get_config_manager
        
        config = get_config_manager()
        max_file_size = config.get('analysis.max_file_size_mb', 500)
        print(f"✓ Config manager successful: max_file_size={max_file_size}MB")
        return True
    except Exception as e:
        print(f"✗ Config manager error: {e}")
        return False

def main():
    """Run all tests"""
    print("Testing Advanced Log Reader Components")
    print("=" * 50)
    
    tests = [
        ("Imports", test_imports),
        ("Log Analyzer", test_log_analyzer),
        ("Utterance Tracer", test_utterance_tracer),
        ("Config Manager", test_config_manager)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        if test_func():
            passed += 1
    
    print("\n" + "=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The application should work correctly.")
        return 0
    else:
        print("⚠️ Some tests failed. Check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())