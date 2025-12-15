#!/usr/bin/env python3
"""
测试运行脚本
用于运行所有单元测试并生成测试报告
"""

import unittest
import sys
import os
import time
from unittest import TextTestRunner, TestResult

# 将项目根目录添加到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class CustomTestResult(TestResult):
    """自定义测试结果类，用于生成更详细的测试报告"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.start_time = 0
        self.end_time = 0
        self.test_cases = []
        self.total_tests = 0
        self.total_passed = 0
        self.total_failed = 0
        self.total_errors = 0
        self.total_skipped = 0
    
    def startTestRun(self):
        """测试运行开始时调用"""
        self.start_time = time.time()
        print("=" * 60)
        print("Wordle 预测模型 - 单元测试报告")
        print("=" * 60)
    
    def stopTestRun(self):
        """测试运行结束时调用"""
        self.end_time = time.time()
        self.total_tests = self.total_passed + self.total_failed + self.total_errors + self.total_skipped
        
        print("\n" + "=" * 60)
        print("测试结果汇总")
        print("=" * 60)
        print(f"测试总数: {self.total_tests}")
        print(f"通过: {self.total_passed}")
        print(f"失败: {self.total_failed}")
        print(f"错误: {self.total_errors}")
        print(f"跳过: {self.total_skipped}")
        print(f"测试用时: {self.end_time - self.start_time:.2f} 秒")
        
        if self.total_failed == 0 and self.total_errors == 0:
            print("\n🎉 所有测试通过!")
        else:
            print(f"\n❌ 测试失败: {self.total_failed} 个失败, {self.total_errors} 个错误")
    
    def addSuccess(self, test):
        """测试通过时调用"""
        super().addSuccess(test)
        self.total_passed += 1
        print(f"✓ {test.id()}")
    
    def addFailure(self, test, err):
        """测试失败时调用"""
        super().addFailure(test, err)
        self.total_failed += 1
        print(f"✗ {test.id()}")
    
    def addError(self, test, err):
        """测试出错时调用"""
        super().addError(test, err)
        self.total_errors += 1
        print(f"! {test.id()}")
    
    def addSkip(self, test, reason):
        """测试跳过时调用"""
        super().addSkip(test, reason)
        self.total_skipped += 1
        print(f"- {test.id()} (跳过: {reason})")

def main():
    """主函数，运行所有测试"""
    # 加载测试用例
    test_directory = os.path.dirname(os.path.abspath(__file__))
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover(test_directory, pattern='test_*.py')
    
    # 创建测试运行器
    test_runner = TextTestRunner(resultclass=CustomTestResult, verbosity=0)
    
    # 运行测试
    result = test_runner.run(test_suite)
    
    # 返回测试结果状态码
    return 0 if result.wasSuccessful() else 1

if __name__ == '__main__':
    sys.exit(main())