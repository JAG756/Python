"""
自动化测试框架
放在: E:\Pythod\Month2\week2\自动化测试脚本.py
"""

from 关于多轮回答 import answer, sessions
from datetime import datetime
import json
import time

class VerticalTester:
    """垂直场景测试器"""
    
    def __init__(self):
        self.results = []
        self.session_id = "auto_test_session"
    
    def test_single(self, name, question, expected_keywords):
        """单轮测试"""
        print(f"  测试: {name} - {question[:30]}...", end=" ")
        
        try:
            answer_text, _ = answer(question, self.session_id, max_len=200, temp=0.3)
            passed = any(kw in answer_text for kw in expected_keywords)
            
            self.results.append({
                "type": "single",
                "name": name,
                "question": question,
                "passed": passed,
                "answer": answer_text[:200],
                "expected": expected_keywords
            })
            
            status = "✅" if passed else "❌"
            print(status)
            return passed
            
        except Exception as e:
            print("❌")
            self.results.append({
                "type": "single",
                "name": name,
                "question": question,
                "passed": False,
                "error": str(e)
            })
            return False
    
    def test_multi_turn(self, name, first, second, expected_keywords):
        """多轮测试（追问）"""
        print(f"  测试: {name}")
        print(f"    第1轮: {first}")
        print(f"    第2轮: {second}", end=" ")
        
        try:
            # 第一轮
            answer(first, self.session_id)
            # 第二轮（应该理解上下文）
            answer_text, _ = answer(second, self.session_id)
            passed = any(kw in answer_text for kw in expected_keywords)
            
            self.results.append({
                "type": "multi_turn",
                "name": name,
                "first": first,
                "second": second,
                "passed": passed,
                "answer": answer_text[:200],
                "expected": expected_keywords
            })
            
            status = "✅" if passed else "❌"
            print(status)
            return passed
            
        except Exception as e:
            print("❌")
            self.results.append({
                "type": "multi_turn",
                "name": name,
                "first": first,
                "second": second,
                "passed": False,
                "error": str(e)
            })
            return False
    
    def run_all_tests(self):
        """运行所有测试"""
        print("\n" + "=" * 70)
        print("🚀 自动化测试开始")
        print("=" * 70)
        print(f"⏰ 时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 70)
        
        # ========== 定义类测试 ==========
        print("\n📚 【定义类测试】")
        definition_tests = [
            ("RAG定义", "什么是RAG？", ["RAG", "检索增强生成"]),
            ("CNN定义", "CNN是什么？", ["CNN", "卷积神经网络"]),
            ("Transformer定义", "Transformer是什么？", ["Transformer", "自注意力"]),
            ("LoRA定义", "LoRA是什么？", ["LoRA", "低秩适配"]),
            ("向量数据库定义", "向量数据库是什么？", ["向量", "Chroma"]),
        ]
        
        for test in definition_tests:
            self.test_single(*test)
            time.sleep(0.3)
        
        # ========== 应用类测试 ==========
        print("\n🔧 【应用类测试】")
        application_tests = [
            ("RAG应用", "RAG有什么用？", ["幻觉", "知识过时"]),
            ("CNN应用", "CNN在哪里应用？", ["图像", "分类"]),
            ("Transformer应用", "Transformer能做什么？", ["客服", "代码"]),
        ]
        
        for test in application_tests:
            self.test_single(*test)
            time.sleep(0.3)
        
        # ========== 多轮测试 ==========
        print("\n💬 【多轮对话测试】")
        multi_turn_tests = [
            ("RAG追问", "什么是RAG？", "它有什么优点？", ["幻觉", "知识过时"]),
            ("CNN追问", "CNN是什么？", "它怎么工作？", ["卷积层", "池化层"]),
        ]
        
        for test in multi_turn_tests:
            self.test_multi_turn(*test)
            time.sleep(0.5)
        
        # ========== 边界测试 ==========
        print("\n🚧 【边界测试】")
        boundary_tests = [
            ("问候语", "你好", ["你好"]),
            ("感谢语", "谢谢", ["不客气", "欢迎"]),
            ("超出领域", "今天天气怎么样", ["未找到"]),
            ("模糊问题", "那个是什么", ["未找到"]),
        ]
        
        for test in boundary_tests:
            self.test_single(*test)
            time.sleep(0.3)
        
        # 清空测试会话
        if self.session_id in sessions:
            del sessions[self.session_id]
        
        # 生成报告
        self.generate_report()
    
    def generate_report(self):
        """生成测试报告"""
        total = len(self.results)
        passed = sum(1 for r in self.results if r["passed"])
        
        print("\n" + "=" * 70)
        print("📊 测试报告")
        print("=" * 70)
        print(f"总计: {total} | ✅ 通过: {passed} | ❌ 失败: {total - passed}")
        print(f"通过率: {passed/total*100:.1f}%")
        
        # 分类统计
        single_tests = [r for r in self.results if r["type"] == "single"]
        multi_tests = [r for r in self.results if r["type"] == "multi_turn"]
        
        if single_tests:
            single_passed = sum(1 for r in single_tests if r["passed"])
            print(f"\n单轮测试: {single_passed}/{len(single_tests)} ({single_passed/len(single_tests)*100:.0f}%)")
        
        if multi_tests:
            multi_passed = sum(1 for r in multi_tests if r["passed"])
            print(f"多轮测试: {multi_passed}/{len(multi_tests)} ({multi_passed/len(multi_tests)*100:.0f}%)")
        
        # 失败详情
        failed = [r for r in self.results if not r["passed"]]
        if failed:
            print("\n" + "-" * 70)
            print("❌ 失败详情:")
            print("-" * 70)
            for r in failed:
                if r["type"] == "single":
                    print(f"  • {r['name']}: {r['question']}")
                    if "answer" in r:
                        print(f"    回答: {r['answer'][:100]}...")
                    if "expected" in r:
                        print(f"    预期: {r['expected']}")
                else:
                    print(f"  • {r['name']}: {r['first']} → {r['second']}")
                    if "answer" in r:
                        print(f"    回答: {r['answer'][:100]}...")
        
        # 保存报告
        report_file = f"test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)
        print(f"\n📄 详细报告已保存: {report_file}")

def quick_test():
    """快速测试单个问题"""
    print("\n" + "=" * 70)
    print("🔍 快速测试")
    print("=" * 70)
    
    question = input("\n请输入问题: ").strip()
    if not question:
        print("问题不能为空")
        return
    
    session_id = "quick_test"
    
    print(f"\n❓ 问题: {question}")
    answer_text, _ = answer(question, session_id)
    print(f"💡 回答: {answer_text}")
    
    # 清空会话
    if session_id in sessions:
        del sessions[session_id]

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("🤖 RAG系统自动化测试工具")
    print("=" * 70)
    
    print("\n请选择模式:")
    print("1. 运行全部自动化测试")
    print("2. 快速测试单个问题")
    
    choice = input("\n请输入选项 (1/2): ").strip()
    
    if choice == "1":
        tester = VerticalTester()
        tester.run_all_tests()
    elif choice == "2":
        quick_test()
    else:
        print("无效选项，运行默认测试...")
        tester = VerticalTester()
        tester.run_all_tests()