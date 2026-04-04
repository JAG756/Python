"""
垂直场景测试脚本
放在: E:\Pythod\Month2\week2\test_vertical.py
"""

from 关于多轮回答 import answer, sessions
from datetime import datetime
import time

def run_vertical_tests():
    """运行垂直场景测试"""
    
    # 测试用例
    tests = [
        # 领域内测试
        ("RAG定义", "什么是RAG？", ["RAG", "检索增强生成"]),
        ("CNN定义", "CNN是什么？", ["CNN", "卷积神经网络"]),
        ("Transformer定义", "Transformer是什么？", ["Transformer", "自注意力"]),
        ("LoRA定义", "LoRA是什么？", ["LoRA", "低秩适配"]),
        ("向量数据库", "向量数据库是什么？", ["向量", "Chroma"]),
        
        # 应用类测试
        ("RAG应用", "RAG有什么用？", ["幻觉", "知识过时"]),
        ("CNN应用", "CNN在哪里应用？", ["图像", "分类"]),
        
        # 边界测试
        ("问候", "你好", ["你好"]),
        ("超出领域", "今天天气怎么样", ["未找到"]),
        ("模糊问题", "那个是什么", ["未找到"]),
    ]
    
    print("\n" + "=" * 70)
    print("🚀 垂直场景问答测试")
    print("=" * 70)
    print(f"⏰ 测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    passed = 0
    total = 0
    session_id = "test_session"
    results = []
    
    for name, question, keywords in tests:
        total += 1
        print(f"\n📝 测试 {total}: {name}")
        print(f"❓ 问题: {question}")
        print(f"🔍 预期关键词: {keywords}")
        
        try:
            # 调用你的RAG函数
            answer_text, _ = answer(question, session_id, max_len=200, temp=0.3)
            print(f"💡 回答: {answer_text[:150]}..." if len(answer_text) > 150 else f"💡 回答: {answer_text}")
            
            # 检查是否包含关键词
            matched = False
            matched_kw = None
            for kw in keywords:
                if kw in answer_text:
                    matched = True
                    matched_kw = kw
                    break
            
            if matched:
                print(f"✅ 通过 (匹配关键词: {matched_kw})")
                passed += 1
                results.append({"name": name, "question": question, "passed": True})
            else:
                print(f"❌ 失败 (未找到预期关键词)")
                results.append({"name": name, "question": question, "passed": False, "answer": answer_text[:100]})
                
        except Exception as e:
            print(f"❌ 错误: {e}")
            results.append({"name": name, "question": question, "passed": False, "error": str(e)})
        
        # 避免请求过快
        time.sleep(0.5)
    
    # 打印总结
    print("\n" + "=" * 70)
    print("📊 测试总结")
    print("=" * 70)
    print(f"总计: {total} | ✅ 通过: {passed} | ❌ 失败: {total - passed}")
    print(f"通过率: {passed/total*100:.1f}%")
    
    # 显示失败详情
    failed_results = [r for r in results if not r["passed"]]
    if failed_results:
        print("\n" + "=" * 70)
        print("❌ 失败详情")
        print("=" * 70)
        for r in failed_results:
            print(f"\n• {r['name']}: {r['question']}")
            print(f"  回答: {r['answer']}...")
    
    # 清空测试会话
    if "test_session" in sessions:
        del sessions["test_session"]
    
    print("\n" + "=" * 70)
    return passed, total

def test_multi_turn():
    """测试多轮对话"""
    print("\n" + "=" * 70)
    print("💬 多轮对话测试")
    print("=" * 70)
    print("测试理解'它'、'这个'等指代能力\n")
    
    session_id = "multi_turn_test"
    
    # 测试序列
    conversations = [
        ("第1轮", "什么是RAG？", ["RAG", "检索增强生成"]),
        ("第2轮", "它有什么优点？", ["幻觉", "知识过时"]),
        ("第3轮", "怎么部署？", ["未找到"])  # 知识库里没有部署相关内容
    ]
    
    print("测试序列:")
    for i, (round_name, q, _) in enumerate(conversations, 1):
        print(f"  {i}. {q}")
    
    print("\n" + "-" * 50)
    
    for round_name, question, keywords in conversations:
        print(f"\n{round_name}: {question}")
        answer_text, _ = answer(question, session_id, max_len=200, temp=0.3)
        print(f"回答: {answer_text[:150]}...")
        
        # 检查上下文理解
        if "它" in question and "RAG" in answer_text:
            print("✅ 上下文理解正确（正确指代RAG）")
        elif "它" in question:
            print("⚠️ 上下文理解可能需要改进")
    
    # 清空测试会话
    if "multi_turn_test" in sessions:
        del sessions["multi_turn_test"]
    
    print("\n" + "=" * 70)

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("🔧 RAG系统测试工具")
    print("=" * 70)
    
    print("\n请选择测试模式:")
    print("1. 完整垂直场景测试")
    print("2. 多轮对话测试")
    print("3. 全部运行")
    
    choice = input("\n请输入选项 (1/2/3): ").strip()
    
    if choice == "1":
        run_vertical_tests()
    elif choice == "2":
        test_multi_turn()
    elif choice == "3":
        run_vertical_tests()
        test_multi_turn()
    else:
        print("无效选项，运行默认测试...")
        run_vertical_tests()