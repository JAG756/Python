import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import warnings
warnings.filterwarnings('ignore')

class ChineseChatBot:
    def __init__(self, model_name="Qwen/Qwen2.5-7B-Instruct"):
        """
        初始化中文聊天模型
        model_name: 可选 "Qwen/Qwen2.5-7B-Instruct" 或 "THUDM/chatglm3-6b"
        """
        print("正在加载模型，请稍候...")
        
        # 设置设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")
        
        # 加载tokenizer和模型
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            padding_side="left"
        )
        
        # 使用4bit量化加载，节省显存
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            load_in_4bit=True,  # 4bit量化，显存占用约4-5GB
            bnb_4bit_compute_dtype=torch.float16
        )
        
        print("模型加载完成！")
        
        # 对话历史
        self.history = []
        
    def chat(self, user_input, max_length=512, temperature=0.7):
        """
        与模型对话
        user_input: 用户输入
        max_length: 最大生成长度
        temperature: 温度参数，控制随机性
        """
        # 构建消息格式（Qwen格式）
        messages = []
        
        # 添加历史对话
        for i, (q, a) in enumerate(self.history[-5:]):  # 保留最近5轮对话
            messages.append({"role": "user", "content": q})
            messages.append({"role": "assistant", "content": a})
        
        # 添加当前输入
        messages.append({"role": "user", "content": user_input})
        
        # 应用聊天模板
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # 编码输入
        inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
        
        # 生成回复
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=temperature,
                do_sample=True,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # 解码输出
        response = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        # 保存历史
        self.history.append((user_input, response))
        
        return response
    
    def clear_history(self):
        """清空对话历史"""
        self.history = []
        print("对话历史已清空")

def main():
    """主函数"""
    print("="*50)
    print("中文AI聊天机器人")
    print("="*50)
    print("输入 'quit' 退出对话")
    print("输入 'clear' 清空对话历史")
    print("="*50)
    
    # 创建聊天机器人实例
    try:
        bot = ChineseChatBot()
        
        while True:
            # 获取用户输入
            user_input = input("\n你: ").strip()
            
            # 退出判断
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("再见！")
                break
            
            # 清空历史
            if user_input.lower() == 'clear':
                bot.clear_history()
                continue
            
            # 跳过空输入
            if not user_input:
                continue
            
            # 获取回复
            try:
                response = bot.chat(user_input)
                print(f"\nAI: {response}")
            except Exception as e:
                print(f"生成回复时出错: {e}")
                
    except KeyboardInterrupt:
        print("\n\n程序被中断")
    except Exception as e:
        print(f"初始化失败: {e}")
        print("请确保已安装必要的库: pip install transformers torch accelerate bitsandbytes")

if __name__ == "__main__":
    main()