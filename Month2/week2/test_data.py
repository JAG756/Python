# test_data.py
test_dataset = {
    "定义类": [
        {"question": "什么是RAG？", "keywords": ["检索增强生成", "幻觉"]},
        {"question": "CNN是什么？", "keywords": ["卷积神经网络", "图像"]},
        {"question": "Transformer是什么？", "keywords": ["自注意力", "架构"]},
        {"question": "LoRA是什么？", "keywords": ["低秩适配", "微调"]},
    ],
    "应用类": [
        {"question": "RAG有什么用？", "keywords": ["幻觉", "知识过时"]},
        {"question": "CNN在哪里应用？", "keywords": ["图像分类", "目标检测"]},
        {"question": "Transformer能做什么？", "keywords": ["智能客服", "代码助手"]},
    ],
    "追问类": [
        {"question": "什么是RAG？", "follow_up": "它有什么优点？", "keywords": ["幻觉", "知识过时"]},
        {"question": "CNN是什么？", "follow_up": "它怎么工作？", "keywords": ["卷积层", "池化层"]},
    ],
    "边界类": [
        {"question": "你好", "expected": ["你好", "Hi", "您好"]},
        {"question": "谢谢", "expected": ["不客气", "欢迎"]},
        {"question": "天气怎么样", "expected": ["未找到", "无法回答", "技术问题"]},
        {"question": "你是谁", "expected": ["助手", "RAG"]},
    ]
}