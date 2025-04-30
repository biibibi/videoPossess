"""
提示词生成器
负责根据不同模型生成相应的提示词
"""

from typing import List, Dict, Optional, Tuple, Any

class PromptGenerator:
    """提示词生成器类"""
    
    @staticmethod
    def generate_prompt(targets: List[str], model_name: str = "minimax") -> str:
        """
        生成模型提示词
        
        Args:
            targets: 搜索目标列表
            model_name: 模型名称，用于根据不同模型生成不同的提示词
            
        Returns:
            str: 生成的提示词
        """
        # 基础提示词模板
        base_prompt = f"分析图片是否包含: {', '.join(targets)}。"
        
        # 不同模型的提示词模板
        prompt_templates = {
            "minimax": base_prompt + """请用JSON格式返回分析结果，格式如下:
{
  "targets": [
    {"name": "目标1", "found": true/false},
    {"name": "目标2", "found": true/false}
  ],
  "description": "对图像的简短描述"
}
在targets数组中，必须包含我指定的每个目标，found字段必须是布尔值true或false。不要使用不同的JSON结构。""",
            "gemini": base_prompt + "返回JSON格式:{'targets':[{'name':'目标名','found':true/false}],'description':'对图像的简短描述'}",
            "qwen": base_prompt + """请用JSON格式返回分析结果，格式如下:
{
  "targets": [
    {"name": "目标1", "found": true/false},
    {"name": "目标2", "found": true/false}
    ...
  ],
  "description": "对图像的简短描述"
}
在targets数组中，必须包含我指定的每个目标，found字段必须是布尔值true或false。
不要使用不同的JSON结构，确保严格按照以上格式回复，不要有任何前缀或后缀解释。""",
            "llama": base_prompt + "返回JSON格式:{'targets':[{'name':'目标名','found':true/false}],'description':'对图像的简短描述'}"
        }
        
        # 返回对应模型的提示词，如果没有对应的模板则使用默认模板
        return prompt_templates.get(model_name.lower(), prompt_templates["minimax"])
    
    @staticmethod
    def generate_system_prompt(model_name: str = "minimax") -> str:
        """
        生成系统提示词
        
        Args:
            model_name: 模型名称
            
        Returns:
            str: 系统提示词
        """
        # 不同模型的系统提示词
        system_prompts = {
            "minimax": "你是一个专业的图像分析助手，请帮助用户分析图像中的内容。你应该使用规范的JSON格式返回分析结果，包含targets数组和description字段。",
            "gemini": "你是一个专业的图像分析助手，请帮助用户分析图像中的内容。",
            "qwen": "你是一个专业的视觉分析助手，请帮助用户分析图像中的内容。你的回答必须是规范的JSON格式，包含targets数组和description字段。不要包含任何其他内容或解释。",
            "llama": "你是一个专业的图像分析助手，请帮助用户分析图像中的内容。"
        }
        
        # 返回对应模型的系统提示词，如果没有对应的模板则使用默认模板
        return system_prompts.get(model_name.lower(), system_prompts["minimax"])
    
    @staticmethod
    def generate_messages(targets: List[str], model_name: str = "minimax") -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        生成模型消息，包括系统消息和用户消息
        
        Args:
            targets: 搜索目标列表
            model_name: 模型名称
            
        Returns:
            Tuple[Dict, Dict]: 系统消息和用户消息的字典
        """
        # 获取系统提示词和用户提示词
        system_prompt = PromptGenerator.generate_system_prompt(model_name)
        user_prompt = PromptGenerator.generate_prompt(targets, model_name)
        
        # 构建系统消息
        system_message = {
            "role": "system",
            "content": system_prompt
        }
        
        # 为不同模型构建适合的用户消息格式
        if model_name.lower() == "minimax" or model_name.lower() == "gemini":
            # MiniMax和Gemini使用特定的内容格式
            user_message = {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    # image将在后续处理中添加
                ]
            }
        elif model_name.lower() == "qwen":
            # 千问模型使用特定的内容格式，确保与process_qwen兼容
            user_message = {
                "role": "user",
                "content": [
                    {"text": user_prompt}
                    # image将在后续处理中添加
                ]
            }
        else:
            # 通用格式
            user_message = {
                "role": "user",
                "content": user_prompt
            }
        
        return system_message, user_message
