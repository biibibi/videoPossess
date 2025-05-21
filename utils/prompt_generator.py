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
        # 检查是否为智能分析模式
        is_analysis_mode = False
        user_query = ""
        
        # 检查targets中是否包含智能分析模式标记
        if len(targets) >= 2 and "mode=intelligent_analysis" in targets:
            is_analysis_mode = True
            # 第一个元素是用户的查询内容
            user_query = targets[0]
            # 移除标记，只保留查询内容
            targets = [user_query]
        
        if is_analysis_mode:
            # 智能分析模式的提示词
            base_prompt = f"请分析图像并回答用户的问题: {user_query}"
            
            # 不同模型的提示词模板
            prompt_templates = {
                "minimax": base_prompt + "。请详细分析图像内容，提供有用的信息来回答用户问题。保持回答简洁明了。",
                "gemini": base_prompt + "。请详细分析图像并提供信息丰富的回答。",
                "qwen": base_prompt + "。请详细分析图像内容并提供专业的回答。",
                "llama": base_prompt + "。请分析图像并给出详细回答。",
                "douban": base_prompt + "。请详细分析图像内容并提供专业的回答。"
            }
        else:
            # 原有的目标搜索模式
            base_prompt = f"分析图片是否包含: {', '.join(targets)}。"
            
            # 不同模型的提示词模板
            prompt_templates = {
                "minimax": base_prompt + "请用JSON格式返回分析结果，格式如下: {\"targets\": [{\"name\": \"目标1\", \"found\": true/false}], \"description\": \"对图像的简短描述\"}。在targets数组中，必须包含我指定的每个目标，found字段必须是布尔值true或false。不要使用不同的JSON结构。",
                "gemini": base_prompt + "返回JSON格式:{\"targets\":[{\"name\":\"目标名\",\"found\":true/false}],\"description\":\"对图像的简短描述\"}",
                "qwen": base_prompt + "请用JSON格式返回分析结果，格式如下: {\"targets\": [{\"name\": \"目标1\", \"found\": true/false}], \"description\": \"对图像的简短描述\"}。在targets数组中，必须包含我指定的每个目标，found字段必须是布尔值true或false。不要使用不同的JSON结构，确保严格按照以上格式回复，不要有任何前缀或后缀解释。",
                "llama": base_prompt + "返回JSON格式:{\"targets\":[{\"name\":\"目标名\",\"found\":true/false}],\"description\":\"对图像的简短描述\"}",
                "douban": base_prompt + "请用JSON格式返回分析结果，格式如下: {\"targets\": [{\"name\": \"目标1\", \"found\": true/false}], \"description\": \"对图像的简短描述\"}。在targets数组中，必须包含我指定的每个目标，found字段必须是布尔值true或false。"
            }
        
        # 返回对应模型的提示词，如果没有对应的模板则使用默认模板
        return prompt_templates.get(model_name.lower(), prompt_templates["minimax"])
    
    @staticmethod
    def generate_system_prompt(model_name: str = "minimax", is_analysis_mode: bool = False) -> str:
        """
        生成系统提示词
        
        Args:
            model_name: 模型名称
            is_analysis_mode: 是否为智能分析模式
            
        Returns:
            str: 系统提示词
        """
        if is_analysis_mode:
            # 智能分析模式的系统提示词
            system_prompts = {
                "minimax": "你是一个专业的图像分析助手，请帮助用户分析图像中的内容并回答问题。提供详细、准确、有用的信息。",
                "gemini": "你是一个专业的图像分析助手，请帮助用户分析图像中的内容并回答问题。",
                "qwen": "你是一个专业的视觉分析助手，请帮助用户分析图像中的内容并回答问题。",
                "llama": "你是一个专业的图像分析助手，请帮助用户分析图像中的内容并回答问题。",
                "douban": "你是一个专业的视觉分析助手，请帮助用户分析图像中的内容并回答问题。"
            }
        else:
            # 目标搜索模式的系统提示词
            system_prompts = {
                "minimax": "你是一个专业的图像分析助手，请帮助用户分析图像中的内容。你应该使用规范的JSON格式返回分析结果，包含targets数组和description字段。",
                "gemini": "你是一个专业的图像分析助手，请帮助用户分析图像中的内容。",
                "qwen": "你是一个专业的视觉分析助手，请帮助用户分析图像中的内容。你的回答必须是规范的JSON格式，包含targets数组和description字段。不要包含任何其他内容或解释。",
                "llama": "你是一个专业的图像分析助手，请帮助用户分析图像中的内容。",
                "douban": "你是一个专业的视觉分析助手，请帮助用户分析图像中的内容。你的回答必须是规范的JSON格式，包含targets数组和description字段。"
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
        # 检查是否为智能分析模式
        is_analysis_mode = False
        user_query = ""
        
        # 检查targets中是否包含智能分析模式标记
        if len(targets) >= 2 and "mode=intelligent_analysis" in targets:
            is_analysis_mode = True
            # 第一个元素是用户的查询内容
            user_query = targets[0]
            # 移除标记，保留用户查询作为targets
            targets = [user_query]
        
        # 获取系统提示词和用户提示词
        system_prompt = PromptGenerator.generate_system_prompt(model_name, is_analysis_mode)
        user_prompt = PromptGenerator.generate_prompt(targets, model_name)
        
        # 构建系统消息
        system_message = {
            "role": "system",
            "content": system_prompt
        }
        
        # 为不同模型构建适合的用户消息格式
        if model_name.lower() in ["minimax", "gemini", "douban"]:
            # MiniMax、Gemini和豆包使用特定的内容格式
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
