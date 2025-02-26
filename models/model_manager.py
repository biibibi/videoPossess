from abc import ABC, abstractmethod
import google.generativeai as genai
from PIL import Image
import base64
import ollama
import io
import os
from config import MODELS, GEMINI_CONFIG, MINIMAX_CONFIG, QWEN_CONFIG
import requests
from openai import OpenAI
import json
import time
from typing import Tuple, Optional
import dashscope
from http import HTTPStatus
from dashscope import MultiModalConversation
import asyncio
import traceback

class BaseModelHandler(ABC):
    """基础模型处理器抽象类"""
    
    @abstractmethod
    async def analyze_image(self, image_path: str, prompt: str) -> Tuple[bool, str, float]:
        """分析图像的抽象方法"""
        pass

    async def preprocess_image(self, image_path: str) -> Tuple[Image.Image, str]:
        """图像预处理方法"""
        try:
            # 打开并优化图像
            image = Image.open(image_path)
            
            # 限制最大尺寸
            max_size = (800, 800)
            image.thumbnail(max_size, Image.Resampling.LANCZOS)
            
            # 转换为 RGB 模式（如果需要）
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # 保存优化后的图像
            buffer = io.BytesIO()
            image.save(buffer, format='JPEG', quality=85, optimize=True)
            image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            return image, image_base64
            
        except Exception as e:
            print(f"图像预处理错误: {str(e)}")
            raise

    @staticmethod
    def parse_response(response_text: str, prompt: str) -> Tuple[bool, str, float]:
        """通用响应解析方法"""
        response_lines = [line.strip() for line in response_text.strip().split('\n') if line.strip()]
        
        answer = False
        description = []
        confidence = 0
        
        for line in response_lines:
            line_lower = line.lower()
            
            if any(keyword in line_lower for keyword in ['回答', '答案', 'answer']):
                answer_text = line.split('：', 1)[1].strip() if '：' in line else (line.split(':', 1)[1].strip() if ':' in line else line)
                answer = any(word in answer_text.lower() for word in ['是', 'yes', '有', '存在', '可以看到'])
                description.append(line)
            
            elif any(keyword in line_lower for keyword in ['描述', '解释', 'description']):
                desc_text = line.split('：', 1)[1].strip() if '：' in line else (line.split(':', 1)[1].strip() if ':' in line else line)
                description.append(line)
            
            elif len(line) > 20 and not any(keyword in line_lower for keyword in ['回答', '答案', '置信度', 'confidence']):
                description.append(line)
            
            elif any(keyword in line_lower for keyword in ['置信度', 'confidence']):
                try:
                    confidence_text = line.split('：', 1)[1].strip() if '：' in line else (line.split(':', 1)[1].strip() if ':' in line else line)
                    confidence = int(''.join(filter(str.isdigit, confidence_text)))
                    description.append(line)
                except ValueError:
                    confidence = 5

        if not description and len(response_text.strip()) > 0:
            description.append(response_text.strip())

        # 检查描述中是否包含所有目标关键词
        keywords = prompt.lower().split()
        description_text = ' '.join(description).lower()
        keywords_found = all(keyword in description_text for keyword in keywords)
        
        # 检查颜色和位置描述
        color_match = False
        if '白' in prompt or '白色' in prompt:
            color_match = any(word in description_text for word in ['白', '白色', '浅色'])
        
        # 综合判断
        is_match = (answer or keywords_found or color_match) and confidence >= 5
        
        return is_match, '\n'.join(description), confidence

class LlamaHandler(BaseModelHandler):
    def __init__(self):
        self.config = MODELS["llama"]
        self.client = ollama.Client(host=self.config["base_url"])
        self.model_name = "llama3.2-vision:latest"
    
    async def analyze_image(self, image_path: str, prompt: str) -> Tuple[bool, str, float]:
        try:
            # 预处理图像
            _, image_base64 = await self.preprocess_image(image_path)
            
            # 构建中文提示词
            prompt_text = f"""请仔细分析这张图片，并回答以下问题：

            问题：这张图片中是否包含 {prompt}？

            请按照以下格式回答：
            回答：[是/否]
            描述：[详细描述图片中的所有内容，包括目标物体（如果存在）的位置和特征，以及其他主要内容]
            置信度：[1-10分]

            注意：
            1. 如果存在目标物体，请详细描述它的位置和特征
            2. 无论是否存在目标物体，都请描述图片中的主要内容
            3. 置信度要根据判断的把握程度给出"""

            # 构建消息格式
            response = await asyncio.to_thread(
                self.client.chat,
                model=self.model_name,
                messages=[{
                    'role': 'user',
                    'content': prompt_text,
                    'images': [image_path]
                }],
                stream=False
            )
            
            if not response or 'message' not in response:
                print(f"[Llama] 无效响应: {response}")
                return False, "模型未返回有效响应", 0
                
            return self.parse_response(response['message']['content'], prompt)
            
        except Exception as e:
            print(f"[Llama] 分析错误: {str(e)}\n{traceback.format_exc()}")
            return False, f"分析过程中出错: {str(e)}", 0

class GeminiHandler(BaseModelHandler):
    def __init__(self):
        self.config = MODELS["gemini"]
        genai.configure(api_key=self.config["api_key"])
        os.environ["HTTPS_PROXY"] = GEMINI_CONFIG["proxy"]["https"]
        os.environ["HTTP_PROXY"] = GEMINI_CONFIG["proxy"]["http"]
        self.model = genai.GenerativeModel(
            self.config["name"],
                generation_config={
                    "temperature": self.config["temperature"],
                    "max_output_tokens": self.config["max_output_tokens"],
                    "top_p": self.config["top_p"],
                    "top_k": self.config["top_k"]
                }
            )
            
    async def analyze_image(self, image_path: str, prompt: str) -> Tuple[bool, str, float]:
        try:
            image = Image.open(image_path)
            
            prompt_text = f"""请分析这张图片并回答以下问题：
            1. 图片中是否存在 {prompt}？
            2. 如果存在，请详细描述它的外观和在图片中的位置。
            3. 如果不存在，请描述你在图片中看到的内容。
            4. 请给出你的判断的置信度（1-10分）。

            请按以下格式回答：
            回答：[是/否]
            描述：[详细描述]
            置信度：[1-10]"""

            # 将图像转换为字节流
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format=image.format or 'JPEG')
            img_byte_arr = img_byte_arr.getvalue()

            response = await asyncio.to_thread(
                self.model.generate_content,
                [prompt_text, {"mime_type": "image/jpeg", "data": img_byte_arr}]
            )
            
            return self.parse_response(response.text, prompt)
            
        except Exception as e:
            print(f"Gemini分析错误: {str(e)}")
            return False, f"分析过程中出错: {str(e)}", 0

class MinimaxHandler(BaseModelHandler):
    def __init__(self):
        self.config = MODELS["minimax"]
        self.client = OpenAI(
            api_key=self.config["api_key"],
            base_url=self.config["base_url"]
        )
    
    async def analyze_image(self, image_path: str, prompt: str) -> Tuple[bool, str, float]:
        try:
            with open(image_path, "rb") as image_file:
                image_base64 = base64.b64encode(image_file.read()).decode('utf-8')
            
            messages = [
                {
                    "role": "system",
                    "content": "MM智能助理是一款由MiniMax自研的，没有调用其他产品的接口的大型语言模型。"
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"""请分析这张图片并回答以下问题：
                            1. 图片中是否存在 {prompt}？
                            2. 如果存在，请详细描述它的外观和在图片中的位置。
                            3. 如果不存在，请描述你在图片中看到的内容。
                            4. 请给出你的判断的置信度（1-10分）。

                            请按以下格式回答：
                            回答：[是/否]
                            描述：[详细描述]
                            置信度：[1-10]"""
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}"
                            }
                        }
                    ]
                }
            ]
            
            completion = self.client.chat.completions.create(
                model=self.config["name"],
                messages=messages,
                max_tokens=4096,
                group_id=self.config["group_id"],
                stream=False,
                temperature=0.7,
                top_p=0.8
            )
            
            if completion.choices:
                return self.parse_response(completion.choices[0].message.content, prompt)
            
            raise Exception("模型未返回有效响应")
            
        except Exception as e:
            print(f"Minimax分析错误: {str(e)}")
            return False, f"分析过程中出错: {str(e)}", 0

class QwenHandler(BaseModelHandler):
    def __init__(self):
        self.config = MODELS["qwen"]
        self.client = OpenAI(
            api_key=self.config["api_key"],
            base_url=self.config["base_url"]
        )
    
    async def analyze_image(self, image_path: str, prompt: str) -> Tuple[bool, str, float]:
        try:
            # 预处理图像
            _, image_base64 = await self.preprocess_image(image_path)
            
            # 构建消息结构
            messages = [{
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"""请分析这张图片并回答：
                        1. 图片中是否存在 {prompt}？
                        2. 详细描述图片中的所有内容，包括：
                           - 如果存在目标物体，描述它的位置和特征
                           - 描述图片中的其他主要内容和场景
                        3. 给出判断的置信度（1-10分）
                        
                        请按以下格式回答：
                        回答：[是/否]
                        描述：[详细描述]
                        置信度：[1-10]"""
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}
                    }
                ]
            }]
            
            # 异步调用 API
            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                model=self.config["name"],
                messages=messages,
                max_tokens=self.config["max_tokens"],
                temperature=self.config["temperature"]
            )
            
            if response.choices:
                return self.parse_response(response.choices[0].message.content, prompt)
            
            return False, "模型未返回有效响应", 0
            
        except Exception as e:
            return False, f"分析错误: {str(e)}", 0

class ModelManager:
    @staticmethod
    def get_model_handler(model_name: str) -> Optional[BaseModelHandler]:
        handlers = {
            "llama": LlamaHandler,
            "gemini": GeminiHandler,
            "minimax": MinimaxHandler,
            "qwen": QwenHandler
        }
        
        handler_class = handlers.get(model_name.lower())
        if handler_class:
            return handler_class()
        return None