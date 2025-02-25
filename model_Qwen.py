from google.generativeai import GenerativeModel
import google.generativeai as genai
from PIL import Image
import base64
import ollama
import io
import os
from config import MODELS
import requests
from openai import OpenAI
import json
import time

class ModelManager:
    @staticmethod
    def get_model_handler(model_name):
        if model_name == "llama":
            return LlamaHandler()
        elif model_name == "gemini":
            return GeminiHandler()
        elif model_name == "minimax":
            return MinimaxHandler()
        elif model_name == "qwen":
            return QwenHandler()
        else:
            raise ValueError(f"不支持的模型: {model_name}")

class BaseAnalyzer:
    """基础分析器类，提供通用的分析方法"""
    
    # 扩展同义词映射
    SYNONYM_MAP = {
        '小孩': ['孩子', '儿童', '幼儿', '小朋友', '男孩', '女孩', '学生', '女孩子'],
        '人': ['人物', '行人', '男人', '女人', '男子', '女子', '成人'],
        '汽车': ['车', '轿车', '小车', '车辆', '机动车'],
        '动物': ['宠物', '猫', '狗', '鸟', '鱼'],
    }
    
    # 否定词列表
    NEGATIVE_WORDS = ['没有', '不是', '不存在', '未见', '看不到', '无', '并未', '找不到']
    
    # 肯定词列表
    POSITIVE_WORDS = ['有', '存在', '可以看到', '出现', '看见', '清晰可见', '是']
    
    # 扩展关键词权重映射
    KEYWORD_WEIGHTS = {
        '位置词': {
            '前面': 1.2, '后面': 1.2, '左边': 1.2, '右边': 1.2,
            '中间': 1.2, '旁边': 1.1, '附近': 1.1
        },
        '描述词': {
            '清晰': 1.3, '明显': 1.3, '确实': 1.3, '准确': 1.3,
            '可能': 0.8, '模糊': 0.7, '不确定': 0.6
        },
        '数量词': {
            '一个': 1.2, '两个': 1.2, '几个': 1.1,
            '很多': 0.9, '一些': 0.8
        }
    }
    
    @staticmethod
    def calculate_description_quality(description):
        """评估描述的质量"""
        score = 1.0
        words = description.lower().split()
        
        # 检查描述长度
        if len(words) < 10:
            score *= 0.8
        elif len(words) > 30:
            score *= 1.2
            
        # 检查关键词权重
        for category, weights in BaseAnalyzer.KEYWORD_WEIGHTS.items():
            for word, weight in weights.items():
                if word in description.lower():
                    score *= weight
                    
        # 检查描述的具体程度
        specific_details = ['穿着', '颜色', '大小', '形状', '位置']
        detail_count = sum(1 for detail in specific_details if detail in description)
        score *= (1 + 0.1 * detail_count)
        
        return score

    @staticmethod
    def analyze_text_for_keywords(text, keywords, context_size=30):
        """增强的关键词分析"""
        if not text or not keywords:
            return False, 0
            
        text = text.lower()
        keyword_list = [k.strip() for k in keywords.lower().split() if k.strip()]
        
        # 扩展关键词
        expanded_keywords = set(keyword_list)
        for keyword in keyword_list:
            if keyword in BaseAnalyzer.SYNONYM_MAP:
                expanded_keywords.update(BaseAnalyzer.SYNONYM_MAP[keyword])
        
        # 分析每个关键词
        keyword_scores = []
        for keyword in expanded_keywords:
            if keyword in text:
                # 获取关键词上下文
                indices = [i for i in range(len(text)) if text.startswith(keyword, i)]
                keyword_score = 0
                
                for index in indices:
                    start = max(0, index - context_size)
                    end = min(len(text), index + len(keyword) + context_size)
                    context = text[start:end]
                    
                    # 计算上下文得分
                    context_score = 1.0
                    
                    # 检查否定词
                    if any(neg in context for neg in BaseAnalyzer.NEGATIVE_WORDS):
                        continue
                    
                    # 检查肯定词
                    if any(pos in context for pos in BaseAnalyzer.POSITIVE_WORDS):
                        context_score *= 1.5
                    
                    # 检查位置描述
                    if any(pos in context for pos in BaseAnalyzer.KEYWORD_WEIGHTS['位置词']):
                        context_score *= 1.2
                    
                    # 检查确定性描述
                    if any(desc in context for desc in BaseAnalyzer.KEYWORD_WEIGHTS['描述词']):
                        for desc, weight in BaseAnalyzer.KEYWORD_WEIGHTS['描述词'].items():
                            if desc in context:
                                context_score *= weight
                    
                    keyword_score = max(keyword_score, context_score)
                
                if keyword_score > 0:
                    keyword_scores.append(keyword_score)
        
        if not keyword_scores:
            return False, 0
            
        # 计算最终得分
        final_score = sum(keyword_scores) / len(expanded_keywords)
        
        # 考虑描述质量
        description_quality = BaseAnalyzer.calculate_description_quality(text)
        final_score *= description_quality
        
        return final_score >= 0.8, final_score

    @staticmethod
    def extract_confidence(text, description):
        """提取并标准化置信度分数"""
        try:
            # 找出所有数字
            confidence_numbers = []
            
            # 从置信度行中提取
            if '置信度' in text or 'confidence' in text.lower():
                numbers = [int(n) for n in text.split() if n.isdigit() and 0 <= int(n) <= 10]
                if numbers:
                    confidence_numbers.append(numbers[-1])
            
            # 从描述中提取
            desc_numbers = [int(n) for n in description.split() if n.isdigit() and 0 <= int(n) <= 10]
            if desc_numbers:
                confidence_numbers.extend(desc_numbers)
            
            if not confidence_numbers:
                return 5  # 默认置信度
            
            # 获取最高的置信度值
            confidence = max(confidence_numbers)
            
            # 根据描述内容调整置信度
            if any(neg in description.lower() for neg in BaseAnalyzer.NEGATIVE_WORDS):
                confidence = min(confidence, 4)  # 有否定词时降低置信度
            elif any(pos in description.lower() for pos in BaseAnalyzer.POSITIVE_WORDS):
                confidence = max(confidence, 7)  # 有肯定词时提高置信度
            
            # 根据描述的具体程度调整
            if len(description.split()) > 20:  # 描述较详细
                confidence = max(confidence, 6)  # 详细描述提高最低置信度
            
            print(f"置信度分析 - 原始值: {confidence_numbers}, 调整后: {confidence}")
            print(f"描述长度: {len(description.split())} 词")
            return confidence
            
        except Exception as e:
            print(f"置信度提取错误: {str(e)}")
            return 5

    @staticmethod
    def parse_response(response_text, prompt):
        """增强的响应解析"""
        result = {
            'answer': 'NO',  # 默认为 NO 可能导致误判
            'description': '无法解析描述',
            'confidence': 5,
            'found_keywords': False,
            'keyword_score': 0,
            'description_quality': 0
        }
        
        # 解析响应文本
        full_text = response_text.strip()
        
        # 首先检查是否有明确的否定回答
        for line in full_text.lower().split('\n'):
            if 'answer:' in line and 'no' in line:
                result['answer'] = 'NO'
                break
            elif 'answer:' in line and 'yes' in line:  # 添加这个判断
                result['answer'] = 'YES'
                break
        
        # 分析关键词匹配
        result['found_keywords'], result['keyword_score'] = BaseAnalyzer.analyze_text_for_keywords(full_text, prompt)
        result['description_quality'] = BaseAnalyzer.calculate_description_quality(full_text)
        
        # 处理每一行
        description_parts = []
        has_negative_context = False
        
        for line in response_text.strip().split('\n'):
            line = line.strip().lower()
            
            # 检查否定上下文
            if any(neg in line for neg in BaseAnalyzer.NEGATIVE_WORDS):
                has_negative_context = True
            
            # 提取描述
            if any(marker in line for marker in ['description:', '描述:', '分析:', '说明:']):
                desc = line.split(':', 1)[1].strip() if ':' in line else line
                if desc:
                    description_parts.append(desc)
            elif len(line) > 10 and ':' not in line:
                description_parts.append(line)
        
        result['description'] = ' '.join(description_parts)
        
        # 判断逻辑优化
        is_match = False
        confidence = result['confidence']
        
        # 1. 检查是否有明确的目标词
        target_words = prompt.lower().split()
        description_lower = result['description'].lower()
        
        target_mentioned = False
        has_position = False
        has_features = False
        
        # 检查目标词及其上下文
        for target in target_words:
            if target in description_lower:
                # 获取目标词的上下文
                start = max(0, description_lower.find(target) - 20)
                end = min(len(description_lower), description_lower.find(target) + len(target) + 20)
                context = description_lower[start:end]
                
                # 如果上下文中没有否定词，则认为找到目标
                if not any(neg in context for neg in BaseAnalyzer.NEGATIVE_WORDS):
                    target_mentioned = True
                    
                    # 检查位置描述
                    position_words = ['位于', '在', '左', '右', '前', '后', '旁边', '角', '中间']
                    if any(pos in context for pos in position_words):
                        has_position = True
                    
                    # 检查特征描述
                    feature_words = ['黑色', '白色', '蓝色', '金属', '塑料', '大小', '形状']
                    if any(feat in context for feat in feature_words):
                        has_features = True
        
        # 最终判断逻辑
        if target_mentioned and not has_negative_context:
            if has_position or has_features:
                is_match = True
                confidence = max(confidence, 7)
            elif result['answer'] == 'YES':
                is_match = True
                confidence = max(confidence, 6)
        
        print(f"\n分析详情:")
        print(f"目标词提及: {target_mentioned}")
        print(f"位置描述: {has_position}")
        print(f"特征描述: {has_features}")
        print(f"否定上下文: {has_negative_context}")
        print(f"Answer: {result['answer']}")
        print(f"最终判断: {'匹配' if is_match else '不匹配'}\n")
        
        return is_match, result['description'], confidence

class LlamaHandler(BaseAnalyzer):
    def __init__(self):
        self.config = MODELS["llama"]
        self.base_url = self.config["base_url"].replace('http://', '')
    
    def analyze_video(self, image_path, prompt):
        print(f"\n[Llama] 使用 {self.config['name']} 模型分析图像...")
        
        prompt_str = f"""请用中文分析这张图片并回答以下问题：

1. 图片中是否有 {prompt}？
2. 如果有，请详细描述它的外观和在图片中的位置。
3. 如果没有，请描述你在图片中看到的内容。
4. 请给出你的回答的置信度（1-10分）。

请按照以下格式回答：
Answer: [YES/NO]
Description: [用中文详细描述]
Confidence: [1-10]

注意：描述必须使用中文。"""

        try:
            os.environ['OLLAMA_HOST'] = self.base_url
            response = ollama.chat(
                model=self.config["name"],
                messages=[{
                    'role': 'user',
                    'content': prompt_str,
                    'images': [image_path]
                }]
            )
            print(f"[Llama] 分析完成")
            return self.parse_response(response['message']['content'], prompt)
        except Exception as e:
            print(f"[Llama] 分析错误: {str(e)}")
            return False, "分析过程中出现错误", 0

    def parse_response(self, response_text, prompt):
        """Llama特定的响应解析方法"""
        # 预处理响应文本
        response_text = response_text.replace('图片质量', '').replace('###', '').replace('**', '')
        print(f"\n[Llama] 原始响应:\n{response_text}\n")
        
        # 1. 初始化结果
        result = {
            'answer': 'NO',
            'description': '',
            'confidence': 5,
            'has_target': False,
            'has_negation': False
        }
        
        # 2. 提取答案和置信度
        for line in response_text.lower().split('\n'):
            line = line.strip()
            # 提取 Answer
            if 'answer:' in line:
                result['answer'] = 'YES' if 'yes' in line else 'NO'
            # 提取置信度
            elif '置信度' in line or 'confidence' in line:
                try:
                    numbers = [int(n) for n in line.split() if n.isdigit() and 0 <= int(n) <= 10]
                    if numbers:
                        result['confidence'] = numbers[-1]
                except:
                    pass
        
        # 3. 提取描述文本
        description_parts = []
        for line in response_text.split('\n'):
            line = line.strip()
            if not line:
                continue
            if any(x in line.lower() for x in ['answer:', 'confidence:', '置信度:']):
                continue
            if ':' in line:
                part = line.split(':', 1)[1].strip()
                if part:
                    description_parts.append(part)
            elif len(line) > 5:
                description_parts.append(line)
        
        result['description'] = ' '.join(description_parts)
        
        # 4. 关键词分析
        target_words = prompt.lower().split()
        description_lower = result['description'].lower()
        
        # 4.1 检查否定词
        negation_patterns = [
            '没有', '不存在', '未见', '看不到', '无', 'no',
            '并未', '找不到', '未找到', '不是'
        ]
class GeminiHandler(BaseAnalyzer):
    def __init__(self):
        self.config = MODELS["gemini"]
        genai.configure(api_key=self.config["api_key"])
        # 设置代理
        os.environ["HTTPS_PROXY"] = self.config["proxy"]["https"]
        os.environ["HTTP_PROXY"] = self.config["proxy"]["http"]
        # 初始化模型
        self.model = genai.GenerativeModel(self.config["model"])
    
    def analyze_video(self, image_path, prompt):
        print(f"\n[Gemini] 使用 {self.config['name']} 模型分析图像...")
        
        try:
            # 读取图片文件
            with open(image_path, 'rb') as image_file:
                image_data = image_file.read()
            
            # 构建提示词
            prompt_text = f"请分析这张图片中是否有{prompt}，如果有请详细描述它的位置和外观特征。"
            
            # 创建请求内容
            contents = [
                prompt_text,
                types.Part.from_bytes(image_data, "image/jpeg")
            ]
            
            # 生成响应
            response = self.model.generate_content(
                contents=contents,
                generation_config={
                    "temperature": self.config["temperature"],
                    "max_output_tokens": self.config["max_output_tokens"],
                    "top_p": self.config["top_p"],
                    "top_k": self.config["top_k"]
                }
            )
            
            print(f"[Gemini] 分析完成")
            return self.parse_response(response.text, prompt)
            
        except Exception as e:
            print(f"[Gemini] 分析错误: {str(e)}")
            return False, "分析过程中出现错误", 0

class MinimaxHandler(BaseAnalyzer):
    def __init__(self):
        self.config = MODELS["minimax"]
        self.client = OpenAI(
            api_key=self.config["api_key"],
            base_url=self.config["base_url"]
        )
    
    def analyze_video(self, image_path, prompt):
        print(f"\n[Minimax] 使用 {self.config['name']} 模型分析图像...")
        
        try:
            # 读取图片并转换为base64
            with open(image_path, "rb") as image_file:
                image_base64 = base64.b64encode(image_file.read()).decode('utf-8')
            
            # 构建消息列表
            messages = [
                {
                    "role": "system",
                    "content": "MM智能助理是一款由MiniMax自研的，没有调用其他产品的接口的大型语言模型。MiniMax是一家中国科技公司，一直致力于进行大模型相关的研究。"
                },
                {
                    "role": "user",
                    "name": "用户",
                    "content": [
                        {
                            "type": "text",
                            "text": f"请分析这张图片中是否有{prompt}，如果有请详细描述它的位置和外观特征。"
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
            
            # 调用API
            completion = self.client.chat.completions.create(
                model=self.config["model"],
                messages=messages,
                max_tokens=4096,
                group_id=self.config["group_id"],
                stream=False,
                temperature=0.7,
                top_p=0.8
            )
            
            if completion.choices:
                print(f"[Minimax] 分析完成")
                return self.parse_response(completion.choices[0].message.content, prompt)
            
            raise Exception("模型未返回有效响应")
        except Exception as e:
            print(f"[Minimax] 分析错误: {str(e)}")
            return False, "分析过程中出现错误", 0

class QwenHandler(BaseAnalyzer):
    def __init__(self):
        self.config = MODELS["qwen"]
        self.client = OpenAI(
            api_key=self.config["api_key"],
            base_url=self.config["base_url"]
        )
    
    def analyze_video(self, image_path, prompt):
        print(f"\n[Qwen] 使用 {self.config['name']} 模型分析图像...")
        
        try:
            # 读取图片文件并转换为base64
            with open(image_path, 'rb') as image_file:
                import base64
                image_data = base64.b64encode(image_file.read()).decode('utf-8')
            
            response = self.client.chat.completions.create(
                model=self.config["name"],
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": f"请分析这张图片中是否有{prompt}，如果有请详细描述它的位置和外观特征。"},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}}
                        ]
                    }
                ],
                temperature=self.config["temperature"],
                max_tokens=self.config["max_tokens"],
                stream=False
            )
            
            if response.choices:
                print(f"[Qwen] 分析完成")
                return self.parse_response(response.choices[0].message.content, prompt)
            
            raise Exception("模型未返回有效响应")
        except Exception as e:
            print(f"[Qwen] 分析错误: {str(e)}")
            return False, "分析过程中出现错误", 0
        
        # 检查整体否定