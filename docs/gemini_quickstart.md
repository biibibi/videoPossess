# Google Gemini API 快速上手指南

## 1. 环境准备

### 设置 Python 3.11 环境

推荐使用我们提供的自动脚本设置环境：

```bash
# 给脚本添加执行权限
chmod +x setup_py311_env.sh

# 运行脚本设置环境
./setup_py311_env.sh
```

脚本将执行以下操作：
- 检查 Python 3.11 是否安装
- 创建并激活虚拟环境
- 安装所需依赖
- 配置镜像源（可选择清华、阿里等国内源）
- 测试 Gemini API 连接

### 手动设置（如果脚本不适用）

```bash
# 创建虚拟环境
python3.11 -m venv .venv

# 激活虚拟环境
source .venv/bin/activate

# 安装依赖
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 安装特定版本的 google-generativeai
pip install google-generativeai==0.3.1 -i https://pypi.tuna.tsinghua.edu.cn/simple --force-reinstall
```

## 2. 配置 API 密钥和代理

编辑 `config.py` 文件，根据需要修改 `GEMINI_CONFIG` 部分：

```python
GEMINI_CONFIG = {
    "api_key": "你的API密钥",  # 替换为有效的API密钥
    "model": "gemini-2.0-flash-exp",  # 可选: gemini-1.5-flash, gemini-1.5-pro 等
    "temperature": 0.7,
    "max_output_tokens": 2048,
    "top_p": 0.8,
    "top_k": 40,
    "proxy": {
        "http": "http://127.0.0.1:10809",  # 根据需要修改代理地址
        "https": "http://127.0.0.1:10809", 
        "socks": "socks5://127.0.0.1:10808"
    }
}
```

## 3. 测试 API 连接

运行测试脚本验证 API 连接：

```bash
python test_gemini.py
```

如果遇到问题，可使用检查脚本诊断：

```bash
./check_gemini_install.sh
```

## 4. 在应用中使用 Gemini

### 直接调用 API 处理图像

```python
import base64
from main import ModelProcessor
import asyncio

async def test_gemini_analysis():
    # 读取图像文件并转为base64
    with open("your_image.jpg", "rb") as img_file:
        image_data = base64.b64encode(img_file.read()).decode('utf-8')
    
    # 设置搜索目标
    targets = ["人物", "电脑", "书籍"]
    
    # 调用处理方法
    start_time = asyncio.get_event_loop().time()
    result = await ModelProcessor.process("gemini", image_data, targets, start_time)
    
    # 输出结果
    print(result)

# 运行测试函数
asyncio.run(test_gemini_analysis())
```

### 通过 Web API 使用

访问 `/api/analyze` 接口，设置请求参数：

```json
{
  "image": "base64编码的图像数据",
  "targets": ["人物", "电脑", "书籍"],
  "model": "gemini"
}
```

## 5. 常见问题解决

### API 密钥错误

确保使用有效的 API 密钥，可以在 [Google AI Studio](https://makersuite.google.com/app/apikey) 获取。

### 网络连接问题

如果在中国大陆地区使用，需要配置有效的代理。确保代理服务器可用且能正常访问 Google 服务。

### 导入错误

如遇到导入错误，尝试重新安装库：

```bash
pip install google-generativeai==0.3.1 --force-reinstall
```

### 模型响应解析错误

如果遇到响应解析问题，检查 API 返回的原始响应格式，可能需要调整解析逻辑。

## 6. 更多资源

- [完整的 Gemini 集成文档](gemini_integration.md)
- [Google Gemini API 官方文档](https://ai.google.dev/docs/gemini_api_overview)
- [项目 README](../README.md) 