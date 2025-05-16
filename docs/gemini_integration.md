# Google Gemini API 集成文档

## 概述

本文档描述了视频分析工具中 Google Gemini API 的集成实现。Gemini 是 Google 提供的先进多模态 AI 模型，能够处理文本、图像等多种形式的输入。

## 配置说明

Google Gemini API 的配置存储在 `config.py` 文件中的 `GEMINI_CONFIG` 部分。主要配置项包括：

```python
GEMINI_CONFIG = {
    "api_key": "YOUR_API_KEY",
    "model": "gemini-2.0-flash-exp",  # 模型版本，可选 gemini-1.5-flash, gemini-2.0-flash-exp 等
    "temperature": 0.7,
    "max_output_tokens": 2048,
    "top_p": 0.8,
    "top_k": 40,
    "proxy": {
        "http": "http://127.0.0.1:10809",  # HTTP 代理
        "https": "http://127.0.0.1:10809", # HTTPS 代理
        "socks": "socks5://127.0.0.1:10808" # SOCKS 代理
    }
}
```

### 关键配置项说明：

- `api_key`: Google Gemini API 的密钥
- `model`: 使用的模型版本
- `temperature`: 生成结果的随机性，取值 0~1，值越低结果越确定
- `max_output_tokens`: 生成结果的最大 token 数量
- `top_p` 和 `top_k`: 影响文本生成多样性的参数
- `proxy`: 代理设置，适用于网络受限环境

## 实现细节

Gemini API 集成在 `main.py` 文件的 `ModelProcessor` 类中通过 `process_gemini` 方法实现：

1. 接收图像数据（Base64 编码）和查询目标
2. 提取用户消息和系统消息内容
3. 创建临时文件存储图像
4. 根据配置初始化 Gemini API 客户端
5. 调用 API 进行图像分析
6. 处理并返回分析结果

## 测试和故障排查

可以使用项目中提供的 `test_gemini.py` 脚本测试 Gemini API 集成：

```bash
python test_gemini.py
```

该脚本会尝试进行文本生成和图像分析测试，以验证 API 连接和功能是否正常。

### 常见问题和解决方案

1. **API 密钥错误**: 确保 `config.py` 中提供了有效的 API 密钥。
   
2. **网络连接问题**: 如果在中国大陆使用，需要设置正确的代理。

3. **导入错误**: 确保安装了最新版本的 `google-generativeai` 库：
   ```bash
   pip install google-generativeai==0.3.1 -i https://pypi.tuna.tsinghua.edu.cn/simple
   ```

4. **响应格式错误**: 检查 API 版本是否变更，可能需要更新处理响应的代码。

## 升级指南

当 Google 更新 Gemini API 时，可能需要更新代码。以下是升级步骤：

1. 更新 `google-generativeai` 库:
   ```bash
   pip install --upgrade google-generativeai
   ```

2. 查看官方文档了解 API 变更
   
3. 更新 `process_gemini` 方法以适应新的 API 结构

## 参考资料

- [Google Generative AI Python SDK](https://github.com/google/generative-ai-python)
- [Gemini API 官方文档](https://ai.google.dev/docs/gemini_api_overview) 