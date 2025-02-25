# 视频内容分析工具 (Video Content Analysis Tool)

一个基于多模型的视频内容智能分析工具，支持多种视觉语言模型，可以对视频内容进行实时分析和目标检测。

## 功能特点

- 支持多种主流视觉语言模型：
  - Llama2-Vision：开源视觉语言模型
  - Google Gemini：Google最新多模态模型
  - MiniMax：国产视觉分析模型
  - Qwen2.5-VL：阿里通义千问视觉模型

- 视频处理功能：
  - 支持多种视频格式（MP4, AVI, MOV, MKV）
  - 自动提取视频关键帧
  - 实时分析和结果展示
  - 支持大文件处理（最大100MB）

- 用户友好界面：
  - 拖放上传支持
  - 实时分析进度显示
  - 清晰的结果展示
  - 响应式设计

## 技术栈

- 后端：
  - FastAPI
  - Python 3.8+
  - OpenCV
  - Ollama
  - asyncio

- 前端：
  - HTML5
  - Bootstrap 5
  - JavaScript (原生)

- 模型集成：
  - Llama2-Vision
  - Google Gemini
  - MiniMax
  - Qwen2.5-VL

## 安装说明

1. 克隆项目：
```bash
git clone https://github.com/yourusername/video-content-analysis.git
cd video-content-analysis
```

2. 创建虚拟环境（推荐）：
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
.\venv\Scripts\activate  # Windows
```

3. 安装依赖：
```bash
pip install -r requirements.txt
```

4. 配置模型API密钥：
- 在 `config.py` 中配置各个模型的API密钥和配置信息
- 确保Ollama服务已启动并可访问

5. 启动服务器：
```bash
python main.py
```

## 使用方法

1. 访问Web界面：
   - 打开浏览器访问 `http://localhost:8000`

2. 上传视频：
   - 点击上传区域或拖放视频文件
   - 支持的格式：MP4, AVI, MOV, MKV
   - 文件大小限制：100MB

3. 选择分析模型：
   - 从四个可用模型中选择一个
   - 每个模型有其特点和优势

4. 设置分析目标：
   - 输入要查找的目标描述
   - 可以包含颜色、形状、位置等特征

5. 查看结果：
   - 实时显示分析进度
   - 每个关键帧的分析结果
   - 包含目标检测结果和详细描述

## 目录结构

```
.
├── main.py              # 主程序入口
├── requirements.txt     # 项目依赖
├── config.py           # 配置文件
├── models/             # 模型处理器
│   └── model_manager.py
├── static/             # 静态文件
│   └── icon/
├── templates/          # 前端模板
│   └── index.html
├── uploads/           # 上传文件临时存储
└── frames/            # 视频帧存储
```

## 注意事项

- 需要预先安装并启动Ollama服务
- 需要配置各个模型的API密钥
- 建议使用Python 3.8或更高版本
- 确保有足够的磁盘空间用于存储临时文件

## 许可证

MIT License

## 贡献指南

欢迎提交Issue和Pull Request来帮助改进项目。

## 更新日志

### v1.0.0 (2024-02-24)
- 初始版本发布
- 支持四种主流视觉语言模型
- 完整的视频分析功能
- 用户友好的Web界面
