# 视频目标检测系统文档

欢迎使用视频目标检测系统文档。以下是各个组件和功能的详细文档。

## API 集成文档

### Google Gemini

- [Gemini API 集成文档](gemini_integration.md) - 详细介绍 Gemini API 的配置、实现和故障排查
- [Gemini API 快速上手指南](gemini_quickstart.md) - 快速入门指南，包括环境设置和基本使用
- [Gemini API 使用示例](gemini_usage_example.md) - 完整的代码示例，展示如何在不同场景中使用 Gemini
- [Gemini 集成改进摘要](changes_summary.md) - 最近对 Gemini 集成的改进和优化

## 使用指南

### 工具脚本

系统提供了以下工具脚本，方便您测试和使用各个功能：

- `test_gemini.py` - 测试 Gemini API 连接和基本功能
  ```bash
  python test_gemini.py
  ```

- `test_video_with_gemini.py` - 使用 Gemini API 分析视频文件
  ```bash
  # 基本用法
  python test_video_with_gemini.py 办公室.mp4
  
  # 指定检测目标
  python test_video_with_gemini.py 办公室.mp4 --targets 人物 电脑 文件 手机
  
  # 调整分析间隔(每60帧分析一次)
  python test_video_with_gemini.py 办公室.mp4 --interval 60
  
  # 自定义输出目录
  python test_video_with_gemini.py 办公室.mp4 --output analysis_results
  ```

## 技术参考

### 配置参数

- `config.py` 文件包含所有可配置参数，包括:
  - API 密钥
  - 代理设置
  - 模型参数
  - 缓存和超时设置

### 代码结构

- `main.py` - 应用主程序
- `ModelProcessor` 类 - 处理不同模型的API调用
  - `process_gemini` - Gemini模型处理方法
  - `process_minimax` - MiniMax模型处理方法
  - `process_qwen` - 阿里云千问模型处理方法

## 故障排查

如遇到问题，请先查看相应模型的集成文档中的故障排查部分。常见问题包括：

1. API 密钥错误
2. 网络连接问题
3. 代理设置不正确
4. 依赖库版本兼容性问题

## 贡献指南

欢迎提交问题报告和改进建议。在提交代码修改前，请确保：

1. 代码符合项目规范
2. 添加适当的注释
3. 更新相关文档
4. 通过基本测试 