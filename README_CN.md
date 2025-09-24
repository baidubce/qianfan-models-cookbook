# 千帆模型应用指南

这是一个代码示例和指南集合，展示了如何使用千帆平台(https://console.bce.baidu.com/qianfan) 托管的大语言模型完成基础任务。您需要一个千帆账户和相关的API密钥才能运行所有示例。我们将在这个应用指南中提供示例文件和提示词。您可以通过设置QIANFAN_TOKEN环境变量来运行所有示例。

请注意，这是一个仅包含Python代码的仓库。

## 最新动态

**2025.09.22**: 百度智能云Qianfan-VL视觉语言模型系列现已开源！
- **多模态大模型**：
  - [Qianfan-VL-3B, Qianfan-VL-8B, Qianfan-VL-70B](qianfan-vl/qianfan_vl_example.ipynb)

面向企业应用，这些多模态模型将出色的通用能力与在 OCR 和教育领域的先进性能相结合。更多信息请参考 GitHub 仓库：https://github.com/baidubce/Qianfan-VL

**2025.06.06**: 千帆慧金-千帆金融行业大模型已上线至ModelBuilder平台（[测试申请链接](https://cloud.baidu.com/survey/qianfanhuijin.html)）:
  - **金融知识增强**: 
    - [QianfanHuijin-70B-32K, QianfanHuijin-8B-32K](qianfan-huijin-llms/qianfan_huijin_cookbook.ipynb)
  - **金融推理增强**: 
    - [QianfanHuijin-Reason-70B-32K, QianfanHuijin-Reason-8B-32K](qianfan-huijin-llms/qianfan_huijin_cookbook.ipynb)

**2025.04.25**：五个全新的千帆系列模型已上线至ModelBuilder平台：
- **文本模型**：
  - [Qianfan-8B，Qianfan-70B](qianfan-llms/qianfan-llms-notebook.ipynb)
- **蒸馏推理模型**：
  - [DeepSeek-Distill-Qianfan-8B, DeepSeek-Distill-Qianfan-70B](deepseek-distilled-qianfan-llms/DeepSeek-Distilled-Qianfan-LLMs.ipynb)

所有模型均支持32K上下文长度。请注意，本次仅开放模型使用权限，模型权重将会在未来的某天开源。
