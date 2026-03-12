# Qianfan Models Cookbook

Example code and guides for basic tasks with LLMs Hosted by Qianfan Platform (https://console.bce.baidu.com/qianfan). You'll need a qianfan account and asscociated API key to accomplish all examples. We will provide example files, prompts within a cookbook. You can run all examples by setting the QIANFAN_TOKEN environment variable.

Note that this is a python code only repository.

## News
**2026.03.12**: [**Qianfan-OCR: A Unified End-to-End Model for Document Intelligence**](qianfan-ocr/qianfan_ocr_report_v0.pdf) is released! Qianfan-OCR (4B+300M parameters) is now available on [Baidu AI Cloud](https://console.bce.baidu.com/qianfan) Open source weights coming soon!

Qianfan-OCR is a unified end-to-end document intelligence model, designed to help enterprises achieve digital transformation and move towards intelligent automation. Key highlights:
- **World Top-1 End-to-End OCR Model** on OmniDocBench v1.5.
- **General OCR**: Top model performance on OCRBench and OCRBench v2.
- **Document Understanding**: Strong performance in document QA and information extraction.
- **Layout-as-Thought**: For documents with complex layouts and non-standard reading orders, Qianfan-OCR can perform layout-analysis-level reasoning via a novel Layout-as-Thought mechanism, achieving superior recognition results.
- **Multilingual OCR**: Supports up to 192 languages with top performance on CC-OCR.

**2025.09.22**: The **Qianfan-VL Vision-Language model series** from Baidu AI Cloud is now open source!
- **Multimodal Large Language Models**:
  - [Qianfan-VL-3B, Qianfan-VL-8B, Qianfan-VL-70B](qianfan-vl/qianfan_vl_example.ipynb)

Designed for enterprise applications, these multimodal models combine excellent general capabilities with advanced performance in OCR and education. For more information, please refer to github repo: https://github.com/baidubce/Qianfan-VL

**2025.06.06**: QianfanHuijin and QianfanHuijin-Reason series financial augmented models have been added to ModelBuilder ([Link to apply for a trial​](https://cloud.baidu.com/survey/qianfanhuijin.html)):
- **Financial Knowledge Augmented Models**:
  - [QianfanHuijin-70B-32K, QianfanHuijin-8B-32K](qianfan-huijin-llms/qianfan_huijin_cookbook.ipynb)
- **Financial Reasoning Augmented Models**:
  - [QianfanHuijin-Reason-70B-32K, QianfanHuijin-Reason-8B-32K](qianfan-huijin-llms/qianfan_huijin_cookbook.ipynb)

**2025.04.25**: Five new Qianfan series models have been added to ModelBuilder:
- **Text Models**:
  - [Qianfan-8B，Qianfan-70B](qianfan-llms/qianfan-llms-notebook.ipynb)
- **Distilled Reasoning Models**:
  - [DeepSeek-Distill-Qianfan-8B, DeepSeek-Distill-Qianfan-70B](deepseek-distilled-qianfan-llms/DeepSeek-Distilled-Qianfan-LLMs.ipynb)

All models feature a 32K context length. Please note that only model access is provided; open sourced model weights coming soon!
