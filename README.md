# Deep Video Discovery: Agentic Search with Tool Use for Long-form Video Understanding

[![arXiv](https://img.shields.io/badge/arXiv-2504.16082-A42C25?style=flat&logo=arXiv&logoColor=A42C25)](https://arxiv.org/abs/2505.18079)
[![License](https://img.shields.io/badge/License-MIT-red.svg)](https://opensource.org/licenses/MIT)


This repository contains the official implementation of the paper [Deep Video Discovery: Agentic Search with Tool Use for Long-form Video Understanding](https://arxiv.org/abs/2505.18079).

## Introduction

**Deep Video Discovery (DVD)** is a research-oriented question answering agent designed for understanding and analyzing long-form videos. Leveraging the powerful capabilities of large language models (LLMs), DVD effectively interprets and processes extensive video content to answer complex user queries.

The core design of DVD includes:

- **Treating segmented video clips as exploration environments**
- **Autonomous planning and reasoning**, dynamically formulating strategies to solve problems efficiently
- **Selecting appropriate multi-granular tools**, iteratively extracting relevant information from the video environment
- **Summarizing and reflecting on observations**, to provide comprehensive and accurate answers to user questions

DVD Achieves state-of-the-art performance by a large margin on multiple long video benchmarks using OpenAI o3.

![radar](./asset/overview.png)

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/microsoft/deepvideodiscovery.git
   cd DeepVideoDiscovery
   ```

2. **Create a virtual environment and install the dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Note: Set up your configuration by updating the variables in  `config.py`.

### Example

The `local_run.py` script provides an example of how to run the Deep Video Discovery agent by providing a youtube url and question about it.

    ```bash
    python local_run.py https://www.youtube.com/watch?v=ktbGziZlt3c "how many animals appear in this video"
    ```

## TODO

- [ ] Support OpenAI API key configuration.
- [ ] Implement MCP server.
- [ ] Release evaluation trajectory data on long video benchmarks.


## Changes

Compared to the original implementation, we have made the following changes:
- Refactored the code for better readability and maintainability.
- In `global_browse_tool` we leverage the textual description (rather than original video pixels) of multiple video clips to provide global overview of the video content to improve efficency.


## Citation

If you find our work useful, please consider citing:

```bibtex
@article{zhang2025deep,
  title={Deep Video Discovery: Agentic Search with Tool Use for Long-form Video Understanding},
  author={Zhang, Xiaoyi and Jia, Zhaoyang and Guo, Zongyu and Li, Jiahao and Li, Bin and Li, Houqiang and Lu, Yan},
  journal={arXiv preprint arXiv:2505.18079},
  year={2025}
}
```
