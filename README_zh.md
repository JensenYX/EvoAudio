# EvoAudio

**面向音频理解的递归自我改进**

[English](README.md) · [论文](https://arxiv.org/abs/2609.27389) · [项目与音频演示](https://jensenyx.github.io/EvoAudio.github.io/)

EvoAudio 根据当前模型的能力构建下一轮训练数据：选择需要练习的技能，控制音频难度，通过音频工具生成问题与可验证答案，再用 GRPO 更新模型。候选模型在固定验证集上取得提升后成为下一轮的起点。

本仓库提供 Qwen2.5-Omni-7B 的数据生成、课程规划、GRPO 训练与评测流程。

[![EvoAudio 框架](assets/framework.png)](https://jensenyx.github.io/EvoAudio.github.io/#demos)

## 工作流程

1. **诊断与规划。** 测量当前模型的技能准确率，结合近期进步和训练奖励变化，确定课程配额与难度。
2. **生成与验证。** 合成语音、渲染 MIDI 或读取已有录音，执行音高、速度、音量和时间结构变换。构造记录确定答案，独立声学测量检查最终波形，Whisper 核验合成语音内容。
3. **筛选与学习。** 去除无需听音频即可回答的问题，对其余候选采样四次，选择答对一至三次的题目进行 GRPO 训练。
4. **验证与演化。** 比较候选 checkpoint 与父模型的固定验证集准确率。提升则晋升，否则保留父模型并重新构建课程。

## 安装

使用 Linux、Python 3.12、CUDA 12.8 Toolkit 和兼容的 NVIDIA 驱动。快速开始的训练配置使用 **8 张 80 GB GPU**；语音合成和推理阶段使用单卡。准备约 **400 GB 可用空间**，用于环境、模型缓存、音频和两个包含优化器状态的 checkpoint。

```bash
git clone https://github.com/JensenYX/EvoAudio.git
cd EvoAudio

sudo apt-get update
sudo apt-get install -y ffmpeg sox libsndfile1 fluidsynth timgm6mb-soundfont build-essential

python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install torch==2.9.0 torchaudio==2.9.0 torchvision==0.24.0 --index-url https://download.pytorch.org/whl/cu128
.venv/bin/python -m pip install -r requirements/train.txt
.venv/bin/python -m pip install packaging ninja wheel setuptools
.venv/bin/python -m pip install flash-attn==2.8.3 --no-build-isolation

python3.12 -m venv .venv-tts
.venv-tts/bin/python -m pip install --upgrade pip
.venv-tts/bin/python -m pip install torch==2.11.0 torchaudio==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
.venv-tts/bin/python -m pip install -r requirements/tts.txt
```

训练与 TTS 使用独立环境，各自的依赖组合保存在安装配置中。首次使用时，模型从 Hugging Face 下载。也可以在 YAML 中将 `model`、`tts_model` 和 `whisper_model` 改为本地模型目录。

## 快速开始

从仓库根目录运行：

```bash
.venv/bin/evoaudio prepare --config configs/quickstart.yaml
.venv/bin/evoaudio generate --config configs/quickstart.yaml --count 20
.venv/bin/evoaudio evolve --config configs/quickstart.yaml
.venv/bin/evoaudio evaluate --config configs/quickstart.yaml
```

快速开始配置使用少量 TTS 与 MIDI 音频、20 道固定验证题和两步 GRPO。课程规划直接使用本地技能统计，无需 API 密钥。模型完成训练后，系统重载两个 checkpoint、比较验证结果，并保存下一轮课程。

输出位于 `runs/quickstart/`：

| 路径 | 内容 |
| --- | --- |
| `samples/items.jsonl`、`samples/audio/` | 独立生成的题目与音频 |
| `heldout/items.jsonl` | 固定验证题目 |
| `round-001/selected.jsonl` | 本轮筛选出的训练题与构造证据 |
| `round-001/train/checkpoints/` | 模型、优化器状态与训练记录 |
| `round-001/decision.json` | 验证结果及晋升决定 |
| `round-001/next_curriculum.json` | 下一轮课程 |
| `state.json` | 当前模型与已完成轮数 |

每条生成数据包含音频路径、问题、选项、答案、难度、构造记录和声学检查结果。

查看命令帮助与可用任务：

```bash
.venv/bin/evoaudio --help
.venv/bin/evoaudio tasks
```

## 运行自演化

复制正式配置并设置输出目录、模型和计算资源：

```bash
cp configs/qwen.yaml local.yaml
read -rsp 'DeepSeek API key: ' DEEPSEEK_API_KEY
export DEEPSEEK_API_KEY
printf '\n'
.venv/bin/evoaudio prepare --config local.yaml
.venv/bin/evoaudio evolve --config local.yaml
```

`proposer` 配置指定 API 地址、模型名称及密钥环境变量。Proposer 接收技能统计并返回课程优先级；答案由音频构造或源数据标签确定。

正式配置的主要参数：

| 参数 | 默认值 |
| --- | --- |
| 演化轮数 | 13 |
| 每轮训练步数 | 64 |
| 候选 checkpoint | 第 32、64 步 |
| 每步问题数 / 每题回答数 | 16 / 16 |
| 学习率 | 恒定 `1e-6` |
| 温度 / KL 系数 | 0.9 / 0.02 |
| 固定验证集 | 2,500 题 |
| 单技能配额上限 / 保留预算下限 | 6% / 12% |

训练更新音频与语言路径参数。每轮重新初始化优化器，并以父模型作为 KL reference。多节点训练时，将 `resources.hosts` 改为各节点地址，设置每个节点使用的 `devices`；节点之间需要免密 SSH，并在相同路径安装代码、环境及共享数据。论文训练使用四个节点、每节点八张 H800。

中断后继续：

```bash
.venv/bin/evoaudio evolve --config local.yaml --resume
```

已完成的生成与推理阶段会复用，训练从完整 checkpoint 继续。开始新的配置或数据划分时，使用新的 `output_dir`。

## 使用已有录音

在 YAML 的 `data` 中配置资源路径。已有录音会与合成资源一起供任务生成器选择。

```yaml
data:
  librispeech: data/LibriSpeech
  fsd50k: data/FSD50K
  manifest: data/recordings.jsonl
```

LibriSpeech 支持原始 FLAC 与转写文件以及 Arrow/Parquet；FSD50K 使用 `FSD50K.ground_truth/dev.csv` 和 `FSD50K.dev_audio/`。自有录音或 AudioSet 录音可以用 JSONL 导入，音频路径相对于清单文件：

```jsonl
{"path":"audio/dog.wav","kind":"sound","label":"dog bark","source_id":"recording-001"}
{"path":"audio/speech.wav","kind":"speech","text":"The train leaves at nine.","source_id":"recording-002"}
{"path":"audio/piano.wav","kind":"music","source_id":"recording-003"}
```

源录音按稳定标识划分为训练与验证两部分。也可以在清单中显式提供 `split`，取值为 `train` 或 `heldout`。清单与音频准备好后，再运行 `prepare`。

## 评测

默认评测当前模型的固定验证集。使用 `--model` 可以指定其他 checkpoint。

```bash
.venv/bin/evoaudio evaluate --config local.yaml --benchmark mmsu --data-root data/MMSU
.venv/bin/evoaudio evaluate --config local.yaml --benchmark mmau_pro --data-root data/MMAU-Pro
.venv/bin/evoaudio evaluate --config local.yaml --benchmark mmar --data-root data/MMAR
```

MMSU 读取官方 Arrow 数据，MMAU-Pro 读取 `test.parquet` 和配套音频，MMAR 读取 `MMAR-meta.json` 或 `MMAR-meta.jsonl`。MMAU-Pro 评测采用具有官方答案的非空间闭集问题。结果写入输出目录下的 `evaluation/`，包含逐题预测、整体准确率和分类准确率。

## 论文结果

下表为论文中 Qwen2.5-Omni-7B 的报告结果。Avg. 为三个 benchmark 总分的平均值。

| 方法 | MMSU | MMAU-Pro | MMAR | Avg. |
| --- | ---: | ---: | ---: | ---: |
| Base | 61.4 | 56.4 | 60.2 | 59.3 |
| EvoAudio (GRPO) | **71.9** | **60.8** | **63.5** | **65.4** |

[![论文中的模型演化曲线](assets/evolution-wide.svg)](https://jensenyx.github.io/EvoAudio.github.io/#evolution)

## 引用

```bibtex
@article{wang2026evoaudio,
  title={EvoAudio: Recursive Self-Improvement for Audio Understanding},
  author={Wang, Yuxiang and Cai, Shengbo and Shen, Yingda and Hsu, Ming-Hao and Ni, Qinke and Zhang, Liqiang and Sun, Teddy and Yevs, Steve and Wu, Zhizheng},
  journal={arXiv preprint arXiv:2609.27389},
  year={2026}
}
```

## 许可证

代码采用 [Apache-2.0](LICENSE) 许可证。模型、数据集和 SoundFont 遵循各自的许可证。
