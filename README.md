# SeCo Context Compression

独立的 context 层面压缩实验目录。来源是相邻 `seco-cgn-modified` 的当前工作区快照（包含其未提交修改），不是仅复制 Git HEAD。原目录保持不变。

**状态：实现与轻量测试阶段；未启动训练，未产生新 checkpoint，也没有效果提升结论。**

当前中间压缩方法的详细说明见 `docs/SeCo中间压缩方法详解.md`，包含完整流程、公式、张量形状、预算示例、训练梯度及实现边界。

## 文件结构

| 文件 | 职责 |
| --- | --- |
| `context_compression.py` | 新增 context 语义分块、信息量预算、连续片段聚合 |
| `compression_modules.py` | 复用 query 编码、context 评分、整数预算；保留 multiscale 对照 |
| `modeling_seco_cluster.py` | SECO encoder/decoder、LoRA、压缩路由、训练与生成接口 |
| `compressor_config.py` | checkpoint 路由校验，防止错误算法加载权重 |
| `instruction_finetune.py`, `training_utils.py` | 训练入口、tokenization、collator、保存与梯度诊断 |
| `ft_inference_all.py`, `eval_utils.py`, `sample_ids.py` | 测评、EM/F1、运行清单、样本身份 |
| `scripts/prepare_data.py` | 转换 QA JSONL、去重、context 不交叉的 train/dev 划分 |
| `scripts/common.sh`, `scripts/train_context.sh`, `scripts/eval_context.sh` | 共享配置及训练/测评启动入口 |
| `tests/` | 新算法测试及保留的模型、query、训练对齐回归测试 |
| `docs/CONTEXT_DESIGN.md` | 优化依据、公式、消融、风险与后续实验方案 |
| `docs/SOURCE_MANIFEST.json` | 原始复制文件哈希与来源版本 |

不复制 `.git`、缓存、历史输出、wandb、日志、论文文档和自动训练队列；不复制大模型权重或数据。模型继续通过本地 Hugging Face 缓存/指定模型路径加载。历史路由在文件内保留是为可重复对照，并非继续扩展 query 算法。

## 环境

本地验证使用 `$HOME/miniconda3/envs/icae_v2/bin/python`（Python 3.10、torch 2.1.1+cu121、transformers 4.43.1）；没有更改环境或系统 CUDA。`requirements.txt` 记录验证环境直接依赖版本，不代表所有模型均支持；默认沿用 Llama-3.2-1B-Instruct。Qwen3.5 与该旧版 transformers 的兼容性未验证，不把源脚本中的 Qwen 模型配置直接作为可用默认值。

```bash
cd ~/proj/clustering/seco-cgn-modified-context
CUDA_VISIBLE_DEVICES='' TRITON_CACHE_DIR=/tmp/seco-context-triton \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  ~/miniconda3/envs/icae_v2/bin/python -m pytest tests -q
```

模型集成测试使用随机微型 Llama 与本地 tokenizer，无权重下载、无训练循环、无 optimizer step；tokenizer 不存在时对应测试会 skip，纯张量测试仍可执行。

## 数据预处理（按需执行，本次未处理正式数据）

只从正式训练源划分开发集，不从 test 中挑选开发样本：

```bash
python scripts/prepare_data.py \
  --input /home/huangzj/proj/clustering/data/mrqa_train_24000_converted.jsonl \
  --output-dir data/processed --dev-fraction 0.1 --seed 42
```

输入支持 `input/prompt/answer` 或 `context/question/answers`，答案可为字符串、字符串列表或 `{"text": [...]}`。输出保留 `subset` 等额外字段，统一为 `input/prompt/answer`；训练使用第一个答案，测评保留多答案。分组哈希按空白归一化后的 context 划分，确保相同 context 的不同问题不横跨 train/dev；这不等于近重复语义去重。输出目录必须不存在，避免覆盖。官方测试集单独保留。

## 后续训练与测评（仅命令示例，尚未执行）

训练脚本默认拒绝执行，只有明确设置 `CONFIRM_TRAIN=1` 才启动：

```bash
CONFIRM_TRAIN=1 CUDA_VISIBLE_DEVICES=0 NUM_GPUS=1 \
TAG=context_v1_s42 SEED=42 bash scripts/train_context.sh

TAG=context_v1_s42 \
RESTORE_FROM="$PWD/output/context_v1_s42/ratio_32/checkpoint-20000/model.safetensors" \
TEST_FILE=/home/huangzj/proj/clustering/data/mrqa_test_58221_9633_converted.jsonl \
  bash scripts/eval_context.sh
```

比较基线时设置 `COMPRESSOR_VERSION=multiscale_budget_v1` 并使用独立 `TAG`。所有实验固定 backbone、query 配置、数据、种子、预算和生成策略；参数变化必须同步到训练与测评。不同消融必须使用不同 `TAG`；新训练输出目录存在时直接拒绝覆盖。`CONFIRM_TRAIN` 仅保护 shell 入口，直接执行 Python 训练入口仍可训练。

context 权重与旧 multiscale 权重不兼容，不能直接把旧权重作为新算法的效果结果；新路线要求完整路由元数据。默认 `MODEL_NAME` 也必须与 checkpoint 一致。
