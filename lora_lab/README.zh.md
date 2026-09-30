# F5-TTS v1 Base：LJSpeech 全量微调与 LoRA 对照实验

本目录扩展上游 F5-TTS，保留原有 `src/` 与许可证。上游完整版本见 `UPSTREAM_COMMIT`。
实际完成的是 **LJSpeech 2 小时、500 步、seed=666** 实验；`prepare_arrow.py`
保留为早期 LibriTTS 章节划分工具，不用于本次报告的结果。

## 实验设置与结果

详见 [完整结果及局限](benchmarks/RESULTS.md)。

| 方法 | WER (%) ↓ | SIM ↑ | UTMOS ↑ |
|---|---:|---:|---:|
| Base | 4.6837 | 0.6470 | 4.2590 |
| Full | 4.7445 | **0.7597** | 4.3046 |
| r8 | 4.5620 | 0.7266 | 4.3318 |
| r16 | 4.5620 | 0.7289 | **4.3589** |
| r32 | **4.5012** | 0.7398 | 4.3475 |

训练 1,066 条，约 120 分钟；验证 100 条，测试 100 条，独立参考 2 条。
24 kHz 单声道，2–12 秒，按音频条目划分并进行规范化文本去重，不是章节隔离。
生成提示音频与 SIM 对比参考使用不同的保留音频。
不能由此证明与官方底座预训练数据完全无重叠。

训练：单卡 RTX3090，microbatch=1，累积8，500次优化器更新，约3.75轮数据遍历；
warmup50，每50步验证/保存；bf16，FP32主权重，激活重计算，AdamW，梯度裁剪1，无EMA。
Full学习率1e-5；LoRA学习率1e-4，rank8/16/32，alpha16/32/64，dropout0。
LoRA在22层attention的Q/K/V/out投影上注入，88个模块，可训练参数分别为
1,441,792 / 2,883,584 / 5,767,168。基础模型原参数为337,096,804。
训练前缓存CPU mel，因此单步计时不含数据预处理、验证和权重保存。
不要把十步试跑的显存/速度当作完整训练基准，或仅由参数更少推断训练更快。

按验证集CFM loss选择best.pt；本次四组best均在update500。测试集不用于选模型或调参。
不同学习率意味着这是具体配置对照，不是仅改变LoRA机制的严格单变量实验。
单训练seed，没有重复训练标准差或显著性结论。WER最高与最低只差4次词错误。

## 1. 环境及本地资源

在项目根目录运行：

```bash
conda activate f5-tts
cp lora_lab/env.example.sh lora_lab/env.local.sh
# 编辑 env.local.sh，填写自己的路径；不要把它提交到Git。
source lora_lab/env.local.sh
```

- BASE_CKPT：官方 F5TTS_v1_Base/model_1250000.safetensors。
- VOCAB：`src/f5_tts/infer/examples/vocab.txt`，必须与官方底座匹配。
- DATA_ROOT、SPLIT_DIR：本次处理后的 `data/ljspeech_2h_seed666`。
- VOCOS_DIR：本地vocos-mel-24khz目录。
- ASR_DIR：本地faster-whisper-large-v3目录。
- SIM_CKPT：本地wavlm_large_finetune.pth。
- UTMOS_HUB：可信的本地SpeechMOS v1.2.0代码目录，包含hubconf.py；权重缓存也须就绪。

本项目代码依赖见根目录pyproject.toml，评测额外依赖见eval可选组。
新环境按上游安装说明配置PyTorch与torchaudio兼容版本；现有已成功环境无需升级。
原服务器已确认torch2.4.1、faster-whisper0.10.1、CTranslate2 4.5.0、
cuDNN9.1.0.70、cuBLAS12.9.2.10；这不是完整锁定环境。
SIM会通过上游加载s3prl/WavLM，首次使用可能访问网络，需要其依赖及权重缓存。
UTMOS的本地hub代码也可能加载缓存缺失的权重；本地代码路径不等于所有依赖离线齐全。

## 2. 保存实际使用的划分与结果（原服务器上执行一次）

不要重新抽样。新增工具从已完成实验直接导出音频编号、顺序、文本哈希、逐条分数及配置：

```bash
python -m lora_lab.publish_snapshot \
  --data-root data/ljspeech_2h_seed666 \
  --split data/ljspeech_2h_seed666 \
  --eval-dir runs/eval_lj2h_test100_seed666 \
  --out lora_lab/benchmarks/lj2h_seed666
```

脚本核对split重叠、提示音频与独立参考关系、100条结果对应关系，以及原训练/推理配置的
清单SHA。输出目录必须不存在；不修改训练目录。导出不含WAV、模型、绝对路径或W&B账号信息。
公共WER逐条记录保留参考文本和识别文本，便于错误分析。

上传的代码审阅包里没有实际split清单或逐条分数，因此更新包不会编造它们。
发布前必须成功运行此导出步骤，GitHub才有本次实验的固定样本定义。

## 3. 新机器准备数据

原服务器已有正确数据，不要重建。新机器取得LJSpeech-1.1后：

```bash
python -m lora_lab.prepare_ljspeech \
  --source-root /path/to/LJSpeech-1.1 \
  --split-ids lora_lab/benchmarks/lj2h_seed666/split_ids.json \
  --out data/ljspeech_2h_seed666
```

严格按已导出的ID与顺序恢复，读取metadata.csv第三列规范化文本并核验哈希。
本次新增脚本使用mono均值、librosa soxr_hq重采样到24kHz、FLOAT WAV。
历史临时重采样脚本未随审阅包提供，不能声称新输出与旧WAV逐字节一致。
保留原服务器已处理音频用于精确复跑；如需更强的重现性，还需补齐历史预处理实现和完整版本记录。
该入口选择“恢复冻结清单”，不重新猜测生成1066条历史样本的随机选择过程。

## 4. 验证与训练

```bash
python -m unittest lora_lab.test_lora lora_lab.test_publication -v
CUDA_VISIBLE_DEVICES=0 python -m lora_lab.verify --base "$BASE_CKPT" --vocab "$VOCAB"
```

实际服务器已经通过真实底座零初始化等价、梯度/冻结、保存恢复与合并验证。
下面每行在独立终端或tmux窗口运行可并行，GPU编号换成分配给自己的卡：

```bash
bash lora_lab/run.sh 0 full lj2h_full_500_seed666
bash lora_lab/run.sh 1 r8   lj2h_r8_500_seed666
bash lora_lab/run.sh 2 r16  lj2h_r16_500_seed666
bash lora_lab/run.sh 3 r32  lj2h_r32_500_seed666
```

每条命令前台运行；同一终端逐条运行则是串行。脚本拒绝覆盖已有实验。
参数覆盖示例：`UPDATES=10 WARMUP=2 SAVE_EVERY=5 bash lora_lab/run.sh 0 r8 smoke_new`。
W&B默认关闭；如需启用设置WANDB_PROJECT。网络不稳时可使用WANDB_MODE=offline。
本次曾出现训练完成后W&B上传未退出，因此是否训练完成以本地日志与checkpoint为准。

`best.pt`按验证loss选择，`deploy_XXXXXX.pt`是各保存步部署权重，
`last.pt`含优化器与RNG用于恢复，只加载自己生成的可信文件。
恢复用`RESUME=1`，但必须原代码/原配置/原预算完全一致，不能只把500改成2000。
**训练器会对lora_lab全部Python文件计算哈希：新增本发布工具也会改变哈希，
因此应用更新后不能直接恢复更新前的旧run。旧部署权重仍可用于推理。**

## 5. 五组推理

```bash
bash lora_lab/infer.sh 0 base
bash lora_lab/infer.sh 1 full
bash lora_lab/infer.sh 2 r8
bash lora_lab/infer.sh 3 r16
bash lora_lab/infer.sh 4 r32
```

默认每组100条，输出`runs/infer_lj2h_test100_方法_seed666/`。
同样需独立终端实现并发；脚本拒绝已有输出目录。已有500条无需重新生成。
固定Euler/NFE32/CFG2/sway-1/speed1、FP32、本地Vocos；每条seed为666+清单行号。
LoRA在推理时未合并，不能把此结果称作合并后速度评测。

## 6. WER / SIM / UTMOS

```bash
source lora_lab/cuda_eval_env.sh
CUDA_VISIBLE_DEVICES=1 python -m lora_lab.evaluate \
  --data-root "$DATA_ROOT" --manifest "$SPLIT_DIR/test.jsonl" \
  --asr "$ASR_DIR" --sim "$SIM_CKPT" --utmos-hub "$UTMOS_HUB"
```

WER以`corpus_wer_percent`为主，另保留逐句WER均值；SIM用sim_ref_audio，
不是生成提示音频；UTMOS为逐句预测分数均值。每项默认五组共500条。
支持`--metrics wer sim`或`--metrics utmos`，也支持`--methods base full`。
已有完整结果且输入没有变化时加`--reuse-existing`；该选项只验证ID与数值，
不会证明历史评测权重和WAV内容未被换掉，所以不要在修改输入后复用。

出现`libcudnn_ops.so.9`找不到时，先在同一终端source上述动态库脚本，
它只补充现有库搜索路径，不安装/降级软件。本次服务器故障已用此方式解决。
英文WER内部会设置CUDA_VISIBLE_DEVICES；新入口保留用户指定的物理GPU值，
SIM在单可见GPU进程内使用cuda:0。

