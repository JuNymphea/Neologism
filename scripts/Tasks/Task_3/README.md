# Task 3 — Lexical-Embedding POS Probe

词性信息能不能**线性地**从输入 embedding 里读出来。不经过模型前向，不需要模型输出 neologism
——这是它相对 Task 1 / Task 2 的关键性质：被测的 token 只要有一行 embedding 就能测。

## 文件

```
extract_embeddings.py   从 safetensors 直接读 embedding 行 → out/embeddings/<name>.npz
task3_probe.py          在 embedding 上训练/评估线性探针 → out/task3_<name>.json
out/task3_splits.json   train/dev/finaltest 词表（冻结）
out/embeddings/         <name>.npz 是主路径产物，<name>_check.npz 是 --via-transformers 的校验产物
```

## 跑

```bash
# 1. 抽 embedding（每个模型一次）
python extract_embeddings.py --model /path/to/gemma-3-4b-it --name gemma

#    校验：换一条完全不同的路径（AutoModel.get_input_embeddings）再抽一遍，应逐位相同
python extract_embeddings.py --model /path/to/gemma-3-4b-it --name gemma_check --via-transformers

# 2. 探针
python task3_probe.py --model gemma
```

两个脚本的默认路径都是 `<脚本所在目录>/out/...`，所以整个 Task_3/ 可以整体挪动。
依赖只有 numpy / scikit-learn / safetensors（`--via-transformers` 才需要 torch+transformers）。

## 切分

975 个三 tokenizer 共有的单 token 词，seed 20260923：

| | noun | verb | adj |
|---|---|---|---|
| train | 175 | 175 | 175 |
| dev | 50 | 50 | 50 |
| finaltest | 100 | 100 | 100 |

`finaltest` 与 `Task_2/out/shared_finaltest.json` **逐词相同**，所以三个任务的 per-word 结果可以直接对比。

## 预处理与超参

`x → (x − train_mean) / ‖x − train_mean‖`，再 L2 正则 logistic regression（lbfgs, max_iter 5000）。
C 在 {0.01, 0.1, 1, 10, 100} 上按 dev 选。

**归一化的后果**：探针只看方向，不看 norm。norm 主要编码词频先验
（gemma ρ=+0.49、aya ρ=+0.58 对 log 词频；qwen 无关），把它去掉是有意的。
代价是 Task 3 对 norm 异常不敏感——见下面的 OOD 检查。

## 三个对照

1. **随机标签**（×100）——打乱 train 标签重训，上界应贴近 1/3。
2. **形态学**——测试集里剔掉带显性派生后缀（-tion/-ness/-ize/-ous…）的词后重测，
   防止探针只是在读后缀。
3. **OOD**——把待测向量对训练分布做 z-score。用在 neologism 上时**必须同时报 norm**：
   探针本身对 norm 不敏感，但 Task 1/2 对它极敏感，norm 跑出已知词范围
   （gemma 0.92–1.08 / qwen 0.93–1.30 / aya 0.97–3.16）时三个任务的不一致是人为的。

## 已知缺口

- `out/task3_splits.json` **没有生成脚本**，是当时临时生成后冻结的。要重现切分得补一个
  `make_splits.py`（输入 `Task_2/out/shared_words.json` + seed 20260923）。
- `task3_probe.py` 选 C 时平手取最大（正则最弱）。gemma 上 C=0.01 给 0.960、C=100 给 0.973，
  qwen 的 dev 五个 C 完全持平——平手规则实际决定了结果，应改成平手取最小 C 并三个模型重跑。
