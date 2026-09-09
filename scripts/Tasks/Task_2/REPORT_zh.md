# Task 2 重新设计：Corpus-Calibrated Syntactic Compatibility Probe

## 0. 一句话

Task 2 从「30 个人工句子的 continuation surprisal probe」改成：

> **从标注语料归纳出高度 POS-selective 的局部句法环境，用最小中性语境实例化，只对该
> frame 所 licensed 的后续 token 计 surprisal，并用已知词校准来判断 neologism 的行为
> 更像哪一类词。**

它只回答一个问题：

> 当 neologism 被放进一个在真实语料中高度选择某个 POS 的局部句法环境时，它引起的后续
> 预测行为，是否更像该 POS 的已知词，而不是其他 POS 的已知词？

## 1. 原来的问题

论文 §3.3.2 的 30 个槽位（Table 5）全部人工撰写，没有任何依据。审稿人可以直接问
「凭什么说这些句子是 noun-selective」。

## 2. 五个组件，各自对应一个明确问题

| 组件 | 回答的问题 | 实现 |
|---|---|---|
| Corpus induction | 哪些局部环境实际上选择某个 POS？ | [`induce.py`](slotgen/induce.py) |
| Minimal realization | 怎样呈现该环境而不引入 concept-specific 语义？ | [`minimal.py`](slotgen/minimal.py) |
| Diagnostic continuation | 这个 construction 真正要求模型预测什么？ | `D_f`，见 §5 |
| Known-word calibration | 什么样的 surprisal 算「像 noun/verb/adjective」？ | **待做**，见 §9 |
| Neologism | 它的行为更接近哪一类已知词？ | **待做** |

## 3. 语料与三分划分

| 角色 | 数据 | 用途 |
|---|---|---|
| Induction | EWT train，12,544 句 / 204,578 词 | 发现并排序 frame |
| Dev | EWT dev，2,001 句 | 否决选择性无法复现的 frame |
| Test（in-domain） | EWT test | 只报告 |
| **External** | **UD-GUM 全部三个 split** | **完全未参与任何决策** |

GUM 现在整体外置。之前把 GUM train 放进 dev 池，会削弱「cross-corpus generalization」
这个 claim；现在外部测试集有 n=3252 / 1217 / 211 个实例，没有一条参与过选择。

## 4. purity 分母改为 all-UPOS（一处关键修正）

原来 `P(p|c)` 只在 {NOUN, VERB, ADJ} 内部归一化。检查后发现这掩盖了真实分布：

```
as {SLOT} as        P(ADJ|c)=1.00   但 open_class_share = 0.28  ← 72% 是副词
NOUN {SLOT} are     P=1.00                                = 0.41
PRON {SLOT} the     P=0.98                                = 0.53
```

`as ___ as` 的主导填充者是副词（*as quickly as*）。三分类任务里它仍能区分名/动/形，
但模型在该位置的期待由一个**根本不在标签集里**的类别主导。

现在分母是全部 UPOS，阈值 0.90。**0.90 all-UPOS 比原来的 0.95 三分更严**，这一改动
自动淘汰了 `as {SLOT} as`、`NOUN {SLOT} are` 等一批 frame。

阈值试算（去重前候选数）：

| all-UPOS 阈值 | NOUN | VERB | ADJ |
|---|---|---|---|
| 0.90 | 431 | 298 | 33 |
| 0.95 | 285 | 209 | **10** |

0.95 时形容词恰好只剩 10 个，去重后必然不够，所以取 0.90。

## 5. 最小实例化 + diagnostic continuation（核心改动）

### 不再拼完整句子

probe item 只是一个**最小的合语法前缀**，加上 frame 自己 licensed 的续接：

```
corpus frame          probe item                          计分 D_f
─────────────────────────────────────────────────────────────────
the {SLOT} of         The {NEOLOGISM} of                  of
to {SLOT} PRON        They tried to {NEOLOGISM} it        it
very {SLOT} NOUN      It was a very {NEOLOGISM} thing     thing
the {SLOT} and NOUN   The {NEOLOGISM} and something       and something
```

**不再固定 n=3/4：**

$$S(w,f)=\frac{1}{|D_f|}\sum_{t\in D_f}-\log P(t\mid \text{prefix containing }w)$$

`|D_f|` 由 construction 决定。当前 30 个 item 中 **23 个长度 1、5 个长度 2、2 个长度
3**。固定取 4 个 token 没有理论依据，而且会把不携带句法证据的 tail 的 surprisal 混
进来。

### 这一步顺带消灭了三个老问题

| 老问题 | 现状 |
|---|---|
| 为凑满 4 个 token 而造的 tail（*at the time* / *more than once*） | 不存在了 |
| tail 在 30 个 item 间大量重复，三类 POS 使用互不相交的 continuation | 不存在了 |
| 自动拼长句容易产生怪英语 | 根本不拼长句 |

### 全部词汇

neologism 之外，30 个 item 一共只用到 **33 个词型**：

```
, a and be can did different do happened have i in is it more n't of one
other same seen similar that the they thing to tried very was when will would
```

没有任何一个词会和某个 FrameNet 概念产生选择性亲和。

## 5b. 三条针对「不自然」的修正

初版 minimal realization 仍有几条明显不自然。三个问题各有不同的成因，分别修掉。

### (1) prefix 本身必须自然 —— 用语料 bigram 检查

```
✗ The same to {NEOLOGISM} the      ← 来自 ADJ to {SLOT} the
```

问题发生在 **neologism 之前**，会污染整个 conditional context。原则：

> prefix 本身必须是自然的英语前缀，prefix + D_f 必须是自然的局部续接。不要求完整
> 句子，但局部不能怪。

判据不用 LM（那是循环论证），而用**语料自身的 bigram 频次**：把 frame 之外添加的相邻
词对拿去合并语料（EWT + GUM，20.9 万个不同 bigram）里查，低于阈值就淘汰。

```
they tried to   → 高频，保留
the  same  to   → 几乎不出现，淘汰
```

阈值取 3。取 1 太松（`the certain to` 也能过），取 8 太紧（连 `They tried to X it`
都会被淘汰）。跨越 slot 的 bigram 不检查。

### (2) 填充词必须适配句法位置，而不只是词类

```
✗ very {NEOLOGISM} and other      ← very happy and other 是不合法的
```

`other` 确实是形容词，POS tagger 也会通过，但它的**句法分布是 attributive-only**，
不能独立出现在并列的谓语形容词位置。

所以 placeholder 不能按 UPOS 固定一个词。改成：

1. 每个 UPOS 给一个**很小、句法分布较广的 generic pool**
   ```
   ADJ  → different / similar / certain / other / same
   NOUN → thing / something / one
   VERB → happened / did / started / changed
   ```
2. 在计数阶段额外记录每个上下文位置的 **dominant deprel**（`context_deprel_at`）
3. 填充时要求该词在语料中**确实以该 (UPOS, deprel) 出现过**（`_fits_position`）

于是 `very {SLOT} and ADJ` 的 ADJ 位置 deprel 是 `conj`，`other` 从未以 conj 形容词
出现，被排除；选中 `similar`：

```
✓ It was very {NEOLOGISM} and similar
```

这一条比「28 个词型算不算多」重要得多 —— 词表大小无所谓，**分布匹配才是关键**。

### (3) a / an 与形态约束

`a {SLOT} to` 这类 frame，如果 control word 是 `idea`，会得到 `a idea`，人为制造
surprisal。现在 `prefix_for()` 按代入词的首音自动切换：

```
A thing to ...        An idea to ...
```

更一般地，每个 slot 现在都带一个 **`morph_requirement`**，直接读自该 frame 在语料中
的 dominant feats：

| 要求 | slot 数 | 含义 |
|---|---|---|
| `Number=Sing` | 10 | 单数可数名词 |
| `Degree=Pos` | 10 | 原级形容词 |
| `VerbForm=Inf` | 7 | 不定式/原形 |
| `Mood=Ind\|...\|Person=1\|Tense=Pres` | 3 | 一般现在时 |

顺带修掉一个不一致：`<s> PRON {SLOT} to` 的 dominant feats 是 1sg，但主语原来被填成
`They`，等于要求 control word 用 1sg 形式却给了复数主语。现在主语按 frame 自己的
person/number 填，得到 `I {NEOLOGISM} to`。

这些要求会在 §9 的 calibration 里用来筛 control words —— **不能只说「这是一个
noun」，还要保证它适合该 slot 的 number / inflection / determiner 环境。**

## 6. 与训练模板 construction-disjoint

neologism 的 embedding 是在这四个模板里训练的（§3.2）：

```
unbiased   Make your answer reflect the following word: {NEOLOGISM}.
verb       Please {NEOLOGISM} your answer.
noun       Please answer this question with a {NEOLOGISM}.
adjective  Your answer should be as {NEOLOGISM} as possible.
```

如果 Task 2 再用 `as {SLOT} as`，测到的可能是对训练 construction 的记忆，而不是
lexical-category generalization。所以加了排除检查
（[`collides_with_training`](slotgen/minimal.py)）。

它目前一条也没触发 —— 因为 all-UPOS purity 在更早阶段就把 `as {SLOT} as` 滤掉了。
检查保留作为守卫。

## 7. 「检查完整句，计分片段」

`The {NEOLOGISM} of` 按设计就是片段，末尾悬空的介词正是要计分的对象。句法分析器无法
判断这种串。

解决办法：**临时**把 item 补成完整句（`of it was there.`）交给检查，补的部分**从不
计分、从不出现在 item 里**。这样既保住了依存层面的语法检查（单一动词性 ROOT、有主语、
无悬空介词、无裸单数名词、主谓一致…），又不需要把 item 撑成句子。

## 8. 结果

pooled `P(p|c)`（总匹配数 / 总实例数）：

| POS | train | EWT test | GUM (external) |
|---|---|---|---|
| Noun | 0.990 (n=1849) | 0.975 (n=198) | 0.993 (n=3197) |
| Verb | 0.992 (n=1351) | 0.994 (n=174) | 0.988 (n=1206) |
| Adjective | 0.994 (n=174) | 1.000 (n=39) | 0.925 (n=159) |

阈值：名词、动词 `Freq≥20, Types≥10`；形容词放松到 `Freq≥8, Types≥5`。**purity 从
不放松**。

## 9. 待办

### 人工 sanity check（步骤 6）

[`out/review_template.json`](out/review_template.json) 每次运行都会重写。只检查三件事：
句子是不是正常英语、有没有 concept-specific 语义、`{NEOLOGISM}` 后面有没有足够
continuation。

人工检查不破坏 corpus-derived 的方法学，因为**人工没有决定这个位置是什么 POS** ——
POS selectivity 已由语料统计独立确立。

目前仍需裁决的五条（`The same to {NEOLOGISM} the` 已被 §5b(1) 的 bigram 检查自动淘汰）：

```
NOUN  7. They have seen the different {NEOLOGISM} it happened  ← 缩合关系从句
VERB  9. It was when it {NEOLOGISM} it                         ← 别扭
ADJ   3. They have seen the same, {NEOLOGISM} and              ← 逗号并列
ADJ   8. The other and {NEOLOGISM} and                         ← 双 and
ADJ  10. The {NEOLOGISM} one is that                           ← 可疑
```

reject 后由语料候选第 11、12 名自动顶替。

### Calibration（步骤 7–8，需要 Gemma）

control words 拆成 **200/POS 拟合 + 100/POS held-out**，三类频率大致匹配、单 token、
词性无歧义。

**除此之外还必须控制形态**（见 §5b(3)）：每个 slot 的 `morph_requirement` 规定了代入
词必须满足的 number / inflection。`They {SLOT} it` 要求能与复数主语搭配的现在时形式，
`A {SLOT} to` 要求单数可数名词。a/an 已由 `prefix_for()` 自动处理，不需要在词表层面
回避元音开头的词。

对每个 slot i，用 calibration controls 拟合
$\mathcal{N}(\mu_{match,i},\sigma^2_{match,i})$ 与
$\mathcal{N}(\mu_{nonmatch,i},\sigma^2_{nonmatch,i})$，保留原来的高斯 LLR：

$$r_i(w)=\log p(s_i(w)\mid match)-\log p(s_i(w)\mid nonmatch)$$

**probe accuracy 必须在没参与拟合的 100/POS 上报告**，这比原来同一批 controls 既拟合
又测准确率严谨。

同时报告每个 slot 的 matched/nonmatched separation 或 AUC。**如果某个 frame 在
held-out controls 上完全不能区分，就在看任何 neologism 结果之前判定为 probe
failure**，由下一名 frame 替换 —— 这一切只能依据 known controls 决定，绝不能依据主
实验结果挑 slot。

这是两种不同的 validity：corpus purity 回答「真实语言里这个环境是不是 POS-selective」，
control-word validation 回答「Gemma 真的感受得到这个区别吗」。

### 正式实验（步骤 9）

输出连续分数而不只是标签：

$$R_p(w)=\frac{1}{10}\sum_{i\in F_p}r_i(w),\qquad \hat{p}=\arg\max_p R_p$$

`R_N, R_V, R_A` 才是 Task 2 最原始的信息；`argmax` 只在需要与 Task 1 / Task 3 合并时
使用。

### 可选 Appendix

old handcrafted slots vs. new corpus-derived slots 的 robustness comparison。主结论
一致 → 说明结果不是人工 slot artifact；形容词结果变化 → 说明旧 probe 对形容词的
sensitivity 确实有问题。

## 10. 一个值得写进论文的附带发现

原 Table 5 的 10 个形容词槽位**全是表语位置**（`seems / sounds / looks / is / was /
became / feels / remained`）。语料归纳出来的以**定语**为主。

这直接关系到度量方式：Task 2 计分槽位**之后**的 token，定语位置后面跟中心名词所以有
诊断力，表语位置形容词接近小句末尾，后面几乎不依赖它。这为 Table 1 中形容词 n=1 时
仅 54% 的准确率、以及 §5.2 形容词 steering 不显著，提供了一个候选解释。

## 11. 走过的弯路（备忘，避免重复）

| 方案 | 为什么放弃 |
|---|---|
| attested 语料原句 | 自然，但把 concept-specific 语义放回来了 |
| 逐位置 argmax 拼句 | "The the {NEOLOGISM} of people and the." |
| 左右 span 各自取最高频再拼 | 两半各自出现过，合起来不成立："Would a more X experience would be extremely appreciated." |
| 用 LM perplexity 判自然度 | **循环论证**：Task 2 测的就是 LM 的 continuation surprisal，用它选槽位会偏向 continuation 高度可预测的槽位 —— 而那恰恰是 match/non-match 区分度最差的 |
| 补 tail 凑满 4 个 token | tail 不携带句法证据，且在 30 个 item 间大量重复 |
| placeholder 按 UPOS 固定一个词（ADJ→other） | POS 对了但句法分布不对："very X and other" |

## 12. 运行

```bash
conda activate py310
python Task_2/run_induction.py
python Task_2/run_induction.py --purity-denominator target3 --min-purity 0.95   # 旧口径
python Task_2/run_induction.py --min-bigram 8                                  # 更严的 prefix 自然度
```

## 13. 产出

| 文件 | 内容 |
|---|---|
| [`out/slots_flat.json`](out/slots_flat.json) | 30 个 item：`prefix`（含 `{NEOLOGISM}` 占位）+ `diagnostic`（D_f）——探针直接消费 |
| [`out/slots.json`](out/slots.json) | 全量：统计、逐语料验证、构式碰撞记录、配置 |
| [`out/slots.md`](out/slots.md) | 人读报告 |
| [`out/slots_table.tex`](out/slots_table.tex) | Table 5 替换 |
| [`out/frames_all.jsonl`](out/frames_all.jsonl) | 全部候选 frame 及其统计量 —— 选择背后的证据表 |
| [`out/review_template.json`](out/review_template.json) | 人工 review 模板 |
