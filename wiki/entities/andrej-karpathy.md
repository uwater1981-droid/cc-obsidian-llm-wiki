---
title: Andrej Karpathy
aliases: [Karpathy, 卡帕西, AK]
type: entity
subtype: person
tags: [ai/llm/framework, ai/llm/paper, learn/course]
sources:
  - raw/articles/2026-04-17-test-llm-wiki.md
  - raw/articles/karpathy/2015-05-21-rnn-unreasonable-effectiveness.md
  - raw/articles/karpathy/2016-09-07-phd-survival-guide.md
  - raw/articles/karpathy/2019-04-25-recipe-for-training-neural-networks.md
  - raw/articles/karpathy/2025-11-17-verifiability.md
  - raw/articles/karpathy/2025-12-19-llm-year-in-review-2025.md
  - raw/transcripts/karpathy/zero-to-hero-01-micrograd-backprop.md
  - raw/gists/karpathy/llm-wiki-original.md
  - raw/gists/karpathy/min-char-rnn.md
  - raw/notes/public/karpathy-threads.md
created: 2026-04-17
updated: 2026-04-17
---

# Andrej Karpathy

> **一句话定义**:前 Tesla AI 总监、OpenAI 创始团队成员、深度学习教育者。2026 年提出 [[llm-wiki]] 概念,把 LLM 重塑为"知识编辑器"。

## 简介

Andrej Karpathy 以**极简教学**项目闻名(micrograd、makemore、nanoGPT、minGPT、min-char-rnn),主张"只有把事情从零实现一遍才真正理解"。经历:Stanford PhD(李飞飞实验室) → OpenAI 创始工程师 → Tesla AI 总监(Autopilot) → 再回 OpenAI → 独立研究。

他的公开输出分三大块:**博客(2011-2026)**、**YouTube 视频课(Zero to Hero + 主题演讲)**、**Twitter/X 精华观点**。本页把本 vault 已归档的原始资料做成一个可点开的学习仪表盘。

---

## 📚 学习仪表盘

### 推荐路径(按领域从浅入深)

**Path A:入门系列(从零实现神经网络)**
1. [[../../raw/transcripts/karpathy/zero-to-hero-01-micrograd-backprop|L1:micrograd 反向传播]]
2. [[../../raw/transcripts/karpathy/zero-to-hero-02-makemore-intro-bigram|L2:makemore bigram]]
3. [[../../raw/transcripts/karpathy/zero-to-hero-03-makemore-mlp|L3:makemore MLP]]
4. [[../../raw/transcripts/karpathy/zero-to-hero-04-makemore-batchnorm|L4:BatchNorm]]
5. 对应代码:[[../../raw/gists/karpathy/min-char-rnn|min-char-rnn 100 行 RNN]] · [[../../raw/gists/karpathy/microgpt-py|microgpt.py 200 行 GPT]]

**Path B:思想脉络(代表文章)**
1. [[../../raw/articles/karpathy/2015-05-21-rnn-unreasonable-effectiveness|RNN 不可思议的效力 (2015)]] — 深度学习启蒙名篇
2. [[../../raw/articles/karpathy/2016-05-31-drl-pong-from-pixels|Pong from Pixels (2016)]] — 强化学习直觉
3. [[../../raw/articles/karpathy/2019-04-25-recipe-for-training-neural-networks|训练神经网络的菜谱 (2019)]] — 工程经验总结
4. [[../../raw/articles/karpathy/2022-03-14-lecun1989-deep-nets-33-years|LeCun 1989 · 33 年前后]] — 历史视角
5. [[../../raw/articles/karpathy/2025-04-27-vibe-coding-menugen|Vibe Coding MenuGen]] — 创造"vibe coding"一词
6. [[../../raw/articles/karpathy/2025-11-17-verifiability|可验证性 (2025)]] — RLVR 思想源头
7. [[../../raw/articles/karpathy/2025-12-19-llm-year-in-review-2025|2025 LLM 年度回顾]] — 他的全年总结

**Path C:生活哲学(关于学习/生产力/思维)**
1. [[../../raw/articles/karpathy/2016-09-07-phd-survival-guide|读博生存指南]]
2. [[../../raw/articles/karpathy/2020-06-11-biohacking-lite|轻量生物黑客]]
3. [[../../raw/articles/karpathy/2025-03-17-digital-hygiene|数字卫生]]
4. [[../../raw/articles/karpathy/2025-03-19-append-and-review-note|追加+复习笔记法]]
5. [[../../raw/articles/karpathy/2025-10-01-animals-vs-ghosts|动物 vs 幽灵]]

**Path D:工具与本项目源起**
- [[../../raw/gists/karpathy/llm-wiki-original|LLM Wiki 原 gist]] — 本 vault 精神祖师
- [[llm-wiki]] · [[knowledge-compounding]] — 本项目对其的演绎

---

## 📊 博客全集(33 篇)

```dataview
TABLE WITHOUT ID file.link AS "文章", title AS "Title", url AS "Source"
FROM "raw/articles/karpathy"
SORT file.name DESC
```

## 🎥 YouTube 视频字幕(4 篇 Whisper · 9 条元数据)

```dataview
TABLE WITHOUT ID file.link AS "视频", title AS "Title", segments AS "段数", type AS "状态"
FROM "raw/transcripts/karpathy"
SORT file.name ASC
```

## 💻 代码 Gists(12 个)

```dataview
TABLE WITHOUT ID file.link AS "Gist", note AS "说明", url AS "GitHub"
FROM "raw/gists/karpathy"
SORT file.name ASC
```

## 💬 Tweets 精选(70 条)

单文件:[[../../raw/notes/public/karpathy-threads|karpathy-threads]]

其中 Part A 70 条来自 [karpathy.ai/tweets.html](https://karpathy.ai/tweets.html) 自选,Part B 为手工登记的爆款线程(如 2025-12 "never felt this behind as programmer")。

---

## 关键事实

- **类型**:person
- **领域**:AI/LLM 研究 & 教育
- **首次出现(本 wiki)**:2026-04-17
- **公开产出规模**:33 blog posts + 13 YT 讲座 + 12 gists + 70+ curated tweets
- **主博客站**:[karpathy.ai](https://karpathy.ai) · [karpathy.bearblog.dev](https://karpathy.bearblog.dev)(2024 后切换)
- **GitHub**:[@karpathy](https://github.com/karpathy)
- **Twitter/X**:[@karpathy](https://x.com/karpathy)

## 相关概念

- [[llm-wiki]] — 他 2026 年 4 月提出、本 vault 的精神原点
- [[knowledge-compounding]] — 本项目把其理念系统化的落地

## 思想主线(从已归档文章提炼的高频主题)

- **"从零实现"(teaching from scratch)**:micrograd · makemore · min-char-rnn · microgpt
- **可验证性(verifiability)**:RLVR 的前奏,贯穿 2025 晚期博客
- **人类-AI 分工(animals vs ghosts)**:2025-10 新视角
- **数字/生物/化学卫生(hygiene 系列)**:把工程思维应用到生活
- **Vibe Coding**:他 2025 年造的词,后成为行业语汇

## Sources(采集来源)

| 类型 | 路径 | 数量 |
|---|---|---|
| 博客原文 | `raw/articles/karpathy/` | 33 |
| Whisper 字幕 | `raw/transcripts/karpathy/` | 4 |
| YT 元数据占位 | `raw/transcripts/karpathy/_raw/` | 9 |
| 代码 Gists | `raw/gists/karpathy/` | 12 |
| 精选 tweets | `raw/notes/public/karpathy-threads.md` | 70 条 |

完整审计链见 `raw/_manifest.jsonl`(第 1065-1124 行为 Karpathy 采集批次)。
