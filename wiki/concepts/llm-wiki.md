---
title: LLM Wiki
aliases: [LLM维基, Karpathy Wiki]
type: concept
tags: [ai/llm/framework, tech/pattern]
sources:
  - raw/articles/2026-04-17-test-llm-wiki.md
created: 2026-04-17
updated: 2026-04-17
---

# LLM Wiki

> **定义**：由 LLM 维护的、持久化的个人知识库 Markdown 百科系统。区别于 RAG 的无状态检索，LLM Wiki 通过"摄取→查询→巡检"的循环让知识**有状态、可审计、可复利**。

## 核心思想

1. **三层架构**：raw/（原始不可变）+ wiki/（AI 编译）+ CLAUDE.md（治理规则）
2. **三大操作**：[[knowledge-compounding|Ingest/Query/Lint]]——从资料到知识、从问题到答案、从陈旧到自愈
3. **LLM 是编辑，人是读者**：AI 负责繁琐的编辑记账（更新引用、保证一致性），人负责提问和治理

## 详解

### 动机

Andrej Karpathy 在 2026 年发起讨论：传统 RAG 像"临时秘书"——每次查询重新检索，知识不沉淀；而 LLM Wiki 像"自更新的百科全书"——每次操作都让知识网络变大变好。

### 机制

- **Ingest**：raw 中新增文件 → AI 抽取实体/概念 → 编译为 wiki 页 → 建立交叉链接 → 追加审计日志
- **Query**：基于 wiki 回答 → 每句附引用 → 如有新知识可归档
- **Lint**：扫描死链/孤岛/陈旧/冗余 → 生成可审查的修复补丁

### 与 RAG 的关键差异

| 维度 | RAG | LLM Wiki |
|---|---|---|
| 状态 | 无状态 | 有状态 |
| 知识积累 | 不积累 | 每次操作都增长 |
| 可审计性 | 黑盒检索 | `git diff` 可见演化 |
| 扩展性 | 依赖向量维度 | Markdown + 索引足够 |

## 应用场景

- 个人知识库（学习笔记、读书摘要）
- 技术参考（API 文档提炼）
- 业务 SOP（跨项目经验沉淀，例如民宿运营）
- 研究追踪（论文 + Gist + 讨论综合）

## 争议与限制

- 对 LLM 上下文窗口有要求（超过 ~100 篇需分层召回）
- 不适合高频小改动（每次 ingest 有成本）
- 需要治理（CLAUDE.md 宪法约束）

## References

- [[../../raw/articles/2026-04-17-test-llm-wiki.md|Test Article about LLM Wiki]]

## See Also

- [[knowledge-compounding]]
