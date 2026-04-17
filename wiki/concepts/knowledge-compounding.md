---
title: Knowledge Compounding
aliases: [知识复利, 知识积累]
type: concept
tags: [ai/llm/framework, learn/note]
sources:
  - raw/articles/2026-04-17-test-llm-wiki.md
created: 2026-04-17
updated: 2026-04-17
---

# Knowledge Compounding

> **定义**：知识像资金一样随时间累积、产生利息的机制。通过 [[llm-wiki]] 实现：每次摄取/查询/巡检都让网络自动变大、变强、自愈。

## 核心思想

1. 每一条新 raw 不只被存储，还被**编译**成可跨页引用的知识节点
2. 每一次查询不只被回答，还可能**归档**为新的综合页
3. 每一次巡检不只发现问题，还**自动修复**死链、标记陈旧

## 对比：线性知识 vs 复利知识

| 模式 | 特征 | 典型工具 |
|---|---|---|
| 线性堆积 | 资料越多越乱，查找变慢 | 堆满硬盘的 PDF |
| 索引存储 | 有搜索但不综合 | Notion 数据库 |
| RAG 检索 | 每次现查，不沉淀 | 一般 RAG 系统 |
| **复利编译** | 自动综合、自动纠错、自动成长 | **LLM Wiki** |

## 度量指标

- **Ingest 吞吐**：每周新 raw 文件数 + 产出 wiki 页数
- **Query 命中率**：问题能被 wiki 回答的比例
- **Lint 健康度**：`wiki/_meta/health.md` 中 CRITICAL + HIGH 数
- **交叉链接密度**：平均每页的 inbound/outbound 链接数

## 应用领域

- 业务沉淀（民宿 SOP、新媒体选题库）
- 学习复盘（书、播客、课程）
- 技术积累（框架迁移经验、踩坑笔记）

## References

- [[../../raw/articles/2026-04-17-test-llm-wiki.md|Test Article about LLM Wiki]]

## See Also

- [[llm-wiki]]
