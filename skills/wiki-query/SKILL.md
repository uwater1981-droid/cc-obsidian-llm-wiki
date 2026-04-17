---
name: wiki-query
description: "基于 wiki 回答问题，带引用，可选归档新知识。触发词：什么是、解释、对比、查询、如何、问 wiki、先查 wiki。"
origin: local
---

# Wiki Query Skill

在 `wiki/` 中搜索，综合答案并附 `[[wiki-page]]` 引用。若答案含 wiki 之外的新知识，询问用户是否归档为新页面。

**宪法依据**：遵守 `CLAUDE.md` 第五节（交叉引用）、第七节（隐私边界）、第八节（质量门禁）。

## 何时激活

- 用户问题以"什么是/解释/对比/如何/为什么"开头
- 用户明确说"问 wiki"、"先查 wiki"、"查询"、"query"
- 用户要求"综述"、"对比"某些主题
- **不激活**：一般问候、无关闲聊、外部 API 文档问题（优先用 /docs）

## 输入契约

| 输入形式 | 示例 |
|---|---|
| 直接提问 | "什么是 LLM Wiki？" |
| 对比 | "对比 RAG 和 LLM Wiki" |
| 综述 | "我都知道哪些关于 prompt engineering 的知识？" |
| 带归档意图 | "查询...并归档答案" |

## 输出契约

**默认输出**：
- 在对话中返回答案（不写文件）
- 每个事实性陈述后附 `[[wiki-page#section]]` 引用
- 若 wiki 不含相关知识，显式说明："当前 wiki 中无此知识"
- 若答案综合了 wiki 外的新知识，询问："是否归档到 wiki？"

**归档模式输出**（用户同意归档）：
- 创建 `wiki/topics/<domain>/<slug>.md` 或 `wiki/concepts/<slug>.md`
- frontmatter 含 `provenance: query-session`、`sources: []`（允许空）
- 追加 `runs/query/YYYY-MM-DD-HHMM.jsonl` 记录

## Workflow

### Phase 1: Retrieve（召回）

1. 解析问题关键词（去停用词，提取实体+概念）
2. 在 `wiki/` 中做三级召回：
   - **L0 index**：读 `wiki/index.md` 的 Dataview 聚合表
   - **L1 topic**：在 `wiki/topics/` 按 frontmatter tags 过滤
   - **L2 entity/concept**：grep 关键词，按命中次数排序
3. 召回 top-K（默认 5），拼接上下文

### Phase 2: Budget（上下文预算）

- 硬上限 **8K tokens** 的 wiki 内容
- 超出时按优先级截断：
  1. 保留 index 摘要（必留）
  2. 保留 topic 页首段（必留）
  3. 截断 entity/concept 页末尾段落
  4. 最后兜底：只留 frontmatter + 首段

### Phase 3: Synthesize（综合）

使用 Claude 生成答案：
- **格式**：分 2-4 段，每段有标题
- **引用**：每个事实陈述后附 `[[wiki-page#section]]`
- **无知识声明**：若 wiki 中确无相关内容，第一段明确写"当前 wiki 中无关于 X 的专页"

**隐私保护**：若召回内容来自 `raw/notes/private/` 间接引用（通过 wiki 页），答案中不得复现原文超过 50 字。

### Phase 4: New Knowledge Detection（新知识检测）

判断答案是否引入 wiki 中不存在的知识：
- 如果答案整体基于 wiki，直接返回答案
- 如果答案有外部补充（如 Claude 自己的知识或 WebSearch 结果），询问：
  ```
  ⚠️ 此答案包含 wiki 之外的新知识：
  - 来源：[Claude 训练数据 | WebSearch]
  - 主要增量点：<列出 1-3 条>
  是否归档为新 wiki 页？(y/n)
  ```

### Phase 5: Archive (optional)

用户同意归档：
1. 基于答案内容，判断归档位置：
   - 单一概念 → `wiki/concepts/<slug>.md`
   - 跨领域综述 → `wiki/topics/<domain>/<slug>.md`
2. 用 `templates/topic.md` 或 `templates/concept.md` 模板
3. frontmatter 特殊字段：
   ```yaml
   provenance: query-session
   sources: []  # query-session 允许为空
   query: "<原问题>"
   ```
4. 追加 `runs/query/YYYY-MM-DD-HHMM.jsonl`：
   ```json
   {"ts": "2026-04-17T10:00:00+08:00", "question": "...", "retrieved": ["wiki/..."], "archived": "wiki/concepts/...", "new_knowledge": true}
   ```

### Phase 6: Git Commit（归档时）

```bash
git add wiki/ runs/query/
git commit -m "docs(query): archive '<slug>'"
```

## 成功条件

- [ ] 答案含 ≥ 1 个 wiki 引用，或显式声明"wiki 中无此知识"
- [ ] 无隐私泄漏（private 原文不超 50 字）
- [ ] 上下文预算内未爆（≤ 8K tokens）
- [ ] 若归档，新页 frontmatter 合规、含 `provenance: query-session`

## 失败处理

- **召回 0 结果**：直接返回 "wiki 中无关于 X 的知识。建议先 ingest 相关资料。"
- **上下文超限**：按 Phase 2 策略逐级截断；极端情况提示用户"wiki 太大，请缩小查询范围"
- **归档冲突**（slug 已存在）：追加数字后缀 `-2`，或合并建议给用户选

## 示例调用

```
用户：什么是 LLM Wiki？

预期行为：
1. 关键词：["LLM Wiki", "Karpathy"]
2. 召回：wiki/concepts/llm-wiki.md、wiki/entities/andrej-karpathy.md
3. 综合答案：
   "LLM Wiki 是 Andrej Karpathy [[andrej-karpathy]] 提出的持久化知识库理念
    [[llm-wiki#定义]]，区别于传统 RAG [[rag]]..."
4. 判断：答案完全基于 wiki，无需归档
5. 直接返回
```

```
用户：LLM Wiki 的思想能用在民宿 SOP 沉淀吗？

预期行为：
1. 关键词：["LLM Wiki", "民宿", "SOP"]
2. 召回：wiki/concepts/llm-wiki.md + wiki/topics/business/（若有 SOP 相关页）
3. 综合答案：引用 LLM Wiki 核心思想 + 推断在民宿场景的应用
4. 检测：有外部推断（wiki 中无 minsu-sop-with-llm-wiki 专页）
5. 询问归档 → 用户同意 → 创建 wiki/topics/business/minsu-sop-with-llm-wiki.md
```

## 依赖

- `wiki/index.md` + Dataview 聚合
- `templates/concept.md`、`templates/topic.md` — 归档模板
- `.reference/karpathy-llm-wiki-upstream/SKILL.md` — 上游参考
