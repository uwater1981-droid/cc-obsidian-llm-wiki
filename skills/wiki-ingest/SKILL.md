---
name: wiki-ingest
description: "摄取 raw/ 新资料并编译为 wiki 页面。触发词：摄取、归档、入库、ingest、新文章、新笔记、处理 raw。自动触发：raw/ 下检测到未登记的新文件。"
origin: local
---

# Wiki Ingest Skill

读取 `raw/` 下的新资料，抽取实体/概念，编译为 `wiki/` 页面，建立交叉链接，并维护审计日志。

**宪法依据**：遵守 `CLAUDE.md` 第三节（分层契约）、第四节（命名约定）、第七节（隐私边界）。

## 何时激活

- 用户说："摄取"、"归档"、"入库"、"ingest"、"处理 raw"、"有新文章"
- 用户把文件拖入 `raw/` 任一子目录后说任何相关词
- 定时任务扫描 `raw/_manifest.jsonl` 发现未登记文件
- **不激活**：Wiki 页手动修改时（那是人工编辑，不是摄取）

## 输入契约

| 输入形式 | 示例 |
|---|---|
| 具体文件路径 | `ingest raw/articles/2026-04-karpathy.md` |
| 目录批量 | `ingest raw/papers/` |
| URL（下载后再摄取） | `ingest https://karpathy.github.io/...` |
| 隐式（raw/ 扫描） | `摄取所有新文件` |

## 输出契约

每次运行成功后：
1. `wiki/entities/` 或 `wiki/concepts/` 或 `wiki/topics/<domain>/` 下有新增/更新页
2. `raw/_manifest.jsonl` 追加一行 JSON（每个新 raw 文件一行）
3. `wiki/index.md` 中的 Dataview 自动刷新（无需手动编辑）
4. `runs/ingest/YYYY-MM-DD-HHMM.md` 新增一份运行报告
5. 如启用 git：自动 commit `feat(ingest): <N> raw files → <M> wiki pages`

## Workflow

### Phase 1: Scan & Diff（扫描去重）

1. 扫描 `raw/**/*.{md,txt,pdf}` 所有文件
2. 对每个文件计算 SHA256（pdf 先 `pdftotext` 提取文本再算）
3. 与 `raw/_manifest.jsonl` 对比，识别：
   - 新增：从未登记过
   - 修改：路径相同但 SHA256 变化（**警告用户**：raw 应只追加）
   - 已存在：跳过

调用 `scripts/dedupe.py`（已提供）。

### Phase 2: Parse Metadata（抽元数据）

对每个新文件：
- 提取标题（第一个 `# ` 或 filename）
- 提取 URL（frontmatter `source:` 或正文第一个链接）
- 提取作者（frontmatter `author:` 或正文署名）
- 提取日期（frontmatter `date:` 或文件名前缀 `YYYY-MM-DD`）
- 按介质分类（目录名已经给出：papers/articles/...）
- 判断领域标签（ai-llm / business / learning / tech-ref）

### Phase 3: Extract Candidates（抽取候选实体/概念）

使用 Claude 读取 raw 全文，生成候选列表：

```
候选实体（Named Entities）：
- 人物：如 Andrej Karpathy
- 工具/产品：如 Claude Code、Obsidian、Dataview
- 组织：如 Anthropic、OpenAI

候选概念（Abstract Concepts）：
- LLM Wiki、RAG、Prompt Engineering、Knowledge Compounding

候选主题页（如果跨多个实体/概念）：
- "知识复利机制" → topics/learning/knowledge-compounding.md
```

**隐私保护**：若 raw 位于 `raw/notes/private/`，跳过外传给 LLM 的步骤，仅做路径级登记（wiki 页不引用正文超过 50 字）。

### Phase 4: Compile Wiki Pages

对每个候选：

**A. 页面已存在**（同名文件在 entities/concepts/topics 下）：
- 在 `## References` 段追加新 raw 的链接
- 如果新 raw 提供了新定义/数据，在对应段追加并用 `> [!note] 来源: <raw>` 标注
- 更新 frontmatter 的 `updated: YYYY-MM-DD`

**B. 页面不存在**：
- 根据类型选模板：
  - 实体 → `templates/entity.md`
  - 概念 → `templates/concept.md`
  - 主题 → `templates/topic.md`
- 填充字段：title、aliases、sources、tags、created、updated
- 生成首段摘要（200 字内）
- 列出 `## References` 段引用 raw 文件

**C. 交叉链接**：
- 页面内首次提及其他实体/概念时插入 `[[slug]]` 双链
- 维护 "See Also" 段（参考 .reference/ 做法）

### Phase 5: Append Manifest & Log

对每个 raw 文件追加到 `raw/_manifest.jsonl`：

```json
{"path": "raw/articles/2026-04-karpathy-llm-wiki.md", "sha256": "abc123...", "source": "https://gist.github.com/karpathy/...", "ingested_at": "2026-04-17T09:45:00+08:00", "wiki_pages": ["wiki/concepts/llm-wiki.md", "wiki/entities/andrej-karpathy.md"]}
```

写入 `runs/ingest/YYYY-MM-DD-HHMM.md`：
- 本次处理的 raw 文件清单
- 新建/更新的 wiki 页清单
- 异常与跳过（例如 SHA256 冲突）

### Phase 6: Git Commit（可选）

若 `.git/` 存在：
```bash
git add raw/_manifest.jsonl wiki/ runs/ingest/
git commit -m "feat(ingest): N raw → M wiki pages" -m "<运行报告摘要>"
```

## 成功条件

- [ ] 每个新 raw 文件都在 `_manifest.jsonl` 有记录
- [ ] 每个新 raw 文件至少关联 1 个 wiki 页
- [ ] 所有新/更新 wiki 页 frontmatter 含 `sources` 字段
- [ ] `runs/ingest/` 下有本次运行报告
- [ ] 无越界引用 `raw/notes/private/` 原文

## 失败与回滚

- **SHA256 冲突**（raw 被修改）：停止处理该文件，报告给用户，**不自动回滚 raw**（raw 只追加原则）
- **LLM 抽取失败**：记录到 `runs/ingest/<date>-errors.md`，跳过该 raw，继续处理其他
- **Wiki 写入失败**（磁盘满/权限）：`git stash` 回滚已写入的 wiki 变更

## 示例调用

```
用户：摄取 raw/articles/2026-04-karpathy-llm-wiki.md

预期行为：
1. 读取该文件，SHA256=abc123
2. 抽取候选：实体 [Andrej Karpathy]，概念 [LLM Wiki, Knowledge Compounding, RAG]
3. 创建 wiki/entities/andrej-karpathy.md（若不存在）
4. 创建 wiki/concepts/llm-wiki.md
5. 创建 wiki/concepts/knowledge-compounding.md
6. 在 wiki/concepts/rag.md 追加 References（假设已存在）
7. 页面间建立双链
8. 追加 manifest + log + git commit
```

## 依赖

- `scripts/dedupe.py` — SHA256 去重
- `templates/entity.md`、`templates/concept.md`、`templates/topic.md` — 页面模板
- `templates/ingest-record.md` — 运行报告模板
- `.reference/karpathy-llm-wiki-upstream/SKILL.md` — 上游参考（只读）
