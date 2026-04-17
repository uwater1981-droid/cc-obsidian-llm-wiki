# LLM Wiki 宪法（知识复利机制 · 不可违反）

> 本文件是整个 vault 的"宪法"。任何 skill、agent、人工操作都必须遵守以下条款。
> 与用户全局 `~/.claude/CLAUDE.md` 的 7 阶段闭环开发规则兼容。

## 一、项目定位

基于 Andrej Karpathy 提出的 LLM Wiki 理念，实现"知识复利"：
- **Raw 层**存原始资料（论文/博文/笔记/字幕），**只追加不修改**
- **Wiki 层**是 AI 编译的百科全书，**可增可改**，由 skill 维护
- **CLAUDE.md 层**（本文件）是治理规则，由人维护

**核心差异**：区别于 RAG 的无状态检索，LLM Wiki 让知识像 Git 一样**可追溯、可审计、可演化**。

参考：
- Karpathy 原始 Gist: https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f
- 本地基准实现: `.reference/karpathy-llm-wiki-upstream/SKILL.md`

## 二、目录契约

```
raw/            # 不可变原始层（只追加）
  papers/       #   学术论文
  articles/     #   博客/新闻剪藏
  gists/        #   GitHub Gist 快照
  notes/
    public/     #   可共享笔记
    private/    #   隐私笔记（.gitignore 排除）
  web-clips/    #   网页剪藏
  transcripts/  #   播客/视频字幕
  _manifest.jsonl  # append-only 入库记录

wiki/           # AI 编译层（可增可改）
  index.md      #   Dataview 聚合的主索引
  entities/     #   实体页（人/组织/工具/产品）
  concepts/     #   概念页（抽象术语）
  topics/       #   主题页（按领域）
    ai-llm/
    business/
    learning/
    tech-ref/
  _meta/
    tags.md     #   标签字典
    glossary.md #   术语表
    health.md   #   Lint 汇总

skills/         # 三大操作 skill
  wiki-ingest/
  wiki-query/
  wiki-lint/

templates/      # 模板（被 skill 读取）
scripts/        # 辅助脚本（Python）
runs/           # 操作日志（对齐云图 Atlas 模式）
  ingest/ query/ lint/
.reference/     # 上游参考实现（只读）
```

## 三、分层契约（不可违反）

1. **raw/** 只读追加
   - Skill 与 agent 不得修改或删除 raw 下已有文件
   - 新增文件必须同时追加一行到 `raw/_manifest.jsonl`
   - 入库记录格式：`{"path": "raw/...", "sha256": "...", "source": "url|local|paste", "ingested_at": "YYYY-MM-DDTHH:MM:SS+08:00"}`

2. **wiki/** 可读可写
   - 每一页 frontmatter 必须含 `sources: [raw/...]` 字段（至少一条）
   - 例外：query-session 归档页标注 `provenance: query-session`
   - 修改 wiki 页时必须同步更新 `updated: YYYY-MM-DD`

3. **禁止跨层污染**
   - raw 文件内容不得引用 wiki 页
   - wiki 引用 raw 用相对路径：`../../raw/<topic>/<file>.md`
   - wiki 内部引用用 Obsidian 双链：`[[concepts/llm-wiki]]` 或短写 `[[llm-wiki]]`

## 四、命名约定

- 所有文件名使用 **kebab-case**（小写连字符），不得含空格、大写、标点
- **中文主题**用拼音 + frontmatter `aliases: [中文名]`
  - 例：`yun-tu-minsu.md` + `aliases: [云图民宿]`
  - 原因：iCloud + Windows shell + git 对中文路径支持不稳定
- 目录层级：
  - `wiki/entities/<slug>.md` — 实体（Karpathy、Claude Code）
  - `wiki/concepts/<slug>.md` — 概念（LLM Wiki、RAG）
  - `wiki/topics/<domain>/<subject>.md` — 主题（business/minsu-sop）
- 模板文件：`templates/<type>.md`

## 五、交叉引用规范

- **Obsidian 双链格式**：`[[page-slug]]` 或 `[[page-slug|显示文本]]`
- 首次提及某实体/概念**必须**建立双链
- 禁止直接循环引用（A→B→A），允许多跳（A→B→C→A）
- 每页至少应有 1 个 outbound 链接和 1 个 inbound 链接（孤岛判定）

## 六、三操作触发规则

| 操作 | 触发词 | 调用 skill |
|---|---|---|
| Ingest | 摄取 / 归档 / 入库 / ingest / 新文章 / 新笔记 / 处理 raw | `skills/wiki-ingest` |
| Query | 什么是 / 解释 / 对比 / 查询 / 如何 / 问 wiki | `skills/wiki-query` |
| Lint | 巡检 / 体检 / lint / 健康检查 / 扫描 wiki | `skills/wiki-lint` |

**Ingest 自动触发条件**：当 `raw/**` 出现新文件（与 manifest 对比发现增量）
**Lint 自动触发条件**：距上次 lint > 7 天时，由用户或定时任务触发

## 七、隐私边界（安全红线）

1. **`raw/notes/private/**` 禁区**
   - 已在 `.gitignore` 中排除，不进 git
   - Wiki 页引用 private 原文不得超过 **50 字**
   - Wiki 页不得在 Sources 中暴露 private 文件路径（用占位符 `[private-source-N]`）

2. **外传禁止**
   - raw 内容禁止发送给非本地 Claude Code（不上传任何 MCP/API）
   - Web 搜索用于 query 时，不得把 private 内容带入 query

3. **与其他项目的隔离**
   - 本 vault **不读写**以下路径：
     - `~/.local/lib/3body/**`
     - `F:/Icloud/iCloudDrive/Anthropic/新媒体发布系统*/`
     - `F:/Icloud/iCloudDrive/Anthropic/云图Atlas工作系统/`
     - `F:/Icloud/iCloudDrive/Anthropic/国际贸易*/`
   - 跨项目知识以内部 wiki 页承载（业务经验归档到 `wiki/topics/business/`）

## 八、Lint 质量门禁（CRITICAL 拒绝合并）

以下任一条不通过则 Lint 报 CRITICAL：
- 存在死链（wiki 页指向不存在的目标）
- 孤岛页（indegree=0 且 outdegree=0）且 mtime > 30 天
- wiki 页 frontmatter 缺少 `sources` 字段（query-session 例外）
- 单页行数 > 800（应拆分）
- wiki 引用 `raw/notes/private/**` 原文超过 50 字（隐私泄漏）

HIGH 级别警告（允许但需修复）：
- 陈旧 TODO：`- [ ] ...` 项超过 90 天未勾选
- 冗余内容：两页相似度 > 0.85
- Dataview 查询失败

## 九、与用户全局 CLAUDE.md 的兼容

本项目 skill 开发仍适用全局 7 阶段闭环：
1. Research → 2. Plan → 3. TDD+Build → 4. Review → 5. Accept → 6. Verify → 7. Learn

特别地：
- Stage 4 Review 并行启动 `code-reviewer` + `security-reviewer` + `python-reviewer`
- Stage 6 Verification 必须跑通 `.reference/karpathy-llm-wiki-upstream/examples/` 等效冒烟测试

## 十、版本与审计

- **Git** 作为主要审计：每次 ingest/lint 后自动 commit（由 skill 负责，commit message 带操作摘要）
- **`_manifest.jsonl`** 作为原始审计：不可修改，只追加
- **`runs/`** 作为过程审计：每次 skill 调用的输入输出日志
- 三者共同支撑 Karpathy 原规范的"可追溯"核心价值

## 十一、迭代规则

- 本 `CLAUDE.md` 每 30 天回顾一次（由 lint 提醒）
- 新增规则前必须先在 `runs/` 留有 ≥ 3 次类似情况的证据
- 删除规则需要至少 7 天观察期（写在 `_meta/health.md` 中等待）
- 重大修改必须在 git commit 中带 `[constitution]` 前缀标签

---

**记住**：Raw 是基石（不可动），Wiki 是知识（会演化），CLAUDE.md 是宪法（由你治理）。
