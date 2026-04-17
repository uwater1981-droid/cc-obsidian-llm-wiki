---
name: wiki-lint
description: "扫描 wiki 的死链、孤岛、陈旧、冗余、隐私泄漏，生成修复补丁。触发词：巡检、体检、lint、健康检查、扫描 wiki。自动触发：距上次 lint > 7 天。"
origin: local
---

# Wiki Lint Skill

分四阶段巡检 `wiki/`，识别死链/孤岛/陈旧/冗余问题，生成可应用的修复补丁。

**宪法依据**：遵守 `CLAUDE.md` 第八节（Lint 质量门禁）。

## 何时激活

- 用户说："巡检"、"体检"、"lint"、"健康检查"、"扫描 wiki"、"找死链"
- 定时任务：距 `wiki/_meta/health.md` 上次更新 > 7 天
- Ingest 结束后（轻量 lint，只检查新增页的死链）
- **不激活**：单页的手动编辑即时检查（不值得跑全量）

## 输入契约

| 输入形式 | 示例 |
|---|---|
| 全量巡检 | `lint` |
| 指定路径 | `lint wiki/topics/ai-llm/` |
| 仅某类检查 | `lint --dead-links` 或 `lint --orphans` |
| 自动修复 | `lint --auto-fix`（仅修复安全项） |

## 输出契约

- 生成 `runs/lint/YYYY-MM-DD-HHMM-report.md`（含 diff 格式修复补丁）
- 更新 `wiki/_meta/health.md`（CRITICAL/HIGH 计数）
- 若 `--auto-fix`：应用修复，更新 wiki 文件
- 若发现 CRITICAL 问题：在对话中报警并列出具体页面

## Workflow（四阶段，参考 skill-stocktake 模式）

### Phase 1: Dead Links Scan（死链扫描）

1. 正则扫描所有 `wiki/**/*.md`：
   - `\[\[([^\]]+)\]\]` — Obsidian 双链
   - `\[([^\]]+)\]\(([^)]+\.md)\)` — 标准 markdown 链接
2. 对每个链接检查目标文件存在性
3. 对每个死链：
   - 在 wiki/ 中搜索同名文件（可能是移动/重命名）
   - 若唯一匹配 → 修复路径（`--auto-fix` 时）
   - 若多/零匹配 → 报告等待用户决定
4. 严重级别：**CRITICAL**

### Phase 2: Orphan Detection（孤岛检测）

1. 构建 wiki 有向图：
   - 节点 = 每个 wiki 页
   - 边 = 页面间的 `[[link]]` 关系
2. 找出 `indegree=0 AND outdegree=0 AND mtime > 30d` 的页
3. 对每个孤岛：
   - 提议 3-5 个可能的相关页（基于 tags + 关键词）
   - 建议用户手动建链或标记 `archived: true`
4. 严重级别：**CRITICAL**（>30 天） / MEDIUM（新页）

调用 `scripts/link-graph.py`。

### Phase 3: Stale Detection（陈旧检测）

1. 提取所有 `- [ ] ...` 任务项
2. 对每个任务项检查：
   - 优先使用 frontmatter `updated:` 字段（与 CLAUDE.md 命名约定一致）
   - 缺少时回退到文件 mtime（注意：iCloud 同步可能更新 mtime）
   - 如果 `today - (updated|mtime) > 90 天` → 提醒
3. 提取 frontmatter `updated:` 字段：
   - 页面超过 180 天未更新 → HIGH 警告
4. 严重级别：HIGH

调用 `scripts/stale-check.py`。

### Phase 4: Redundancy & Privacy Check（冗余与隐私）

1. **冗余检测**：
   - 对同目录页做 title+frontmatter `aliases` 的 shingle 相似度
   - 相似度 > 0.85 → 建议合并
2. **Frontmatter 合规**：
   - 缺 `sources` 字段（非 `provenance: query-session` 页）→ CRITICAL
   - `provenance: query-session` 允许 `sources: []`（CLAUDE.md 例外条款）
   - 单页行数 > 800 → CRITICAL（建议拆分）
3. **隐私泄漏检测（不读私密文件）**：
   - 纯脚本方案：**不把 private 原文读入 LLM 上下文**
   - grep wiki 页：Sources / References 字段中是否出现 `raw/notes/private/` 路径字符串 → CRITICAL
   - 长度检测：wiki 页中紧跟 `raw/notes/private/` 引用标记之后的连续引用块 > 50 字 → HIGH
   - 如需深度比对（50 字片段匹配），通过独立 Python 脚本对比 SHA256 哈希前缀与字符数，**比对结果（计数+路径）**传入 LLM，**原文不传入**
   - 严重级别：**CRITICAL**（路径暴露）/ HIGH（长度超限）

### Phase 5: Report Generation

生成 `runs/lint/YYYY-MM-DD-HHMM-report.md`：

```markdown
# Wiki Lint Report — 2026-04-17 10:30

## Summary
- CRITICAL: 2
- HIGH: 5
- MEDIUM: 3
- AUTO-FIXED: 7

## CRITICAL Issues

### 死链
- `wiki/topics/ai-llm/prompt-engineering.md:42` → `[[不存在页]]`
  建议：搜索结果有 2 个同名候选，请选择：
    1. wiki/concepts/not-exist.md
    2. wiki/topics/tech-ref/not-exist.md

### 隐私泄漏
- `wiki/topics/business/minsu.md` 引用 `raw/notes/private/minsu-financial.md` 原文 87 字
  建议：改为 `[private-source-1]` 占位符，或移至 public/

## HIGH Issues
...

## Auto-fixed (已应用)
- wiki/xxx.md: 修复死链 [[yyy]] → [[concepts/yyy]]
- ...

## Diff Patch (待用户确认)
```diff
--- a/wiki/topics/business/minsu.md
+++ b/wiki/topics/business/minsu.md
@@ -10,5 +10,5 @@
-<隐私原文 87 字>
+[private-source-1]
```

### Phase 6: Health Summary & Commit

更新 `wiki/_meta/health.md`：

```markdown
# Wiki 健康状态

## 最近一次巡检
- 时间：2026-04-17 10:30
- 死链：2 (CRITICAL)
- 孤岛：1 (CRITICAL)
- 陈旧 TODO：5 (HIGH)
- 冗余页：0
- 隐私泄漏：1 (CRITICAL)

## 历史记录
- 2026-04-17: 9 问题 (3 CRITICAL, 5 HIGH, 1 MEDIUM), 7 auto-fixed
- ...
```

若有 `--auto-fix`：
```bash
git add wiki/ runs/lint/
git commit -m "fix(lint): auto-fix N issues (dead links, orphan cross-refs)"
```

## 成功条件

- [ ] 四个 phase 全部执行完（或按 `--<category>` 过滤）
- [ ] 报告写入 `runs/lint/`
- [ ] `wiki/_meta/health.md` 已更新
- [ ] 若无 CRITICAL，标记绿灯
- [ ] 所有 auto-fix 必须可通过 `git diff` 审查

## 失败处理

- 文件编码错误（中文/BOM）→ 跳过并报告，不终止
- Python 脚本依赖缺失 → 在 `runs/lint/errors.md` 提示安装
- 大量 CRITICAL（>50）→ 建议用户先手动清理再全量 lint

## 示例调用

```
用户：lint

预期行为：
1. Phase 1-4 全跑
2. 生成报告 runs/lint/2026-04-17-1030-report.md
3. 在对话中摘要：
   "巡检完成：2 CRITICAL, 5 HIGH, 3 MEDIUM
    - 1 个死链（已自动修复）
    - 1 个孤岛（需手动处理：wiki/topics/learning/obsolete-concept.md）
    - 1 个隐私泄漏（需手动修复）
    完整报告：runs/lint/2026-04-17-1030-report.md"
4. 若用户说 "apply auto-fix" → 应用安全修复
```

## 依赖

- `scripts/link-graph.py` — 构建链接图
- `scripts/stale-check.py` — 识别陈旧内容
- `wiki/_meta/health.md` — 健康汇总
- Obsidian Find Unlinked Files 插件（可选，二次验证）
