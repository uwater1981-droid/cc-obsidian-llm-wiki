---
name: wiki-deploy-check
description: "用 7 阶段闭环法检查 LLM Wiki 部署状态，输出合规报告和修复建议。触发词：部署检查、闭环检查、deploy check、7 stage check、健康体检（完整版）。"
origin: local-meta
---

# Wiki Deploy Check Skill

对本项目的部署状态做**全链路七阶段闭环**自检，识别缺失、陷阱、改进点。
区别于 `wiki-lint`（只看 wiki 内容健康），本 skill 检查**整个部署 + 治理体系**。

**宪法依据**：遵守 `CLAUDE.md` 第九节（兼容用户全局 7 阶段闭环）。

## 何时激活

- 用户说："部署检查"、"闭环检查"、"deploy check"、"健康体检（完整）"
- 大改动后（修改 CLAUDE.md / 新增 skill / 升级 Obsidian 插件栈）
- 项目上线前、迁移前、季度复盘时
- **不激活**：日常 ingest/query/lint 已经覆盖（本 skill 更重）

## 输入契约

| 输入 | 示例 |
|---|---|
| 无参全检 | `部署检查` |
| 指定阶段 | `部署检查 --stage=4`（只跑 Review 阶段） |
| 严格模式 | `部署检查 --strict`（未达标直接 exit 1）|

## 输出契约

- `runs/lint/deploy-check-YYYY-MM-DD-HHMM.md` 带通过/失败清单
- 更新 `wiki/_meta/health.md` 的"部署闭环"段
- CRITICAL 缺口 → 对话中报警
- 给出**下一步修复动作清单**

## 七阶段检查清单

### Stage 1 — Research（理论/参考已就位）

| 检查项 | 证据 | 期望 |
|---|---|---|
| 理论基础声明 | `CLAUDE.md` 第一节含 Karpathy Gist 链接 | 有 |
| 上游参考实现 | `.reference/karpathy-llm-wiki-upstream/SKILL.md` 存在 | 有 |
| 相关开源调研记录 | `plan` 文件中列出 ≥ 3 个 fork 对照 | 有 |

### Stage 2 — Plan（方案文档化）

| 检查项 | 证据 | 期望 |
|---|---|---|
| 部署 Plan 文件 | `~/.claude/plans/karpathy-llm-wiki-*.md` | 有 |
| CLAUDE.md 宪法 | 含 11 节结构（分层/命名/隐私/门禁...）| 行数 ≥ 120 |
| 目录契约落地 | `raw/` `wiki/` `skills/` `templates/` `scripts/` `runs/` `.reference/` | 全部存在 |

### Stage 3 — TDD + Build（脚本与 skill 可用）

| 检查项 | 命令 | 期望 |
|---|---|---|
| Python 脚本可运行 | `python scripts/dedupe.py .` | exit 0 |
| 链接图脚本无异常 | `python scripts/link-graph.py .` | 0 dead |
| 陈旧检测可运行 | `python scripts/stale-check.py .` | exit 0 |
| 锁机制原子性 | `icloud-lock.py acquire/release` | 双方都 exit 0 |
| 三 skill SKILL.md 完备 | `skills/wiki-{ingest,query,lint}/SKILL.md` | 全部 ≥ 80 行 |
| 四模板存在 | `templates/{entity,concept,topic,ingest-record}.md` | 全部有 |
| ⚠️ pytest 单元测试 | `tests/test_*.py` | **当前缺口**（已登记）|

### Stage 4 — Review（并行评审已过）

| 检查项 | 证据 | 期望 |
|---|---|---|
| code-reviewer 跑过 | git log 含 `fix(review)` | 至少 1 次 |
| security-reviewer 跑过 | CRITICAL 安全项已闭环 | 0 CRITICAL 未修 |
| python-reviewer 跑过 | `.claude/settings.local.json` 有 deny | 6+ 条 deny |
| Pre-commit hook 防护 | `.git/hooks/pre-commit` 存在且拒绝 private notes | 测试通过 |

### Stage 5 — Acceptance（端到端跑通）

| 检查项 | 证据 | 期望 |
|---|---|---|
| 冒烟测试记录 | `runs/ingest/*smoke-test*.md` | 至少 1 份 |
| 示例 wiki 页 | `wiki/entities/` + `wiki/concepts/` 非空 | ≥ 2 页 |
| Manifest 与实际 raw 一致 | `dedupe.py` Modified=0 | 一致 |
| Obsidian vault 可打开 | `.obsidian/community-plugins.json` 列 6 个插件 | 有 |
| 核心三插件装好 | `.obsidian/plugins/{dataview,templater-obsidian,obsidian-tasks-plugin}/manifest.json` 存在 | 全部有 |
| BRAT + 2 beta 插件 | `.obsidian/plugins/{obsidian42-brat,find-unlinked-files,broken-links-cleaner}/` | 全部有 |

### Stage 6 — Verification（质量门禁）

| 检查项 | 命令 | 期望 |
|---|---|---|
| 死链 | `link-graph.py` | 0 |
| 孤岛 | `link-graph.py` | 0 |
| 陈旧 TODO | `stale-check.py` | < 5 |
| `_manifest.jsonl` 非空 | `wc -l raw/_manifest.jsonl` | ≥ 1 |
| `.gitignore` 排除 private | `grep "raw/notes/private" .gitignore` | 命中 |
| 项目隔离 deny 完备 | `settings.local.json` deny 含 3body/Atlas/新媒体/国际贸易 | 4 条 |

### Stage 7 — Learn（经验沉淀）

| 检查项 | 证据 | 期望 |
|---|---|---|
| 项目记忆文件 | `~/.claude/projects/<hash>/memory/MEMORY.md` | 有 |
| 项目状态记录 | 同上的 `project_*.md` 详情页 | 有 |
| runs/ 审计日志 | `runs/ingest/` + `runs/lint/` 非空 | 非空 |
| 部署经验 skill | 本文件本身（wiki-deploy-check） | 有 |
| Git 提交链完整 | `git log --oneline \| wc -l` | ≥ 5 |

---

## Workflow（执行步骤）

```
Phase 1: 收集证据
  → 读各阶段对应的文件 / 运行脚本
  → 整理为结构化 JSON

Phase 2: 断言与打分
  → 对每个检查项标记 PASS / WARN / FAIL
  → CRITICAL（Stage 1-6 的 PASS 缺失）
  → HIGH（质量门禁 WARN）
  → MEDIUM（Stage 7 WARN）

Phase 3: 生成报告
  → runs/lint/deploy-check-<date>.md
  → 按阶段分节，每节 表格 + 摘要
  → 末尾给"修复动作清单"

Phase 4: 更新健康状态
  → wiki/_meta/health.md 追加一行

Phase 5: Git commit（可选）
  → git add runs/lint/ wiki/_meta/health.md
  → git commit -m "chore(health): deploy-check YYYY-MM-DD"
```

## 成功条件

- [ ] 所有 7 阶段 PASS，或 FAIL 项都有登记的改进计划
- [ ] 报告写入 `runs/lint/`
- [ ] `wiki/_meta/health.md` 更新
- [ ] 无未修复 CRITICAL

## 常见陷阱（本次部署踩过）

| 陷阱 | 发生阶段 | 修复 |
|---|---|---|
| iCloud + Windows 中文目录编码问题 | Build | 强制 kebab-case 拼音 + aliases |
| Vault 选错（在主 vault 跑 skill 导致隔离失效）| Accept | 每次启动 Claude 前验证 `pwd` |
| 模板 `[[X]]` 占位符被 Broken Links Cleaner 判为死链 | Verify | 占位符包代码块 + HTML 注释 |
| BRAT 追踪 ≠ 自动下载 | Accept | 需手动 Command Palette 触发 |
| 插件名差异（manifest id vs GitHub repo name） | Accept | 以 `plugins/<dir>/manifest.json` 的 id 为准 |
| `Read(./**)` 过宽暴露 private notes | Review | 改成具体子路径 allow + private deny |
| link-graph 把跨层 `[[../../raw/...]]` 当死链 | Build | 脚本加 vault containment check |

## 示例调用

```
用户：部署检查

预期行为：
1. 跑完 7 阶段全量检查（约 30 秒）
2. 生成 runs/lint/deploy-check-2026-04-17-1200.md
3. 对话中摘要：
   "部署检查完成（2026-04-17 12:00）：
    - Stage 1-2 PASS
    - Stage 3 WARN：缺 pytest 单元测试
    - Stage 4-6 PASS
    - Stage 7 PASS
    - 0 CRITICAL / 1 WARN / 0 FAIL
    完整报告：runs/lint/deploy-check-2026-04-17-1200.md
    建议下次 ingest 前补 tests/ 目录"
```

## 依赖

- `scripts/dedupe.py`、`link-graph.py`、`stale-check.py`、`icloud-lock.py`
- `.reference/karpathy-llm-wiki-upstream/` 上游参考
- 用户全局 `~/.claude/CLAUDE.md` 第 7 阶段规范
