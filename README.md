# CC + Obsidian 学习复利机制

> 基于 Andrej Karpathy 的 [LLM Wiki](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f) 理念，搭建的持久化个人知识库。配合 Claude Code + Obsidian 实现**知识复利**：每次摄取、查询、巡检都让知识网络自动增长和自愈。

---

## 一、核心理念

传统 **RAG** 是**无状态**的——每次查询重新检索，知识不沉淀。

**LLM Wiki** 是**有状态**的——AI 像编辑一样维护 Markdown 百科：

```
Ingest (摄取)  ─→  Query (查询)  ─→  Lint (巡检)
    ↑                                    │
    └────────────── 知识复利 ─────────────┘
```

三层架构：

| 层 | 目录 | 特性 | 谁来写 |
|---|---|---|---|
| Raw | `raw/` | 不可变原始资料（论文、博文、笔记） | 人类追加 |
| Wiki | `wiki/` | AI 编译的百科（实体/概念/主题页） | Skill 维护 |
| Rule | `CLAUDE.md` | 治理宪法（规则、边界、质量门禁） | 人类演化 |

三大操作：

| 操作 | 输入 | 输出 | 典型触发词 |
|---|---|---|---|
| **Ingest** | raw/ 新文件 | wiki/ 新/更新页 + manifest | "摄取"、"归档" |
| **Query** | 问题 | 带引用的答案 + 可选归档 | "什么是"、"对比" |
| **Lint** | 当前 wiki 状态 | 死链/孤岛/陈旧/冗余报告 | "巡检"、"lint" |

---

## 二、快速开始

### 前置条件

- [Claude Code](https://claude.com/claude-code) 已安装
- [Obsidian](https://obsidian.md) 已安装（可选但推荐）
- Python 3.10+（用于辅助脚本）
- Git（推荐）

### 5 分钟启动

```bash
# 1. 用 Obsidian 打开本目录作为 vault
#    Open folder as vault → 选中 cc+obisidian学习复利机制/
#    在 Settings → Community plugins 中启用五个推荐插件（见下）

# 2. 用 Claude Code 打开本目录作为工作目录
cd "F:/Icloud/iCloudDrive/Anthropic/cc+obisidian学习复利机制"

# 3. 放入第一份原始资料
cp <某篇文章.md> raw/articles/

# 4. 在 Claude Code 里说："摄取"
#    Claude 会调用 skills/wiki-ingest/，生成 wiki/ 下的实体/概念页
```

### 推荐 Obsidian 插件（已预配置在 `.obsidian/community-plugins.json`）

首次启动需在 Settings → Community plugins → Browse 安装：

- **Dataview** — `wiki/index.md` 自动聚合四领域最新页
- **Templater** — 新页自动套模板
- **Tasks** — 识别 `- [ ] TODO` 支持陈旧检测
- **Find Unlinked Files** — 孤岛二次验证
- **Broken Links Cleaner** — 死链批量清理

---

## 三、目录结构

```
.
├── CLAUDE.md               # 治理宪法（核心文件！）
├── README.md               # 本文件
├── .gitignore              # 排除 raw/notes/private、Obsidian workspace
├── .obsidian/              # Obsidian vault 配置
├── .claude/                # Claude Code 权限配置
│
├── raw/                    # 不可变原始层（只追加）
│   ├── papers/   articles/   gists/
│   ├── notes/public/   notes/private/
│   ├── web-clips/   transcripts/
│   └── _manifest.jsonl     # 入库审计链
│
├── wiki/                   # AI 编译的知识层
│   ├── index.md            # Dataview 聚合主索引
│   ├── entities/           # 人/组织/工具/产品
│   ├── concepts/           # 抽象术语
│   ├── topics/
│   │   ├── ai-llm/         # AI/LLM 研究与工程
│   │   ├── business/       # 民宿/新媒体/外贸
│   │   ├── learning/       # 个人学习笔记
│   │   └── tech-ref/       # 技术教程/API 速查
│   └── _meta/              # tags/glossary/health
│
├── skills/                 # 三大操作 skill
│   ├── wiki-ingest/SKILL.md
│   ├── wiki-query/SKILL.md
│   └── wiki-lint/SKILL.md
│
├── templates/              # 页面模板（entity/concept/topic/ingest-record）
├── scripts/                # Python 辅助（dedupe/link-graph/stale-check/icloud-lock）
├── runs/                   # 操作日志（ingest/query/lint）
└── .reference/             # 上游实现（只读参考）
    └── karpathy-llm-wiki-upstream/
```

---

## 四、日常使用

### 摄取一篇文章

```bash
# 方式 1：直接拖入文件
cp article.md raw/articles/

# 方式 2：在 Claude Code 中说
你：摄取 raw/articles/article.md
Claude：→ 调用 skills/wiki-ingest → 生成 wiki/ 页面
```

### 查询知识

```
你：什么是 LLM Wiki？
Claude：→ 调用 skills/wiki-query → 返回带 [[ref]] 的答案

你：LLM Wiki 能用在民宿 SOP 沉淀吗？
Claude：→ 查到现有知识，推断应用场景，询问是否归档
```

### 周度巡检

```
你：lint
Claude：→ 调用 skills/wiki-lint → 生成 runs/lint/<date>-report.md
        报告死链/孤岛/陈旧/冗余，可选 --auto-fix
```

### Obsidian 图谱

打开 Graph view → 看到四领域知识网络逐渐丰富。点击 `[[llm-wiki]]` 跳转。

---

## 五、四个知识领域

| 领域 | 对应目录 | 典型内容 |
|---|---|---|
| AI/LLM 研究与工程 | `wiki/topics/ai-llm/` | 论文、Gist、框架、Prompt 技巧、Skill 模式 |
| 民宿/新媒体/外贸业务 | `wiki/topics/business/` | SOP、定价、运营心得（不读其他项目内部） |
| 个人学习笔记 | `wiki/topics/learning/` | 书摘、播客、课程、思维模型 |
| 技术教程/API 参考 | `wiki/topics/tech-ref/` | 框架文档提炼、架构案例、API 速查 |

---

## 六、宪法要点（详见 `CLAUDE.md`）

- **raw/ 只读追加**：任何 skill 不得修改/删除 raw 文件
- **wiki/ 必须注明来源**：frontmatter `sources: [raw/...]`
- **命名规范**：kebab-case 英文/拼音，中文放 aliases
- **隐私边界**：raw/notes/private/ 不进 git，wiki 引用不超 50 字
- **项目隔离**：不读写 3body/新媒体发布系统/云图 Atlas 等相邻项目内部
- **质量门禁**：死链、孤岛 >30 天、缺 sources、单页 >800 行都是 CRITICAL

---

## 七、7 阶段闭环开发（搭建 skill 时遵守）

与用户全局 `~/.claude/CLAUDE.md` 兼容：

1. Research → 2. Plan → 3. TDD+Build → 4. Review（并行）→ 5. Accept → 6. Verify → 7. Learn

---

## 八、参考

- [Karpathy 原始 Gist](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f)（本项目理论基础）
- [Astro-Han/karpathy-llm-wiki](https://github.com/Astro-Han/karpathy-llm-wiki)（本项目骨架基准，已 clone 到 `.reference/`）
- [lucasastorian/llmwiki](https://github.com/lucasastorian/llmwiki)（Web + MCP 对照参考）
- [kytmanov/obsidian-llm-wiki-local](https://github.com/kytmanov/obsidian-llm-wiki-local)（Obsidian 本地离线方案）

---

## 九、路线图

- [x] 骨架初始化（raw/wiki/skills/templates/scripts/runs）
- [x] CLAUDE.md 宪法 + 三 SKILL.md
- [x] 四模板 + 四 Python 脚本
- [x] Obsidian 配置 + .claude 权限
- [ ] 冒烟测试（3 篇样例 raw → ingest → query → lint）
- [ ] 四域首批种子页（每域 2-3 页启动）
- [ ] 定时 lint（/schedule 周度触发）
- [ ] 30 天后第一次宪法回顾（`CLAUDE.md` v2）
