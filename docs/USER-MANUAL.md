# LLM Wiki 使用手册

> 基于 Karpathy LLM Wiki 理念，Claude Code + Obsidian 的个人知识复利系统。

---

## 目录

- [零. 系统构成速览](#零-系统构成速览)
- [一. 每次启动](#一-每次启动)
- [二. 三大操作](#二-三大操作)
- [三. 典型场景](#三-典型场景)
- [四. Obsidian 端配合](#四-obsidian-端配合)
- [五. 日常维护节奏](#五-日常维护节奏)
- [六. 故障排查 FAQ](#六-故障排查-faq)
- [七. 进阶玩法](#七-进阶玩法)
- [八. 命令速查卡](#八-命令速查卡)

---

## 零. 系统构成速览

```
raw/           # 原始资料（只追加，不改动）
  papers/      #   学术论文
  articles/    #   博客/新闻
  gists/       #   Gist 快照
  notes/
    public/    #   可共享笔记
    private/   #   私密（不进 git）
  web-clips/   #   网页剪藏
  transcripts/ #   音视频字幕

wiki/          # AI 编译的知识库
  entities/    #   人/工具/组织
  concepts/    #   抽象概念
  topics/
    ai-llm/    #   AI/LLM 领域
    business/  #   业务领域
    learning/  #   个人学习
    tech-ref/  #   技术参考
  _meta/       #   标签/术语/健康状态
  index.md     #   Dataview 聚合主页

skills/        # 四个操作 skill
  wiki-ingest/ #   摄取
  wiki-query/  #   查询
  wiki-lint/   #   巡检
  wiki-deploy-check/  # 七阶段自检
```

**三者关系**：
- **Raw 只追加** — AI 和人都不能改已有 raw 文件
- **Wiki 可增改** — AI（skill）维护，人偶尔编辑
- **CLAUDE.md 是宪法** — 定义规则，只有人能改

---

## 一. 每次启动

### 1. 打开 Claude Code

三选一（推荐第一个，最稳）：

**A. 桌面快捷方式（一次配置永久用）**

桌面空白处右键 → 新建 → 快捷方式，目标填：

```
cmd.exe /k "cd /d F:\Icloud\iCloudDrive\Anthropic\cc+obisidian学习复利机制 && claude"
```

名字叫 `LLM Wiki`。以后双击就自动进正确目录启动 Claude Code。

**B. 文件管理器右键**

打开 `F:\Icloud\iCloudDrive\Anthropic\cc+obisidian学习复利机制\` → 空白处右键 → **Open in Terminal** → 输入 `claude`

**C. 终端手动**

```bash
cd "F:/Icloud/iCloudDrive/Anthropic/cc+obisidian学习复利机制"
claude
```

### 2. 打开 Obsidian

左下角 vault 切换图标 → **Open another vault** → 选 LLM Wiki 目录

**验证对了**：左侧文件树只看到 `raw/ wiki/ templates/ skills/ runs/` 等，不应该有 `云图Atlas / 新媒体发布系统` 等其他项目文件夹。

### 3. 验证工作目录（首次启动或怀疑时）

在 Claude 里输入：

```
当前工作目录是什么
```

应显示 `.../cc+obisidian学习复利机制`。若不是，退出重启。

---

## 二. 三大操作

### 操作 1：Ingest（摄取）— 把资料变成知识

**触发词**：摄取、归档、入库、ingest、处理 raw、新文章

**三步走**：

```
Step 1 — 把资料放到 raw/ 对应子目录
  论文 → raw/papers/<YYYY-MM-DD>-标题.md（或 .pdf）
  博文 → raw/articles/
  Gist → raw/gists/
  笔记 → raw/notes/public/（或 private/）
  网页 → raw/web-clips/
  字幕 → raw/transcripts/

Step 2 — 在 Claude 里说
  "摄取所有新文件"
  或
  "ingest raw/articles/<文件名>.md"

Step 3 — AI 自动完成
  ✓ 抽取实体/概念
  ✓ 生成 wiki 页
  ✓ 建立双链
  ✓ 追加 _manifest.jsonl
  ✓ git commit
```

**文件命名规范**：

```
raw/articles/2026-04-17-karpathy-llm-wiki.md   ← 日期 + kebab-case
raw/papers/attention-is-all-you-need.md        ← 无日期可以
raw/notes/public/reading-hpmor-ch1-10.md
```

中文主题用拼音：

```
raw/notes/public/yuntu-minsu-sop.md
# frontmatter 加 aliases: [云图民宿 SOP]
```

### 操作 2：Query（查询）— 从知识库问答

**触发词**：什么是 / 解释 / 对比 / 如何 / 综述 / 问 wiki / 查

**四种问法**：

| 类型 | 示例 |
|---|---|
| 定义 | "什么是 LLM Wiki？" |
| 对比 | "对比 RAG 和 LLM Wiki" |
| 综述 | "我都知道哪些关于 prompt engineering 的知识？" |
| 应用 | "LLM Wiki 能用在民宿 SOP 沉淀上吗？" |

**输出格式**：

每个事实带 `[[wiki 页]]` 引用，例如：

> LLM Wiki 是 Andrej Karpathy [[andrej-karpathy]] 提出的持久化知识库理念 [[llm-wiki#定义]]，区别于传统 RAG [[rag]]...

**归档新知识**：

当问题答案涉及 wiki 中不存在的新知识，Claude 会问：

```
⚠️ 此答案包含 wiki 之外的新知识：
- 来源：[Claude 训练数据 | WebSearch]
- 主要增量：...
是否归档为新 wiki 页？(y/n)
```

回答 `y` → 自动创建 `wiki/topics/<领域>/<slug>.md`，这就是**知识复利的来源**。

### 操作 3：Lint（巡检）— 维护知识健康

**触发词**：巡检 / 体检 / lint / 健康检查

**常规扫描**：

```
"lint"
```

自动跑四阶段：
1. 死链扫描（`[[不存在页]]`）
2. 孤岛检测（> 30 天无链接）
3. 陈旧检测（> 90 天未勾选的 TODO、> 180 天未更新的页）
4. 冗余/隐私检测

**自动修复（安全项）**：

```
"lint --auto-fix"
```

只修复确定性项：死链重命名、缺失索引补全。其他仍输出人工确认列表。

**完整部署自检**（七阶段闭环）：

```
"部署检查"
```

跑 `wiki-deploy-check` skill，不止看 wiki 健康，还检查整个部署合规性（脚本/评审/测试/审计）。

---

## 三. 典型场景

### 场景 1：读完一篇论文想归档

```
1. 保存论文 PDF 或 md 摘要到 raw/papers/attention-is-all-you-need.md
   （PDF 大文件建议保留 PDF + 写一份精简 md 摘要）

2. 在 Claude 里说："ingest raw/papers/attention-is-all-you-need.md"

3. 看输出：
   - 新建 wiki/concepts/transformer.md
   - 新建 wiki/concepts/attention-mechanism.md
   - 新建 wiki/entities/ashish-vaswani.md
   - 更新 wiki/_meta/tags.md
   - 自动 git commit

4. 在 Obsidian 里打开 Graph view，看新节点如何和已有知识连接
```

### 场景 2：想查某个概念，发现 wiki 里还没有

```
问："什么是 Mamba 架构？"

Claude 答："当前 wiki 中无关于 Mamba 架构的专页。
           根据外部知识：Mamba 是 2024 年提出的状态空间模型..."
        → 问"是否归档？"

你答："y，归档到 ai-llm topic"

→ 创建 wiki/topics/ai-llm/mamba-architecture.md
  frontmatter: provenance: query-session, sources: []
  同时建立与 [[transformer]] 的对比链接

下次有人问"Mamba 和 Transformer 区别"时，wiki 里就有现成答案。
```

### 场景 3：跨领域联想

```
问："LLM Wiki 的知识积累模式能用在云图民宿 SOP 沉淀上吗？"

Claude 会：
  1. 召回 wiki/concepts/llm-wiki.md 和 wiki/concepts/knowledge-compounding.md
  2. 召回 wiki/topics/business/（如果已有 SOP 相关页）
  3. 综合类比：raw/ ↔ 历史 SOP 文档，wiki/ ↔ 精炼 SOP 手册
  4. 给出具体适配建议，附引用

通过归档，wiki 成为跨 AI/业务 两个领域的桥梁。
```

### 场景 4：周度维护

```
周日晚上在 Claude 里说：
"周度巡检"

跑完会给报告：
- 2 个死链（上周改名遗留）→ 一键 apply auto-fix
- 1 个孤岛页 → 建议手动建链或标记 archived
- 5 个陈旧 TODO（> 90 天）→ 看要不要清理
- 1 个冗余建议：A 页和 B 页相似度 0.87 → 决定是否合并

全部处理后 wiki/_meta/health.md 显示绿灯，git 有本周的清理提交。
```

---

## 四. Obsidian 端配合

### Graph View（图谱）

**打开**：左侧栏第一排图标 → 点地球图标

**怎么用**：
- 看孤岛：没和主图连接的点
- 看枢纽：连接多的节点（重要概念）
- 点击节点 → 跳转到该页
- 悬停 → 预览内容

**最佳实践**：每月看一次 Graph，识别该拆的大页和该连的孤岛。

### 搜索（Ctrl+Shift+F）

- 支持 frontmatter 过滤：`tag:#ai/llm/framework`
- 支持路径：`path:wiki/concepts/`
- 支持双链反向：`[[andrej-karpathy]]` → 找引用

### 模板快速新建页（Templater 插件）

Ctrl+P → `Templater: Create new note from template` → 选 entity / concept / topic

填充 `{{title}}` 后自动套模板结构。

### Dataview DQL

`wiki/index.md` 里已有示例。自定义 DQL：

````markdown
```dataview
TABLE sources, updated, tags
FROM "wiki/topics/ai-llm"
WHERE contains(tags, "ai/llm/framework")
SORT updated DESC
LIMIT 10
```
````

### Find Unlinked Files

Ctrl+P → `Find orphaned files and broken links: Find orphaned files`

生成孤岛列表供人工复核（lint skill 的双保险）。

### Broken Links Cleaner

Ctrl+P → `Broken Links Cleaner: Scan vault and generate report`

生成 `broken links output.md` 文件。看完删掉即可，不随 git 走。

---

## 五. 日常维护节奏

### 每次操作后

✅ Claude 自动 git commit，无需手动操作

### 每周一次（推荐周日）

```
"lint"  → 处理 CRITICAL / HIGH
"plink-auto-fix"  → 应用安全修复
```

耗时：5-10 分钟

### 每月一次

```
1. 打开 Obsidian Graph View 看拓扑
2. 查 wiki/_meta/health.md 看趋势
3. 处理累积的 MEDIUM / LOW 警告
4. 清理 broken links output.md 之类的扫描产物
```

耗时：30 分钟

### 每季一次

```
"部署检查"  # 跑 wiki-deploy-check
```

输出 `runs/lint/deploy-check-<date>.md`，对比上季，决定是否：
- 升级 CLAUDE.md（新增规则、调整门禁）
- 补齐 tests/（如果仍缺）
- 迁移陈旧页到 archived/
- 更新推荐插件栈

耗时：1 小时

### 30 天回顾 CLAUDE.md

CLAUDE.md 宪法每 30 天回顾一次（`runs/` 里有足够证据时才改规则，至少 3 次类似情况）。

---

## 六. 故障排查 FAQ

### Q1：Claude 说找不到 raw/ 或 wiki/

**原因**：工作目录错了（在主 vault 或其他项目里启动了 Claude）
**解决**：退出 Claude，用"每次启动"那节的方法重新启动

### Q2：摄取时说"权限被拒绝"

**原因**：尝试写 raw/ 下的只读文件
**解决**：raw 是不可变层。要改 raw → 先删除原文件再用新文件 ingest（manifest 会记录变化）

### Q3：Obsidian 搜不到新建的 wiki 页

**原因**：Obsidian 索引延迟
**解决**：Ctrl+P → `Reload app without saving` → 等 5 秒

### Q4：Broken Links Cleaner 扫出大量死链

**原因**：vault 选错，扫到别的项目
**解决**：检查 Obsidian 左下角 vault 名，确保在 cc-obsidian-llm-wiki 里

### Q5：git commit 失败 "pre-commit hook rejected"

**原因**：raw/notes/private/ 的文件被暂存
**解决**：`git reset HEAD raw/notes/private/` 然后只 add 其他文件

### Q6：iCloud 同步冲突（多设备）

**原因**：两台设备同时写 wiki 页
**解决**：先用 `python scripts/icloud-lock.py acquire . edit` 加锁再改

### Q7：BRAT 没自动更新 beta 插件

**原因**：GitHub API 限流
**解决**：GitHub → Settings → Tokens 生成 public_repo token → 粘贴到 BRAT 设置

### Q8：wiki 页太长（> 800 行）

**原因**：单一主题堆积太多
**解决**：lint 会报 CRITICAL → 按"子主题"段拆成多页，主页保留索引

### Q9：摄取后发现生成的 wiki 页质量差

**解决**：直接在 Obsidian 里编辑该页。下次摄取相关资料时，AI 会保留你的修改并追加新信息（不会覆盖）。

### Q10：想导出某个主题给别人看

```bash
# 打 tar 包只含 wiki/topics/ai-llm/
tar czf export.tar.gz wiki/topics/ai-llm/ wiki/entities/ wiki/concepts/
```

或直接 share GitHub 仓库链接。

---

## 七. 进阶玩法

### 1. 添加新 skill

```
1. skills/<new-skill>/SKILL.md 按现有三个 skill 格式写
2. 更新 CLAUDE.md 第六节触发词表
3. 写相应的 Python 脚本到 scripts/（可选）
4. 跑 "部署检查" 验证没问题
```

### 2. 扩展模板

`templates/` 下加新类型，例如 `templates/meeting-note.md`（会议记录），然后：

```yaml
---
name: meeting-note
type: meeting
attendees: []
action-items: []
---
```

在 Obsidian 里 Templater 会自动识别。

### 3. 定时任务

**Windows 任务计划**：每周日 23:00 跑：

```bash
claude -p "部署检查" --project "F:\Icloud\iCloudDrive\Anthropic\cc+obisidian学习复利机制"
```

或用 Claude Code 的 `/schedule` 功能（更优雅）。

### 4. 多设备同步

**推荐**：Git + iCloud 双通道
- Git：wiki 内容（权威）
- iCloud：临时同步、大文件（PDF）
- raw/notes/private：**只 iCloud，不 git**

**冲突解决**：
- 代码（wiki/md）冲突 → git merge 处理
- 插件配置冲突 → 以某一设备为准（BRAT 不同设备各自配置）

### 5. 分享 wiki 到公网

当前 GitHub 仓库已 public。想让别人直接浏览 wiki：
- **GitHub Pages**：Settings → Pages → Source: master → /docs
- **Obsidian Publish**（付费）：一键发布
- **quartz**：开源静态站点生成器

### 6. 与其他项目联动

**业务项目 → LLM Wiki**：云图 Atlas 的定价经验教训 → `wiki/topics/business/pricing-lessons.md`
- 通过 `wiki/` 的内部链接承载，**不直接读业务项目内部文件**（隔离原则）

**LLM Wiki → 业务项目**：业务项目的 CLAUDE.md 可引用 `[[path/to/llm-wiki/wiki/concepts/...]]`
- 单向只读引用，不修改 wiki

### 7. 对接 MCP

未来可加的扩展：
- `firecrawl` MCP → 从 URL 自动摄取到 raw/
- `exa` MCP → query 时打通外网检索
- 自建 MCP server → 把 wiki 暴露给其他 Claude 实例

---

## 八. 命令速查卡

在 Claude Code 里说：

| 想做什么 | 说什么 |
|---|---|
| 摄取一个文件 | `ingest raw/articles/<文件>.md` |
| 摄取所有新文件 | `摄取所有 raw 新文件` |
| 查询概念 | `什么是 <概念>？` |
| 对比 | `对比 A 和 B` |
| 综述 | `我都知道哪些关于 <主题> 的？` |
| 归档查询答案 | `查询 ... 并归档到 <领域>` |
| 巡检 | `lint` |
| 自动修复 | `lint --auto-fix` |
| 七阶段自检 | `部署检查` |
| 查看工作目录 | `pwd` |
| 看 git 历史 | `git log` |

Obsidian 命令面板（Ctrl+P）：

| 想做什么 | 命令 |
|---|---|
| 新建页套模板 | `Templater: Create new note from template` |
| 找孤岛 | `Find orphaned files and broken links: Find orphaned files` |
| 扫死链 | `Broken Links Cleaner: Scan vault and generate report` |
| 重载 app | `Reload app without saving` |
| 全文搜索 | `Search: Search in all files` |

Python 脚本（终端）：

```bash
python scripts/dedupe.py .         # 扫描 raw/ 去重
python scripts/link-graph.py .     # 死链 + 孤岛
python scripts/stale-check.py .    # 陈旧检测
python scripts/icloud-lock.py acquire . <op>  # 加锁
python scripts/icloud-lock.py release .       # 解锁
```

---

## 九. 学习路径建议

第一周：
- [ ] 摄取 5 篇你最近读过的文章（分在 raw/articles/ 和 raw/papers/）
- [ ] 查 3 个问题，体验"先查自建库"的心理转变
- [ ] 周末跑一次 lint

第一个月：
- [ ] wiki/entities/ ≥ 10 页（关键人物/工具）
- [ ] wiki/concepts/ ≥ 20 页（核心术语）
- [ ] wiki/topics/ 四个域各 ≥ 3 页
- [ ] Graph View 开始有形状

三个月：
- [ ] wiki 总页数 ≥ 100
- [ ] 每周 ingest 2-5 篇
- [ ] 至少一次业务决策参考了 wiki 内容（复利已生效）
- [ ] CLAUDE.md 迭代到 v2

---

## 十. 参考与支持

- **理论原点**：[Karpathy LLM Wiki Gist](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f)
- **本项目 GitHub**：https://github.com/uwater1981-droid/cc-obsidian-llm-wiki
- **宪法**：本仓库 `CLAUDE.md`
- **部署检查 skill**：`skills/wiki-deploy-check/SKILL.md`
- **上游参考**：`.reference/karpathy-llm-wiki-upstream/`（本地 clone）
- **Obsidian 插件文档**：
  - [Dataview](https://blacksmithgu.github.io/obsidian-dataview/)
  - [Templater](https://silentvoid13.github.io/Templater/)
  - [Tasks](https://publish.obsidian.md/tasks/)

---

**记住三条铁律**：

1. **Raw 只追加**（历史不可篡改）
2. **Wiki 靠 skill**（让 AI 做苦力，你只读和提问）
3. **CLAUDE.md 治理**（每 30 天回顾，不随便改规则）

开始你的知识复利之旅。
