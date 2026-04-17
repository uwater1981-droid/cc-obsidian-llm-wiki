# BRAT 一键补装缺失插件指南

> **为什么需要 BRAT**：`Find Unlinked Files` 和 `Broken Links Cleaner` 在部分地区 Obsidian 社区市场可能搜不到。BRAT（Beta Reviewers Auto-update Tool）可以从 GitHub 直接安装插件，**无需出现在官方市场**。

## 一、已预配置的内容

本 vault 已经预置：

1. **`.obsidian/community-plugins.json`** 列出了 `obsidian42-brat`（BRAT 的插件 ID）
2. **`.obsidian/plugins/obsidian42-brat/data.json`** 预填了两个 beta 插件的 GitHub 地址：
   - `Vinzent03/find-unlinked-files`
   - `sarwarkaiser/obsidian-broken-links-cleaner`

只要你装上 BRAT，它会自动读取 data.json 并补装这两个。

## 二、三步操作

### Step 1：安装 BRAT

1. 打开 Obsidian → Settings（设置）→ Community plugins（社区插件）→ Browse（浏览）
2. 搜索 **`BRAT`**（全大写 4 个字母）
3. 作者是 **TfTHacker**，完整名字 `Obsidian42 - BRAT`
4. 点 **Install** → **Enable**

### Step 2：让 BRAT 读取预配置

1. Settings → Community plugins → 找到 **BRAT** → 点齿轮图标打开设置
2. BRAT 会看到 `data.json` 中已有两条 `pluginList` 记录
3. 找到 "Check for updates at startup" 选项，确保已开启（默认开启）
4. **重启 Obsidian**（或点 BRAT 顶部 "Check for updates to all beta plugins and themes"）

### Step 3：BRAT 自动下载并启用

BRAT 会：
1. 连接 GitHub Releases
2. 下载 `Vinzent03/find-unlinked-files` 的最新 release（main.js + manifest.json）
3. 解压到 `.obsidian/plugins/find-unlinked-files/`
4. 同样处理 `broken-links-cleaner`
5. 因为 `enableAfterInstall: true`，**装完自动启用**

装完后 Settings → Community plugins 列表里就能看到这俩已启用。

## 三、验证

```bash
ls .obsidian/plugins/
# 预期看到：
# obsidian42-brat/
# find-unlinked-files/
# obsidian-broken-links-cleaner/
# dataview/
# templater-obsidian/
# obsidian-tasks-plugin/
```

## 四、如果下载失败

常见原因是 GitHub API 速率限制或访问问题：

1. **加 Personal Access Token**（提升速率）：
   - GitHub → Settings → Developer settings → Personal access tokens → Generate（只需 public_repo 读权限）
   - 复制 token，粘贴到 BRAT 设置的 `Personal Access Token` 字段

2. **手动触发**：BRAT 设置页底部有 "Check for updates to all beta plugins" 按钮，点它重试

3. **实在不行**：从 GitHub Release 下载 zip，手动解压到 `.obsidian/plugins/<plugin-name>/` 下

## 五、以后还有其他 beta 插件

BRAT 界面里点 "Add beta plugin" → 粘贴 `作者/仓库名` → 回车。BRAT 会自动记录到 data.json 并下载。

---

## 六、可选：完全跳过这俩插件

本 vault 的 `wiki-lint` skill + `scripts/link-graph.py` 已经实现了：
- 死链检测（Phase 1）
- 孤岛检测（Phase 2）

Obsidian 内置的 **Graph View**（左侧栏工具栏的图标）肉眼也能看出孤立节点。

所以这俩插件只是**双保险**，不装也能用。装了 BRAT 之后它们是锦上添花。
