<div align="center">

# Engram

**为 AI 编码 agent 设计的向量记忆系统 —— 语义化、低成本、海马体启发。**

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Status](https://img.shields.io/badge/status-v0.4%20beta-orange)](#路线图)
[![Platforms](https://img.shields.io/badge/platforms-Win%20x64%20%7C%20Linux%20x64%20%7C%20macOS%20arm64-lightgrey)](#安装)

[English](README.md) · [中文](README_CN.md) · [安装指南](INSTALL_CN.md) · [完整基准](eval/RESULTS.md)

</div>

---

Engram 是一个为 AI 编码 agent 设计的**向量数据库记忆后端**。**一份 `.pst` 存储**，四家 host 通用：**Claude Code**、**OpenCode**、**Codex**、**OpenClaw**。它原地替换老的 `MEMORY.md` / 单文件 markdown 记忆系统，用本地 HNSW 索引上的语义 KNN 搜索完成同一件事——并附赠模式分离、睡眠 replay、Ebbinghaus 衰减。

```bash
pip install git+https://github.com/shannonxu-2018/Engram.git
engram install --agent claude-code   # 或：opencode | codex | all
```

---

## 为什么用 Engram

- **更准的召回，更轻的上下文。** **Recall@3 = 100 %**（自适应 top-K），而老 MD 索引是 92 %，**token 成本相差 26 倍**。([基准](#基准))
- **一份存储，四家 agent。** 适配 **Claude Code**、**OpenCode**、**Codex**（这三家都是 skill **+** MCP），以及 **OpenClaw**（只装 skill——官方文档没覆盖 MCP）。个人偏好跨 agent 跟随，项目事实留在项目内。
- **海马体启发。** 模式分离拒绝近似重复写入，Ebbinghaus 衰减让长期不用的记忆淡出，importance × recency 复合重排把"当下重要"的内容顶上去，睡眠 replay 周期性整理库。
- **常驻、暖热、低延迟。** `engram-mcp` 服务进程把嵌入器常驻在内存里——召回延迟从冷启动脚本的 ~3 秒降到常驻进程的 ~10 毫秒。
- **本地优先。** 默认 `multilingual-e5-small`（384 维，多语言），首次下载 ~471 MB 之后完全离线。一个 env var 切到 OpenAI 或自建后端。
- **单 wheel、无 daemon。** `pip install engram` + 一条 CLI。PistaDB 原生库（Windows x64、Linux x64、macOS Apple Silicon）随 wheel 一起发。

## 基准

40 条合成记忆、25 条查询（10 简单 / 10 中等 / 5 困难），真 `multilingual-e5-small` 嵌入，`tiktoken cl100k_base` 计数，amortized 到 50 轮会话。

| 指标                              | MD 基线 (top-3) | Engram (k=3) | Δ |
|----------------------------------|-----------------|--------------|------|
| **Recall@3**                     | 92.0 %          | **100.0 %**  | +8.0 pp |
| **MRR**                          | 0.880           | **0.973**    | +0.093 |
| **每轮平均 token（amortized）**  | 1430.8          | **54.7**     | **便宜 26.1 倍** |
| 基线（常驻上下文）               | 1331 tokens     | **0 tokens** | — |
| 平均每查询 token                 | 199.6           | 109.4        | — |

> **双向收益**：Engram 既**更准**（三档难度全 100 % Recall@3），又**更省 token**（量级差）。没有常驻的 `MEMORY.md` 索引；recall 返回的是紧凑的"一行一条"。完整对照：[`eval/RESULTS.md`](eval/RESULTS.md)。

复现：

```bash
pip install "engram[eval]"
python -m eval.runner
```

## 支持的 agent

| Agent | 集成方式 | 指令文件 | 发现点 |
|-------|---------|---------|--------|
| **Claude Code** | 文件式 skill **+** MCP | `~/.claude/CLAUDE.md` | `~/.claude/skills/engram/` 和 `~/.claude.json` |
| **OpenCode**    | 文件式 skill **+** MCP | `~/.config/opencode/AGENTS.md`（XDG） | `~/.config/opencode/skills/engram/` 和 `opencode.json` |
| **Codex**       | 文件式 skill **+** MCP | `~/.codex/AGENTS.md` | `~/.agents/skills/engram/` 和 `~/.codex/config.toml` |
| **OpenClaw**    | 文件式 skill（官方未文档化 MCP）| `~/.openclaw/AGENTS.md` | `~/.openclaw/skills/engram/` |

> **跨生态发现。** OpenCode 和 OpenClaw 都会扫描多个 skill 目录（`~/.config/opencode/skills/` 或 `~/.openclaw/skills/`，加上 `~/.claude/skills/` 和 `~/.agents/skills/`）——所以你只要装过 `engram install --agent claude-code` 或 `--agent codex`，这两家 agent 都能自动发现 skill。专门 `--agent <name>` 安装只是为了给你一个干净的 per-agent 卸载点。

按需安装：

```bash
engram install --agent claude-code   # skill + MCP
engram install --agent opencode      # skill + MCP
engram install --agent codex         # skill + MCP
engram install --agent openclaw      # 只装 skill
engram install --agent all           # 全部一次到位
```

> **MCP 传输层：** `engram-mcp` 这个 console script 随包发布。即使没装官方 `mcp` Python SDK，它也会落回到内置的最小 stdio JSON-RPC 实现（覆盖 `initialize` / `tools/list` / `tools/call`），所有支持 MCP 的 host 都接受。需要 spec 完美帧时再 `pip install "engram[mcp]"`。OpenClaw 这家只装 skill 路径——它的 skills 文档里不覆盖 MCP，我们也不擅自猜测格式。

## 安装

任意支持的 agent，两条命令：

```bash
pip install git+https://github.com/shannonxu-2018/Engram.git
engram install --agent claude-code
```

开发者——clone + bootstrap：

```bash
git clone https://github.com/shannonxu-2018/Engram.git && cd Engram
./bootstrap.ps1     # Windows
./bootstrap.sh      # Linux / macOS
```

> 完整选项、支持的 OS 矩阵、env vars、故障排查和卸载见 [`INSTALL_CN.md`](INSTALL_CN.md)。

## 工作原理

### 四种记忆类型

类型不是自由文本——模型每次保存都对照这份清单分类。与老 MD 策略的四元组完全一致，所以**既有的"应不应该存"启发法可以直接平移**。

| 类型 | 捕获的内容 | 触发例子 |
|------|-----------|---------|
| `user`      | 用户身份 / 背景 / 偏好               | *"我是做数据科学的，主要看日志"* |
| `feedback`  | 纠正**或**确认了非显然的做法         | *"别 mock 数据库"* / *"对，单 PR 打包就是对的"* |
| `project`   | 决策的"为什么"、deadline、动机     | *"auth 重写是合规驱动的，不是技术债清理"* |
| `reference` | 指向外部系统 / 文档 / 看板的指针    | *"管线 bug 在 Linear INGEST 项目"* |

### 两级存储

按 type 自动路由。个人偏好跨工程跟随用户，项目事实留在项目内。

| 层级 | 路径 | 容纳的 type | 作用域 |
|------|------|-------------|--------|
| **global** | `~/.claude/engram/global.{pst,pcc}` | `user`、`feedback` | 跨工程跟随用户 |
| **local**  | `<project>/.claude/engram/local.{pst,pcc}` | `project`、`reference` | 项目内隔离 |

> **local 层应该加进项目的 `.gitignore`**——向量库不是源代码。

### 记忆策略（何时存什么）

Engram 是**被动存储**，不是自动入库器。本仓库里没有任何代码会监听对话、灌入转录或在会话结尾压缩成记忆。`.pst` 里出现一行**只可能是因为模型主动调用了 `save`**，遵循下面这套策略。

| 信号 | 类型 | 例子 |
|------|------|------|
| 用户透露身份 / 背景 / 偏好                   | `user`      | *"我是做数据科学的，主要看日志"* |
| 用户纠正了某个做法，**或**确认了非显然的做法 | `feedback`  | *"别 mock 数据库"* / *"对，单 PR 打包就是对的"* |
| 项目背景、动机、deadline、决策的"为什么"   | `project`   | *"auth 重写是合规驱动的"* |
| 指向外部系统的指针                           | `reference` | *"管线 bug 都在 Linear INGEST 项目"* |

**绝对不存**（即使用户明示要求也要劝阻）：代码模式、文件路径、git 历史、调试方案、临时任务状态、`CLAUDE.md` / `AGENTS.md` 里已经写过的内容。这些从代码里现读比存起来便宜。

**`importance` 不决定是否写入，而是决定如何排序。** `save` 可传 `--importance 0.7`（默认 `0.5`）。召回时 composite = `distance − β·importance_eff − γ·log1p(hits)`，重要或常用的记忆会越过等距但陈旧的对手。每次 `expand` 都会 `hits++` 并施加 boost。配合 30 天 Ebbinghaus 衰减，注意力自动聚焦到真正有用的部分——但永远不会自动删数据。

**写入时去重抑制。** 插入前 `save` 会做一次 KNN 检查；如果存在近似项（cosine < 0.08）就返回 `MERGE_SUGGESTION` 而不写入。要么 `--update` 覆盖旧条目，要么 `--force` 强插，要么跳过。

用户显式覆盖：

- *"记一下，……"* → 强制立即保存
- *"忘掉……"* → `forget --name <slug>`

### 海马体启发的机制

| 大脑机制 | Engram 的对应实现 |
|---------|------------------|
| **模式分离** | `save()` 做一次 KNN 检查，若 `dist < 0.08` 拒绝写入近似副本，提示合并 |
| **模式补全** | `recall()` 用查询做 KNN——从部分线索中找回完整记忆 |
| **痕迹强化** | 每次召回/展开都 `hits++`、刷新 `accessed_at`，并对 importance 做 boost |
| **衰减** | importance 按 Ebbinghaus 衰减：`eff = base · exp(-Δt/τ)`，τ 默认 30 天 |
| **睡眠 replay** | `consolidate` 周期性合并近似项 + 重新应用衰减 |
| **激活引起的重排** | 召回 composite = `distance − β·importance_eff − γ·log1p(hits)` —— 经常被访问的重要记忆越过等距但陈旧的对手 |

## 工程级控制

agent 在某个工程里**是否真的用** Engram，取决于**两个信号**：

1. **可见性**——这个 agent 是否注册了 skill / MCP server？
2. **指令**——工程级（或全局）指令文件是否告诉它去用？

<details>
<summary><b>在工程里启用 Engram</b></summary>

**(a) 单工程（最常见）。** 在工程的指令文件里加：

```markdown
# Memory

Use the Engram skill at `~/.claude/skills/engram/` for all memory
operations in this project. Follow the protocol in its SKILL.md.

The local tier writes to `.claude/engram/local.pst` — keep that in
.gitignore.
```

然后把 `.claude/engram/` 加进 `.gitignore`。

**(b) 全局——所有未来工程自动启用。** 一次性追加到全局指令文件：

```bash
engram install --agent claude-code --print-instructions-snippet \
    >> ~/.claude/CLAUDE.md
```

**(c) 绑进工程（团队共享）。** 把 skill 装到工程内部，所有 clone 的人都能用：

```bash
cd <project>
engram install --agent claude-code --target ./.claude/skills/engram
```

队友仍需要 `pip install engram` 来解决 Python 依赖。

</details>

<details>
<summary><b>在工程里禁用 Engram</b></summary>

三档力度，从轻到重：

**(a) 什么都不做。** 没加全局 snippet 的情况下，agent 只在用户显式触发时用（*"记一下……"*）——这本身就是软 opt-out。

**(b) 明文告诉 agent 不要用。** 工程级指令文件：

```markdown
# Memory

This project does NOT use the engram skill. Ignore any global
instruction to use it. Fall back to default memory behavior.
```

工程级文件优先级高于全局。

**(c) 彻底屏蔽脚本调用。** 工程 `.claude/settings.local.json`：

```json
{
  "permissions": {
    "deny": [
      "Bash(*engram*)",
      "Bash(*~/.claude/skills/engram*)"
    ]
  }
}
```

harness 在 agent 尝试之前就拦住——最硬核。

</details>

<details>
<summary><b>用户级 vs 工程级 skill 优先级</b></summary>

如果 `<project>/.claude/skills/engram/` 和 `~/.claude/skills/engram/` **两个都存在**，**工程级生效**（Claude Code 先读工程 skill）。所以"工程级覆盖用户级默认"随时可做。

| 目标 | 装在哪 |
|------|--------|
| "任何地方都能用 Engram" | 用户级（`engram install` 的默认）|
| "团队要通过 git 共享固定版本" | 工程级（`--target ./.claude/skills/engram`，**不**带 `--dev`）|
| "这个项目要定制 SKILL.md" | 工程级覆盖 |

</details>

## 架构

```
┌──────────────────────────────────────────────────────────────────┐
│   Claude Code  /  OpenCode  /  Codex  /  OpenClaw                 │
│   （指令文件告诉 agent 去用 Engram）                                 │
└──────────────────────────────────────────────────────────────────┘
        │                                            ▲
        │ skill 脚本 / MCP 工具调用                    │ 紧凑文本回流
        ▼                                            │
┌───────────────────────────────────────────────────────┴──────────┐
│   ~/.claude/skills/engram/scripts/*.py    （skill 类 agent）        │
│   engram-mcp  （长驻 stdio MCP server，支持 MCP 的 agent 共用）      │
└──────────────────────────────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────────────────────────┐
│   engram （Python 包，在 site-packages）                            │
│   embedder.py   ─►  4 个嵌入后端（默认 LocalE5、OpenAI、HTTP、         │
│                                    Hash 仅测试）                      │
│   store.py      ─►  Schema + 两级路径路由                            │
│   memory.py     ─►  MemoryManager.save / recall / expand / forget   │
│   decay.py      ─►  importance_eff、boost_on_access、rerank          │
│   consolidate.py ►  睡眠 replay（merge_pass + decay_pass）            │
│   agents.py     ─►  AgentProfile 注册表（v0.4）                       │
│   install.py    ─►  多 agent 安装器（v0.4）                           │
│   mcp_server.py ─►  engram-mcp 入口（v0.4）                           │
└──────────────────────────────────────────────────────────────────┘
        │                                  │
        ▼                                  ▼
┌─────────────────────────────┐   ┌─────────────────────────────┐
│ pistadb（内嵌）             │   │ HuggingFace 缓存            │
│   ctypes 封装 + bundled     │   │ ~/.cache/huggingface/       │
│   .dll/.so/.dylib           │   │ multilingual-e5-small (384) │
│   HNSW + COSINE             │   │ （首次使用按需下载）         │
└─────────────────────────────┘   └─────────────────────────────┘
```

<details>
<summary><b>Schema</b>（Milvus 风格，定义在 <code>engram/store.py</code>）</summary>

```python
[
  FieldSchema("mem_id",      INT64,        is_primary=True, auto_id=True),
  FieldSchema("type",        VARCHAR, max_length=20),
  FieldSchema("name",        VARCHAR, max_length=128),
  FieldSchema("description", VARCHAR, max_length=512),    # 要旨（recall 返回的就是它）
  FieldSchema("content",     VARCHAR, max_length=8192),   # 完整正文（通过 expand 懒取）
  FieldSchema("tags",        JSON),                       # [[link]] / 关键词
  FieldSchema("importance",  FLOAT),
  FieldSchema("hits",        INT64),
  FieldSchema("created_at",  INT64),
  FieldSchema("accessed_at", INT64),
  FieldSchema("vector",      FLOAT_VECTOR, dim=384),
]
# Metric.COSINE + Index.HNSW
```

</details>

<details>
<summary><b>嵌入器后端（7 个内置 + 可扩展）</b></summary>

`Embedder` 是 `Protocol`，任何供应商都能接入。用 `ENGRAM_EMBEDDER` 的
**URI spec** 切换：

```bash
ENGRAM_EMBEDDER='openai:text-embedding-3-large?dim=2048'
ENGRAM_EMBEDDER='ollama:nomic-embed-text'                       # 本地，免 API key
ENGRAM_EMBEDDER='cohere:embed-multilingual-v3.0'
ENGRAM_EMBEDDER='voyage:voyage-3'
ENGRAM_EMBEDDER='http://localhost:8080/embeddings?dim=1024'     # 任意 OpenAI 兼容端点
ENGRAM_EMBEDDER='hash?dim=384'                                  # 仅测试
```

URI 形式：`scheme:target?key=val&key=val` —— 通用 HTTP 后端也可以直接
贴 URL。空 / 未设 = 默认本地 e5。

| Scheme | 后端类 | 必填 env / 参数 | 默认 dim |
|--------|-------|----------------|----------|
| `local` *(默认)*  | `LocalE5Embedder` | （无 — 可选 `?device=cuda` / `?model_path=…`）          | 384 |
| `openai`         | `OpenAIEmbedder`  | `OPENAI_API_KEY`；`?dim=…&base_url=…` 可选              | 1536 / 3072 |
| `ollama`         | `OllamaEmbedder`  | 本地 Ollama；`?host=http://...&dim=…` 可选              | 768 |
| `cohere`         | `CohereEmbedder`  | `COHERE_API_KEY`                                          | 1024 |
| `voyage`         | `VoyageEmbedder`  | `VOYAGE_API_KEY`                                          | 1024 |
| `http`           | `HTTPEmbedder`    | URL + `?dim=N`                                            | （你给）|
| `hash`           | `HashEmbedder`    | （无 — 可选 `?dim=…`）                                 | 384 |

向量维度在 `.pst` 首次创建时锁定——一个存储用一个后端，要换就**删 tier
文件重建**。启动时维度不匹配会给出三条修复路径的清晰提示。

**第三方后端**接入很干净：

```python
from engram.embedder import register_embedder, ParsedSpec

def _bedrock_factory(spec: ParsedSpec):
    return MyBedrockEmbedder(
        model=spec.target,
        region=spec.params.get("region", "us-east-1"),
    )

register_embedder("bedrock", _bedrock_factory)
# 然后任何地方都可以:  ENGRAM_EMBEDDER='bedrock:cohere.embed-multilingual-v3?region=us-west-2'
```

> v0.3 的散 env var 还能用——`ENGRAM_EMBEDDER=openai` + `ENGRAM_OPENAI_MODEL=…`
> 以及 `ENGRAM_EMBEDDER=http` + `ENGRAM_HTTP_URL=…` + `ENGRAM_HTTP_DIM=…`
> 在加载时被自动改写成新的 URI 形式。

</details>

## CLI 速查

跑 `engram install` 之后，脚本都在 `~/.claude/skills/engram/scripts/` 下。OpenCode 和 Codex 通过 MCP 工具调同样的操作（`engram_recall` / `engram_save` / …）。

| 脚本 | 用途 |
|------|------|
| `recall.py <查询> [--k N] [--type T] [--tier T] [--no-rerank] [--json]` | 语义检索。`--k` 默认 **adaptive**（距离断点检测，2-10 条）|
| `save.py <type> <name> <desc> [<content>] [--tag X] [--importance F] [--force\|--update]` | 写入记忆；自动去重 |
| `expand.py <id> [--json]` | 取一条记忆的完整正文（顺手 `hits++`） |
| `list.py [--type T] [--tier T] [--limit N] [--json]` | 不花 embedding 的全量枚举 |
| `forget.py (--name <slug> \| --id <n> \| --older-than DAYS) [--type T] [--tier T] [--dry-run]` | 按名字、id 或年龄删除。`--older-than` 比的是 `accessed_at`。`--dry-run` 只预览 |
| `patch.py <id-or-name> [--description X] [--content X] [--importance F] [--type T] [--rename X] [--add-tag X] [--remove-tag X] [--set-tag X]` | 修改单条记忆的部分字段。只改 metadata ⇒ id 不变；改 desc/content/tier ⇒ id 变 |
| `consolidate.py [--dry-run] [--no-merge\|--no-decay] [--threshold F]` | 睡眠 replay（合并 + 衰减） |
| `related.py <name-or-id> [--depth N] [--with-content]` | 沿 `[[name]]` 标签 BFS |
| `migrate_md.py <source-dir> [--dry-run] [--force]` | 从老 MD 系统一次性导入 |

**调用时机**（agent 的启发式）：

- **召回**——用户提到之前的对话、问起关于自己的事、或要给出的建议依赖于他们的偏好/反馈时。
- **保存**——学到关于用户、反馈、项目或外部参考的**长期**事实时。
- **展开**——单看 description 不够时才用。
- **整理**——每周一次或一次大批量 save 之后。

<details>
<summary><b>详细示例</b></summary>

#### `recall` —— 语义检索

```bash
$ recall.py "用户的偏好"
# adaptive k=2
    9 | l | reference | disable-pattern | d=0.082 | 在某工程禁用 engram 的三档方法 ...
    1 | g | user      | lang-chinese    | d=0.103 | 用户偏好用简体中文沟通 ...
```

列：`id | tier(g/l) | type | name | distance | description`。distance 越小越相关。

**自适应 k**（默认）：脚本拉一个宽裕的候选池，按距离排序，在第一个明显的距离跳跃处切开（边界 K_MIN=2、K_MAX=10）。需要固定 K 时传 `--k 5`。

#### `save` —— 写入记忆

```bash
$ save.py feedback no-mock-db \
    "integration tests must hit a real database, not mocks" \
    "Why: 之前一次 mock/prod 行为分歧掩盖了 migration bug. \
     How to apply: tests/integration/** 下任何 DB 调用都不要 mock。"

inserted: id=11 tier=global
```

近似项（cosine < 0.08）存在时脚本拒绝写入并打印 `MERGE_SUGGESTION`——可以 `--update` 覆盖、`--force` 强插或跳过。

#### `expand` —— 取完整正文 + 强化痕迹

```bash
$ expand.py 9
id   : 9
tier : local
type : reference
name : disable-pattern
hits : 3
---
Why: 用户级安装的 skill 会出现在所有工程，需要 per-project opt-out.
How to apply: 看 README §"工程级控制" 那一节.
```

副作用：`hits++`、`accessed_at = now`。省着用。

#### `related` —— 沿 `[[name]]` 图边遍历

存 `compliance-deadline` 时带了 `--tag "[[auth-rewrite-driver]]"`：

```bash
$ related.py auth-rewrite-driver --depth 1
   12 | l | project | compliance-deadline | d=0.000 | auth rewrite 必须在 2026-04-01 前 ship ...
```

纯元数据图遍历——零 embedding 成本。

</details>

## Python API

```python
from engram import MemoryManager

with MemoryManager() as mgr:
    mgr.save(
        type="feedback",
        name="terse-replies",
        description="用户希望回复简洁，不要末尾总结",
        content="Why: 审美偏好. How to apply: 不再在每轮末尾来一段 recap.",
        tags=["style", "[[role-data-sci]]"],
    )

    hits = mgr.recall("我该怎么收尾这次回复？", k=5)
    for h in hits:
        print(h.to_compact())

    neighbours = mgr.recall_related("terse-replies", depth=1)

# 维护
from engram import consolidate
with MemoryManager() as mgr:
    report = consolidate(mgr, dry_run=True)
    print(report.to_dict())
```

## 配置

所有旋钮都是环境变量，每次调用现读：

| 变量 | 默认 | 含义 |
|------|------|------|
| `ENGRAM_EMBEDDER` | `""`（=`local`）| **URI spec** 选嵌入器后端。例：`openai:text-embedding-3-small`、`ollama:nomic-embed-text`、`cohere:embed-multilingual-v3.0`、`voyage:voyage-3`、`http://host:port/embed?dim=N`、`hash?dim=384`。见上面"嵌入器后端"一节。|
| `ENGRAM_E5_MODEL_PATH` | — | 本地 e5 快照绝对路径（离线场景覆盖）|
| `OPENAI_API_KEY` | — | `openai:…` spec 必填 |
| `COHERE_API_KEY` | — | `cohere:…` spec 必填 |
| `VOYAGE_API_KEY` | — | `voyage:…` spec 必填 |
| `ENGRAM_OPENAI_MODEL` | — | v0.3 兼容：当 `ENGRAM_EMBEDDER=openai`（光秃秃）时此项指定模型，等价于新形式 `openai:<model>`。|
| `ENGRAM_HTTP_URL` / `ENGRAM_HTTP_DIM` | — | v0.3 兼容：`ENGRAM_EMBEDDER=http` 时用，等价于 `http://URL?dim=DIM`。|
| `ENGRAM_DECAY_TAU_DAYS` | `30` | Ebbinghaus 衰减 τ（天）|
| `ENGRAM_RANK_BETA` | `0.10` | 重排公式 `importance_eff` 权重 |
| `ENGRAM_RANK_GAMMA` | `0.02` | 重排公式 `log1p(hits)` 权重 |
| `ENGRAM_HOME` | `~/.claude` | global 层根目录 *(v0.4，优先级高于 `CLAUDE_HOME`)* |
| `CLAUDE_HOME` | `~/.claude` | `ENGRAM_HOME` 的 v0.3 兼容别名 |
| `OPENCODE_HOME` | XDG / `%APPDATA%\opencode` | OpenCode 配置目录 |
| `CODEX_HOME` | `~/.codex` | Codex 配置目录（`AGENTS.md`、`config.toml`）|
| `OPENCLAW_HOME` | `~/.openclaw` | OpenClaw 配置 & skill 目录 |
| `AGENTS_SKILLS_HOME` | `~/.agents/skills` | agent 无关的 `.agents/skills` 约定（Codex 用）的用户级 skill 目录 |
| `PISTADB_LIB_PATH` | — | PistaDB 原生库的绝对路径覆盖 |

## 工程结构

```
engram/                                  (仓库根)
├── README.md / README_CN.md             项目介绍
├── INSTALL.md / INSTALL_CN.md           部署 / 配置 / 故障排查
├── CLAUDE.md                            本仓库的 dev 约定
├── pyproject.toml                       pip 可装，src 布局
├── bootstrap.ps1 / bootstrap.sh         dev 一键安装
├── src/
│   ├── engram/                          本包
│   │   ├── embedder.py / store.py / memory.py
│   │   ├── decay.py / consolidate.py    (v2 机制)
│   │   ├── agents.py / install.py       (v0.4 多 agent)
│   │   ├── mcp_server.py                (v0.4 `engram-mcp`)
│   │   ├── cli.py / install_skill.py    (`engram` CLI + v0.3 别名)
│   │   └── _skill_files/                作为 package data 打包
│   │       ├── SKILL.md
│   │       └── scripts/{recall,save,expand,list,forget,related,
│   │                    consolidate,migrate_md,_common}.py
│   └── pistadb/                         内嵌向量数据库
│       └── pistadb.dll / libpistadb.so / libpistadb.dylib
├── tests/
│   ├── smoke_test.py                    端到端，HashEmbedder
│   ├── unit_test.py                     v0.3 review 修复
│   └── agents_test.py                   v0.4 多 agent 层
└── eval/                                对照 MD 基线的基准
    └── RESULTS.md                       (生成)
```

## Built on PistaDB

Engram 的向量存储用的是 **[PistaDB](https://github.com/shannonxu-2018/PistaDB)** —— *"the embedded vector database for LLM-native applications"*（面向 LLM 原生应用的嵌入式向量数据库）。它提供 HNSW + cosine 索引、单 `.pst` 文件格式（可选 `.wal` 做崩溃恢复），核心是零外部依赖的 C99 实现。Engram 在 `src/pistadb/` 内嵌了 PistaDB 以及 Windows x64、Linux x64、macOS Apple Silicon 的预编译原生库——所以 `pip install engram` 就够了，**不需要**额外起服务，也**不需要**单独下载任何库。

为什么选 PistaDB ：

- **单文件、嵌入式。** 一个 tier 一个 `.pst`——global 存储就是 `~/.claude/engram/` 下的一个文件，local 存储就是工程目录下的一个文件。备份、同步、`.gitignore` 都是 1 行的事。
- **零外部依赖。** 没 daemon、没 Docker、没托管服务。C99 核心编译成一个小巧的 `.dll`/`.so`/`.dylib`，Engram 用 `ctypes` 直接加载。
- **索引可选。** HNSW（RAG 场景推荐）加另外 7 种索引，5 种距离度量——Engram 选了 `HNSW + COSINE`，但 schema 可以按你的工作负载切换。
- **多语言。** PistaDB 有 Python、Go、C++、Java、Swift、Rust、C# 的封装——所以 Engram 写出的 `.pst` 文件你随时可以从其他语言读出来。

想读或贡献向量库本身？看上游仓库：<https://github.com/shannonxu-2018/PistaDB>。

## 路线图

| 版本 | 状态 | 主要内容 |
|------|------|---------|
| **v0.3** | 已发布 | 单 agent（Claude Code skill）、四种记忆类型、两级存储、海马体机制（模式分离、衰减、睡眠 replay、importance 重排）|
| **v0.4** | 已发布 | **多 agent**：同一份存储被 Claude Code、OpenCode、Codex（三家都是文件式 skill + MCP）以及 OpenClaw（只装文件式 skill——官方未文档化 MCP）共享。OpenCode 和 OpenClaw 额外支持跨生态发现（`~/.claude/skills/`、`~/.agents/skills/`）。新增 `engram-mcp` server，以及 `ENGRAM_HOME` / `AGENTS_SKILLS_HOME` / `OPENCLAW_HOME` env vars。所有 v0.3 CLI 保留向后兼容 |
| **v0.5** | 规划 | 跨嵌入器迁移（换后端时重新嵌入）。主动 `replay.py`——LLM 给热门记忆簇做摘要再回写。每条记忆的 ACL（私有 vs 共享）|
| **v0.6** | 规划 | 跨工程知识图谱（跨 tier 的 `[[link]]` 解析）。浏览存储的 Web UI。Anthropic memory API 稳定后做对接 |

<details>
<summary><b>已知限制</b>（当前版本）</summary>

- **并发写入下 `save()` 去重只是 best-effort。** PistaDB 单 `.pst` 单写；`MemoryManager.save()` 里"检查 + 写入"两步不是原子的。Claude Code 今天"一个会话一个 store"模式下没事，等 server 模式落地再处理。
- **几个管理操作是 O(n) 全 tier 扫描。** `list.py`、recall 触发的 `_bump_access`、`recall_related` 的图遍历都会重写或重读整份 sidecar JSON。1000 条以内顺滑。
- **Windows 下 `--dev` 需要开开发者模式**（或 admin）才能 symlink。installer 故意不静默退化为拷贝。
- **macOS Intel (x86_64) 不支持。** wheel 里只带 Apple Silicon 的 `libpistadb.dylib`。

</details>

## License

MIT —— 见 [`LICENSE`](LICENSE)。
