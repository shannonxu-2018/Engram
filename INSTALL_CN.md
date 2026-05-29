<div align="center">

# 安装指南

Engram 在 **Claude Code**、**OpenCode**、**Codex** 上的安装、配置、故障排查。

[English](INSTALL.md) · [中文](INSTALL_CN.md) · [项目说明](README_CN.md)

</div>

---

## TL;DR

```bash
pip install git+https://github.com/shannonxu-2018/Engram.git
engram install --agent claude-code   # 或：opencode | codex | openclaw | all
```

大多数用户到这就够了。`--agent` 决定集成方式，下面都是参考资料。

v0.3 的 `engram install-skill` 仍然可用，等价于 `engram install --agent claude-code --no-mcp`（只装文件式 skill）。

## 目录

1. [系统要求](#1-系统要求)
2. [安装包](#2-安装包)
3. [为某个 agent 安装](#3-为某个-agent-安装)
4. [（可选）全局启用](#4可选全局启用)
5. [验证](#5-验证)
6. [数据存放位置](#6-数据存放位置)
7. [配置（环境变量）](#7-配置环境变量)
8. [卸载](#8-卸载)
9. [开发者（在 repo 内工作）](#9-开发者在-repo-内工作)
10. [故障排查](#10-故障排查)

---

## 1. 系统要求

- **Python** ≥ 3.10
- **网络**：首次使用 `recall` / `save` 时需要联网，自动从 HuggingFace 下载
  `intfloat/multilingual-e5-small`（约 471 MB），之后完全离线
- **操作系统 / 架构**：

  | 平台 | 提供的二进制 | 支持 |
  |------|------------|------|
  | Windows x64           | `pistadb.dll`        | ✅ |
  | Linux x64             | `libpistadb.so`      | ✅ |
  | macOS Apple Silicon (arm64) | `libpistadb.dylib` | ✅ |
  | macOS Intel (x86_64)  | —                    | ❌ 不支持 |
  | Linux aarch64 / 其他  | —                    | ❌ 不支持 |

  其他平台想跑的话，从上游
  [shannonxu-2018/PistaDB](https://github.com/shannonxu-2018/PistaDB)
  自行构建对应二进制，并设 `PISTADB_LIB_PATH=/abs/path/to/lib`。

---

## 2. 安装包

私有 git 仓库，按你环境选 ssh 或 https：

```bash
# HTTPS（不需要 SSH key 也能用）
pip install git+https://github.com/shannonxu-2018/Engram.git

# SSH（如果你配了 GitHub SSH key，推荐用这个）
pip install git+ssh://git@github.com/shannonxu-2018/Engram.git

# 指定版本/分支/tag
pip install "engram @ git+https://github.com/shannonxu-2018/Engram.git@v0.4.0"
```

### 可选嵌入后端

按需加 extras：

```bash
pip install "engram[local]"   # sentence-transformers，本地 e5（默认推荐）
pip install "engram[openai]"  # OpenAI text-embedding-3-small/large
pip install "engram[eval]"    # tiktoken，跑评估时需要
pip install "engram[mcp]"     # 官方 MCP SDK，给 engram-mcp 跑标准帧
pip install "engram[all]"     # 以上全部
```

不装 extras 也能 import，但只能用内置的 `hash` embedder（仅用于 smoke 测试，
无语义）。

> `mcp` 这个 extra **可选**。`engram-mcp` 自带一个最小的 stdio JSON-RPC 实现，
> Claude Code / OpenCode / Codex 这三家走 MCP 路径的 agent 都能用。只有
> 当你需要严格符合 MCP spec 的帧（例如对接其他 host）时再装官方 SDK。
> OpenClaw 走 skill 路径，不用 MCP，所以这个 extra 对它没意义。

---

## 3. 为某个 agent 安装

```bash
# Claude Code：文件式 skill + MCP server 注册
engram install --agent claude-code

# OpenCode：MCP server 注册到 opencode.json
engram install --agent opencode

# Codex：MCP server 注册到 ~/.codex/config.toml
engram install --agent codex

# OpenClaw：只装文件式 skill（官方未文档化 MCP）
engram install --agent openclaw

# 四家一起装
engram install --agent all
```

`engram install --agent claude-code` 做两件事：

1. 创建 `~/.claude/skills/engram/`，把 pip 包里的 `SKILL.md` 和
   `scripts/*.py` 拷过去（文件式 skill 通道——和 v0.3 的
   `install-skill` 行为一致）。
2. 在 `~/.claude.json` 的 `mcpServers.engram` 下注册 `engram-mcp` 命令
   （MCP 通道——v0.4 新增）。

`engram install --agent codex` 用 Codex 自己的约定做同样两件事：

1. 在 `~/.agents/skills/engram/` 安装 skill（按 OpenAI 的 Codex skills 文档
   —— Codex 会扫描 `$HOME/.agents/skills/` 下所有带 `SKILL.md` 的目录）。
2. 在 `~/.codex/config.toml` 的 `[mcp_servers.engram]` 下注册 `engram-mcp`。

`engram install --agent opencode` 是对称的：

1. 在 `~/.config/opencode/skills/engram/` 安装 skill（Windows 上是
   `%APPDATA%\opencode\skills\engram\`）。OpenCode 还会扫描
   `~/.claude/skills/` 和 `~/.agents/skills/`——所以你只要装过 Claude Code
   或 Codex，OpenCode 就能自动发现那两份。专门 `--agent opencode` 安装
   只是为了给一个干净的 per-agent 卸载点。
2. 在 `~/.config/opencode/opencode.json` 的 `mcpServers.engram` 下
   注册 `engram-mcp`。

`engram install --agent openclaw` 是**只装 skill**（依据
[`docs.openclaw.ai/tools/skills`](https://docs.openclaw.ai/tools/skills)）：

1. 在 `~/.openclaw/skills/engram/` 安装 skill。和 OpenCode 一样，
   OpenClaw 还会扫描 `~/.claude/skills/` 和 `~/.agents/skills/`，所以
   之前装过的 skill 也能跨生态被发现。
2. **不做 MCP 注册。** OpenClaw 的 skills 文档没说明 MCP server 配置
   位置——我们不猜。如果你要手工接 MCP，把 `engram-mcp`（包自带的可执行
   入口）挂上去即可。

任一半都可以用 `--no-skill` / `--no-mcp` 单独关掉（OpenClaw 这种只装
skill 的 agent 用 `--no-mcp` 是 no-op）。

可选参数：

| 参数 | 作用 |
|------|------|
| `--agent NAME` | 目标 agent：`claude-code`（默认）/ `opencode` / `codex` / `openclaw` / `all`。|
| `--dev` | 用符号链接代替拷贝（Win 需开发者模式）。只对支持文件式 skill 的 agent 有意义。|
| `--target DIR` | 覆盖 skill 目标目录。|
| `--force` | 覆盖已存在的安装。|
| `--remove` | 卸载该 agent 的安装（文件 skill + MCP 注册都清掉）。|
| `--check` | 只检查不写入，输出当前状态。|
| `--no-skill` | 只装 MCP，跳过文件 skill。|
| `--no-mcp` | 只装文件 skill，跳过 MCP 注册（v0.3 的行为）。|
| `--print-instructions-snippet` | 打印对应 agent 的指令片段（追加到 `CLAUDE.md` / `AGENTS.md` 用）。|
| `--snippet-kind {skill,mcp}` | 覆盖指令片段的描述风格。默认 = agent 的主集成（Claude Code 和 Codex 是 `skill`，OpenCode 是 `mcp`）。|

v0.3 的旧命令保留：

```bash
engram install-skill          # 等价于 `install --agent claude-code --no-mcp`
engram install-skill --check  # 等价于 `install --agent claude-code --check`
```

查看支持的 agent 和它们的配置位置：

```bash
engram agents
```

---

## 4.（可选）全局启用

默认情况下，agent 只有在工程级指令文件（`CLAUDE.md` / `AGENTS.md`）提到
Engram 时才会主动用它。想让**所有**新会话都默认走 Engram，把对应的
per-agent snippet 追加到全局指令文件：

```bash
# Claude Code
engram install --agent claude-code --print-instructions-snippet \
    >> ~/.claude/CLAUDE.md

# OpenCode
engram install --agent opencode --print-instructions-snippet \
    >> ~/.config/opencode/AGENTS.md

# Codex
engram install --agent codex --print-instructions-snippet \
    >> ~/.codex/AGENTS.md

# OpenClaw
engram install --agent openclaw --print-instructions-snippet \
    >> ~/.openclaw/AGENTS.md
```

`--print-global-snippet` 保留为 `--print-instructions-snippet` 的废弃别名。

snippet 内容是按 agent 定制的：Claude Code 版本指向 skill 脚本路径，
OpenCode / Codex 版本描述 MCP 工具（`engram_recall` / `engram_save` …）。

---

## 5. 验证

```bash
# 包能 import 吗？
python -c "import engram; print(engram.__version__)"

# CLI 都在 PATH 上吗？
engram version
engram-mcp --help

# 每个 agent 单独检查
engram install --agent claude-code --check
engram install --agent opencode    --check
engram install --agent codex       --check
engram install --agent openclaw    --check

# 真跑一次 list（应该是空的，或显示你之前的数据）
python ~/.claude/skills/engram/scripts/list.py
```

上面这些都顺利就算部署完成。

---

## 6. 数据存放位置

| 层 | 路径 | 容纳的 type |
|----|------|------------|
| **global** | `~/.claude/engram/global.{pst,pcc}` | `user`, `feedback`（跨工程跟随用户）|
| **local**  | `<project>/.claude/engram/local.{pst,pcc}` | `project`, `reference`（项目隔离）|

第一次 save 时自动创建。**强烈建议**把 `.claude/engram/` 加进项目的
`.gitignore`，避免向量库被 commit。

---

## 7. 配置（环境变量）

| 变量 | 默认 | 含义 |
|------|------|------|
| `ENGRAM_EMBEDDER` | `""`（=`local`）| **URI spec** 选嵌入器后端。详见下面的"嵌入器 spec"一节 |
| `OPENAI_API_KEY` | — | `openai:…` 必填 |
| `COHERE_API_KEY` | — | `cohere:…` 必填 |
| `VOYAGE_API_KEY` | — | `voyage:…` 必填 |
| `ENGRAM_OPENAI_MODEL` | — | v0.3 兼容：光秃秃的 `openai` spec 的模型名 |
| `ENGRAM_HTTP_URL` / `ENGRAM_HTTP_DIM` | — | v0.3 兼容：光秃秃的 `http` spec 的 endpoint + dim |
| `ENGRAM_DECAY_TAU_DAYS` | `30` | Ebbinghaus 衰减时间常数（天） |
| `ENGRAM_RANK_BETA` | `0.10` | 重排公式 importance 权重 |
| `ENGRAM_RANK_GAMMA` | `0.02` | 重排公式 hits 权重 |
| `ENGRAM_HOME` | `~/.claude` | global 层根目录（v0.4 新增，优先级高于 `CLAUDE_HOME`）|
| `CLAUDE_HOME` | `~/.claude` | `ENGRAM_HOME` 的 v0.3 兼容别名 |
| `OPENCODE_HOME` | XDG / `%APPDATA%\opencode` | OpenCode 配置目录（`AGENTS.md` + `opencode.json` 都在这里）|
| `CODEX_HOME` | `~/.codex` | Codex 配置目录（`AGENTS.md` + `config.toml` 都在这里）|
| `OPENCLAW_HOME` | `~/.openclaw` | OpenClaw 配置 & skill 目录 |
| `AGENTS_SKILLS_HOME` | `~/.agents/skills` | `.agents/skills` 约定的用户级 skill 目录（Codex 用）|
| `PISTADB_LIB_PATH` | — | PistaDB 原生库的绝对路径覆盖 |

---

## 7b. 嵌入器 spec（`ENGRAM_EMBEDDER`）

一个 env var 选后端。语法：`scheme:target?key=val&key=val`，通用 HTTP
后端也可以直接贴 URL。

| Spec 示例 | 后端 | 必填 env |
|----------|------|---------|
| *(未设)* / `local` | 本地 `multilingual-e5-small`（默认）| — |
| `local?device=cuda&model_path=/abs/path` | 本地，可覆盖设备 / 模型路径 | — |
| `openai:text-embedding-3-small` | OpenAI / OpenAI 兼容 | `OPENAI_API_KEY` |
| `openai:text-embedding-3-large?dim=2048` | OpenAI 显式 dim | `OPENAI_API_KEY` |
| `ollama:nomic-embed-text` | 本地 Ollama @ `localhost:11434` | — |
| `ollama:bge-m3?host=http://my-box:11434&dim=1024` | 远程 Ollama | — |
| `cohere:embed-multilingual-v3.0` | Cohere | `COHERE_API_KEY` |
| `voyage:voyage-3` | Voyage AI | `VOYAGE_API_KEY` |
| `http://localhost:8080/embeddings?dim=1024` | 任何 OpenAI 兼容端点 | — |
| `hash?dim=384` | 确定性 hash（仅测试） | — |

**每个 `.pst` 存储只挑一次后端**——向量维度在首次创建时锁定。换后端
但维度不匹配时启动会失败，提示三档修复路径（换回原后端 / 删 tier 文件 /
全部 re-embed）。

**第三方后端**接入很干净：

```python
from engram.embedder import register_embedder, ParsedSpec

def _my_factory(spec: ParsedSpec):
    return MyEmbedder(model=spec.target, **spec.params)

register_embedder("my-backend", _my_factory)
# 现在 ENGRAM_EMBEDDER='my-backend:foo?param=bar' 就能用了
```

> v0.3 的旧 env var（`ENGRAM_OPENAI_MODEL` / `ENGRAM_HTTP_URL` /
> `ENGRAM_HTTP_DIM`）仍然有效——会在加载时自动改写成新的 URI 形式。

---

## 8. 卸载

```bash
# 单独卸载某个 agent（同时清理 skill 文件和 MCP 注册）：
engram install --agent claude-code --remove
engram install --agent opencode    --remove
engram install --agent codex       --remove
engram install --agent openclaw    --remove

# 或者一次性全卸：
engram install --agent all --remove

pip uninstall engram

# 想连数据一起干掉（可选）
rm -rf ~/.claude/engram/             # 全局向量库
# 各工程的本地向量库需要逐个删：
rm -rf <project>/.claude/engram/
```

v0.3 的 `engram install-skill --remove` 仍然能用（只清 Claude Code skill）。

---

## 9. 开发者（在 repo 内工作）

```bash
git clone https://github.com/shannonxu-2018/Engram.git
cd Engram
pip install -e ".[all]"                # editable + 全部 extras

# dev 模式：symlink 而非拷贝，改源码即时生效
engram install-skill --dev

# 跑 smoke 测试（无外部依赖）
ENGRAM_EMBEDDER=hash python tests/smoke_test.py

# 跑评估
python -m eval.runner
```

或者直接用脚本一把梭：

```powershell
./bootstrap.ps1                         # Windows：pip install -e . + skill --dev
```

```bash
./bootstrap.sh                          # Linux/macOS 等价物（待补）
```

---

## 10. 故障排查

### `OSError: PistaDB shared library not found`

你的 OS/架构不在内置二进制里。两个选项：

1. 从上游
   [shannonxu-2018/PistaDB](https://github.com/shannonxu-2018/PistaDB)
   构建/获取对应平台的二进制，设 `PISTADB_LIB_PATH=/abs/path/to/lib`
2. 把二进制丢到 site-packages 的 `pistadb/` 目录下，让自动搜索找到

### `ImportError: sentence_transformers`

默认 embedder 需要这个包。修：

```bash
pip install "engram[local]"
```

或者换后端：

```bash
export ENGRAM_EMBEDDER=openai
export OPENAI_API_KEY=sk-...
```

### 首次跑非常慢，并且看不到下载进度

第一次实例化 `LocalE5Embedder` 时 sentence-transformers 去 HuggingFace 拉
e5-small（~471 MB），下载进度打到 stderr。可以预热：

```bash
python -c "from sentence_transformers import SentenceTransformer; \
           SentenceTransformer('intfloat/multilingual-e5-small')"
```

之后所有调用走本地缓存（`~/.cache/huggingface/`）。

### Claude 找不到 skill

确认：

```bash
ls ~/.claude/skills/engram/SKILL.md  # 文件存在吗？
```

不存在就重新跑 `engram install --agent claude-code`。如果跑了还是不存在，
加 `--check` 看报告。

### OpenClaw 看不到 engram skill

```bash
ls ~/.openclaw/skills/engram/SKILL.md   # 文件存在吗？
```

不存在就重新跑 `engram install --agent openclaw`。如果存在但 OpenClaw
依然没显示——按官方 skills 文档，OpenClaw 在会话开始时 snapshot 一次
可用的 skill 列表，所以可能需要重启会话。

### OpenCode / Codex 看不到 `engram` MCP server

检查注册表文件：

```bash
# OpenCode
cat ~/.config/opencode/opencode.json   # Windows 上是 %APPDATA%\opencode\opencode.json

# Codex
cat ~/.codex/config.toml
```

应该看到 `engram` 在 `mcpServers` 下面（OpenCode）或者
`[mcp_servers.engram]`（Codex）。没看到就重新跑
`engram install --agent <name>`。看到了但 agent 还是不调，那就需要把对应
的指令片段也写进 `AGENTS.md`（见第 4 节）——host 需要**两个信号**：注册
表 + prompt 才会真的用 Engram。

### `engram-mcp: command not found`

`engram-mcp` 是包提供的 console script，`pip install` 应该已经放进 PATH。
确认：

```bash
which engram-mcp                       # POSIX
where engram-mcp                       # Windows
python -m engram.mcp_server --help     # 兜底
```

如果只有 `python -m …` 能跑，说明 shell PATH 里没有 venv 的 scripts 目录。
要么用 `engram-mcp` 的绝对路径重装，要么在启动 agent 之前先激活 venv。

### 想在公司内网用，但没法访问 HuggingFace

两个出路：

1. 在能上网的机器先预热（见上），把 `~/.cache/huggingface/hub/` 拷到目标机
2. 改用 `ENGRAM_EMBEDDER=openai` 或 `http`（指向公司内部 embedding 服务）

---

## 相关文档

| 文件 | 读者 | 用途 |
|------|------|------|
| [`README.md`](README.md) / [`README_CN.md`](README_CN.md) | 终端用户 | Engram 是什么、技术设计、基准测试 |
| [`INSTALL.md`](INSTALL.md) / `INSTALL_CN.md`（本文） | 部署/运维 | 安装、配置、故障排查、卸载 |
| [`CLAUDE.md`](CLAUDE.md) | 贡献者 | 本 repo 的开发约定 |
| `SKILL.md`（pip 包里） | AI agent | 通用的 skill 协议，由 Claude 在任何工程中读取 |
