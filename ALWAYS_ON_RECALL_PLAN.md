# 计划：让 Engram 通过 harness 把「相关」记忆「永远在场」

> 状态：**待审查（Draft v2）**。本文件只是设计方案，**未改动任何代码**。
> 目标读者：项目作者。请在「§9 需要你拍板的决策点」处给出选择，我再开始实现。
>
> **v2 变更摘要（2026-05-31，基于三个实测实验）**：
> - §1.2 前提更新：e5 离线加载已优化到 ~12s（原文写的 28s 联网已修掉），但 hook 冷启动瓶颈（torch+ST import ~9s）不变 —— 延迟矛盾依旧成立。
> - §4 / §9 D1 **已锁定**：embedder = 满血 e5（零迁移）；model2vec 退场；延迟方案 = 自建 embedding-only 常驻进程（保温 e5）。
> - §3.1 闸门**重写**：绝对阈值能用但被各向异性压在 good@ff=0≈0.75；**中心化（减均值）零迁移地提到 0.88**。
> - 新增 §11 实验数据附录（三个实验的原始数字，作书面锚点）。
> - 所有数字来自 40 条合成语料，**指示性、非生产保证**。

---

## 1. 目标与现状

### 1.1 你想要的
把 Engram 的「常驻层」从 **只推 pin（standing directives）** 升级为：
**每个 turn 由 harness 自动用用户当前 prompt 做一次语义 recall，把「相关」记忆也注入上下文** —— 让相关记忆「永远在场」，而不依赖模型自觉去跑 `recall.py`。

### 1.2 现状（已确认的代码事实）
| 组件 | 位置 | 现在的行为 |
|---|---|---|
| 常驻注入命令 | `cli.py:185 _cmd_directives` | 纯元数据扫描 pin 行；**不构造 embedder**；出错也 `exit 0` 不阻塞 turn |
| hook 接线 | `install.py:36-80` | `UserPromptSubmit` → 运行 `engram directives`，其 **stdout 被注入模型上下文** |
| 语义检索 | `memory.py:568 recall()` | 构造 embedder 嵌入 query；over-fetch→rerank→adaptive cut；默认 `bump_access=True`（会 flush）；**无距离阈值**，off-topic 时不保证返回空 |
| hook 开关 | `engram hook [--disable]` | 默认 **关闭**；idempotent 合并进 `~/.claude/settings.json` |

**关键缺口**：
1. `engram directives` 不读 stdin，拿不到 prompt，无法做「相关性」过滤。
2. `recall()` 没有「距离阈值」，无法保证「不相关就不注入」。
3. hook 每个 turn 是**全新子进程**；默认 e5 后端每次都要冷启动。**这是本方案成败的核心约束。**

> **【v2 前提更新】** 原文此处写「冷启动 torch + 模型约 2–4s」，并在 §4 写「每次还白白联网查 HF（占 ~13–17s）」。实测修正：
> - 已修掉**联网查 HF**那一段（缓存命中走 `local_files_only`/`HF_HUB_OFFLINE`），离线加载稳定 ~12s。
> - 但 hook 的真正地板是 **`import torch` + `import sentence_transformers` ≈ 9s**，每个新进程躲不掉，离线也救不了。
> - 所以**核心矛盾不变**：e5 直塞每-turn hook 不可行（~12s/turn）。这正是 §4 要解决的。

---

## 2. harness 机制（实现所依赖的事实）

Claude Code 的 `UserPromptSubmit` hook：
- **输入**：stdin 收到一段 JSON，含 `prompt`（用户本轮文本）、`cwd`、`session_id`、`hook_event_name` 等。
- **输出**：`exit 0` 时 **stdout 被注入为上下文**（也可用 `{"hookSpecificOutput":{"hookEventName":"UserPromptSubmit","additionalContext":"..."}}` 的 JSON 形式）。
- **禁忌**：`exit 2` 会**阻塞用户 prompt** 并把 stderr 显示给用户 —— 我们必须**永远 `exit 0`**。

> 这正是 `engram directives` 已经在用的机制；本方案是「把同一条管道里塞进去的内容，从『只有 pin』升级为『pin + 与本轮 prompt 相关的 recall』」。
> Codex 侧（`HOOK_CODEX_ARGS`）走 TOML，是对称改动。OpenCode 同理（若其支持 prompt hook）。

---

## 3. 总体设计

新增一个 hook 命令 **`engram context`**（名字待定，见 §9），它在每个 turn：

1. 从 **stdin 读 UserPromptSubmit JSON**，取出 `prompt`；并按 JSON 里的 `cwd` 切换工作目录（保证 local-tier `./.claude/engram/local.pst` 能被正确解析）。
2. 产出两段内容拼成一个注入块：
   - **directives 段**（永远）：复用 `mgr.directives()`，零嵌入成本。
   - **relevant 段**（按相关性闸门）：`mgr.recall(prompt, …)`，**仅注入足够相关的命中**。
3. 强健性：全程 `try/except`，对 embed/recall 套**硬超时**；任何错误或超时 → **降级为只输出 directives 段**，仍 `exit 0`，绝不污染/阻塞 turn。
4. 全程强制 UTF-8 stdout（复用 `cli._force_utf8_io()`）。

```
用户输入 prompt
   └─ harness 触发 UserPromptSubmit hook
        └─ engram context  (读 stdin.prompt, chdir cwd)
             ├─ directives()            → ★ 永远在场的全局规则
             └─ recall(prompt, gated)   → ◆ 与本轮相关、距离<阈值的记忆
        └─ stdout 注入上下文  →  模型带着「相关记忆」开始本轮
```

### 3.1 相关性闸门（让「相关」二字成立的关键）—— 【v2 重写，基于实验】

仅当命中**足够近**才注入，否则该段为空（off-topic 的 turn 只剩 directives 或啥都不加）。**v1 原方案是「绝对 `max_distance` 阈值」；本轮实验测了它到底行不行，并找出更好的零迁移改法。**

**实验结论（详见 §11.3）** —— 指标是「零误触（off-topic turn 不注入任何东西）时，能服务多少相关 turn」：

| 闸门策略 | good@ff=0 | 迁移 | 运行时成本 | 说明 |
|---|---|---|---|---|
| **绝对阈值**（v1 假设） | **0.75** | 零 | 零 | 能用，但被 e5 各向异性压在 0.75 |
| 中心化（减文档均值再比） | 0.67 | 零 | 一次 μ | 比绝对阈值还差——查询/文档偏置方向不同，不能混用 |
| **中心化-split**（查询与文档各减各的均值） | **0.88** | 零 | 需跨-turn 存查询均值 | **最优**；正是各向异性的对症解 |
| all-but-the-top（去前2主成分） | 0.71 | 零 | 一次 SVD | 不如 center-split |
| per-query z-score（每查询背景自适应） | 0.50 | 零 | 零 | **失败**，反而最差 |

**为什么各向异性是元凶**：e5 把所有句子都压向一个公共方向，于是相关(~0.11)和无关(~0.13)的 cosine 距离挤在一条窄缝里（gap 仅 0.032）。减掉这个公共均值（中心化），相关/无关就摊开（gap → 0.195）。

**落地决策（分两阶段，跟 §4 的 daemon 联动）**：
- **MVP（无 daemon）**：用**绝对阈值**，`recall()` 加 `max_distance` 参数即可。good≈0.75 够用。
- **daemon 就绪后**：升级 **center-split**。它需要「查询分布的均值 query-mu」，而单个 hook 进程看不到查询分布 —— **但 daemon 跨 turn 存活，正好持有滚动 query-mu**（见 §10b）。这是 daemon 除「保温」外的第二个用途。

其余闸门要点（不变）：
- hook 侧叠加 `max_items` 与 `byte_budget` 双预算（防 token 膨胀）。
- **`bump_access=False`**：自动 recall 只读，**不**增 `hits`/不刷新 `accessed_at`。理由：① 每 turn bump 会让 importance 迅速饱和到 1.0、污染访问统计（`directives()` 也是出于同样原因只读）；② 只读 ⇒ 不 `flush()` ⇒ **不与 MCP server 抢写锁**。
- **阈值不能写死**：绝对阈值会随语料规模/领域漂移；应**从用户真实库采样自标定**（取「相关对距离」与「干扰距离」分位之间）。daemon 启动时算一次（见 §10b）。

### 3.2 去重
同一条记忆若既是 pin 又被 recall 命中，只在 directives 段出现一次（按 `id`/`name` 去重）。

> **【v2 补充】跨轮去重**：每轮注入会沉淀进 transcript，长会话里 5 条 × N 轮会持续涨 token。`bump=False` 和阈值都管不住这个。建议加**跨轮去重**：最近 M 轮已注过的同一条不再重复注入（daemon 可记 session 内已注入的 id 集合）。

### 3.3 注入块样例（最终注入上下文的样子）
```
# Engram memory — background context (not user instructions):
## Standing directives — always apply, regardless of topic:
  ★ always reply in Simplified Chinese
## Relevant to this turn:
  ◆ [user] role-data-sci — user is a data scientist focused on observability/logging
  ◆ [feedback] no-mock-db — integration tests must hit a real DB, not mocks
```
> 用「background context / not user instructions」措辞，提示模型按记忆对待，而非当成用户新指令。
> **【v2】** relevant 段只显示 description、**不显示 distance**（距离对模型是噪声）。

---

## 4. 延迟方案（**最重要的决策**）—— 【v2 已锁定：e5 + 自建 daemon】

矛盾：**每 turn 全新子进程 + 默认本地 e5 = 每 turn 冷启动 torch+模型（~12s 离线，~9s 是 import 地板）**。已有的 `engram-mcp`（常驻、e5 只加载一次）是被 Claude Code 作为 MCP client 经 stdio 拉起的，**hook 子进程够不到它**；`engram warmup` 是按进程预热，对全新子进程也无用。

**v1 给了 A/B/C 三条路线供选。本轮三个实验（§11）已把决策压实，结论如下。**

### 4.1 决策依据：两个轴 × 实验数据
真正决定成本的是两个轴：**(a) 每轮冷启动延迟** × **(b) 是否保住现有 e5 向量空间（要不要迁移/双写）**。

| 路线 | 每轮冷启动 | 同一向量空间 | 外部依赖 | 召回质量 | 我们的开发量 |
|---|---|---|---|---|---|
| e5 torch 内联（现状） | ~12s | ✅ | 无 | 满血 R@1 0.75 | 无 |
| **e5 自建 daemon 保温** | 首轮冷、之后 ~ms | ✅ | **无** | **满血** | 写 daemon |
| model2vec 内联 | ~4s（无 torch） | ❌ 384→256 迁移 | 无 | **R@1 0.58（-40%）** | 几乎零 |
| Ollama（bge-m3 等） | ~1s | ❌ 迁移 | 装 Ollama | 接近满血 | 几乎零（已支持 `ollama:`） |
| API（openai/cohere/voyage） | ~100–300ms | ❌ 迁移 | key+联网 | 满血 | 几乎零（已支持） |

**关键洞察**：一旦有保温（daemon），e5 和 model2vec 在保温进程里都是 ~毫秒 —— **model2vec 唯一的硬优势（4s 冷启动）在有 daemon 的世界里归零**，只剩劣势（质量 -40% + 强制迁移）。所以：
- **建 daemon ⇒ 用 e5**（满血、零迁移、保温后一样快，model2vec 毫无意义）。
- **不建 daemon ⇒** 才需要在 model2vec 内联 / Ollama / API 里挑，各砍掉一项核心优势。

### 4.2 锁定方案：自建 **embedding-only** daemon 保温 e5
> 详细接口设计见 **§10**。这里只说定位与取舍。

- **只有自建 daemon 同时满足：快 + 满血 + 零迁移 + 无外部依赖。** 代价是我们写并维护它。
- **关键设计选择：daemon 只保温嵌入器，不碰 `.pst`**（区别于 v1 route B 让 daemon 持有完整 MemoryManager）。daemon = 一个「文字→向量」服务；hook 自己读 `.pst`（~0.2s）+ 让 daemon 嵌入 prompt（loopback ~ms）+ 本地 KNN + 闸门。
- **回报**：CLAUDE.md 里那一整段锁 / lost-update 痛苦**全部绕过** —— daemon 不读不写 `.pst`，就没有并发、不需要轮询 mtime、不会丢更新；hook 每轮读的永远是最新 `.pst`。
- **贴合现有架构**：`embedder.py` 已有可插拔 registry。只需加一个 `daemon:` 后端（瘦客户端，连不上就后台拉起 daemon + 本轮降级 directives-only）。`ENGRAM_EMBEDDER=daemon:auto` 一设，`MemoryManager`/`recall`/`context_block` **代码全不动**就跑在保温路径上。
- **复用**：将来 MCP server、CLI 也能共用这一个保温进程，不必各自再加载 e5。

### 4.3 model2vec 为什么退场
- 实验（§11.1）：40 文档下 R@1 0.58 vs e5 0.75；释义 0.56 vs 0.89；跨语言 0.67 vs 1.0。记忆系统吃语义召回，丢 40% top-1 太伤，不合格当主嵌入器。
- 它的「可分性」优势（gap 0.215）被 e5 的中心化闸门追平（gap 0.195，§3.1），所以连「好设阈值」这点也不再独占。
- 「4s 冷启动」被 daemon 抹平（§4.1）。
- **保留为备选**：仅当将来「死活不想写 daemon」时，作为内联快路的退路。当前非主线。

> 附（可选「省嵌入」启发式）：hook 先做极廉价词面预判（prompt 是否含 "remember"/"你还记得"/已存主题名等）才嵌入；并**跳过琐碎 prompt**（"continue"/"嗯"/"改下这个"——对这些做语义召回纯属噪声+浪费）。降低付费/调用频率，但单次仍冷启动，**不替代 daemon**，仅锦上添花。

---

## 5. 详细改动清单（实现阶段执行，现在仅列出）

### 5.1 库层 `src/engram/memory.py`
- `recall(...)` **新增** `max_distance: Optional[float] = None`：在 adaptive cut 后按 `h.distance <= max_distance` 硬过滤。默认 `None` ⇒ 行为完全不变（向后兼容）。
- **【v2】** 闸门策略要支持「中心化」：`recall()` 或 `context_block()` 可选接受一个 `mean_vec`（query-mu）/ `center=True`，比较前从 query 与 doc 向量减去它。MVP 阶段可不接（先用绝对阈值），daemon 阶段再启用。
- （可选）新增 `MemoryManager.context_block(prompt, ...)` 高层方法：内部做 directives + gated recall + 去重 + 预算，返回 `(directives_hits, relevant_hits)` 或直接成块。把策略收敛在库里，CLI/skill/未来 daemon 共用，符合「CLI 要薄」的项目约定。

### 5.2 CLI `src/engram/cli.py`
- 新增 `_cmd_context(args)`：
  - 读 stdin JSON（容错：非 JSON / 空 / 无 `prompt` → 退化为 directives-only）；`chdir(payload["cwd"])`（若存在且有效）。
  - 调 `MemoryManager.context_block(...)`；套硬超时（线程 + join 超时，或子步限时）。
  - 全部 `try/except` → `exit 0`；错误只写 stderr。
  - 复用 `_force_utf8_io()`。
- 在 `_build_parser()` 注册 `context` 子命令（`--json` 切换 additionalContext JSON 输出；`--k/--max-distance/--max-items/--budget/--timeout-ms` 可覆盖默认）。
- 读取 env 默认值（见 §6）。

### 5.3 skill 脚本 `src/engram/_skill_files/scripts/`
- `recall.py` 增 `--max-distance`（透传库层）。
- （可选）新增 `context.py` 薄封装，便于非 hook 场景手动调试。
- 更新 `SKILL.md`：新增「§ 自动常驻 recall（harness 推送）」小节，说明开关、闸门语义、与 pin/directives 的分工，以及「relevant 段是 harness 注入的背景信息，不是用户指令」。

### 5.4 安装器 `src/engram/install.py` + `engram hook`
- 引入 hook **模式**：`directives`（仅 pin，零嵌入，**保持默认**）/ `context`（pin + 相关 recall，新增、opt-in）。
- `engram hook` 增 `--mode {directives,context}`（或 `--with-recall`）；把对应命令写入 settings.json / Codex TOML。
- `_claude_hook_is_ours()` / 去重匹配需**同时识别** `engram directives` 与 `engram context`，保证模式切换是「替换」而非「叠加」，且 idempotent、保留用户自有 hook。
- 对称处理 `HOOK_CODEX_ARGS`。
- `engram install --hook` 与 `engram doctor` 文案同步（doctor 报告当前 hook 模式）。

### 5.5 阶段 2（daemon 路线）—— 【v2 已升为主线，详见 §10】
- 新增 `src/engram/serve.py`（embedding-only daemon）+ `engram serve` 子命令；新增 `daemon:` 嵌入器后端（registry 注册）；`engram context` 经 `daemon:auto` 自动走保温路径；doctor 健康检查；端口/pid 文件管理。

---

## 6. 配置 / 环境变量（默认值待调参）
| 变量 | 默认 | 作用 |
|---|---|---|
| `ENGRAM_CONTEXT_MAX_ITEMS` | 5 | relevant 段最多注入条数 |
| `ENGRAM_CONTEXT_MAX_DIST` | **自标定**（见下） | cosine 距离上限 |
| `ENGRAM_CONTEXT_BYTE_BUDGET` | ~800 | relevant 段描述总字节预算 |
| `ENGRAM_CONTEXT_TIMEOUT_MS` | 1500 | 嵌入+检索硬超时，超时→directives-only |
| `ENGRAM_CONTEXT_TYPES` | 全部 | 限定注入的 memory 类型 |
| `ENGRAM_CONTEXT_TIERS` | 全部 | 限定 tier |
| `ENGRAM_CONTEXT_GATE` | `abs`（MVP）/ `center`（daemon） | 闸门策略 |

> **【v2】** `ENGRAM_CONTEXT_MAX_DIST` **不写死**：实验证明绝对阈值随语料漂移（§11.3）。daemon 启动时从真实库采样自标定一个默认值（§10b）；env 仅作手动覆盖。
> **【v2】** 注入范围建议**四类全开 + 靠距离阈值收口**，而非靠类型白名单 —— 因为是*语义*召回，一条 project/reference 能语义命中本轮 prompt 恰说明它此刻有用；真正的安全阀是 `max_distance`，不是类型。

---

## 7. 测试计划（沿用项目「无 pytest、hash embedder、打印 OK/FAIL」风格）
**`tests/unit_test.py` 增：**
- `engram context`：喂造 stdin JSON + `_StubEmbedder` → 断言含 directives + 闸门内 recall；**off-topic prompt → 只剩 directives**；注入错误 → `exit 0`、stdout 安全。
- `recall(max_distance=…)` 过滤正确；`bump_access=False` 时 `hits`/`accessed_at` 不变、tier 未 flush。
- **【v2】** 中心化闸门：给定带公共偏置的造向量，`center` 路径能把相关/无关分开而绝对阈值不能。
- pin 与 recall 命中的**去重**；**【v2】** 跨轮去重。
- stdin 为空/非 JSON/无 prompt → directives-only，不崩。

**`tests/agents_test.py` 增：**
- `engram hook --mode context` 写入 `engram context`；与 `--mode directives` 互相**替换且 idempotent**；识别两种命令形态；保留用户自有 hook；Codex TOML 对称。

**daemon（§10）：** 起停 + 一次 embed 往返 smoke；daemon 不在时客户端降级路径；**Windows detached 进程拉起+存活+连上**（最高风险，先验证）。

全部尽量 hash 后端、零网络。

---

## 8. 向后兼容与安全
- **默认仍关闭**；`context` 模式 opt-in（`engram hook --mode context`）。`engram directives` 与现有安装零影响。
- 自动 recall **只读**（`bump_access=False`）⇒ 不与 MCP server 抢写锁、不放大 importance。
- **【v2】** daemon **不碰 `.pst`** ⇒ 从根上不参与 lost-update；只绑 `127.0.0.1` + 随机端口 + 仅当前用户可读的端口文件。
- hook **永不 `exit 2`**、永不在出错时写 stdout、有硬超时 ⇒ 不阻塞、不拖慢、不污染 turn。
- 相关性闸门 + 双预算 + 跨轮去重 ⇒ 控制每 turn token 成本，防止退化成「又一个臃肿 MEMORY.md」。

---

## 9. 需要你拍板的决策点
1. ~~**延迟路线**：A｜B｜C？~~ → **【v2 已定】** C 的实质：先 MVP（绝对阈值闸门，e5 内联 + 超时降级），再上自建 embedding-only daemon 保温 e5。model2vec 退场。
2. ~~**默认 e5 用户体验**~~ → **【v2 已定】** MVP 阶段 e5 内联会超时降级为 directives-only（功能优雅退化）；daemon 就绪后 relevant 段才真正常开。可接受（你已认可分阶段）。
3. **命令命名**：`engram context`？还是 `engram recall-hook` / 扩展 `engram directives` 使其在 `--mode context` 下读 stdin？ ← **仍待定**
4. ~~**自动 recall 是否完全不 bump**~~ → **【v2 已定】** 完全只读 `bump_access=False`。
5. **注入输出形态**：纯 markdown 块（与 directives 一致，简单）｜ `hookSpecificOutput.additionalContext` JSON？ ← **仍待定**
6. ~~**注入范围**~~ → **【v2 倾向已定】** 四类全开 + 距离阈值收口（见 §6）。仍可由 `ENGRAM_CONTEXT_TYPES` 覆盖。
7. **是否同时加 `SessionStart` 注入**：会话开场推一次 directives？ ← **仍待定**
8. **【v2 新增】daemon 接口决策**：见 §10 末尾的开放问题（协议形态、端口 vs 命名管道、idle 超时、token 与否）。

---

## 10. daemon 设计草案（`engram serve`）—— 【v2 新增，详见单独草拟】

> 本节为 daemon 的详细接口设计，单独草拟。占位指向；正文随讨论补全。

### 10a. 定位
embedding-only 常驻进程：保温 e5（只加载一次），监听 loopback，收「文字」回「向量」。**不碰 `.pst`。**

### 10b. daemon 的三项职责
1. **保温嵌入器**：进程内 e5 常驻，首轮冷（~12s）、之后每次 embed ~ms。
2. **持有 query-mu**：跨 turn 累积查询向量的滚动均值，供 §3.1 的 center-split 闸门用（单 hook 进程拿不到查询分布，daemon 能）。
3. **自标定阈值**：启动时从真实库采样，算一个 `max_distance` 默认值（§3.1 / §6）。

### 10c. 待补：协议、客户端降级、Windows detached 拉起、生命周期、安全。

---

## 11. 实验数据附录（2026-05-30 / 31，书面锚点）

> 环境：用户机 Windows 11 / Py 3.13 / e5 multilingual-e5-small（满血）/ CPU。
> 语料：40 条合成中文记忆（覆盖 user/feedback/project/reference 四类，含密集同域簇）。
> **所有数字指示性，非生产保证**；合成语料、干扰项有限，仅用于分支取舍。

### 11.1 model2vec vs e5 召回质量（40 docs / 24 targeted + 5 distractor）
| 指标 | e5(满血,384d) | model2vec(potion-ml,256d) |
|---|---|---|
| R@1 | **0.75** | 0.58 |
| R@3 / MRR | 0.92 / 0.83 | 0.71 / 0.63 |
| 释义 paraphrase | **0.89** | 0.56 |
| 跨语言 cross-ling | **1.00** | 0.67 |
| 闸门 gap | 0.014（窄） | 0.215（宽） |
> 14 文档 pilot 曾给 m2v R@1=0.82（高估）；扩到 40 文档密集同域后掉到 0.58 —— 静态向量在「细分辨」上崩。结论：m2v 召回不足以当主嵌入器。

### 11.2 冷启动分解（全新进程到首个向量）
| 路径 | 耗时 | 备注 |
|---|---|---|
| e5（离线，缓存命中） | ~12s | 大头 `import torch`(~2.3s)+`import sentence_transformers`(~6.5s)≈9s 地板 + 加载 1–3s |
| e5（联网，旧行为） | ~24–28s | 多一次 HF Hub revision 往返（已修） |
| model2vec(potion) | ~4s | import 0.24s + 加载 3.5s + encode 3ms；**不引入 torch** |
| hash（纯 numpy 参照） | ~0.23s | 进程启动地板 |
> 推论：任何「每-turn 新进程 + torch 系」都躲不掉 ~9s import 地板 → 必须保温（daemon），不能靠换 e5 自身。

### 11.3 e5 闸门策略（40 docs / 24 targeted + 15 distractor，纯 numpy，零模型迁移）
| 策略 | gap | **good@ff=0** | 落地性 |
|---|---|---|---|
| 绝对阈值（v1 假设） | 0.032 | **0.75** | 直接可用（MVP） |
| 中心化（混用文档均值） | 0.130 | 0.67 | 反而更差 |
| **中心化-split（查询/文档各减各均值）** | 0.195 | **0.88** | 需 daemon 存 query-mu |
| all-but-the-top（去前2主成分） | 0.137 | 0.71 | 不如 split |
| per-query z-score | 0.032 | 0.50 | 失败 |
> 绝对阈值能撑到 0.75（修正了之前「e5 几乎没法设阈值」的过严判断）；中心化-split 零迁移地提到 0.88，但依赖跨-turn 的 query-mu（daemon 持有）。

---

> 确认 §9 剩余决策（3/5/7/8）后我再动代码；在此之前不做任何源码改动。
> 附：本文件曾于 2026-05-31 09:02 被外部改写为残桩，已从对话上下文完整还原并升级为 Draft v2。
