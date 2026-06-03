# engram serve — 保温嵌入器 daemon 设计草案

> 状态：**设计草案 v1（待审查）**。本文件只是设计，**未改动任何代码**。
> 配套：上层路线见 [`ALWAYS_ON_RECALL_PLAN.md`](./ALWAYS_ON_RECALL_PLAN.md)（§4 已锁定「e5 + 自建 embedding-only daemon」，§10 指向本文件）。
> 决策点集中在 §9，给出选择我再实现。

---

## 1. 它解决什么（一句话）

每个 turn 的 hook 是全新子进程，默认 e5 每次冷启动 ~12s（其中 ~9s 是 `import torch`+`sentence_transformers` 的地板，离线也躲不掉，见 plan §11.2）。**daemon 让 e5 在 turn 之间一直醒着**：首轮付一次 ~12s，之后每次 embed 只要 loopback 往返 ~毫秒。这是「自动 recall」在默认 e5 下真正可用的前提。

> 已有的 `engram-mcp` 也常驻、也只加载一次 e5，但它是被 Claude Code 当 MCP client 经 **stdio** 拉起的，**hook 子进程够不到**。本 daemon 是不同的传输（网络回环）+ 不同的生命周期（自拉起、自存活）。§8 谈将来如何让两者复用同一保温进程。

---

## 2. 范围与非目标（最重要的设计约束）

**daemon 只做一件事：把「文字」变成「向量」。它绝不碰 `.pst`。**

| | daemon | hook（瘦客户端） |
|---|---|---|
| 加载 e5 | ✅ 一次，常驻 | ❌ 从不 import torch |
| 读 `.pst` | ❌ **从不** | ✅ 每轮读最新（~0.2s） |
| 写 `.pst` | ❌ **从不** | ❌（自动 recall 只读） |
| KNN / 闸门 | ❌ | ✅ 本地 numpy |
| 持有 query-mu / 标定阈值 | ✅（跨 turn 状态） | ❌（单进程看不到分布） |

**为什么 embedding-only 是关键决策**（区别于 plan v1 route B 让 daemon 持有完整 MemoryManager）：
- daemon 不读不写 `.pst` ⇒ **CLAUDE.md 里那一整段锁 / lost-update 痛苦全部绕过**。没有并发写、不需要轮询 `.pst` mtime、不会丢更新。
- hook 每轮自己读 `.pst` ⇒ 永远拿到最新数据，没有「daemon 内存副本陈旧」问题。
- daemon 状态极小（一个 e5 + 一个 384 维均值向量 + 几个标量），崩了重启代价低、无数据风险。

**非目标**：不做通用 RPC、不做多模型路由、不暴露记忆 CRUD、不跨机器（只 localhost）。

---

## 3. 架构与数据流

```
每个 turn:
  UserPromptSubmit hook 子进程 (engram context)
    │
    ├─(1) 读 stdin JSON → prompt, cwd ; chdir(cwd)
    │
    ├─(2) embed prompt:  ENGRAM_EMBEDDER=daemon:auto
    │        └─ daemon 客户端 ──TCP 127.0.0.1:port──▶ engram serve（常驻，e5 已热）
    │                                                    └─ 回 query 向量 + 当前 query-mu
    │        └─ 连不上 ⇒ 后台拉起 daemon，本轮抛 DaemonCold → 降级 directives-only
    │
    ├─(3) 读 .pst（本地，~0.2s）+ numpy KNN + 闸门（中心化用 query-mu）
    │
    └─(4) 拼 directives + relevant → stdout 注入，exit 0
```

要点：**torch 只活在 daemon 里**；hook 进程只有 numpy + 读文件 + 一次网络往返，本身亚秒级。

---

## 4. 协议

### 4.1 传输
**loopback TCP：`127.0.0.1` + 随机高位端口。** 选它的理由：
- 跨平台一致（Windows 无 unix socket；命名管道 API 与 socket 差异大，徒增分支）。
- Python `socketserver` / `asyncio` 直接可用，零依赖。
- 绑定 `127.0.0.1` 外部网络不可达（见 §7 安全）。

### 4.2 发现：端口文件
daemon 启动后把连接信息写到 **`~/.claude/engram/serve.json`**（`ENGRAM_HOME` 感知，与 store 同根）：
```json
{
  "pid": 12345,
  "port": 51789,
  "token": "<32字节随机hex>",
  "embedder": "local",          // daemon 实际加载的 embedder spec
  "dim": 384,
  "started_at": 1780200000,
  "proto": 1
}
```
客户端读它来连接 + 校验。文件权限 `0600`（见 §7）。

### 4.3 线格式：JSON Lines（一行一消息）
请求 / 响应都是单行 UTF-8 JSON + `\n`。无需重框架、无需额外依赖。

**embed**（最热路径）：
```
→ {"op":"embed","token":"...","kind":"query","texts":["怎么防止并发写坏存储"]}
← {"ok":true,"dim":384,"vecs":[[...384 floats...]],"mu":[...384...],"mu_n":417}
```
- `kind` = `query`|`passage`（e5 前缀语义，daemon 转给 embedder）。
- 响应带 `mu`（当前滚动 query-mu）+ `mu_n`（累计样本数）——hook 拿去做 center-split 闸门，无需额外往返。

**ping**（健康检查 / doctor）：
```
→ {"op":"ping"}
← {"ok":true,"pid":12345,"dim":384,"embedder":"local","uptime_s":83.2,"served":417}
```

**stats**（取标定阈值等）：
```
→ {"op":"stats","token":"..."}
← {"ok":true,"calibrated_max_dist":0.118,"gate":"center","mu_n":417}
```

**shutdown**（`engram serve --stop` 用）：
```
→ {"op":"shutdown","token":"..."}
← {"ok":true}
```

错误一律 `{"ok":false,"error":"...","code":"BAD_TOKEN|BAD_OP|DIM_MISMATCH|..."}`，**daemon 永不因一个坏请求退出**。

---

## 5. `daemon:` 嵌入器后端（registry 集成）

复用 `embedder.py` 已有的 `register_embedder` 机制，新增 scheme：
```
ENGRAM_EMBEDDER=daemon:auto      # 连不上则自动拉起（hook 默认）
ENGRAM_EMBEDDER=daemon:e5        # 同上，显式声明底层模型
ENGRAM_EMBEDDER=daemon?port=51789&autostart=0   # 只连不拉（调试/CI）
```
`DaemonEmbedder` 实现 `Embedder` 协议（`dim` / `embed` / `embed_batch`），内部是瘦客户端：
1. 读 `serve.json` → 连接 → 发 `embed` → 返回向量。**进程内不 import torch。**
2. `dim`：从 `serve.json` 读（连不上且需要 dim 时，回退到已知默认 384 或拉起后再问）。

### 5.1 冷 daemon 的处理（关键张力）
`Embedder.embed()` 协议必须返回一个向量，但「降级为 directives-only」是 **hook 层策略**，不是 embedder 层能决定的。解决：

- `daemon:auto` 连不上时：**后台拉起 daemon**（detached，§6），然后**抛一个带类型的 `DaemonColdError`**。
- `context_block()` / `_cmd_context` **捕获** `DaemonColdError` → 本轮降级 directives-only（仍 `exit 0`）。下一轮 daemon 已热。
- **非 hook 调用者**（如 `engram recall` 交互式）：捕获后可自行选择「inline 回退到 `LocalE5Embedder`（慢但正确）」或「等待 daemon 就绪」。即同一个后端，hook 选降级、CLI 选等待，策略在调用方。

> 这样 `MemoryManager`/`recall`/`context_block` 的核心代码**完全不动**——只是 `ENGRAM_EMBEDDER` 换成 `daemon:auto`，加一个 `DaemonColdError` 的捕获点。

---

## 6. 拉起 detached 进程（**最高风险，先做 spike**）

hook 子进程必须能拉起一个**在 hook 自己退出后仍存活**的 daemon。这是整个方案 Windows 上最容易翻车的一环，**应在写其余代码前先单独验证一条链**：

> **spike 验收**：进程 A `Popen` 拉起 daemon → A 立即退出 → daemon 仍在 → daemon 绑定端口、写 `serve.json` → 全新进程 B 能连上并 embed 成功。

实现要点：
- **Windows**：`subprocess.Popen([...], creationflags=DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP, close_fds=True)`；用 **`pythonw.exe`** 避免弹控制台窗口；`stdin/stdout/stderr` 重定向到 `NUL` 或日志文件（不能继承 hook 的管道，否则 hook 的 stdout 被污染 → 注入垃圾）。
- **POSIX**：`start_new_session=True`（脱离进程组），stdio 重定向到 `/dev/null` 或日志。
- **单实例**：拉起前用**原子方式**抢 `serve.json`（或一个 `serve.lock`，复用 `_lock.py` 的 `FileLock`），避免两个 hook 同时拉起两个 daemon 抢同一逻辑端口。抢不到锁就等一小会儿再读 `serve.json` 连。
- **拉起后不阻塞本轮**：hook 拉起 daemon 后**不等它热**（要 ~12s），直接本轮降级 directives-only，让 daemon 在后台慢慢热；下一轮自然连上。

---

## 7. 安全

- **只绑 `127.0.0.1`**，绝不 `0.0.0.0`。外部网络不可达。
- **`serve.json` 权限 `0600`**（仅当前用户可读）。Windows 上靠用户 profile 目录 ACL + 可选显式 `icacls`。
- **token**：`serve.json` 里放一个随机 token，写操作类请求（embed/stats/shutdown）需带。防同机其它用户 / 进程乱连那个端口。单用户开发机风险低，但 token 近乎零成本，**默认带上**。
- **资源**：限制单请求 `texts` 条数与总字节（复用 `_truncate_for_remote` 的思路），防止一个超大请求拖垮 daemon。

---

## 7b. 生命周期模型与开关语义（**核心：默认零进程**）

### 7b.1 不变式
**「记忆长留」默认关闭。不开启 ⇒ 机器上永远不存在这个 daemon 进程**（零内存、零意外）。装了 engram 但没手动开的用户，环境里没有任何常驻 python。

### 7b.2 开关 = 单一真相源
daemon 只被 `engram context` 自动拉起；`engram context` 只被 **context 模式的 UserPromptSubmit hook** 触发；该 hook 只有用户显式 CLI 命令才会装上。所以 **「装没装 context hook」本身就是开关**，daemon 是它的下游结果 —— 不再额外引入独立的 config flag（两个开关会不一致，用户困惑「我关了它怎么还在」）。

**面向用户的入口（已定）**：
```
engram autorecall on       # 装 context hook → 「记忆长留」开启（下个 turn 起惰性拉 daemon）
engram autorecall off      # 卸 context hook，退回 ② directives；顺手 serve --stop
engram autorecall status   # 报当前档位 + daemon 是否在跑 + serve.json 新鲜度
```
`autorecall` 是面向用户的友好叙事入口，**底层转调** `engram hook --mode {context,directives}`（hook 仍是底层机制，见 plan §5.4）。两套入口的真相源都是「settings.json 里装的是哪个 hook」，不另存状态。

### 7b.3 三档阶梯（与现有架构一脉相承）
| 档位 | 注入 | daemon | 成本 | 开启方式 |
|---|---|---|---|---|
| ① 全关（默认） | 无 | ❌ | 0 | —— |
| ② directives | 仅 pin 常驻规则 | ❌ | ~0.2s 纯元数据 | `engram hook`（现有，亦 opt-in） |
| ③ **记忆长留**（context） | pin + 每轮相关 recall | ✅ 惰性 | 首轮冷、后续 ~ms | `engram autorecall on`（本次新增） |
> `autorecall off` **退回 ②**（保留 pin 常驻规则，只去掉每轮 recall + daemon）—— 符合「我不想要语义 recall，但全局规则还要」的常见诉求。要彻底退到 ① 用 `engram hook --disable`。

### 7b.4 生命周期状态机（self-managed，A 模型）
```
   ① ABSENT ──hook 连不上→抢锁→Popen detached──► ② STARTING(冷启~12s,写serve.json)
                                                        │
                                                        ▼
                                                   ③ WARM(服务中, 每请求重置 idle 计时器)
                                                        │
        ┌──────────────┬──────────────┬────────────────┤
   idle 30min 自退   --stop 优雅退    机器重启进程消失   崩溃/OOM(留陈旧 serve.json)
        │              │              │                │
        ▼              ▼              ▼                ▼
   ① ABSENT       ① ABSENT       ① ABSENT      下个 hook 自愈：连不上→检测pid死→删陈旧→重拉
```
**关键性质**：
- **死后不复活看门狗**，而是**下一个 hook 惰性自愈**（连不上 → pid 不存活 → 删 `serve.json` → 重拉）。无需常驻守护、无需开机自启。
- **一机一 daemon，跨会话/跨项目共享**：嵌入器「文字→向量」与项目无关（不像 `.pst` 每项目一个），多个 Claude Code 窗口/多项目**共享同一 daemon**，e5 内存里只一份。这正是 embedding-only 设计的红利（若持有 MemoryManager 就会逼出「每项目一 daemon」）。
- **孤儿被 idle 兜住**：关掉所有 client 后 daemon 最多再空转 idle 超时就自退，不会永久占内存。
- **重启后第一轮降级**：机器重启后首个会话的头一个 turn 没有 relevant 段（daemon 正在被拉起冷启动），下一轮即热。换来「零系统服务、零开机负担」，可接受。

### 7b.5 为什么是 A（self-managed）而非 OS 服务
| 模型 | 谁管 | 取舍 |
|---|---|---|
| **A. self-managed**（采用） | hook 惰性拉、idle 自退、hook 自愈 | 零安装/零权限/自愈/不用不占；代价：冷启动等一轮、重启后首轮降级 |
| B. OS 服务（将来 opt-in） | 开机自启，systemd/launchd/Win service | 永远热；代价：安装重、要权限、「装记忆工具却被塞个系统服务」观感差 |
| C. parent-managed（否决） | client 起停（像 engram-mcp） | 生命周期干净，但 **hook 够不到**，这正是要自建 daemon 的原因 |
> A 契合 engram「本地优先、零负担」的调性（per-turn hook 本就 opt-in）。B 留作将来 `engram serve --install-service` 给重度用户。与 `engram-mcp`（parent-managed 写入主力）形成清晰分工：daemon 是 self-managed 只读保温嵌入器，两者生命周期独立。

### 7b.6 升级与内存（运维诚实告知）
- **版本一致性**：`pip install -U engram` 后后台可能还跑旧代码 daemon。`serve.json` 记 `proto`/`embedder`/版本；客户端连上先校验，不匹配 → 让旧的退、重起新的；doctor 报「daemon 版本与已装不一致」。
- **内存占用**：e5+torch 常驻约数百 MB。idle 超时保证「不用就不占」。该写进文档让用户知情。

---

## 8. 运维命令与 doctor

- **`engram serve`**：前台运行（调试用，日志打 stderr）。
- **`engram serve --detach`**：后台 detached 启动（§6）。
- **`engram serve --stop`**：读 `serve.json` 的 pid，发 `shutdown`（或终止 pid），清理 `serve.json`。
- **`engram serve --status`**：`ping` 一下，报 uptime / served / dim。
- **idle 自退**：空闲 **30min**（已定）无请求自动退出，免得常驻吃内存。每个请求重置计时器。
- **陈旧清理**：客户端连不上 / pid 不存活 ⇒ 视 `serve.json` 为陈旧，删除并重新拉起（§7b.4 自愈）。
- **状态持久化（待定，§9）**：query-mu 和标定阈值要不要存盘？存 → daemon 重启不丢累积（写一个小 `serve_state.npz`，但增一点复杂度）；不存 → 每次重启 query-mu 从零累计（前几轮 center-split 退化为近似 center）。
- **doctor 集成**：新增一项健康检查 ——「daemon 在跑吗 / `ping` 通吗 / `dim` 与 store 匹配吗 / `serve.json` 新鲜吗」，失败给可复制的修复提示（`engram serve --detach`）。

---

## 9. 决策点（✅ 已定 / ⬜ 待定）

- ✅ **生命周期模型**：A（self-managed，惰性拉起 + idle 自退 + hook 自愈）。见 §7b。
- ✅ **默认开启**：关闭。「记忆长留」需 `engram autorecall on` 手动开启；不开则零进程。见 §7b.1。
- ✅ **开关入口**：`engram autorecall on/off/status`（底层转调 `engram hook --mode`）；单一真相源 = 装的是哪个 hook。见 §7b.2。
- ✅ **关闭语义**：`autorecall off` 退回 ② directives（保留 pin）。见 §7b.3。
- ✅ **idle 自退超时**：30min。
- ✅ **传输**：loopback TCP（127.0.0.1 + 随机端口）。**spike 已验证可行**（§9b）。
- ✅ **token**：默认带上（serve.json 内随机 token，写操作类请求校验）。spike 已验证拒绝逻辑可用。
- ✅ **query-mu / 阈值持久化**：存盘跨重启（小 `serve_state` 文件）。daemon 重启不丢累积，体验更稳。
- ✅ **冷 daemon 本轮策略**：降级 directives-only + 后台拉起（与 §5.1/§7b.4 一致）。绝不同步阻塞 turn。
- ✅ **命名**：daemon 运维子命令 `engram serve`；面向用户开关 `engram autorecall`（两者分工见 §7b.2）。
- ✅ **MVP 范围**：先只做「保温 embed」（职责①）。query-mu（②）+ 标定阈值（③）留到 daemon 跑通后接入（§10 step 4），避免一次性铺太大。

> **§9 全部决策已定，无待定项。** 实现前的唯一硬闸门（Windows detached spike）已通过（§9b），可进入 §10 实现。

---

## 9b. spike 验证记录（go/no-go 闸门，2026-05-31，Windows 11 / Py 3.13）

在写任何主干代码前跑了两个一次性 spike（已删，不入库），验证 A 模型在 Windows 上的可行性。

**spike 1 — 完整生命周期（11/11 PASS）**：detached 拉起 → 释放句柄 → daemon 存活并写 `serve.json`（带 port+token）→ 全新进程连上 + embed 往返 → 坏 token 被拒 → idle 超时自退 + 清理 `serve.json` → 自愈重拉（新 pid≠旧 pid）。

**spike 2 — 严格父死验证（6/6 PASS）**：spawner 拉起 daemon 后 `os._exit(0)` 硬退（模拟 hook 结束），独立进程确认 spawner pid 已死、daemon pid 仍存活、孤儿 daemon 仍可连上 + embed。这是最该证伪、Windows 上最易翻车的一环。

**关键可移植结论**（直接用于 `serve.py`）：
- 拉起 flags：`DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW` + stdio 全部重定向 `DEVNULL`（**不可继承 hook 的管道**，否则污染注入 stdout）。
- `serve.json` 原子发布：写 `.tmp` 再 `os.replace()`。
- 协议：JSON Lines（一行一消息）over loopback TCP，零依赖。
- idle watchdog：独立线程，每请求重置计时器，超时后自连一次 nudge 醒 accept 循环再退。

**边界**：仅验 Windows（主力平台）。POSIX（`start_new_session=True`）那遍未跑，属成熟套路、风险低，留待实现期 CI 补，不阻塞。

---

## 10. 建议实现顺序（你批准后）

0. ✅ **（前置）spike — go/no-go 闸门【已通过，见 §9b】**：A 模型命门一环已在 Windows 验证。
1. `serve.py`：embedding-only daemon（loopback TCP + JSON Lines + `embed`/`ping`/`stats`/`shutdown` + idle 自退 + `serve.json` + 单实例锁）。
2. `DaemonEmbedder` 后端 + registry 注册 + `DaemonColdError` + 客户端拉起/自愈逻辑。
3. `engram serve [--detach|--stop|--status]`（底层运维）+ **`engram autorecall on/off/status`**（面向用户开关，转调 hook --mode）+ doctor 健康检查。
4. 接进 `context_block`：center-split 用响应里的 `mu`；阈值用 `stats` 的标定值。
5. 测试：JSON 协议往返（hash 后端，零网络）；客户端降级路径；`autorecall on/off` 装卸 hook 幂等且与 `hook --mode` 真相源一致；spike 的自动化版（起停 + 一次往返 + 自愈）。

> 全部尽量 hash 后端、零网络；torch 相关只在 daemon 进程，测试用 hash embedder 验证协议与生命周期，不验 e5 本身。
> **下一步**：§9 全部已定 + §9b 闸门已过 → 可从 step 1（`serve.py`）开始编码。
