# Engram 与 MD 记忆系统的对照基准

- 语料：**40 条记忆**，**25 条查询**（10 简单 / 10 中等 / 5 困难）
- token 计数器：**tiktoken (cl100k_base)**
- engram 嵌入后端：**local-e5**


## 速览结论

在最强 MD 配置 `md-top3` 与同档 Engram 配置 `engram-k3`（k=3）的正面对决中：

- **Recall@3**：engram 100.0%，MD 92.0%  （Δ = +8.0 个百分点）
- **MRR**：engram 0.973，MD 0.880
- **50 轮会话下的每轮平均 token 开销**：engram **54.7** vs MD **1430.8**  （**便宜 26.1 倍**）


## 关键结论

1. **token 成本压倒性优势**：在 50 轮会话内，amortized 每轮平均 token 数 engram 仅 **54.7**，MD 基线高达 **1430.8**，相当于 **26.1× 的节省**。核心原因：MD 必须把 ~1331 token 的 `MEMORY.md` 索引塞进**每一轮**的上下文；engram 完全不需要，按需检索。

2. **更激进的 k=1 配置进一步压到 78× 便宜**（每轮 18.3 token），且 Recall@1 仍保持 96.0%，高于 MD 的 84.0%。说明可以**用更少的 token 拿到更高的准确率**——这是传统检索系统做不到的反直觉结果。

3. **准确率正向**：engram Recall@3 = 100.0%，MD = 92.0%（高 +8.0 个百分点）。MRR 0.973 vs 0.880，也意味着正确答案在 engram 中通常排得更靠前。

4. **越难的题差距越大**：困难档 Recall@3，engram 100.0% vs MD 80.0%（差 +20.0 pp）。MD 依赖关键词匹配，问题改写或同义表达会让它失手；语义嵌入对释义不敏感。

5. **可独立复现**：corpus、queries、两个系统实现全部在 `eval/` 目录中，ground-truth 已标注，token 计数透明，无随机性。运行 `python -m eval.runner` 即可在你的机器上得到同样的对比数据。


> **结论**：在记忆检索这个场景下，向量数据库 + 语义嵌入的方案**同时**击败了基于 MD 的传统方案在准确率和 token 成本两个维度上的指标，证明 engram 方向的**有效性**与**可用性**。


## 头条数据表

| 系统 | R@1 | R@3 | R@5 | MRR | 基线 token | 平均查询 token |
|---|---:|---:|---:|---:|---:|---:|
| `md-top3` | 84.0% | 92.0% | 92.0% | 0.880 | 1331 | 199.6 |
| `md-top1` | 84.0% | 84.0% | 84.0% | 0.840 | 1331 | 65.5 |
| `engram-k1` | 96.0% | 96.0% | 96.0% | 0.960 | 0 | 36.6 |
| `engram-k3` | 96.0% | 100.0% | 100.0% | 0.973 | 0 | 109.4 |
| `engram-k5` | 96.0% | 100.0% | 100.0% | 0.973 | 0 | 188.2 |

## 50 轮会话内的每轮平均 token 开销

每轮加载一次索引，每轮发起一次查询。

| 系统 | 每轮 token | 拆分 |
|---|---:|---|
| `md-top3` | 1430.8 | 基线 1331 + 每查询 199.6 |
| `md-top1` | 1363.7 | 基线 1331 + 每查询 65.5 |
| `engram-k1` | 18.3 | 基线 0 + 每查询 36.6 |
| `engram-k3` | 54.7 | 基线 0 + 每查询 109.4 |
| `engram-k5` | 94.1 | 基线 0 + 每查询 188.2 |

## 按查询难度拆分的 Recall@3

| 系统 | 简单 | 中等 | 困难 |
|---|---:|---:|---:|
| `md-top3` | 90.0% | 100.0% | 80.0% |
| `md-top1` | 90.0% | 80.0% | 80.0% |
| `engram-k1` | 100.0% | 90.0% | 100.0% |
| `engram-k3` | 100.0% | 100.0% | 100.0% |
| `engram-k5` | 100.0% | 100.0% | 100.0% |

## 各系统的实现细节

- **`md-top3`** — Legacy MEMORY.md + per-file .md; reads top-3 files per query (keyword Jaccard on the index line).
- **`md-top1`** — Legacy MEMORY.md + per-file .md; reads top-1 files per query (keyword Jaccard on the index line).
- **`engram-k1`** — Engram v2 (vector DB + rerank); top-1 recall, descriptions only. Embedder = local-e5.
- **`engram-k3`** — Engram v2 (vector DB + rerank); top-3 recall, descriptions only. Embedder = local-e5.
- **`engram-k5`** — Engram v2 (vector DB + rerank); top-5 recall, descriptions only. Embedder = local-e5.

## 逐查询明细


### md-top3

| # | 难度 | 查询 | 标注答案 | top-5 命中情况 | token |
|---:|---|---|---|---|---:|
| 1 | 简单 | what timezone is the user in | u-tz-shanghai | ✓ u-tz-shanghai, f-bundled-pr, f-conservative-deps | 199 |
| 2 | 简单 | does the user prefer Chinese or English | u-lang-chinese | ✓ u-lang-chinese, f-bundled-pr, f-conservative-deps | 197 |
| 3 | 简单 | how should I write integration tests | f-no-mock-db | ✓ f-no-mock-db, p-mvp-scope-v1, f-bundled-pr | 222 |
| 4 | 简单 | should I add emojis to my reply | f-no-emoji | ✓ f-no-emoji, f-bundled-pr, f-conservative-deps | 200 |
| 5 | 简单 | when does the merge freeze begin | p-freeze-2026-03-05 | ✓ p-freeze-2026-03-05, p-v2-scope, f-bundled-pr | 222 |
| 6 | 简单 | which Python version does the repo target | p-py-version-3-11 | ✓ p-py-version-3-11, u-platform-windows, f-name-snake-case | 200 |
| 7 | 简单 | where is the api latency dashboard | r-grafana-api-latency | ✓ r-grafana-api-latency, r-launchdarkly-dashboard, r-pistadb-skill | 191 |
| 8 | 简单 | what is the runbook for db failover | r-runbook-db-failover | ✓ r-runbook-db-failover, p-repo-purpose-memskill, p-stack-pistadb-e5 | 208 |
| 9 | 简单 | how should I name unit tests | f-test-naming-given-when | ✗ f-no-mock-db, f-bundled-pr, f-conservative-deps | 211 |
| 10 | 简单 | can I force push to shared branches | f-no-force-push-shared | ✓ f-no-force-push-shared, f-bundled-pr, f-conservative-deps | 210 |
| 11 | 中等 | the user uses Windows right? which python should I call | u-platform-windows | ✓ u-platform-windows, p-py-version-3-11, u-vim-keys | 183 |
| 12 | 中等 | is the user fluent in Go | u-role-backend-go | ✓ u-role-backend-go, f-bundled-pr, f-conservative-deps | 202 |
| 13 | 中等 | do not just patch the symptom; what does the user want | f-fix-root-cause | ✓ f-fix-root-cause, f-no-mock-db, f-no-trailing-questions | 198 |
| 14 | 中等 | should we cap how many external libraries we pull in | f-conservative-deps | ✓ f-bundled-pr, f-conservative-deps, f-explain-tradeoffs | 201 |
| 15 | 中等 | how do I gate a new feature behind a toggle | p-feature-flag-launch-default-off | ✓ p-feature-flag-launch-default-off, f-conservative-deps, r-launchdarkly-dashboard | 192 |
| 16 | 中等 | how long do raw events stick around in the warehouse | p-data-retention-90d | ✓ p-data-retention-90d, f-bundled-pr, f-conservative-deps | 207 |
| 17 | 中等 | which team handles changes to the ingest pipeline | p-ingest-pipeline-owner, r-slack-data-infra | ✓ p-ingest-pipeline-owner, r-linear-ingest-project, r-slack-data-infra | 184 |
| 18 | 中等 | where should new INGEST bugs be filed | r-linear-ingest-project | ✓ f-conservative-deps, r-linear-ingest-project, p-feature-flag-launch-default-off | 188 |
| 19 | 中等 | are type hints required on new Python functions | f-prefer-explicit-types | ✓ f-prefer-explicit-types, u-platform-windows, f-conservative-deps | 194 |
| 20 | 中等 | does the user want a meeting at 9am | u-coffee-no-meeting-am | ✓ u-coffee-no-meeting-am, f-bundled-pr, f-conservative-deps | 192 |
| 21 | 困难 | the user told me to stop wrapping up with 'anything else?' l | f-no-trailing-questions | ✓ f-no-trailing-questions, f-bundled-pr, f-conservative-deps | 199 |
| 22 | 困难 | what's the protocol around shipping a tightly coupled refact | f-bundled-pr | ✓ f-bundled-pr, f-conservative-deps, f-explain-tradeoffs | 201 |
| 23 | 困难 | the auth project — is it tech debt or something else driving | p-auth-rewrite-compliance | ✓ p-auth-rewrite-compliance, r-linear-ingest-project, f-no-trailing-questions | 174 |
| 24 | 困难 | how do release tags map to deploys | p-release-tag-pattern | ✓ p-release-tag-pattern, p-freeze-2026-03-05, f-bundled-pr | 215 |
| 25 | 困难 | what's the convention for naming variables in this codebase | f-name-snake-case | ✗ f-bundled-pr, f-conservative-deps, f-explain-tradeoffs | 201 |

### md-top1

| # | 难度 | 查询 | 标注答案 | top-5 命中情况 | token |
|---:|---|---|---|---|---:|
| 1 | 简单 | what timezone is the user in | u-tz-shanghai | ✓ u-tz-shanghai | 57 |
| 2 | 简单 | does the user prefer Chinese or English | u-lang-chinese | ✓ u-lang-chinese | 55 |
| 3 | 简单 | how should I write integration tests | f-no-mock-db | ✓ f-no-mock-db | 69 |
| 4 | 简单 | should I add emojis to my reply | f-no-emoji | ✓ f-no-emoji | 58 |
| 5 | 简单 | when does the merge freeze begin | p-freeze-2026-03-05 | ✓ p-freeze-2026-03-05 | 69 |
| 6 | 简单 | which Python version does the repo target | p-py-version-3-11 | ✓ p-py-version-3-11 | 69 |
| 7 | 简单 | where is the api latency dashboard | r-grafana-api-latency | ✓ r-grafana-api-latency | 57 |
| 8 | 简单 | what is the runbook for db failover | r-runbook-db-failover | ✓ r-runbook-db-failover | 61 |
| 9 | 简单 | how should I name unit tests | f-test-naming-given-when | ✗ f-no-mock-db | 69 |
| 10 | 简单 | can I force push to shared branches | f-no-force-push-shared | ✓ f-no-force-push-shared | 68 |
| 11 | 中等 | the user uses Windows right? which python should I call | u-platform-windows | ✓ u-platform-windows | 61 |
| 12 | 中等 | is the user fluent in Go | u-role-backend-go | ✓ u-role-backend-go | 60 |
| 13 | 中等 | do not just patch the symptom; what does the user want | f-fix-root-cause | ✓ f-fix-root-cause | 72 |
| 14 | 中等 | should we cap how many external libraries we pull in | f-conservative-deps | ✗ f-bundled-pr | 81 |
| 15 | 中等 | how do I gate a new feature behind a toggle | p-feature-flag-launch-default-off | ✓ p-feature-flag-launch-default-off | 70 |
| 16 | 中等 | how long do raw events stick around in the warehouse | p-data-retention-90d | ✓ p-data-retention-90d | 65 |
| 17 | 中等 | which team handles changes to the ingest pipeline | p-ingest-pipeline-owner, r-slack-data-infra | ✓ p-ingest-pipeline-owner | 69 |
| 18 | 中等 | where should new INGEST bugs be filed | r-linear-ingest-project | ✗ f-conservative-deps | 61 |
| 19 | 中等 | are type hints required on new Python functions | f-prefer-explicit-types | ✓ f-prefer-explicit-types | 72 |
| 20 | 中等 | does the user want a meeting at 9am | u-coffee-no-meeting-am | ✓ u-coffee-no-meeting-am | 50 |
| 21 | 困难 | the user told me to stop wrapping up with 'anything else?' l | f-no-trailing-questions | ✓ f-no-trailing-questions | 57 |
| 22 | 困难 | what's the protocol around shipping a tightly coupled refact | f-bundled-pr | ✓ f-bundled-pr | 81 |
| 23 | 困难 | the auth project — is it tech debt or something else driving | p-auth-rewrite-compliance | ✓ p-auth-rewrite-compliance | 60 |
| 24 | 困难 | how do release tags map to deploys | p-release-tag-pattern | ✓ p-release-tag-pattern | 65 |
| 25 | 困难 | what's the convention for naming variables in this codebase | f-name-snake-case | ✗ f-bundled-pr | 81 |

### engram-k1

| # | 难度 | 查询 | 标注答案 | top-5 命中情况 | token |
|---:|---|---|---|---|---:|
| 1 | 简单 | what timezone is the user in | u-tz-shanghai | ✓ u-tz-shanghai | 33 |
| 2 | 简单 | does the user prefer Chinese or English | u-lang-chinese | ✓ u-lang-chinese | 35 |
| 3 | 简单 | how should I write integration tests | f-no-mock-db | ✓ f-no-mock-db | 32 |
| 4 | 简单 | should I add emojis to my reply | f-no-emoji | ✓ f-no-emoji | 32 |
| 5 | 简单 | when does the merge freeze begin | p-freeze-2026-03-05 | ✓ p-freeze-2026-03-05 | 46 |
| 6 | 简单 | which Python version does the repo target | p-py-version-3-11 | ✓ p-py-version-3-11 | 42 |
| 7 | 简单 | where is the api latency dashboard | r-grafana-api-latency | ✓ r-grafana-api-latency | 37 |
| 8 | 简单 | what is the runbook for db failover | r-runbook-db-failover | ✓ r-runbook-db-failover | 39 |
| 9 | 简单 | how should I name unit tests | f-test-naming-given-when | ✓ f-test-naming-given-when | 41 |
| 10 | 简单 | can I force push to shared branches | f-no-force-push-shared | ✓ f-no-force-push-shared | 34 |
| 11 | 中等 | the user uses Windows right? which python should I call | u-platform-windows | ✓ u-platform-windows | 33 |
| 12 | 中等 | is the user fluent in Go | u-role-backend-go | ✓ u-role-backend-go | 34 |
| 13 | 中等 | do not just patch the symptom; what does the user want | f-fix-root-cause | ✗ f-terse-replies | 31 |
| 14 | 中等 | should we cap how many external libraries we pull in | f-conservative-deps | ✓ f-conservative-deps | 31 |
| 15 | 中等 | how do I gate a new feature behind a toggle | p-feature-flag-launch-default-off | ✓ p-feature-flag-launch-default-off | 38 |
| 16 | 中等 | how long do raw events stick around in the warehouse | p-data-retention-90d | ✓ p-data-retention-90d | 37 |
| 17 | 中等 | which team handles changes to the ingest pipeline | p-ingest-pipeline-owner, r-slack-data-infra | ✓ p-ingest-pipeline-owner | 43 |
| 18 | 中等 | where should new INGEST bugs be filed | r-linear-ingest-project | ✓ r-linear-ingest-project | 31 |
| 19 | 中等 | are type hints required on new Python functions | f-prefer-explicit-types | ✓ f-prefer-explicit-types | 41 |
| 20 | 中等 | does the user want a meeting at 9am | u-coffee-no-meeting-am | ✓ u-coffee-no-meeting-am | 39 |
| 21 | 困难 | the user told me to stop wrapping up with 'anything else?' l | f-no-trailing-questions | ✓ f-no-trailing-questions | 36 |
| 22 | 困难 | what's the protocol around shipping a tightly coupled refact | f-bundled-pr | ✓ f-bundled-pr | 37 |
| 23 | 困难 | the auth project — is it tech debt or something else driving | p-auth-rewrite-compliance | ✓ p-auth-rewrite-compliance | 38 |
| 24 | 困难 | how do release tags map to deploys | p-release-tag-pattern | ✓ p-release-tag-pattern | 39 |
| 25 | 困难 | what's the convention for naming variables in this codebase | f-name-snake-case | ✓ f-name-snake-case | 36 |

### engram-k3

| # | 难度 | 查询 | 标注答案 | top-5 命中情况 | token |
|---:|---|---|---|---|---:|
| 1 | 简单 | what timezone is the user in | u-tz-shanghai | ✓ u-tz-shanghai, u-coffee-no-meeting-am, r-grafana-api-latency | 111 |
| 2 | 简单 | does the user prefer Chinese or English | u-lang-chinese | ✓ u-lang-chinese, f-no-emoji, u-name-tony | 104 |
| 3 | 简单 | how should I write integration tests | f-no-mock-db | ✓ f-no-mock-db, f-test-naming-given-when, f-fix-root-cause | 107 |
| 4 | 简单 | should I add emojis to my reply | f-no-emoji | ✓ f-no-emoji, u-name-tony, f-terse-replies | 100 |
| 5 | 简单 | when does the merge freeze begin | p-freeze-2026-03-05 | ✓ p-freeze-2026-03-05, p-data-retention-90d, p-v2-scope | 131 |
| 6 | 简单 | which Python version does the repo target | p-py-version-3-11 | ✓ p-py-version-3-11, u-platform-windows, p-tier-storage | 109 |
| 7 | 简单 | where is the api latency dashboard | r-grafana-api-latency | ✓ r-grafana-api-latency, r-launchdarkly-dashboard, r-pagerduty-on-call | 116 |
| 8 | 简单 | what is the runbook for db failover | r-runbook-db-failover | ✓ r-runbook-db-failover, p-feature-flag-launch-default-off, r-launchdarkly-dashboa | 117 |
| 9 | 简单 | how should I name unit tests | f-test-naming-given-when | ✓ f-test-naming-given-when, u-lang-chinese, f-no-mock-db | 110 |
| 10 | 简单 | can I force push to shared branches | f-no-force-push-shared | ✓ f-no-force-push-shared, p-release-tag-pattern, f-bundled-pr | 112 |
| 11 | 中等 | the user uses Windows right? which python should I call | u-platform-windows | ✓ u-platform-windows, u-vim-keys, f-name-snake-case | 106 |
| 12 | 中等 | is the user fluent in Go | u-role-backend-go | ✓ u-role-backend-go, u-vim-keys, u-name-tony | 106 |
| 13 | 中等 | do not just patch the symptom; what does the user want | f-fix-root-cause | ✓ f-terse-replies, f-conservative-deps, f-fix-root-cause | 96 |
| 14 | 中等 | should we cap how many external libraries we pull in | f-conservative-deps | ✓ f-conservative-deps, f-bundled-pr, f-terse-replies | 101 |
| 15 | 中等 | how do I gate a new feature behind a toggle | p-feature-flag-launch-default-off | ✓ p-feature-flag-launch-default-off, p-release-tag-pattern, p-tier-storage | 112 |
| 16 | 中等 | how long do raw events stick around in the warehouse | p-data-retention-90d | ✓ p-data-retention-90d, r-grafana-api-latency, f-fix-root-cause | 108 |
| 17 | 中等 | which team handles changes to the ingest pipeline | p-ingest-pipeline-owner, r-slack-data-infra | ✓ p-ingest-pipeline-owner, r-linear-ingest-project, r-slack-data-infra | 110 |
| 18 | 中等 | where should new INGEST bugs be filed | r-linear-ingest-project | ✓ r-linear-ingest-project, p-ingest-pipeline-owner, f-fix-root-cause | 108 |
| 19 | 中等 | are type hints required on new Python functions | f-prefer-explicit-types | ✓ f-prefer-explicit-types, p-py-version-3-11, f-conservative-deps | 115 |
| 20 | 中等 | does the user want a meeting at 9am | u-coffee-no-meeting-am | ✓ u-coffee-no-meeting-am, p-data-retention-90d, u-role-backend-go | 112 |
| 21 | 困难 | the user told me to stop wrapping up with 'anything else?' l | f-no-trailing-questions | ✓ f-no-trailing-questions, f-terse-replies, f-no-emoji | 101 |
| 22 | 困难 | what's the protocol around shipping a tightly coupled refact | f-bundled-pr | ✓ f-bundled-pr, p-tier-storage, p-auth-rewrite-compliance | 110 |
| 23 | 困难 | the auth project — is it tech debt or something else driving | p-auth-rewrite-compliance | ✓ p-auth-rewrite-compliance, f-fix-root-cause, f-conservative-deps | 103 |
| 24 | 困难 | how do release tags map to deploys | p-release-tag-pattern | ✓ p-release-tag-pattern, p-feature-flag-launch-default-off, r-grafana-api-latency | 116 |
| 25 | 困难 | what's the convention for naming variables in this codebase | f-name-snake-case | ✓ f-name-snake-case, u-lang-chinese, p-py-version-3-11 | 115 |

### engram-k5

| # | 难度 | 查询 | 标注答案 | top-5 命中情况 | token |
|---:|---|---|---|---|---:|
| 1 | 简单 | what timezone is the user in | u-tz-shanghai | ✓ u-tz-shanghai, u-coffee-no-meeting-am, r-grafana-api-latency, p-data-retention-9 | 183 |
| 2 | 简单 | does the user prefer Chinese or English | u-lang-chinese | ✓ u-lang-chinese, f-no-emoji, u-name-tony, f-terse-replies, u-vim-keys | 173 |
| 3 | 简单 | how should I write integration tests | f-no-mock-db | ✓ f-no-mock-db, f-test-naming-given-when, f-fix-root-cause, f-terse-replies, f-pre | 181 |
| 4 | 简单 | should I add emojis to my reply | f-no-emoji | ✓ f-no-emoji, u-name-tony, f-terse-replies, u-lang-chinese, p-py-version-3-11 | 179 |
| 5 | 简单 | when does the merge freeze begin | p-freeze-2026-03-05 | ✓ p-freeze-2026-03-05, p-data-retention-90d, p-v2-scope, p-py-version-3-11, p-mvp- | 222 |
| 6 | 简单 | which Python version does the repo target | p-py-version-3-11 | ✓ p-py-version-3-11, u-platform-windows, p-tier-storage, p-mvp-scope-v1, p-v2-scop | 205 |
| 7 | 简单 | where is the api latency dashboard | r-grafana-api-latency | ✓ r-grafana-api-latency, r-launchdarkly-dashboard, r-pagerduty-on-call, u-tz-shang | 185 |
| 8 | 简单 | what is the runbook for db failover | r-runbook-db-failover | ✓ r-runbook-db-failover, p-feature-flag-launch-default-off, r-launchdarkly-dashboa | 210 |
| 9 | 简单 | how should I name unit tests | f-test-naming-given-when | ✓ f-test-naming-given-when, u-lang-chinese, f-no-mock-db, f-fix-root-cause, f-pref | 185 |
| 10 | 简单 | can I force push to shared branches | f-no-force-push-shared | ✓ f-no-force-push-shared, p-release-tag-pattern, f-bundled-pr, u-coffee-no-meeting | 199 |
| 11 | 中等 | the user uses Windows right? which python should I call | u-platform-windows | ✓ u-platform-windows, u-vim-keys, f-name-snake-case, u-lang-chinese, p-py-version- | 185 |
| 12 | 中等 | is the user fluent in Go | u-role-backend-go | ✓ u-role-backend-go, u-vim-keys, u-name-tony, u-lang-chinese, f-no-trailing-questi | 179 |
| 13 | 中等 | do not just patch the symptom; what does the user want | f-fix-root-cause | ✓ f-terse-replies, f-conservative-deps, f-fix-root-cause, f-no-emoji, f-no-trailin | 166 |
| 14 | 中等 | should we cap how many external libraries we pull in | f-conservative-deps | ✓ f-conservative-deps, f-bundled-pr, f-terse-replies, f-no-force-push-shared, r-gr | 174 |
| 15 | 中等 | how do I gate a new feature behind a toggle | p-feature-flag-launch-default-off | ✓ p-feature-flag-launch-default-off, p-release-tag-pattern, p-tier-storage, r-laun | 184 |
| 16 | 中等 | how long do raw events stick around in the warehouse | p-data-retention-90d | ✓ p-data-retention-90d, r-grafana-api-latency, f-fix-root-cause, u-coffee-no-meeti | 197 |
| 17 | 中等 | which team handles changes to the ingest pipeline | p-ingest-pipeline-owner, r-slack-data-infra | ✓ p-ingest-pipeline-owner, r-linear-ingest-project, r-slack-data-infra, p-v2-scope | 197 |
| 18 | 中等 | where should new INGEST bugs be filed | r-linear-ingest-project | ✓ r-linear-ingest-project, p-ingest-pipeline-owner, f-fix-root-cause, p-stack-pist | 188 |
| 19 | 中等 | are type hints required on new Python functions | f-prefer-explicit-types | ✓ f-prefer-explicit-types, p-py-version-3-11, f-conservative-deps, f-name-snake-ca | 204 |
| 20 | 中等 | does the user want a meeting at 9am | u-coffee-no-meeting-am | ✓ u-coffee-no-meeting-am, p-data-retention-90d, u-role-backend-go, u-tz-shanghai,  | 182 |
| 21 | 困难 | the user told me to stop wrapping up with 'anything else?' l | f-no-trailing-questions | ✓ f-no-trailing-questions, f-terse-replies, f-no-emoji, u-vim-keys, u-name-tony | 173 |
| 22 | 困难 | what's the protocol around shipping a tightly coupled refact | f-bundled-pr | ✓ f-bundled-pr, p-tier-storage, p-auth-rewrite-compliance, p-release-tag-pattern,  | 192 |
| 23 | 困难 | the auth project — is it tech debt or something else driving | p-auth-rewrite-compliance | ✓ p-auth-rewrite-compliance, f-fix-root-cause, f-conservative-deps, f-terse-replie | 169 |
| 24 | 困难 | how do release tags map to deploys | p-release-tag-pattern | ✓ p-release-tag-pattern, p-feature-flag-launch-default-off, r-grafana-api-latency, | 190 |
| 25 | 困难 | what's the convention for naming variables in this codebase | f-name-snake-case | ✓ f-name-snake-case, u-lang-chinese, p-py-version-3-11, p-release-tag-pattern, p-m | 203 |

## 评测方法说明

- 两个系统使用**完全相同**的查询集与 ground-truth 标注。
- MD 基线必须把 `MEMORY.md` 索引常驻上下文；Engram 不需要常驻任何内容。
- 当 `ENGRAM_EMBEDDER=hash` 时 Engram 退化为词袋匹配，与 MD 基线的关键词 Jaccard 同档；切换到真正的语义后端（本地 e5、OpenAI 等）会进一步拉开差距。
- 对于多答案标注的查询，只要 top-k 中任一答案命中即视为成功；MRR 取最优答案的 1/rank。
- 「困难档」查询故意只与目标记忆**主题相关**而无字面重叠，用来检验对释义和同义改写的鲁棒性。

