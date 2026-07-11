# Changelog

所有显著变更都记录在此文件。本项目遵循 [Semantic Versioning](https://semver.org/lang/zh-CN/)。

## [Unreleased]

## [0.1.0] - 2026-07-11

### Added

- **Chunk-level hybrid retrieval**：文档按 Markdown 结构切分为带稳定 ID 的 chunk，结合 BM25 关键词和 OpenAI-compatible Embedding 混合检索。
- **Markdown-aware chunking**：自动剥离 YAML frontmatter 和前置元数据表，优先按标题/段落/列表边界切分，超长块使用滑动窗口。
- **SQLite ChunkStore**：`chunks.sqlite3` 作为 chunk 元数据与内容的权威存储，支持按文档增删改生命周期管理。
- **Chroma vector index**：持久化向量索引，collection 名称和元数据记录 embedding 模型与维度，避免不兼容向量复用。
- **BM25-style keyword index**：针对 chunk 的中文 unigram/bigram、英文/数字/URL 分词，支持标题加权、短语加权和覆盖度奖励。
- **混合打分与可靠性门槛**：`final_score = 0.5 * vector_relevance + 0.5 * keyword_score`，同一文档限制返回 chunk 数量，合并相邻 chunk；最终结果需同时满足分数阈值和证据门槛才视为可靠。
- **管理员调试命令**：`/nju_search` 输出查询拆分、关键词候选、向量候选、合并分数与阈值。
- **chunk 大小配置**：`chunk_size`（默认 1200）和 `chunk_overlap`（默认 180）。

### Changed

- 检索结果以 chunk 为单位引用来源，替代整篇文档引用，提升答案精确度。
- Agent 回答策略收紧：涉及校园事实的问题必须基于可靠来源；来源不可靠时统一返回“知识库中暂未找到可靠资料”，禁止用模型常识补充具体流程、链接或联系方式。
- Grounding 材料从整篇 `document.body` 改为 chunk 片段，减少无关内容干扰。

### Fixed

- 修复 `search_knowledge_base` 工具因 `ChunkResult` 缺少 `slug` 字段导致的 `AttributeError`。
- 修复 `read_doc` / `get_doc_details` 因 `file_path` 已包含 `docs_root` 前缀而解析失败的问题。
- 修复 Agent 在无可靠资料时可能给出不一致的通用回答的问题。

### Tests

- 新增 chunk 切分、chunk 生命周期、混合检索、模型/维度一致性、批量失败可见性等端到端测试。
- 现有测试更新为适配 chunk store 和可靠来源判断。
