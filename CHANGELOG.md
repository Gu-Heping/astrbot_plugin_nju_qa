# Changelog

所有显著变更都记录在此文件。本项目遵循 [Semantic Versioning](https://semver.org/lang/zh-CN/)。

## [Unreleased]

### Added

- `/nju_grep`：全文搜索本地 Markdown，长中文查询未命中时自动按二字切分兜底。
- `/nju_debug`：管理员命令，用于查看 AstrBot 对 `/nju` 命令的解析结果。
- 后台任务启动反馈：`/nju_sync`、`/nju_index rebuild` 立即返回“已启动后台...”提示，避免用户误以为无响应。
- Agent 关键步骤日志：provider、检索源数量、可靠源数量、grounding 源数量、引用数量。
- 输出 Markdown 转纯文本：所有命令和普通消息回复在发送前都会去掉 Markdown 标记（标题、加粗、代码块、图片、链接、表格等），适配 QQ 个人号聊天页的纯文本显示。
- Agent 自我介绍中自然包含“由 NOVA 开发”说明，用于社团宣传且不干扰正常问答。
- `grep_local_docs` 改为逐行扫描本地 Markdown，返回匹配行号与上下文片段；`read_doc` 新增 `start_line`/`end_line` 参数，可按行号精确读取。
- 新增配置项 `enable_vector_search`：可在 `config.yaml` 中关闭向量检索，完全依赖本地关键词/grep 检索以加快响应。

### Changed

- 同步阶段即调用 `clean_document_body` 清洗 Yuque HTML 标签、Markdown 图片等噪声，chunk 索引和本地文件均基于干净文本，行号与 grep 结果一致。
- Agent 提示词明确检索优先级：具体事实问题优先使用 `grep_local_docs` 并按行号精读，向量检索仅作为兜底；同时加强对 QA 知识库的利用。
- 针对“详细整理/汇总/全面介绍”类问题，Agent 必须拆分为 2-4 个子主题分别检索，避免单次检索覆盖不全。
- 引用来源限制为最多 5 条，减少答案被大量来源淹没的情况。
- Grounding 材料来源上限从 5 提升到 7，优先保证事实性问题有足够依据。
- 向量检索与关键词检索改为并发执行，降低 `search_knowledge_base` 的整体耗时。

### Fixed

- 修复部分 AstrBot 实例只发出命令 handler 第一个 `yield` 导致最终答案丢失的问题；`/nju`、`/nju_grep`、`/nju_search` 等改为单次 yield 返回。
- 修复 `read_doc` / `get_doc_details` 因数据库中 `path` 列已包含 `docs_root` 前缀而报 `invalid document path` 的问题，并统一 `DocumentStore` 后续写入使用相对 `docs_root` 的路径。
- 修复 `grep_local_docs` 对长中文关键词无法命中时的兜底检索逻辑。
- 修复 `search_knowledge_base` 返回片段和 `read_doc` / `get_doc_details` 正文中残留 Yuque HTML 标签、Markdown 图片、颜色字体等未清洗内容的问题。
- 修复与其他插件（如 `astrbot_plugin_nju_qq_audit`）的 `/` 指令冲突：本插件的 `on_message` 现在会检查原始消息文本，若真实内容以 `/` 开头则不再当作普通提问处理。
- 修复 `main.py` 中 `_plain()` 辅助函数因批量替换而递归调用自身导致的 `RecursionError`。
- 修复 `DocumentStore.path_for()` 改为返回相对路径后，重复文档名去重失效的问题。

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
