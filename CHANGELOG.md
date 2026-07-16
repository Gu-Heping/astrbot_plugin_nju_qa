# Changelog

所有显著变更都记录在此文件。本项目遵循 [Semantic Versioning](https://semver.org/lang/zh-CN/)。

## [Unreleased]

### Added

- 证据优先两阶段 Agent：研究阶段调用工具收集证据，回答阶段只能使用实际读取的证据片段，并用 `[E#]` 内部标记绑定引用。
- `EvidenceExcerpt` 统一证据模型，记录来源、行号、内容、QA 状态、历史资料标记和分数。
- `grep_local_docs` 新增 `required_phrases` 参数，可强制命中片段同时包含指定短语。
- `grep_local_docs` 返回 `recommended_read_range`，提示模型按行号精读相关上下文。
- `read_doc`、`get_doc_details`、`get_doc_outline` 及结构导航工具将结果记录为证据片段。
- Agent 支持近似名称处理：精确搜索结果不足时提取稳定关键词搜索标题/路径/正文，读取正文验证后仅在证据明确指向同一对象时说明正式名称对应关系。
- 启动时日志记录插件版本号与当前 git commit SHA。
- 新增群聊白名单：启用 `enable_group_whitelist` 后仅响应 `group_whitelist` 中的群 ID；白名单列表为空时拒绝所有群聊。
- `enable_private_chat` 统一控制所有私聊入口，关闭后私聊的 `/nju`、`/nju_grep` 等命令均静默，不调用 Agent、检索或限流。

### Changed

- 重构 `nju_qa/agent.py`：移除规则驱动的状态机，改为 `NjuQaAgent` 两阶段调度；研究阶段忽略模型生成的自然语言，仅收集工具证据。
- 移除 `QueryEvidenceMode`、`RetrievalPlan`、`RetrievalExecutor`、`CoverageStatus`、Python 正则子问题拆分和 `core_terms` 证据门控。
- 简化实体处理：不再依赖复杂正则实体提取作为回答前提，Agent 通过工具自行定位相关文档。
- `search_knowledge_base` 仅注册候选来源；最终回答必须基于 `read_doc` 等工具产生的证据。
- 更新系统提示词为研究/回答/寒暄三份独立提示，明确证据约束和引用规则。
- 版本号提升至 0.3.0。

### Removed

- 删除 `nju_qa/entities.py`、`nju_qa/retrieval_plan.py`、`nju_qa/retrieval_executor.py` 及其对应测试文件。

### Added

- `/nju_grep`：全文搜索本地 Markdown，长中文查询未命中时自动按二字切分兜底。
- `/nju_debug`：管理员命令，用于查看 AstrBot 对 `/nju` 命令的解析结果。
- 后台任务启动反馈：`/nju_sync`、`/nju_index rebuild` 立即返回“已启动后台...”提示，避免用户误以为无响应。
- Agent 关键步骤日志：provider、检索源数量、可靠源数量、grounding 源数量、引用数量。
- 输出 Markdown 转纯文本：所有命令和普通消息回复在发送前都会去掉 Markdown 标记（标题、加粗、代码块、图片、链接、表格等），适配 QQ 个人号聊天页的纯文本显示。
- Agent 自我介绍中自然包含“由 NOVA 开发”说明，用于社团宣传且不干扰正常问答。
- `grep_local_docs` 改为逐行扫描本地 Markdown，返回匹配行号与上下文片段；`read_doc` 新增 `start_line`/`end_line` 参数，可按行号精确读取。
- 新增 `table_font_path` 配置：可显式指定表格图片使用的中文字体文件。
- 新增 `auto_download_table_font` 配置：未找到可用系统字体时，自动下载开源 Noto Sans CJK SC 字体到插件数据目录，避免表格图片出现乱码；下载失败时仍自动回退为纯文本表格。
- 新增 `table_font_download_timeout` 配置：控制字体自动下载超时，默认 30 秒。
- 字体下载优先使用 jsDelivr 镜像，失败后再尝试 Gitee/GitHub，减少大陆网络环境下的下载卡顿。
- 新增按聊天上下文限流：`group_rate_limit` / `private_rate_limit` 及对应窗口配置，限制 `/nju`、`/nju_grep` 和普通消息触发的回答频率；群聊达到上限后首次鼓励私聊提问，私聊达到上限后首次提示稍后再试，之后同一窗口内不再重复回复。管理员命令不受限流影响。
- 新增 `render_tables_as_images` 配置：将回答中的 Markdown 表格渲染为 PNG 图片插入回复，默认开启。
- 新增路由安全回归测试，覆盖未知 slash 抑制、跨插件命令放行、自身命令匹配、@/唤醒词/私聊入口等场景。

### Changed

- 同步阶段即调用 `clean_document_body` 清洗 Yuque HTML 标签、Markdown 图片等噪声，chunk 索引和本地文件均基于干净文本，行号与 grep 结果一致。
- Agent 提示词明确检索优先级：具体事实问题优先使用 `grep_local_docs` 并按行号精读，向量检索仅作为兜底；同时加强对 QA 知识库的利用。
- 针对“详细整理/汇总/全面介绍”类问题，Agent 必须拆分为 2-4 个子主题分别检索，避免单次检索覆盖不全。
- 引用来源限制为最多 5 条，减少答案被大量来源淹没的情况。
- Grounding 材料来源上限从 5 提升到 7，优先保证事实性问题有足够依据。
- 向量检索与关键词检索改为并发执行，降低 `search_knowledge_base` 的整体耗时。

### Fixed

- 修复未知 `/...` 指令会回退到 AstrBot 默认 LLM 的问题：`on_message` 现在通过 `handlers_parsed_params` 和 `activated_handlers` 检测是否已匹配注册命令；对未匹配的 `/...` 调用 `event.should_call_llm(True)` 阻止默认 LLM，同时不 `stop_event`，保证其他插件的 ALL-message handler 仍能处理。
- 修复部分 AstrBot 实例只发出命令 handler 第一个 `yield` 导致最终答案丢失的问题；`/nju`、`/nju_grep`、`/nju_search` 等改为单次 yield 返回。
- 修复 `read_doc` / `get_doc_details` 因数据库中 `path` 列已包含 `docs_root` 前缀而报 `invalid document path` 的问题，并统一 `DocumentStore` 后续写入使用相对 `docs_root` 的路径。
- 修复 `grep_local_docs` 对长中文关键词无法命中时的兜底检索逻辑。
- 修复 `search_knowledge_base` 返回片段和 `read_doc` / `get_doc_details` 正文中残留 Yuque HTML 标签、Markdown 图片、颜色字体等未清洗内容的问题。
- 修复与其他插件（如 `astrbot_plugin_nju_qq_audit`）的 `/` 指令冲突：本插件的 `on_message` 现在会检查原始消息文本，若真实内容以 `/` 开头则不再当作普通提问处理。
- 修复所有命令 handler 在部分 AstrBot 环境下会响应任意 / 指令的问题：每个 handler 现在先校验 event.message_str 是否以自身命令名开头，不匹配时直接返回，不再占用事件或输出内容。
- 修复 `on_message` 在部分 AstrBot 环境下无法识别其他插件 `/` 指令的问题：增加 `event.is_command` 检查，任何已被识别为命令的消息都不会再进入普通问答流程。
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
