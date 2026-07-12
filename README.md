# NJU QA

南京大学知识库问答助手：把**配置中明确指定的**语雀知识库同步到插件数据目录，按 Markdown 结构切分成带稳定 ID 的 chunk，使用本地 BM25 关键词和可选的 OpenAI-compatible Embedding 进行 chunk-level 混合检索，并为每个回答附上实际检索到的语雀来源。

> 本项目为非官方开源项目，与南京大学官方无隶属或授权关系。具体政策以南京大学官方最新通知为准。

## 范围与兼容性

需要 AstrBot `>=4.16,<5`。v0.1 支持私聊、群聊 @/唤醒词、指定知识库全量同步、SQLite 元数据与 chunk 索引、来源检索和管理员命令。不会同步 Token 可见的全部知识库。

不包含用户绑定、画像、记忆、学习/社区功能、自动推送、Git 操作、群聊旁听或 Webhook。Webhook 已评估为后续工作；v0.1 不启动任何 HTTP 服务，也没有相关配置项。

## 安装与配置

在 AstrBot 插件目录安装本仓库，安装 `requirements.txt`，重载插件后在 WebUI 填写配置。`yuque_repositories` 使用 namespace 字符串或对象列表，例如：

```json
[
  {"namespace": "nju/student-guide", "name": "学生指南"},
  "nju/academic-affairs"
]
```

namespace 是语雀文档 URL `https://www.yuque.com/<namespace>/<slug>` 中的 `<namespace>`；请在同步前确认 Token 对其有读取权限。

| 配置 | 必填 | 说明 |
| --- | --- | --- |
| `yuque_token` | 是 | 仅用于 API 读取，勿提交到仓库。 |
| `yuque_base_url` | 否 | 默认 `https://www.yuque.com/api/v2`。 |
| `yuque_repositories` | 是 | 仅这些指定知识库会被同步。 |
| `embedding_api_key` / `embedding_base_url` / `embedding_model` | 否 | OpenAI-compatible Embedding；为空时仍可关键词检索。 |
| `enable_vector_search` | 否 | 默认 `true`；设为 `false` 可关闭联网向量检索，完全使用本地 grep/关键词，响应更快。 |
| `chunk_size` | 否 | 单个 chunk 目标字符数，默认 `1200`，最小 `200`。 |
| `chunk_overlap` | 否 | 相邻 chunk 重叠字符数，默认 `180`。 |
| `wake_words` | 否 | 默认 `南大助手,南小答,nju`。 |
| `enable_private_chat` / `enable_group_at` | 否 | 私聊、群聊显式触发开关。 |
| `retrieval_top_k` / `score_threshold` | 否 | 检索数量和阈值。 |
| `group_rate_limit` | 否 | 群聊每小时最多响应次数，默认 `30`，`0` 表示不限。 |
| `group_rate_limit_window` | 否 | 群聊限流窗口秒数，默认 `3600`（1 小时）。 |
| `private_rate_limit` | 否 | 私聊每小时最多响应次数，默认 `20`，`0` 表示不限。 |
| `private_rate_limit_window` | 否 | 私聊限流窗口秒数，默认 `3600`（1 小时）。 |
| `render_tables_as_images` | 否 | 将回答中的 Markdown 表格渲染为图片插入回复，默认 `true`。需要系统装有中文字体。 |

首次使用：配置完成后执行 `/nju_sync`，待完成后使用 `/nju_sync status`。同步会自动切分文档并建立可选的向量索引。

## 命令

- `/nju <问题>`：提问；回答只会引用实际检索到的 chunk 来源。
- `/nju help`：帮助。
- `/nju source <关键词>`：查看相关来源。
- `/nju_grep <关键词>`：全文搜索本地 Markdown，长中文词会自动按二字切分兜底。
- `/nju_sync`：管理员启动后台全量同步；启动后会立即返回状态提示。
- `/nju_sync status`：管理员查看同步与 chunk/向量索引状态。
- `/nju_index rebuild`：管理员重建 chunk 向量索引；启动后会立即返回状态提示。
- `/nju_search <关键词>`：管理员查看混合检索调试信息（候选、分数、阈值）。
- `/nju_debug`：管理员查看 AstrBot 对 `/nju` 命令的解析结果（用于排查命令路由）。

当材料不足时，机器人会明确回复“知识库中暂未找到可靠资料”。高风险或易变信息应以来源的更新时间和南京大学官方最新通知为准。

`/nju`、`/nju_grep` 和普通消息触发的回答会按聊天上下文限流：群聊超过配置次数后首次提示大家私聊提问，之后在同一限流窗口内不再回复；私聊超过次数后首次提示稍后再试，之后不再回复。管理员命令不受限流影响。

当回答包含 Markdown 表格时，默认会将表格渲染为一张 PNG 图片并插入到回复中；无表格时仍按纯文本输出。可在配置中关闭 `render_tables_as_images`。

## 检索架构

- **Chunk 切分**：Markdown 正文剥离 YAML frontmatter 和前置元数据表后，按标题/段落/列表边界切分；超长块使用滑动窗口。每个 chunk 有稳定 ID 和完整元数据。
- **BM25 关键词索引**：针对 chunk 做中文 unigram/bigram、英文/数字/URL 分词，支持标题加权、短语加权和覆盖度奖励。
- **向量索引**：使用 Chroma 持久化存储，collection 名称和元数据记录 embedding 模型与维度，避免不兼容向量复用。
- **混合打分**：`final_score = 0.5 * vector_relevance + 0.5 * keyword_score`，并限制同一文档返回 chunk 数量，合并相邻 chunk。
- **可靠性门槛**：最终分数必须超过阈值，且有关键词命中或强向量相似度，才视为可靠来源。

## 数据与隐私

AstrBot 插件数据目录下保存：

- `documents/`：Markdown 源文件。
- `nju_qa.sqlite3`：文档元数据、状态及旧版文档向量。
- `chunks.sqlite3`：chunk 元数据与内容的权威存储。
- `vectors/`：Chroma 持久化向量索引。

Markdown frontmatter 与 chunk 元数据包含语雀文档 ID、标题、知识库名、namespace、slug、原文 URL、创建/更新时间。数据、密钥、SQLite WAL/SHM 和缓存均被 `.gitignore` 排除。

## 与旧项目的关系

旧的语雀社团助手仓库仅作为实现参考，不是本项目的依赖或运行数据来源。详见 [MIGRATION.md](MIGRATION.md)：本项目重写了可靠的同步/检索闭环，并移除了社团运营和个人成长功能。

## 开发与排查

```powershell
python -m pip install -r requirements.txt
python -m pytest -q
python -m ruff check .
```

若同步失败，确认 Token、指定 namespace、网络访问和语雀权限；429/5xx 会有限次数退避重试。Embedding 失败时可先留空其三项配置，关键词检索仍可工作。单篇文档失败会计入结果，不会让整次同步静默失败。

## 更新日志

见 [CHANGELOG.md](CHANGELOG.md)。
