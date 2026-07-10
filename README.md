# NJU QA

南京大学知识库问答助手：把**配置中明确指定的**语雀知识库同步到插件数据目录，使用本地关键词和可选的 OpenAI-compatible Embedding 混合检索，并为每个回答附上实际检索到的语雀来源。

> 本项目为非官方开源项目，与南京大学官方无隶属或授权关系。具体政策以南京大学官方最新通知为准。

## 范围与兼容性

需要 AstrBot `>=4.16,<5`。v0.1 支持私聊、群聊 @/唤醒词、指定知识库全量同步、SQLite 元数据与向量索引、来源检索和管理员命令。不会同步 Token 可见的全部知识库。

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
| `wake_words` | 否 | 默认 `南大助手,南小答,nju`。 |
| `enable_private_chat` / `enable_group_at` | 否 | 私聊、群聊显式触发开关。 |
| `retrieval_top_k` / `score_threshold` | 否 | 检索数量和阈值。 |

首次使用：配置完成后执行 `/nju_sync`，待完成后使用 `/nju_sync status`。同步会自动重建可选向量索引。

## 命令

- `/nju <问题>`：提问；回答只会引用实际检索到的文档。
- `/nju help`：帮助。
- `/nju source <关键词>`：查看相关来源。
- `/nju_sync`：管理员启动后台全量同步。
- `/nju_sync status`：管理员查看同步状态。
- `/nju_index rebuild`：管理员重建向量索引。
- `/nju_search <关键词>`：管理员搜索来源。

当材料不足时，机器人会明确回复“知识库中暂未找到可靠答案”。高风险或易变信息应以来源的更新时间和南京大学官方最新通知为准。

## 数据与隐私

AstrBot 插件数据目录下保存 `documents/` 中的 Markdown 和 `nju_qa.sqlite3` 中的元数据、状态及向量。Markdown frontmatter 包含语雀文档 ID、标题、知识库名、namespace、slug、原文 URL、创建/更新时间。数据、密钥、SQLite WAL/SHM 和缓存均被 `.gitignore` 排除。

## 与旧项目的关系

旧的语雀社团助手仓库仅作为实现参考，不是本项目的依赖或运行数据来源。详见 [MIGRATION.md](MIGRATION.md)：本项目重写了可靠的同步/检索闭环，并移除了社团运营和个人成长功能。

## 开发与排查

```powershell
python -m pip install -r requirements.txt
python -m pytest -q
python -m ruff check .
```

若同步失败，确认 Token、指定 namespace、网络访问和语雀权限；429/5xx 会有限次数退避重试。Embedding 失败时可先留空其三项配置，关键词检索仍可工作。单篇文档失败会计入结果，不会让整次同步静默失败。
