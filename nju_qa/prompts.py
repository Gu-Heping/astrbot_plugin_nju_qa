SYSTEM_PROMPT = """你是南京大学校园知识库问答助手。只能根据提供的检索材料回答事实性校园问题，不得把常识当作知识库事实。材料不足时必须仅回答“知识库中暂未找到可靠答案”。招生、学籍、奖助、考试、住宿、医疗、安全等信息可能变化，提醒用户核对文档更新时间和官方最新通知。不要编造来源或链接。"""


def build_prompt(question: str, sources) -> str:
    material = "\n\n".join(
        f"[{s.source_id}] 标题：{s.document.title}\n更新时间：{s.document.updated_at}\n内容：{s.document.body[:3500]}"
        for s in sources
    )
    return f"问题：{question}\n\n检索材料：\n{material}\n\n仅依据材料回答；不要输出引用列表。"
