"""
AI 工作流示例：多角色协作的测验生成器。
运行方式：python ai_workflow_demo.py
"""
from __future__ import annotations

import json
import os
import random
from typing import Any, Dict, List, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from langchain_deepseek import ChatDeepSeek


# ===================== 1) 状态定义：工作流各节点共享的数据 =====================
class QuizState(TypedDict):
    topic: str
    difficulty: str
    angle: str
    question: str
    options: List[str]
    answer: str
    explanation: str
    user_answer: str
    score: int
    retries: int
    qa_decision: str
    trace: List[str]


# ===================== 2) 统一的透明化日志输出 =====================
def trace(state: QuizState, text: str) -> None:
    state["trace"].append(text)
    print(text)


# ===================== 3) LLM 初始化（无 Key 时用简易兜底） =====================
class FallbackLLM:
    """没有 API Key 时的兜底生成器，让流程可运行。"""

    def invoke(self, messages: List[Any]) -> AIMessage:
        _ = messages
        sample = {
            "question": "哪一种动物以黑白相间的条纹著称？",
            "options": ["斑马", "企鹅", "熊猫", "海豚"],
            "answer": "A",
            "explanation": "斑马最典型的特征就是黑白条纹。",
        }
        return AIMessage(content=json.dumps(sample, ensure_ascii=False))


def get_llm() -> Any:
    load_dotenv()
    api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        return FallbackLLM()
    return ChatDeepSeek(model="deepseek-v4-pro", temperature=0.3, api_key=api_key)


LLM = get_llm()


# ===================== 4) 节点实现：多角色协作 =====================

def planner_node(state: QuizState) -> QuizState:
    """策划：确定难度与有趣角度。"""
    trace(state, "[Planner] 规划主题与角度...")
    topic = state["topic"]
    prompt = (
        "你是测验策划，请给主题一个有趣角度，并决定难度（easy/medium/hard）。\n"
        f"主题：{topic}\n"
        "输出 JSON：{\"angle\":..., \"difficulty\":...}"
    )
    result = LLM.invoke([SystemMessage(content="你是策划。"), HumanMessage(content=prompt)])
    try:
        data = json.loads(result.content)
        state["angle"] = data.get("angle", "冷知识")
        state["difficulty"] = data.get("difficulty", "medium")
    except Exception:
        state["angle"] = "冷知识"
        state["difficulty"] = "medium"
    trace(state, f"[Planner] angle='{state['angle']}', difficulty='{state['difficulty']}'")
    return state


def writer_node(state: QuizState) -> QuizState:
    """写手：生成选择题。"""
    trace(state, "[Writer] 生成题目...")
    prompt = (
        "你是出题人，请基于主题和角度出 1 道四选一选择题。\n"
        f"主题：{state['topic']}\n"
        f"角度：{state['angle']}\n"
        f"难度：{state['difficulty']}\n"
        "要求输出 JSON："
        "{\"question\":..., \"options\":[...], \"answer\":\"A/B/C/D\", \"explanation\":...}"
    )
    result = LLM.invoke([SystemMessage(content="你是出题人。"), HumanMessage(content=prompt)])
    try:
        data = json.loads(result.content)
        state["question"] = data["question"]
        state["options"] = data["options"]
        state["answer"] = data["answer"].strip().upper()
        state["explanation"] = data["explanation"]
    except Exception:
        state["question"] = "请问太阳系中离太阳最近的行星是？"
        state["options"] = ["水星", "金星", "地球", "火星"]
        state["answer"] = "A"
        state["explanation"] = "水星离太阳最近。"
    return state


def quality_gate_node(state: QuizState) -> QuizState:
    """质检：判断题目是否合格，不合格则重试。"""
    trace(state, "[QA] 质量检查...")
    if len(state["options"]) != 4 or state["answer"] not in ("A", "B", "C", "D"):
        state["retries"] += 1
        trace(state, "[QA] 结构不合格，触发重试。")
        state["qa_decision"] = "retry"
        return state
    if state["retries"] >= 2:
        trace(state, "[QA] 达到重试上限，强制通过。")
        state["qa_decision"] = "pass"
        return state
    trace(state, "[QA] 通过。")
    state["qa_decision"] = "pass"
    return state


def ask_user_node(state: QuizState) -> QuizState:
    """主持：向用户提问并收集答案。"""
    trace(state, "[Host] 开始提问。")
    letters = ["A", "B", "C", "D"]
    print("\n题目：", state["question"])
    for i, opt in enumerate(state["options"]):
        print(f"  {letters[i]}. {opt}")
    user_answer = input("\n你的答案（A/B/C/D）：").strip().upper() or "A"
    state["user_answer"] = user_answer
    return state


def grader_node(state: QuizState) -> QuizState:
    """评审：判分并给出反馈。"""
    trace(state, "[Grader] 判分中...")
    if state["user_answer"] == state["answer"]:
        state["score"] = 1
    else:
        state["score"] = 0
    return state


def coach_node(state: QuizState) -> QuizState:
    """教练：给出简短点评与解释。"""
    trace(state, "[Coach] 给出点评...")
    if state["score"] == 1:
        print("\n✅ 回答正确！")
    else:
        print("\n❌ 回答错误。")
    print("解释：", state["explanation"])
    return state


# ===================== 5) 构建工作流图（节点 + 分支） =====================
workflow = StateGraph(QuizState)
workflow.add_node("planner", planner_node)
workflow.add_node("writer", writer_node)
workflow.add_node("quality_gate", quality_gate_node)
workflow.add_node("ask_user", ask_user_node)
workflow.add_node("grader", grader_node)
workflow.add_node("coach", coach_node)

workflow.set_entry_point("planner")
workflow.add_edge("planner", "writer")
workflow.add_edge("writer", "quality_gate")
workflow.add_conditional_edges(
    "quality_gate",
    lambda state: state["qa_decision"],
    {"retry": "writer", "pass": "ask_user"},
)
workflow.add_edge("ask_user", "grader")
workflow.add_edge("grader", "coach")
workflow.add_edge("coach", END)

app = workflow.compile()


# ===================== 6) 运行入口 =====================
if __name__ == "__main__":
    # 初始状态
    topic = input("请输入测验主题（例如：太空/美食/电影）：").strip() or "太空"
    initial_state: QuizState = {
        "topic": topic,
        "difficulty": "medium",
        "angle": "",
        "question": "",
        "options": [],
        "answer": "A",
        "explanation": "",
        "user_answer": "",
        "score": 0,
        "retries": 0,
        "qa_decision": "pass",
        "trace": [],
    }

    print("\n=== AI Workflow: 多角色测验生成器 ===")
    result = app.invoke(initial_state)

    print("\n=== 执行日志 ===")
    for line in result["trace"]:
        print(line)

    print("\n最终得分：", result["score"])
