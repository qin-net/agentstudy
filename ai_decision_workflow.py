"""
AI 工作流示例：AI 决策分支的互动侦探故事。
运行方式：python ai_decision_workflow.py
"""
from __future__ import annotations

import json
import os
from typing import Any, List, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from langchain_deepseek import ChatDeepSeek


# ===================== 1) 状态定义 =====================
class StoryState(TypedDict):
    seed: str
    task_type: str
    clue: str
    suspect: str
    twist: str
    scene: str
    ai_route: str
    user_guess: str
    verdict: str
    score: int
    trace: List[str]


# ===================== 2) 透明化日志 =====================
def trace(state: StoryState, text: str) -> None:
    state["trace"].append(text)
    print(text)


# ===================== 3) LLM 初始化（无 Key 时直接报错） =====================
def get_llm() -> Any:
    load_dotenv()
    api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("未设置 DEEPSEEK_API_KEY，无法调用 LLM。")
    return ChatDeepSeek(model="deepseek-v4-pro", temperature=0.4, api_key=api_key)


LLM = get_llm()


# ===================== 4) 节点实现 =====================

def _invoke_llm(state: StoryState, messages: List[Any], context: str) -> AIMessage:
    try:
        return LLM.invoke(messages)
    except Exception as exc:
        trace(state, f"[Error] {context} 调用失败：{exc}")
        raise


def _parse_json(state: StoryState, content: str, context: str) -> dict:
    raw = content.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.startswith("json"):
            raw = raw[4:].lstrip()
    if raw and raw[0] != "{":
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            raw = raw[start : end + 1]
    try:
        return json.loads(raw)
    except Exception as exc:
        trace(state, f"[Error] {context} JSON 解析失败：{exc}; 原始输出：{content}")
        raise

def _build_scene(state: StoryState, role: str, genre_hint: str) -> StoryState:
    trace(state, f"[Scene] 生成开场（{state['task_type']}）...")
    prompt = (
        f"你是{role}，请生成一个{genre_hint}的开场与线索。\n"
        f"灵感词：{state['seed']}\n"
        "输出 JSON：{\"scene\":..., \"clue\":..., \"suspect\":...}"
    )
    result = _invoke_llm(
        state,
        [SystemMessage(content="你是故事引擎。"), HumanMessage(content=prompt)],
        "场景生成",
    )
    data = _parse_json(state, result.content, "场景生成")
    state["scene"] = data["scene"]
    state["clue"] = data["clue"]
    suspect_value = data["suspect"]
    if isinstance(suspect_value, list):
        state["suspect"] = "；".join(str(item) for item in suspect_value)
    else:
        state["suspect"] = str(suspect_value)
    return state


def scene_detective(state: StoryState) -> StoryState:
    """侦探分支场景。"""
    return _build_scene(state, "侦探小说开场生成器", "经典推理风格")


def scene_heist(state: StoryState) -> StoryState:
    """盗窃分支场景。"""
    return _build_scene(state, "盗窃题材编剧", "高智商盗窃风格")


def scene_paranormal(state: StoryState) -> StoryState:
    """超自然分支场景。"""
    return _build_scene(state, "超自然故事叙述者", "诡异但不恐怖")


def scene_tech(state: StoryState) -> StoryState:
    """科技分支场景。"""
    return _build_scene(state, "科技悬疑编剧", "高科技谜案风格")


def task_router_node(state: StoryState) -> StoryState:
    """AI 路由：根据灵感词判断任务类型。"""
    trace(state, "[Router] 识别任务类型...")
    prompt = (
        "你是任务分发器，请根据灵感词判断故事类型。\n"
        "可选类型：detective / heist / paranormal / tech\n"
        f"灵感词：{state['seed']}\n"
        "只输出 JSON：{\"task_type\":\"detective|heist|paranormal|tech\"}"
    )
    result = _invoke_llm(
        state,
        [SystemMessage(content="你是任务分发器。"), HumanMessage(content=prompt)],
        "任务路由",
    )
    data = _parse_json(state, result.content, "任务路由")
    task_type = str(data.get("task_type", "detective")).strip().lower()
    if task_type not in ("detective", "heist", "paranormal", "tech"):
        task_type = "detective"
    state["task_type"] = task_type
    trace(state, f"[Router] task_type='{task_type}'")
    return state


def ask_guess_node(state: StoryState) -> StoryState:
    """向用户抛出问题，收集猜测。"""
    trace(state, "[Host] 抛出问题并收集猜测...")
    print("\n=== 案件开场 ===")
    print(state["scene"])
    print("线索：", state["clue"])
    print("\n问题：你认为谁是嫌疑人？给出一句理由。")
    guess = input("你的猜测：").strip()
    state["user_guess"] = guess if guess else "（未作答）"
    return state


def ai_judge_node(state: StoryState) -> StoryState:
    """AI 评判用户回答并打分。"""
    trace(state, "[Judge] 评估用户回答...")
    prompt = (
        "你是侦探评审，请根据场景、线索和嫌疑人信息评价用户猜测。\n"
        "评分范围 0-10，简短评语。\n"
        f"场景：{state['scene']}\n"
        f"线索：{state['clue']}\n"
        f"嫌疑人：{state['suspect']}\n"
        f"用户猜测：{state['user_guess']}\n"
        "只输出 JSON：{\"score\":0-10, \"verdict\":...}"
    )
    result = _invoke_llm(
        state,
        [SystemMessage(content="你是公平的评审。"), HumanMessage(content=prompt)],
        "评审打分",
    )
    data = _parse_json(state, result.content, "评审打分")
    score = int(data.get("score", 5))
    verdict = str(data.get("verdict", "推理有亮点，但证据不足。")).strip()
    score = max(0, min(10, score))
    state["score"] = score
    state["verdict"] = verdict
    trace(state, f"[Judge] score={score}, verdict='{verdict}'")
    return state


def ai_decision_gate(state: StoryState) -> StoryState:
    """AI 决策：根据线索决定下一步调查方向。"""
    trace(state, "[AI-Gate] 决策下一步...")
    prompt = (
        "你是侦探助手，请根据场景和线索选择下一步行动。\n"
        "可选路线：interrogate(审问嫌疑人) 或 analyze(分析线索)。\n"
        f"场景：{state['scene']}\n"
        f"线索：{state['clue']}\n"
        "只输出 JSON：{\"route\":\"interrogate|analyze\"}"
    )
    result = _invoke_llm(
        state,
        [SystemMessage(content="你是冷静的侦探助手。"), HumanMessage(content=prompt)],
        "路线决策",
    )
    data = _parse_json(state, result.content, "路线决策")
    route = str(data.get("route", "analyze")).strip().lower()
    if route not in ("interrogate", "analyze"):
        route = "analyze"
    state["ai_route"] = route
    trace(state, f"[AI-Gate] route='{route}'")
    return state


def interrogate_branch(state: StoryState) -> StoryState:
    """审问分支：从嫌疑人处获取转折。"""
    trace(state, "[Branch] 审问嫌疑人...")
    prompt = (
        "你是侦探，请给出一个戏剧性转折。\n"
        f"嫌疑人：{state['suspect']}\n"
        "输出一句话转折："
    )
    result = _invoke_llm(
        state,
        [SystemMessage(content="你是侦探。"), HumanMessage(content=prompt)],
        "审问分支",
    )
    state["twist"] = (result.content or "嫌疑人承认是被人胁迫。").strip()
    return state


def analyze_branch(state: StoryState) -> StoryState:
    """分析分支：从线索中推断幕后目标。"""
    trace(state, "[Branch] 分析线索...")
    prompt = (
        "你是法证专家，请基于线索给出推断。\n"
        f"线索：{state['clue']}\n"
        "输出一句话推断："
    )
    result = _invoke_llm(
        state,
        [SystemMessage(content="你是法证专家。"), HumanMessage(content=prompt)],
        "分析分支",
    )
    state["twist"] = (result.content or "线索指向内部人士作案。").strip()
    return state


def finale_node(state: StoryState) -> StoryState:
    """结局输出：汇总剧情。"""
    trace(state, "[Finale] 输出结局...")
    trace(state, f"[Finale] 评分结果 score={state['score']}, verdict='{state['verdict']}'")
    print("\n=== 侦探结案（分步骤） ===")
    print("步骤 1/5：回顾案发现场")
    print("  - ", state["scene"])
    print("步骤 2/5：锁定关键线索")
    print("  - ", state["clue"])
    print("步骤 3/5：确认嫌疑人")
    print("  - ", state["suspect"])
    print("步骤 4/5：揭示转折")
    print("  - ", state["twist"])
    print("步骤 5/5：记录调查路线")
    print("  - ", state["ai_route"])
    print("\n=== 用户猜测评分 ===")
    print("评分：", state["score"], "/ 10")
    print("评语：", state["verdict"])
    print("答案：", state["suspect"])
    return state


# ===================== 5) 构建工作流图 =====================
workflow = StateGraph(StoryState)
workflow.add_node("router", task_router_node)
workflow.add_node("scene_detective", scene_detective)
workflow.add_node("scene_heist", scene_heist)
workflow.add_node("scene_paranormal", scene_paranormal)
workflow.add_node("scene_tech", scene_tech)
workflow.add_node("ask_guess", ask_guess_node)
workflow.add_node("ai_judge", ai_judge_node)
workflow.add_node("ai_gate", ai_decision_gate)
workflow.add_node("interrogate", interrogate_branch)
workflow.add_node("analyze", analyze_branch)
workflow.add_node("finale", finale_node)

workflow.set_entry_point("router")
workflow.add_conditional_edges(
    "router",
    lambda state: state["task_type"],
    {
        "detective": "scene_detective",
        "heist": "scene_heist",
        "paranormal": "scene_paranormal",
        "tech": "scene_tech",
    },
)
workflow.add_edge("scene_detective", "ask_guess")
workflow.add_edge("scene_heist", "ask_guess")
workflow.add_edge("scene_paranormal", "ask_guess")
workflow.add_edge("scene_tech", "ask_guess")
workflow.add_edge("ask_guess", "ai_judge")
workflow.add_edge("ai_judge", "ai_gate")
workflow.add_conditional_edges(
    "ai_gate",
    lambda state: state["ai_route"],
    {"interrogate": "interrogate", "analyze": "analyze"},
)
workflow.add_edge("interrogate", "finale")
workflow.add_edge("analyze", "finale")
workflow.add_edge("finale", END)

app = workflow.compile()


# ===================== 6) 运行入口 =====================
if __name__ == "__main__":
    seed = input("请输入故事灵感词（例如：博物馆/地铁/游乐园）：").strip() or "博物馆"
    state: StoryState = {
        "seed": seed,
        "clue": "",
        "suspect": "",
        "twist": "",
        "scene": "",
        "ai_route": "analyze",
        "user_guess": "",
        "verdict": "",
        "score": 0,
        "trace": [],
    }

    print("\n=== AI Workflow: 侦探故事 ===")
    result = app.invoke(state)

    print("\n=== 执行日志 ===")
    for line in result["trace"]:
        print(line)
