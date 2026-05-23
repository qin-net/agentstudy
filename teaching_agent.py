"""
============================================================
LangChain 教学版 Agent
适用：零基础 → 理解 Agent 的 思考→调用工具→回答 循环
依赖：pip install langchain langchain-core langchain-deepseek
============================================================
"""
from typing import ClassVar, Dict
import os
import math
import json
import urllib.parse
import urllib.request
import uuid
from datetime import datetime
from typing import Any

from dotenv import load_dotenv

# ===================== 1. 标准 Tool 基类 =====================
load_dotenv()
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field


class StandardTool(BaseTool):
    """
    统一工具基类 —— 所有工具必须继承此类。
    只需要：
      1. 给 name（工具名）
      2. 给 description（用自然语言描述何时调用）
      3. 实现 _run 方法
    注意：description 的写法直接影响 LLM 能否在正确时机选中你的工具！
    """

    def _run(self, **kwargs) -> str:
        raise NotImplementedError("子类必须实现 _run 方法")

    async def _arun(self, **kwargs) -> str:
        return self._run(**kwargs)


# ===================== 2. 现成可用的 10 个工具 =====================

class CalculatorTool(StandardTool):
    """安全计算器 —— 用 eval 但加了安全检查"""
    name: str = "calculator"
    description: str = (
        "数学计算器，支持 + - * / ** sqrt() abs() 等运算。"
        "输入一个数学表达式字符串，例如 '2 + 3 * 4' 或 'sqrt(144)'"
    )

    def _run(self, expression: str) -> str:
        allowed = set("0123456789+-*/().% eabsqrt")
        if not set(expression).issubset(allowed):
            return "错误：表达式包含非法字符，仅支持基础数学运算"
        try:
            result = eval(expression, {"__builtins__": {}}, {
                "sqrt": math.sqrt, "abs": abs, "sin": math.sin,
                "cos": math.cos, "pi": math.pi, "e": math.e,
            })
            return f"计算结果：{result}"
        except Exception as e:
            return f"计算出错：{e}"


class DateTimeTool(StandardTool):
    """当前日期时间"""
    name: str = "datetime"
    description: str = "获取当前日期和时间。不需要任何输入参数。"

    def _run(self, query: str = "") -> str:
        now = datetime.now()
        return f"现在是 {now.strftime('%Y年%m月%d日 %H:%M:%S')}，星期{['一','二','三','四','五','六','日'][now.weekday()]}"


class WordCountTool(StandardTool):
    """文本统计"""
    name: str = "word_counter"
    description: str = "统计一段文本的字数、字符数。输入要统计的文本。"

    def _run(self, text: str) -> str:
        chars = len(text)
        words = len(text.split()) if text.strip() else 0
        cn_chars = sum(1 for c in text if '一' <= c <= '鿿')
        return f"总字符数：{chars}，单词数：{words}，中文字数：{cn_chars}"


class DiceTool(StandardTool):
    """掷骰子"""
    name: str = "dice"
    description: str = (
        "掷骰子工具。输入格式如 '2d6' 表示掷2个6面骰子，"
        "或 '1d20' 表示掷1个20面骰子。默认是 1d6。"
    )

    def _run(self, formula: str = "1d6") -> str:
        import random
        try:
            count, sides = formula.lower().split("d")
            count, sides = int(count), int(sides)
            if count > 100 or sides > 1000:
                return "骰子数量或面数太大，最多100个骰子，1000面"
            rolls = [random.randint(1, sides) for _ in range(count)]
            total = sum(rolls)
            detail = " + ".join(str(r) for r in rolls)
            return f"掷{count}个{sides}面骰子：{detail} = {total}"
        except Exception:
            return "格式错误，请使用类似 '2d6' 的格式"


class UnitConvertTool(StandardTool):
    """单位换算"""
    name: str = "unit_converter"
    description: str = (
        "单位换算工具。目前支持：长度(km/m/cm/mm/mile/inch/feet)、"
        "重量(kg/g/mg/lb/oz)、温度(c/f)。"
        "输入格式：'数值 原单位 转 目标单位'，例如 '100 km 转 mile' 或 '32 c 转 f'"
    )

    # 换算表（都以基准单位存储）
    LENGTH_TO_METER: ClassVar[Dict[str, float]] = {
        "km": 1000, "m": 1, "cm": 0.01, "mm": 0.001,
        "mile": 1609.344, "feet": 0.3048, "inch": 0.0254,
    }
    WEIGHT_TO_KG: ClassVar[Dict[str, float]] = {
        "kg": 1, "g": 0.001, "mg": 0.000001,
        "lb": 0.453592, "oz": 0.0283495,
    }

    def _run(self, query: str) -> str:
        try:
            parts = query.strip().split()
            if len(parts) == 4 and parts[2] == "转":
                value, from_unit, _, to_unit = float(parts[0]), parts[1], parts[2], parts[3]
            else:
                value, from_unit, to_unit = float(parts[0]), parts[1], parts[2]

            # 温度特殊处理
            if from_unit in ("c", "f") and to_unit in ("c", "f"):
                if from_unit == "c" and to_unit == "f":
                    result = value * 9 / 5 + 32
                elif from_unit == "f" and to_unit == "c":
                    result = (value - 32) * 5 / 9
                else:
                    result = value
                return f"{value}°{from_unit.upper()} = {result:.2f}°{to_unit.upper()}"

            # 长度
            if from_unit in self.LENGTH_TO_METER and to_unit in self.LENGTH_TO_METER:
                meters = value * self.LENGTH_TO_METER[from_unit]
                result = meters / self.LENGTH_TO_METER[to_unit]
                return f"{value} {from_unit} = {result:.4f} {to_unit}"

            # 重量
            if from_unit in self.WEIGHT_TO_KG and to_unit in self.WEIGHT_TO_KG:
                kg = value * self.WEIGHT_TO_KG[from_unit]
                result = kg / self.WEIGHT_TO_KG[to_unit]
                return f"{value} {from_unit} = {result:.4f} {to_unit}"

            return "不支持的单位换算，支持：长度(km/m/cm/mm/mile/feet/inch)、重量(kg/g/mg/lb/oz)、温度(c/f)"
        except Exception as e:
            return f"换算出错：{e}，正确格式如 '100 km 转 mile'"


class RandomPickerTool(StandardTool):
    """随机选择器"""
    name: str = "random_picker"
    description: str = (
        "从一组选项中随机选择一个或多个。"
        "输入格式：'选项1, 选项2, 选项3 抽N个'，例如 '吃火锅, 吃日料, 吃烧烤 抽1个'"
    )

    def _run(self, query: str) -> str:
        import random
        try:
            if "抽" in query:
                items_str, count_str = query.rsplit("抽", 1)
                count = int(count_str.replace("个", "").strip())
            else:
                items_str, count = query, 1
            items = [x.strip() for x in items_str.split(",") if x.strip()]
            if count > len(items):
                count = len(items)
            picked = random.sample(items, count)
            return f"从{len(items)}个选项中随机选出：{'、'.join(picked)}"
        except Exception as e:
            return f"随机选择出错：{e}"


class PasswordGeneratorTool(StandardTool):
    """随机密码生成"""
    name: str = "password_generator"
    description: str = (
        "生成随机密码。输入密码长度（默认16位），例如 '12' 生成12位密码。"
        "最大支持64位。"
    )

    def _run(self, length: str = "16") -> str:
        import random
        import string
        try:
            n = min(int(length), 64)
        except ValueError:
            n = 16
        chars = string.ascii_letters + string.digits + "!@#$%^&*"
        pwd = "".join(random.choice(chars) for _ in range(n))
        return f"生成{n}位随机密码：{pwd}"


class TodoListTool(StandardTool):
    """简易待办清单（会话内存储）"""
    name: str = "todo_list"
    description: str = (
        "管理待办事项。输入格式："
        "'添加 买菜' 添加事项，'完成 1' 标记第1项完成，'删除 1' 删除第1项，'查看' 显示所有事项。"
    )
    todos: list = []  # 注意：这是类级别共享的，教学演示用

    def _run(self, action: str) -> str:
        parts = action.split(maxsplit=1)
        cmd = parts[0]
        arg = parts[1] if len(parts) > 1 else ""

        if cmd == "添加":
            self.todos.append({"task": arg, "done": False})
            return f"已添加：{arg}（当前共{len(self.todos)}项）"

        elif cmd == "完成":
            try:
                idx = int(arg) - 1
                self.todos[idx]["done"] = True
                return f"已完成：{self.todos[idx]['task']}"
            except (IndexError, ValueError):
                return "序号无效，请用'查看'确认序号"

        elif cmd == "删除":
            try:
                idx = int(arg) - 1
                removed = self.todos.pop(idx)
                return f"已删除：{removed['task']}（剩余{len(self.todos)}项）"
            except (IndexError, ValueError):
                return "序号无效"

        elif cmd == "查看":
            if not self.todos:
                return "待办清单为空"
            lines = []
            for i, t in enumerate(self.todos, 1):
                status = "✓" if t["done"] else "○"
                lines.append(f"  {i}. [{status}] {t['task']}")
            return "待办清单：\n" + "\n".join(lines)

        return "格式错误，支持：添加/完成/删除/查看"


class JsonFormatterTool(StandardTool):
    """JSON 格式化"""
    name: str = "json_formatter"
    description: str = "格式化或压缩 JSON 字符串。输入一个 JSON 字符串，自动美化输出。"

    def _run(self, json_str: str) -> str:
        try:
            data = json.loads(json_str)
            return json.dumps(data, ensure_ascii=False, indent=2)
        except json.JSONDecodeError as e:
            return f"JSON 解析错误：{e}"


class WebSearchTool(StandardTool):
    """网络搜索（Tavily API）"""
    name: str = "web_search"
    description: str = (
        "网络搜索工具（Tavily）。输入搜索关键词，返回简要摘要和前几条相关结果。"
        "需要设置环境变量 TAVILY_API_KEY。"
    )

    def _run(self, query: str) -> str:
        if not query.strip():
            return "请输入要搜索的关键词"

        api_key = os.getenv("TAVILY_API_KEY", "").strip()
        if not api_key:
            return "搜索失败：未设置 TAVILY_API_KEY 环境变量"

        payload = json.dumps(
            {
                "api_key": api_key,
                "query": query,
                "search_depth": "basic",
                "max_results": 5,
                "include_answer": True,
                "include_raw_content": False,
                "include_images": False,
            }
        ).encode("utf-8")

        req = urllib.request.Request(
            "https://api.tavily.com/search",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "User-Agent": "Mozilla/5.0 (compatible; TeachingAgent/1.0)",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode("utf-8"))

            answer = (data.get("answer") or "").strip()
            results = data.get("results", []) or []

            lines = []
            if answer:
                lines.append(f"摘要：{answer}")

            if results:
                lines.append("相关结果：")
                for i, item in enumerate(results[:5], 1):
                    title = (item.get("title") or "").strip()
                    url = (item.get("url") or "").strip()
                    snippet = (item.get("content") or "").strip()
                    text = title or snippet or "(无标题)"
                    if url:
                        lines.append(f"  {i}. {text} - {url}")
                    else:
                        lines.append(f"  {i}. {text}")
            else:
                lines.append("未找到相关结果，建议换个关键词再试。")

            return "\n".join(lines)
        except Exception as e:
            return f"搜索失败：{e}"


# ===================== 3. 🎯 工具区 —— 只改这里！ =====================
# 把你写的工具实例加入这个列表就能跑：
#   - 可以继承 StandardTool 自己写
#   - 也可以用 @tool 装饰器快速创建
#   - 也可以直接放现成的 LangChain 社区工具

tools = [
    CalculatorTool(),
    DateTimeTool(),
    WordCountTool(),
    DiceTool(),
    UnitConvertTool(),
    RandomPickerTool(),
    PasswordGeneratorTool(),
    TodoListTool(),
    JsonFormatterTool(),
    WebSearchTool(),
]

# ===================== 4. Agent 核心（不需要改动） =====================
from langchain_deepseek import ChatDeepSeek
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

# ----- 4.1 初始化 LLM -----
# DeepSeek thinking 模式会要求回传 reasoning_content，这里补齐请求字段
class ChatDeepSeekWithReasoning(ChatDeepSeek):
    def _get_request_payload(self, input_, *, stop=None, **kwargs) -> dict:
        messages = self._convert_input(input_).to_messages()
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)
        if "messages" in payload:
            for msg_dict, lc_msg in zip(payload["messages"], messages):
                if msg_dict.get("role") == "assistant" and isinstance(lc_msg, AIMessage):
                    reasoning_content = lc_msg.additional_kwargs.get("reasoning_content")
                    if reasoning_content:
                        msg_dict["reasoning_content"] = reasoning_content
        return payload

# 从环境变量读取 DeepSeek API Key，或在代码中直接设置
# export DEEPSEEK_API_KEY="sk-xxxxx"
llm = ChatDeepSeekWithReasoning(
    model="deepseek-v4-pro",  
    temperature=0.3,         # 低温度让工具选择更稳定
    api_key=os.getenv("DEEPSEEK_API_KEY"),
)

# ----- 4.2 系统提示词 -----
SYSTEM_PROMPT = """你是一个友好的教学助手 Agent。

行为准则：
1. 用户的问题如果需要计算、查询或操作，先使用对应工具获取结果
2. 如果工具返回的结果足够回答，直接基于结果回答，不要虚构信息
3. 如果用户的问题不需要任何工具，直接回答即可
4. 回答使用中文，简洁清晰
5. 如果你无法完成某个任务，诚实告诉用户并说明原因
"""

# ----- 4.3 创建 Agent（一行代码！）-----
# create_react_agent 是 LangGraph 最新推荐方式
# 它内部自动处理：思考→调用工具→观察结果→再思考→回答 的循环
agent = create_react_agent(
    model=llm,
    tools=tools,
    prompt=SYSTEM_PROMPT,
    checkpointer=MemorySaver(),  # 短期记忆，会话内保持上下文
)

# ===================== 5. 运行入口 =====================
def chat():
    """交互式对话循环"""
    print("=" * 60)
    print("  🎓 LangChain 教学版 Agent")
    print(f"  已加载 {len(tools)} 个工具：")
    for t in tools:
        print(f"    • {t.name}: {t.description[:40]}...")
    print("  输入 'quit' 退出，输入 'clear' 清除记忆")
    print("=" * 60)

    config = {"configurable": {"thread_id": "teaching_session"}}
    summary = ""
    max_messages_before_compress = 24

    def build_summary(messages: list) -> str:
        dialogue_lines = []
        for msg in messages:
            if isinstance(msg, HumanMessage) and msg.content:
                dialogue_lines.append(f"用户：{msg.content}")
            elif isinstance(msg, AIMessage) and msg.content:
                dialogue_lines.append(f"助手：{msg.content}")

        if not dialogue_lines:
            return ""

        summary_prompt = (
            "请把以下对话压缩为简洁摘要，保留：关键事实、用户偏好、未完成事项、工具结论。\n"
            "只输出摘要正文，不要加标题或多余说明。\n\n"
            + "\n".join(dialogue_lines)
        )
        summary_result = llm.invoke(
            [
                {"role": "system", "content": "你是一个负责压缩对话上下文的助手。"},
                {"role": "user", "content": summary_prompt},
            ]
        )
        return (summary_result.content or "").strip()

    while True:
        try:
            user_input = input("\n👤 你：").strip()
            if not user_input:
                continue
            if user_input.lower() == "quit":
                print("👋 再见！")
                break
            if user_input.lower() == "clear":
                # 换一个 thread_id 就是新会话，旧记忆清空
                import uuid
                config["configurable"]["thread_id"] = str(uuid.uuid4())
                print("🧹 记忆已清除！")
                continue

            print("🤖 Agent 思考中...")
            input_messages = []
            if summary:
                input_messages.append(
                    {
                        "role": "system",
                        "content": f"这是对话压缩摘要：\n{summary}\n请基于摘要继续对话。",
                    }
                )
            input_messages.append({"role": "user", "content": user_input})
            result = agent.invoke({"messages": input_messages}, config=config)

            # 透明化显示工具调用过程
            messages = result.get("messages", [])
            for msg in messages:
                if isinstance(msg, AIMessage) and msg.tool_calls:
                    print("\n🛠️ 工具调用：")
                    for call in msg.tool_calls:
                        name = call.get("name", "<unknown>")
                        args = call.get("args", {})
                        print(f"  - {name}({args})")
                if isinstance(msg, ToolMessage):
                    tool_name = msg.name or msg.additional_kwargs.get("name") or "<unknown>"
                    print(f"  ↳ 结果[{tool_name}]：{msg.content}")

            # 取最后一条 AI 消息作为回复
            for msg in reversed(messages):
                if isinstance(msg, AIMessage) and msg.content:
                    print(f"\n🤖 Agent：{msg.content}")
                    break

            # 触发上下文压缩
            if len(messages) >= max_messages_before_compress:
                summary = build_summary(messages)
                config["configurable"]["thread_id"] = str(uuid.uuid4())
                print("\n🧠 已压缩上下文并开启新会话线程。")

        except KeyboardInterrupt:
            print("\n👋 再见！")
            break
        except Exception as e:
            print(f"\n❌ 出错了：{e}")


if __name__ == "__main__":
    chat()
