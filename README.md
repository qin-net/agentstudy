# AI 工作流与教学 Agent 示例

一组 LangGraph/LangChain 的小型示例：

- `ai_decision_workflow.py`：带 AI 路由与分支的互动侦探故事。
- `ai_workflow_demo.py`：多角色协作的测验生成器，带 QA 重试逻辑。
- `teaching_agent.py`：教学风格的 ReAct Agent，包含工具集与记忆。

## 依赖

安装依赖：

```bash
pip install -r requirements.txt
```

可选环境变量：

- `DEEPSEEK_API_KEY`：启用 DeepSeek LLM 调用；未设置时会使用兜底输出，脚本仍可运行。
- `TAVILY_API_KEY`：仅在 `teaching_agent.py` 的联网搜索工具中需要。

## 运行方式

### 1) 侦探工作流（决策分支）

```bash
python ai_decision_workflow.py
```

功能说明：
- 根据灵感词路由故事类型（`detective/heist/paranormal/tech`）。
- 生成开场场景并向用户提问，对答案进行评分。
- 通过 AI 决策选择 `interrogate` 或 `analyze`，输出最终回顾。

### 2) 测验工作流（多角色）

```bash
python ai_workflow_demo.py
```

功能说明：
- 策划角色确定有趣角度与难度。
- 出题角色生成四选一问题。
- QA 关卡检查结构，失败则重试（最多 2 次）。
- 主持提问、评审打分、教练讲解。

### 3) 教学 Agent（工具 + 记忆）

```bash
python teaching_agent.py
```

功能说明：
- ReAct 风格 Agent，可调用工具（计算器、时间、字数统计、单位换算、联网搜索等）。
- 终端会显示工具调用过程。
- 使用 `MemorySaver` 保持短期记忆，并在对话过长时压缩上下文。

## 备注

- 所有示例以可读性与教学性为主。
- 想更换模型或新增工具，可从 `teaching_agent.py` 的工具列表处入手。
- 需要修改工作流，请调整两个工作流脚本中的节点函数与图连接。
