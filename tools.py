# [imports]
# Loads helpers used by the time, calculator, delete, memory, and docs tools.
import ast
import operator
from datetime import datetime

from langchain.tools import tool

from docs_store import search_local_docs
from memory_store import get_facts, set_fact


# [get_current_time]
# Returns the local date and time for the agent.
@tool
def get_current_time() -> str:
    """Return the current local date and time."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# [calculate]
# Evaluates a simple math expression for the agent.
# 1. Reject expressions that use anything except numbers and basic operators.
# 2. Evaluate the expression and return the result as text.
@tool
def calculate(expression: str) -> str:
    """Evaluate a basic math expression using +, -, *, /, **, and parentheses.

    Args:
        expression: A math expression such as "17 * 24" or "(3 + 5) / 2".
    """
    # [calculate-1] Reject expressions that use anything except numbers and basic operators.
    allowed_operators = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }

    def eval_node(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return eval_node(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in allowed_operators:
            return allowed_operators[type(node.op)](
                eval_node(node.left),
                eval_node(node.right),
            )
        if isinstance(node, ast.UnaryOp) and type(node.op) in allowed_operators:
            return allowed_operators[type(node.op)](eval_node(node.operand))
        raise ValueError("Only numbers and +, -, *, /, **, and parentheses are allowed.")

    # [calculate-2] Evaluate the expression and return the result as text.
    try:
        parsed = ast.parse(expression, mode="eval")
        result = eval_node(parsed)
    except Exception as error:
        return f"Could not calculate that: {error}"
    return str(result)


# [delete_note]
# Pretends to delete a note. Used to practice human approval before a risky tool.
@tool
def delete_note(title: str) -> str:
    """Pretend to delete a saved note by title. Does not delete real files.

    Args:
        title: The note title the user asked to delete.
    """
    return f"Pretend-deleted note: {title}"


# [remember_fact]
# Saves a lasting fact about the user to the local memory file.
@tool
def remember_fact(key: str, value: str) -> str:
    """Save a lasting fact about the user such as hobby or favorite color.

    Args:
        key: Short label for the fact, such as "hobby" or "favorite_color".
        value: The fact to remember.
    """
    return set_fact(key, value)


# [recall_facts]
# Returns all lasting facts saved about the user.
@tool
def recall_facts() -> str:
    """Return all saved lasting facts about the user."""
    return get_facts()


# [search_docs]
# Searches local project markdown docs and returns matching snippets with sources.
@tool
def search_docs(query: str) -> str:
    """Search local markdown docs (cafe, library, park, and any other files in docs/).

    Args:
        query: What to look for in the docs.
    """
    return search_local_docs(query)


# [SAFE_TOOLS]
# Tools that can run without asking the user first.
SAFE_TOOLS = [
    get_current_time,
    calculate,
    remember_fact,
    recall_facts,
    search_docs,
]

# [TOOLS]
# All tools, including the fake risky delete.
TOOLS = [*SAFE_TOOLS, delete_note]
