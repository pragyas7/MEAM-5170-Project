"""Action schema and the plan parser.

Plans are JSON lists. The parser accepts a raw list, a fenced ```json block,
or an object {"plan": [...]} . Anything else is a structured malformed_plan error.
"""

from __future__ import annotations

import json
import re
from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field, TypeAdapter, ValidationError


class PlanParseError(Exception):
    def __init__(self, message: str, details: dict | None = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def as_dict(self) -> dict:
        return {"ok": False, "error": "malformed_plan", "message": self.message, **self.details}


class NavigateAction(BaseModel):
    action: Literal["navigate"]
    region: str


class TargetPose(BaseModel):
    x: float
    y: float
    theta: float = 0.0


class PushAction(BaseModel):
    action: Literal["push"]
    box: str
    target_region: str | None = None
    target_pose: TargetPose | None = None

    def model_post_init(self, _context) -> None:
        has_region = self.target_region is not None
        has_pose = self.target_pose is not None
        if has_region == has_pose:
            raise ValueError("push requires exactly one of target_region or target_pose")


class WaitAction(BaseModel):
    action: Literal["wait"]


class DoneAction(BaseModel):
    action: Literal["done"]


Action = Annotated[
    Union[NavigateAction, PushAction, WaitAction, DoneAction],
    Field(discriminator="action"),
]
_ADAPTER = TypeAdapter(list[Action])


def parse_plan(text: str) -> list[Action]:
    """Parse an LLM response into actions. Raises PlanParseError on failure."""
    if text is None or not str(text).strip():
        raise PlanParseError("empty plan", {"raw": text or ""})
    raw = str(text).strip()
    payload = _extract_json(raw)
    if payload is None:
        raise PlanParseError("response does not contain a JSON plan", {"raw": raw[:500]})
    if isinstance(payload, dict):
        if "plan" not in payload:
            raise PlanParseError("JSON object has no 'plan' list", {"keys": list(payload.keys())})
        payload = payload["plan"]
    if not isinstance(payload, list):
        raise PlanParseError("plan must be a JSON list", {"type": type(payload).__name__})
    try:
        actions = _ADAPTER.validate_python(payload)
    except ValidationError as exc:
        raise PlanParseError(
            "plan failed schema validation",
            {"errors": exc.errors()[:6], "raw": raw[:500]},
        ) from exc
    return actions


def plan_to_json(actions: list) -> str:
    dumped = []
    for action in actions:
        if hasattr(action, "model_dump"):
            dumped.append(action.model_dump(exclude_none=True))
        else:
            dumped.append(action)
    return json.dumps(dumped, separators=(",", ":"))


def _extract_json(raw: str):
    fenced = re.search(r"```(?:json)?\s*(.*?)```", raw, flags=re.DOTALL | re.IGNORECASE)
    candidates = []
    if fenced:
        candidates.append(fenced.group(1).strip())
    candidates.append(raw)
    # Largest bracketed array, then object, so prose around a plan still parses.
    array = _balanced(raw, "[", "]")
    obj = _balanced(raw, "{", "}")
    if array:
        candidates.append(array)
    if obj:
        candidates.append(obj)
    for cand in candidates:
        try:
            return json.loads(cand)
        except json.JSONDecodeError:
            continue
    return None


def _balanced(text: str, open_ch: str, close_ch: str) -> str | None:
    start = text.find(open_ch)
    if start < 0:
        return None
    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None
