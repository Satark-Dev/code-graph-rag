from __future__ import annotations

from . import constants as cs


def style(
    text: str, color: cs.Color, modifier: cs.StyleModifier = cs.StyleModifier.BOLD
) -> str:
    if modifier == cs.StyleModifier.NONE:
        return f"[{color}]{text}[/{color}]"
    return f"[{modifier} {color}]{text}[/{modifier} {color}]"


def dim(text: str) -> str:
    return f"[{cs.StyleModifier.DIM}]{text}[/{cs.StyleModifier.DIM}]"

