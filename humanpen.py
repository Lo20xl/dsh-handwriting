"""
humanpen.py - human-like handwriting for AI agents.

What it does
------------
Turns text (and LaTeX-style math) into pen trajectories and replays them with a realistic
velocity profile: 2/3 power-law speed, bell-shaped stroke starts/ends, min-jerk pen-up moves,
pauses, fatigue, ligatures and self-corrections. Use it on drawing surfaces: whiteboards,
signature pads, Excalidraw/Miro-style canvases, "draw your answer" boxes. For ordinary
<input>/<textarea> fields just use type_text - a pen can't type.

Backends (pick with --backend or HUMANPEN_BACKEND)
--------------------------------------------------
* win32   - SendInput on the Windows desktop. Any app, no VM, no Cua. Coordinates are physical
            screen pixels, y down.
* cua     - a Cua `computer.interface` (mouse_down / move_cursor / mouse_up), for VMs and hosted
            Cua computers. Pass the Computer object to `make_agent_tool`.
* rec     - records events to a list instead of touching the pointer (tests).
* none    - plan only; nothing is drawn (see dry_run).

Why it doesn't look like a font
-------------------------------
* hand="bank" (default): letters are REAL pen strokes traced from a scan of a person's handwriting
  (embedded in this file, x-height = 1). Each letter has several real samples; every time a letter
  is written the generator picks one according to the writer's habits and often morphs between two
  samples, so no two "a"s are alike. Letters missing from the bank fall back to procedural glyphs.
* Variation is coherent, not per-point jitter: slow drift of size / slant / baseline / width / speed,
  one affine per letter, a smooth warp per letter, per-word size/tilt/rise, fatigue over long text.
* Optional ligatures: neighbouring letters sometimes join with the pen still down.
* hand="casual" / "cursive" are procedural alternatives.

Math
----
Wrap LaTeX in $...$ inside the text. Supported: ^ _ \\frac{}{} \\sqrt{} \\sqrt[n]{} \\vec \\hat \\bar
\\overline \\left( \\right) \\sum \\int \\prod with limits, greek letters, \\pm \\times \\div \\cdot
\\leq \\geq \\neq \\approx \\infty \\to \\Rightarrow \\partial \\nabla \\angle \\degree \\checkmark
\\sin \\cos \\log ... \\text{}.  Bare unicode (x\\u00b2, \\u221ax, \\u2264, \\u03c0, \\u00b0) is converted automatically.

Usage with Cua
--------------
    from computer import Computer
    from agent import ComputerAgent
    from humanpen import make_agent_tool

    async with Computer(...) as computer:
        handwrite = make_agent_tool(computer, seed=42)        # one persistent "writer"
        agent = ComputerAgent(model="...", tools=[computer, handwrite])

or, framework-agnostic:  TOOL_SCHEMAS + `await handle_tool_call(computer, tool_input)`.

Usage without Cua (this machine's own pointer)
----------------------------------------------
    from humanpen import get_backend, Handwriter
    await Handwriter(seed=42).write(get_backend("win32"), "hi", 200, 300)

CLI
---
    python humanpen.py preview "Area $A=\\pi r^2$" --png out.png --seed 7   # no pointer needed
    python humanpen.py write "Note: $v = u + at$" --x 200 --y 300 --backend win32
    python humanpen.py draw circle --x0 100 --y0 100 --x1 200 --y1 160
    python humanpen.py doctor
    python humanpen.py mcp                                                   # MCP stdio server
"""
from __future__ import annotations

import argparse
import asyncio
import inspect
import json
import math
import os
import sys
import time
import dataclasses
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

__version__ = "2.0.0"

# =============================================================================
# 1. Glyph skeletons.  Units: x-height = 1, baseline y = 0, y up, advance `adv`.
#    Cursive glyphs are ONE pen path from entry (0,0) to exit (adv, ~0).
#    hi=True  -> exit is at x-height (o, v, w, r): the join arrives high.
#    skip=k   -> when entered high, drop the first k points (skip the entry upstroke).
# =============================================================================


@dataclass
class Glyph:
    pts: Optional[list] = None          # cursive: single connectable path
    adv: float = 1.0
    hi: bool = False
    skip: int = 0
    marks: tuple = ()                   # ("dot", x, y) / ("bar", x, y, halfwidth)
    strokes: Optional[list] = None      # print glyphs: list of independent strokes
    ys: float = 1.0                     # vertical scale (print capitals are 1.8x x-height)
    ji: bool = True                     # may a join arrive INTO this glyph
    jo: bool = True                     # may a join leave FROM this glyph
    bank: bool = False                  # strokes come from the real-handwriting bank

    @property
    def cursive(self) -> bool:
        return self.pts is not None


C = Glyph
GLYPHS: dict[str, Glyph] = {
    "a": C([(0, 0), (.30, .45), (.62, .88), (.38, 1.0), (.12, .72), (.10, .30), (.32, .04),
            (.58, .26), (.74, .82), (.78, .38), (.83, .06), (1.08, .04)], 1.08, skip=2),
    "b": C([(0, 0), (.26, .6), (.5, 1.4), (.6, 1.95), (.45, 2.1), (.3, 1.8), (.36, 1.0),
            (.42, .3), (.46, .03), (.78, .22), (.9, .55), (.72, .86), (.5, .7), (.46, .5),
            (.62, .1), (.95, .04)], .95, skip=3),
    "c": C([(0, 0), (.28, .5), (.6, .88), (.38, 1.0), (.12, .74), (.1, .35), (.3, .04),
            (.58, .12), (.86, .1)], .86, skip=2),
    "d": C([(0, 0), (.3, .45), (.62, .88), (.38, 1.0), (.12, .72), (.1, .3), (.32, .04),
            (.6, .3), (.72, .9), (.84, 1.6), (.94, 2.05), (.8, 2.12), (.74, 1.5), (.75, .7),
            (.8, .08), (1.1, .04)], 1.1, skip=2),
    "e": C([(0, 0), (.22, .42), (.48, .8), (.5, .98), (.3, 1.0), (.26, .8), (.5, .62),
            (.28, .5), (.18, .22), (.4, .05), (.7, .1)], .72, skip=2),
    "f": C([(0, 0), (.3, .8), (.5, 1.6), (.6, 2.1), (.4, 2.2), (.3, 1.5), (.25, .5),
            (.12, -.5), (-.15, -1.0), (-.32, -.8), (-.1, -.3), (.2, .3), (.6, .5),
            (.82, .08)], .82, marks=(("bar", .2, 1.05, .35),), skip=2),
    "g": C([(0, 0), (.3, .45), (.62, .88), (.38, 1.0), (.12, .72), (.1, .3), (.32, .04),
            (.6, .3), (.72, .9), (.72, .2), (.68, -.5), (.45, -1.0), (.15, -.92), (.08, -.55),
            (.5, -.2), (.9, .05)], .92, skip=2),
    "h": C([(0, 0), (.28, .9), (.5, 1.6), (.56, 2.1), (.38, 2.15), (.28, 1.6), (.27, .8),
            (.27, .05), (.3, .5), (.55, .9), (.78, .75), (.8, .4), (.85, .05), (1.1, .05)],
           1.1, skip=3),
    "i": C([(0, 0), (.22, .6), (.32, .98), (.3, .5), (.34, .06), (.62, .04)], .62,
           marks=(("dot", .3, 1.42),), skip=2),
    "j": C([(0, 0), (.22, .6), (.32, .98), (.3, .3), (.22, -.6), (.0, -1.0), (-.25, -.8),
            (-.1, -.4), (.2, 0.0), (.55, .05)], .55, marks=(("dot", .3, 1.42),), skip=2),
    "k": C([(0, 0), (.3, .9), (.5, 1.8), (.55, 2.1), (.38, 2.1), (.3, 1.2), (.28, .4),
            (.3, .05), (.32, .45), (.62, .8), (.42, .55), (.55, .3), (.62, .05), (.9, .05)],
           .9, skip=3),
    "l": C([(0, 0), (.3, .7), (.5, 1.5), (.6, 2.1), (.42, 2.16), (.32, 1.7), (.34, .8),
            (.36, .1), (.64, .05)], .64, skip=2),
    "m": C([(0, 0), (.2, .75), (.28, 1.0), (.22, .5), (.2, .05), (.3, .6), (.5, .95),
            (.65, .8), (.66, .4), (.66, .05), (.78, .6), (.98, .95), (1.15, .8), (1.17, .4),
            (1.2, .05), (1.46, .05)], 1.46, skip=2),
    "n": C([(0, 0), (.2, .75), (.28, 1.0), (.22, .5), (.2, .05), (.3, .6), (.55, .95),
            (.78, .8), (.8, .4), (.85, .05), (1.1, .05)], 1.1, skip=2),
    "o": C([(0, 0), (.25, .55), (.5, .98), (.25, 1.0), (.08, .65), (.12, .2), (.35, .02),
            (.6, .25), (.62, .7), (.5, .98), (.78, .97), (1.02, .92)], 1.02, hi=True, skip=2),
    "p": C([(0, 0), (.2, .7), (.3, 1.0), (.28, .3), (.25, -.6), (.2, -1.0), (.24, -.3),
            (.3, .4), (.58, .92), (.85, .7), (.75, .25), (.45, .05), (.3, .3), (.55, .08),
            (.95, .05)], .95, skip=2),
    "q": C([(0, 0), (.3, .45), (.62, .88), (.38, 1.0), (.12, .72), (.1, .3), (.32, .04),
            (.6, .3), (.72, .9), (.72, .2), (.7, -.6), (.55, -1.0), (.35, -.85), (.55, -.3),
            (.95, .05)], .95, skip=2),
    "r": C([(0, 0), (.2, .7), (.3, 1.0), (.24, .5), (.22, .05), (.35, .6), (.55, .88),
            (.78, .88)], .8, hi=True, skip=2),
    "t": C([(0, 0), (.2, .5), (.32, 1.2), (.36, 1.5), (.34, .9), (.34, .3), (.5, .05),
            (.78, .05)], .78, marks=(("bar", .35, 1.0, .34),), skip=3),
    "u": C([(0, 0), (.1, .6), (.15, .95), (.15, .4), (.3, .05), (.52, .2), (.62, .6),
            (.7, .95), (.68, .4), (.7, .05), (.98, .05)], .98, skip=2),
    "v": C([(0, 0), (.1, .6), (.15, .95), (.2, .4), (.35, .02), (.55, .5), (.7, .95),
            (.95, .95), (1.12, .9)], 1.12, hi=True, skip=2),
    "w": C([(0, 0), (.1, .6), (.15, .95), (.2, .4), (.35, .02), (.55, .6), (.75, .02),
            (.95, .95), (1.2, .95), (1.4, .9)], 1.4, hi=True, skip=2),
    "y": C([(0, 0), (.1, .6), (.15, .95), (.2, .4), (.35, .02), (.55, .5), (.7, .95),
            (.65, .3), (.45, -.6), (.2, -1.0), (-.05, -.82), (.1, -.45), (.45, 0.0),
            (.78, .05)], .78, skip=2),
    "x": Glyph(strokes=[[(0, .95), (.35, .5), (.78, 0)], [(.78, .95), (.4, .5), (0, 0)]], adv=.95),
    "z": Glyph(strokes=[[(.05, .95), (.7, .95), (.05, 0), (.78, 0)]], adv=.95),
    "s": Glyph(strokes=[[(.66, .9), (.44, 1.0), (.16, .9), (.12, .68), (.36, .52), (.6, .38), (.64, .15), (.4, .02), (.1, .1)]], adv=.8),
}

_P = lambda *s: list(s)  # noqa: E731  (tiny helper for readability below)
_PRINT = {
    # capitals (height 2.0 -> scaled by ys)
    "A": ([[(0, 0), (.6, 2), (1.2, 0)], [(.25, .7), (.95, .7)]], 1.4),
    "B": ([[(0, 0), (0, 2)], [(0, 2), (.7, 2), (.85, 1.6), (.6, 1.1), (0, 1.05)],
           [(0, 1.05), (.8, 1.0), (.95, .5), (.7, .05), (0, 0)]], 1.2),
    "C": ([[(1.0, 1.6), (.7, 2.0), (.25, 1.9), (0, 1.0), (.25, .1), (.7, 0), (1.05, .35)]], 1.2),
    "D": ([[(0, 0), (0, 2)], [(0, 2), (.8, 1.9), (1.0, 1.0), (.8, .1), (0, 0)]], 1.3),
    "E": ([[(1, 2), (0, 2), (0, 0), (1, 0)], [(0, 1), (.7, 1)]], 1.15),
    "F": ([[(1, 2), (0, 2), (0, 0)], [(0, 1), (.7, 1)]], 1.1),
    "G": ([[(1.0, 1.6), (.7, 2.0), (.25, 1.9), (0, 1.0), (.25, .1), (.7, 0), (1.0, .4), (1.0, .9),
            (.65, .9)]], 1.25),
    "H": ([[(0, 0), (0, 2)], [(1, 0), (1, 2)], [(0, 1), (1, 1)]], 1.3),
    "I": ([[(0, 2), (.6, 2)], [(.3, 2), (.3, 0)], [(0, 0), (.6, 0)]], .8),
    "J": ([[(.8, 2), (.8, .4), (.6, 0), (.2, .1), (0, .4)]], 1.0),
    "K": ([[(0, 0), (0, 2)], [(1, 2), (0, .9)], [(.3, 1.25), (1, 0)]], 1.2),
    "L": ([[(0, 2), (0, 0), (.9, 0)]], 1.0),
    "M": ([[(0, 0), (0, 2), (.6, .7), (1.2, 2), (1.2, 0)]], 1.5),
    "N": ([[(0, 0), (0, 2), (1, 0), (1, 2)]], 1.3),
    "O": ([[(.5, 2), (.1, 1.6), (0, 1.0), (.15, .3), (.5, 0), (.9, .3), (1.0, 1.0), (.9, 1.6),
            (.5, 2.0)]], 1.3),
    "P": ([[(0, 0), (0, 2), (.8, 1.95), (.95, 1.5), (.7, 1.05), (0, 1)]], 1.15),
    "Q": ([[(.5, 2), (.1, 1.6), (0, 1.0), (.15, .3), (.5, 0), (.9, .3), (1.0, 1.0), (.9, 1.6),
            (.5, 2.0)], [(.6, .5), (1.05, -.25)]], 1.3),
    "R": ([[(0, 0), (0, 2), (.8, 1.95), (.95, 1.5), (.7, 1.05), (0, 1)], [(.4, 1.0), (1.0, 0)]], 1.2),
    "S": ([[(.95, 1.7), (.65, 2.0), (.2, 1.9), (.1, 1.4), (.5, 1.05), (.9, .6), (.85, .1), (.4, 0),
            (.05, .3)]], 1.1),
    "T": ([[(0, 2), (1, 2)], [(.5, 2), (.5, 0)]], 1.2),
    "U": ([[(0, 2), (0, .5), (.2, .05), (.7, 0), (1.0, .5), (1.0, 2)]], 1.3),
    "V": ([[(0, 2), (.55, 0), (1.1, 2)]], 1.3),
    "W": ([[(0, 2), (.3, 0), (.65, 1.4), (1.0, 0), (1.3, 2)]], 1.55),
    "X": ([[(0, 2), (1, 0)], [(1, 2), (0, 0)]], 1.2),
    "Y": ([[(0, 2), (.5, 1.0), (1, 2)], [(.5, 1.0), (.5, 0)]], 1.2),
    "Z": ([[(0, 2), (1, 2), (0, 0), (1, 0)]], 1.2),
    # digits
    "0": ([[(.45, 2), (.1, 1.6), (0, 1.0), (.1, .35), (.45, 0), (.8, .35), (.9, 1.0), (.8, 1.6),
            (.45, 2)]], 1.1),
    "1": ([[(.05, 1.6), (.45, 2.0), (.45, 0)]], .8),
    "2": ([[(.05, 1.6), (.3, 2.0), (.7, 2.0), (.9, 1.6), (.7, 1.1), (0, 0), (.95, 0)]], 1.1),
    "3": ([[(.05, 1.8), (.4, 2.0), (.8, 1.85), (.8, 1.4), (.4, 1.05), (.85, .8), (.85, .25),
            (.45, 0), (0, .2)]], 1.1),
    "4": ([[(.7, 0), (.7, 2), (0, .6), (.95, .6)]], 1.1),
    "5": ([[(.85, 2), (.15, 2), (.1, 1.1), (.5, 1.2), (.85, .9), (.85, .3), (.5, 0), (0, .2)]], 1.1),
    "6": ([[(.8, 1.9), (.4, 2.0), (.1, 1.3), (.05, .5), (.3, 0), (.7, .1), (.85, .55), (.55, 1.0),
            (.15, .8)]], 1.1),
    "7": ([[(0, 2), (.9, 2), (.35, 0)]], 1.0),
    "8": ([[(.45, 1.05), (.1, 1.4), (.3, 1.95), (.65, 1.9), (.8, 1.4), (.45, 1.05), (.05, .5),
            (.25, 0), (.65, .05), (.85, .5), (.45, 1.05)]], 1.1),
    "9": ([[(.8, 1.4), (.5, 1.0), (.15, 1.2), (.1, 1.75), (.45, 2.0), (.8, 1.8), (.85, 1.2),
            (.7, .4), (.35, 0), (.1, .1)]], 1.1),
    # punctuation
    ".": ([[(0, .0), (.08, .1), (.14, 0)]], .35),
    ",": ([[(.12, .12), (.05, -.2), (-.05, -.4)]], .4),
    "!": ([[(.1, 2), (.1, .6)], [(.1, .12), (.14, .02)]], .45),
    "?": ([[(0, 1.6), (.3, 2.0), (.8, 1.8), (.7, 1.3), (.4, 1.0), (.4, .6)], [(.4, .12), (.44, .02)]], 1.0),
    "-": ([[(0, .8), (.6, .8)]], .75),
    "_": ([[(0, -.15), (1, -.15)]], 1.1),
    ":": ([[(.1, 1.0), (.14, .92)], [(.1, .12), (.14, .02)]], .4),
    ";": ([[(.1, 1.0), (.14, .92)], [(.14, .12), (.05, -.3)]], .45),
    "'": ([[(.1, 2.0), (.05, 1.5)]], .3),
    '"': ([[(.1, 2.0), (.05, 1.5)], [(.4, 2.0), (.35, 1.5)]], .65),
    "/": ([[(.6, 2), (0, -.2)]], .75),
    "(": ([[(.45, 2.0), (.1, 1.0), (.4, -.15)]], .6),
    ")": ([[(.05, 2.0), (.4, 1.0), (.1, -.15)]], .6),
    "+": ([[(.0, 1.0), (.8, 1.0)], [(.4, 1.4), (.4, .6)]], 1.0),
    "=": ([[(0, 1.2), (.8, 1.2)], [(0, .7), (.8, .7)]], 1.0),
    "*": ([[(.1, 1.8), (.7, 1.0)], [(.7, 1.8), (.1, 1.0)]], .9),
}
for _ch, (_strokes, _adv) in _PRINT.items():
    GLYPHS[_ch] = Glyph(strokes=_strokes, adv=_adv, ys=0.9 if (_ch.isupper() or _ch.isdigit()) else 0.85)
GLYPHS[" "] = Glyph(strokes=[], adv=0.0)
CURSIVE = dict(GLYPHS)

_LOWER = {
    "a": C([(.68, .9), (.42, 1.0), (.14, .78), (.1, .35), (.3, .05), (.58, .2), (.72, .6), (.72, .96),
            (.74, .5), (.77, .05), (.97, .04)], .97),
    "b": C([(.1, 1.8), (.1, .9), (.08, .1), (.1, .8), (.35, 1.0), (.66, .78), (.68, .3), (.42, .03),
            (.1, .12)], .82, ji=False, jo=False),
    "c": C([(.72, .82), (.45, 1.0), (.14, .8), (.08, .4), (.3, .05), (.62, .1), (.8, .2)], .82),
    "d": C([(.68, .9), (.42, 1.0), (.14, .78), (.1, .35), (.3, .05), (.6, .22), (.74, .6), (.78, 1.8),
            (.76, .9), (.78, .06), (.97, .05)], .97),
    "e": C([(.1, .52), (.4, .56), (.72, .66), (.62, .92), (.35, 1.0), (.12, .75), (.08, .35), (.3, .05),
            (.62, .08), (.8, .2)], .82),
    "f": C([(.72, 1.68), (.5, 1.86), (.3, 1.64), (.28, .9), (.26, .05)], .62, ji=False, jo=False,
           marks=(("bar", .28, 1.02, .3),)),
    "g": C([(.7, .9), (.42, 1.0), (.14, .78), (.1, .35), (.3, .05), (.6, .2), (.72, .75), (.72, 1.0),
            (.72, .3), (.7, -.4), (.5, -.75), (.2, -.7), (.1, -.5)], .95, jo=False),
    "h": C([(.1, 1.85), (.1, .9), (.08, .05), (.1, .6), (.35, .95), (.62, .8), (.7, .4), (.72, .05),
            (.9, .04)], .9, ji=False),
    "i": C([(.12, .95), (.12, .5), (.14, .05), (.34, .04)], .5, marks=(("dot", .12, 1.4),)),
    "j": C([(.14, .95), (.14, .3), (.1, -.4), (-.05, -.72), (-.28, -.6)], .4, marks=(("dot", .14, 1.4),),
           jo=False),
    "k": C([(.1, 1.85), (.1, .9), (.09, .04), (.1, .4), (.38, .65), (.68, .95), (.38, .62), (.5, .35),
            (.76, .04)], .88, ji=False),
    "l": C([(.1, 1.85), (.1, .9), (.12, .06), (.32, .04)], .42, ji=False),
    "m": C([(.1, .95), (.1, .5), (.09, .04), (.1, .6), (.25, .95), (.42, .9), (.46, .5), (.47, .05),
            (.5, .6), (.65, .95), (.83, .88), (.87, .5), (.88, .05), (1.06, .04)], 1.1),
    "n": C([(.1, .95), (.1, .5), (.09, .04), (.1, .6), (.28, .95), (.55, .88), (.62, .5), (.63, .05),
            (.82, .04)], .84),
    "o": C([(.5, 1.0), (.2, .85), (.08, .45), (.25, .08), (.55, .05), (.75, .4), (.65, .85), (.5, 1.0),
            (.76, .97), (.98, .92)], 1.0, hi=True),
    "p": C([(.1, .95), (.1, .2), (.1, -.7), (.12, .3), (.1, .7), (.35, .98), (.66, .8), (.7, .4),
            (.42, .04), (.12, .15)], .8, jo=False),
    "q": C([(.7, .9), (.42, 1.0), (.14, .78), (.1, .35), (.3, .05), (.6, .2), (.72, .75), (.72, 1.0),
            (.72, .3), (.72, -.7), (.9, -.62)], .95, jo=False),
    "r": C([(.1, .95), (.1, .5), (.09, .04), (.1, .55), (.3, .92), (.55, .9), (.68, .8)], .74, hi=True),
    "s": C([(.66, .9), (.44, 1.0), (.16, .9), (.12, .68), (.36, .52), (.6, .38), (.64, .15), (.4, .02),
            (.1, .1)], .8, jo=False),
    "t": C([(.3, 1.55), (.3, .9), (.3, .2), (.45, .04), (.64, .1)], .64, ji=False,
           marks=(("bar", .3, 1.05, .36),)),
    "u": C([(.1, .95), (.1, .45), (.28, .05), (.55, .12), (.7, .5), (.72, .95), (.72, .4), (.74, .05),
            (.93, .04)], .87),
    "v": C([(.05, .95), (.3, .04), (.62, .95), (.86, .98)], .86, hi=True),
    "w": C([(.05, .95), (.22, .05), (.4, .7), (.6, .05), (.82, .95), (1.02, .98)], 1.05, hi=True),
    "y": C([(.05, .95), (.3, .05), (.62, .95), (.42, -.2), (.12, -.72)], .75, jo=False),
    "x": GLYPHS["x"], "z": GLYPHS["z"],
}
CASUAL = {k: v for k, v in GLYPHS.items() if not k.islower()}
CASUAL.update(_LOWER)


# =============================================================================
# 2. Writer personality
# =============================================================================


@dataclass
class WriterStyle:
    hand: str = "bank"              # "bank" (real strokes traced from a scan) | "casual" | "cursive"
    bank_ratio: float = 1.0          # P(use a bank sample when one exists)
    gap_min: float = .85             # word gap, x-heights (+ gap_var * U(0,1))
    gap_var: float = .5
    letter_base: float = .04         # per-letter baseline offset, x-heights
    slant_deg: float = 12.0          # forward lean
    width_scale: float = 1.0
    connectivity: float = 0.85       # P(keep the join between two letters)
    word_join: float = 0.08          # P(word's exit stroke runs on into the next word)
    loop_up: float = 1.0             # ascender loop fullness
    loop_down: float = 1.0           # descender loop fullness
    speed: float = 24.0              # base pen speed, x-heights per second
    line_slope_deg: float = 0.6      # lines drift upward/downward
    shape_sigma: float = 0.03        # smooth skeleton variation (glyph units)
    size_drift: float = 0.07
    slant_drift_deg: float = 3.0
    baseline_drift: float = 0.10     # in x-heights
    width_drift: float = 0.08
    speed_drift: float = 0.15
    size_jit: float = 0.07           # per-letter-instance scale
    dot_forward: float = 0.12        # i-dots sit ahead of the stem by this many x-heights
    miss_rate: float = 0.0           # P(forget an i-dot / t-bar)

    @classmethod
    def from_seed(cls, seed: Optional[int] = None, hand: str = "bank") -> "WriterStyle":
        r = np.random.default_rng(seed)
        if hand == "cursive":
            return cls(
                hand="cursive", gap_min=.62, gap_var=.35, size_jit=.035, letter_base=.0,
                slant_deg=float(np.clip(r.normal(12, 5), -4, 25)),
                width_scale=float(np.clip(r.lognormal(0, .11), .8, 1.3)),
                connectivity=float(r.uniform(.72, .95)), word_join=float(r.uniform(.03, .12)),
                loop_up=float(r.uniform(.85, 1.25)), loop_down=float(r.uniform(.8, 1.2)),
                speed=float(np.clip(r.normal(24, 3), 16, 34)), line_slope_deg=float(r.normal(.4, .8)),
                shape_sigma=float(r.uniform(.02, .045)), dot_forward=float(r.uniform(.04, .22)))
        return cls(
            hand=hand,
            slant_deg=float(np.clip(r.normal(5 if hand != "bank" else 1, 3.5 if hand != "bank" else 2.5), -4, 14)),
            width_scale=float(np.clip(r.lognormal(0, .1), .85, 1.25)),
            connectivity=float(r.uniform(.22, .5)), word_join=float(r.uniform(0, .03)),
            loop_up=float(r.uniform(.9, 1.1)), loop_down=float(r.uniform(.85, 1.15)),
            speed=float(np.clip(r.normal(22, 3), 15, 30)), line_slope_deg=float(r.normal(.5, 1.0)),
            shape_sigma=float(r.uniform(.03, .05)), dot_forward=float(r.uniform(.05, .3)),
            gap_min=float(r.uniform(.8, 1.0)), size_drift=.09, baseline_drift=.12)


@dataclass
class Controls:
    """Per-call humanisation knobs (all optional). 1.0 = the writer's natural variation."""
    messiness: float = 1.0           # global multiplier on every kind of variation (0 = robotic, 2 = scrawl)
    size_variation: float = 1.0      # letter / word size wobble
    position_variation: float = 1.0  # baseline wander, letter spacing noise, margin jitter, word rise/fall
    slant_variation: float = 1.0
    speed_variation: float = 1.0
    slant: Optional[float] = None    # forward lean override (degrees)
    connectivity: Optional[float] = None   # 0..1 chance neighbouring letters are joined
    line_slope: Optional[float] = None     # degrees; lines drift up (+) / down (-)
    align: str = "left"              # left | center | right
    mistakes: float = 0.0            # 0..1 chance a word is mis-written, struck out and rewritten
    fatigue: float = .5              # 0..1 letters shrink / get sloppier along the text
    flourish: float = .3             # 0..1 chance of a lift-off flick at the end of words

    @classmethod
    def from_kwargs(cls, **kw):
        names = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in kw.items() if k in names and v is not None})


class _Drift:
    """Smooth 1-D random field: knots every `scale` letters, cosine interpolation."""

    def __init__(self, rng, n, scale, amp):
        self.scale = float(scale)
        self.k = rng.normal(0, amp, int(n / scale) + 4)

    def __call__(self, g):
        i = int(g // self.scale)
        f = (g / self.scale) - i
        f = (1 - math.cos(math.pi * f)) / 2
        i = min(i, len(self.k) - 2)
        return self.k[i] * (1 - f) + self.k[i + 1] * f


# =============================================================================
# 3. Geometry / dynamics helpers
# =============================================================================


def _dedupe(P, eps=1e-3):
    P = np.asarray(P, float)
    keep = [0]
    for i in range(1, len(P)):
        if np.hypot(*(P[i] - P[keep[-1]])) > eps:
            keep.append(i)
    return P[keep]


def catmull_rom(pts, step=0.6):
    """Centripetal Catmull-Rom (Barry-Goldman) through `pts`, ~`step` px spacing."""
    P = _dedupe(pts)
    n = len(P)
    if n == 1:
        return P
    if n == 2:
        m = max(2, int(np.hypot(*(P[1] - P[0])) / step) + 1)
        return P[0] + (P[1] - P[0]) * np.linspace(0, 1, m)[:, None]
    E = np.vstack([2 * P[0] - P[1], P, 2 * P[-1] - P[-2]])
    out = []
    for i in range(1, n):
        p0, p1, p2, p3 = E[i - 1], E[i], E[i + 1], E[i + 2]
        t0 = 0.0
        t1 = t0 + max(np.hypot(*(p1 - p0)), 1e-6) ** .5
        t2 = t1 + max(np.hypot(*(p2 - p1)), 1e-6) ** .5
        t3 = t2 + max(np.hypot(*(p3 - p2)), 1e-6) ** .5
        m = max(2, int(np.hypot(*(p2 - p1)) * 1.25 / step) + 1)
        u = np.linspace(t1, t2, m, endpoint=(i == n - 1))[:, None]
        A1 = (t1 - u) / (t1 - t0) * p0 + (u - t0) / (t1 - t0) * p1
        A2 = (t2 - u) / (t2 - t1) * p1 + (u - t1) / (t2 - t1) * p2
        A3 = (t3 - u) / (t3 - t2) * p2 + (u - t2) / (t3 - t2) * p3
        B1 = (t2 - u) / (t2 - t0) * A1 + (u - t0) / (t2 - t0) * A2
        B2 = (t3 - u) / (t3 - t1) * A2 + (u - t1) / (t3 - t1) * A3
        out.append((t2 - u) / (t2 - t1) * B1 + (u - t1) / (t2 - t1) * B2)
    return np.vstack(out)


def _bow(stk, rng, k=.04):
    """Humans never draw a perfectly straight line: bow long straight segments slightly."""
    P = np.asarray(stk, float)
    out = [P[0]]
    for a, b in zip(P[:-1], P[1:]):
        d = b - a
        L = float(np.hypot(*d))
        if L > .6:
            n = np.array([-d[1], d[0]]) / L
            out.append((a + b) / 2 + n * rng.normal(0, k) * L)
        out.append(b)
    return np.array(out)


def _smooth(x, w):
    w = max(1, int(w))
    if w == 1 or len(x) < 3:
        return x
    k = np.ones(w) / w
    pad = np.pad(x, (w // 2, w - 1 - w // 2), mode="edge")
    return np.convolve(pad, k, mode="valid")


def _smoothstep(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


def stroke_times(xy, vbase, xh):
    """Time stamps for a dense polyline: 2/3 power law + bell-shaped start/end."""
    if len(xy) < 2:
        return np.zeros(len(xy))
    d = np.diff(xy, axis=0)
    ds = np.maximum(np.hypot(d[:, 0], d[:, 1]), 1e-6)
    ang = np.arctan2(d[:, 1], d[:, 0])
    dth = np.zeros(len(ds))
    if len(ds) > 1:
        dth[1:] = np.abs(np.angle(np.exp(1j * np.diff(ang))))
    kappa = _smooth(dth / ds, max(3, int(.3 * xh / .6)))
    r = 1.0 / (kappa + 1e-4)
    v = vbase * np.clip((r / (.9 * xh)) ** (1 / 3), .28, 1.0)
    v = _smooth(v, max(3, int(.2 * xh / .6)))
    s = np.cumsum(ds) - ds / 2
    L = s[-1] + ds[-1] / 2
    ramp = max(min(.5 * xh, L / 3), 1.0)
    v = v * (.30 + .70 * _smoothstep(s / ramp)) * (.30 + .70 * _smoothstep((L - s) / ramp))
    return np.concatenate([[0], np.cumsum(ds / np.maximum(v, 1e-3))])


def rdp_indices(xy, tol):
    n = len(xy)
    keep = np.zeros(n, bool)
    keep[0] = keep[-1] = True
    stack = [(0, n - 1)]
    while stack:
        a, b = stack.pop()
        if b <= a + 1:
            continue
        p, A, B = xy[a + 1:b], xy[a], xy[b]
        AB = B - A
        L = np.hypot(*AB)
        dist = np.hypot(*(p - A).T) if L < 1e-9 else np.abs(AB[0] * (p[:, 1] - A[1]) - AB[1] * (p[:, 0] - A[0])) / L
        k = int(np.argmax(dist))
        if dist[k] > tol:
            idx = a + 1 + k
            keep[idx] = True
            stack += [(a, idx), (idx, b)]
    return np.nonzero(keep)[0]


def min_jerk_path(a, b, dur, rng, n=None):
    a, b = np.asarray(a, float), np.asarray(b, float)
    dist = float(np.hypot(*(b - a)))
    n = n or max(6, int(dist / 10))
    tau = np.linspace(0, 1, n)
    s = 10 * tau**3 - 15 * tau**4 + 6 * tau**5
    d = (b - a) / max(dist, 1e-6)
    nrm = np.array([-d[1], d[0]])
    bow = rng.normal(0, .035) * dist
    xy = a + np.outer(s, b - a) + np.outer(np.sin(np.pi * s) * bow, nrm)
    return xy, tau * dur


# =============================================================================
# 4. Plan data structures
# =============================================================================


@dataclass
class Seg:
    down: bool
    xy: np.ndarray            # waypoints (int-valued floats), screen px
    t: np.ndarray             # seconds since segment start, per waypoint
    pause: float = 0.0        # idle time before this segment
    dense: Optional[np.ndarray] = None   # original dense polyline (preview only)
    dense_v: Optional[np.ndarray] = None # pen speed along dense (preview only)
    corner: Optional[np.ndarray] = None  # bool per waypoint: never skip these


@dataclass
class Plan:
    segs: list
    bbox: tuple
    skipped: list
    text: str
    next_y: float
    size: float = 24.0
    requested_size: float = 24.0

    @property
    def duration(self) -> float:
        return sum(s.pause + (s.t[-1] if len(s.t) else 0) for s in self.segs)

    @property
    def n_strokes(self) -> int:
        return sum(1 for s in self.segs if s.down)

    def render_png(self, path, pad=20, scale=2, width=4.0, bg=(255, 255, 255), ink=(25, 30, 60)):
        """Rough preview with a felt-pen look: line gets thicker where the pen slowed down.
        (On a real site the stroke width is whatever the site's brush gives you.)"""
        from PIL import Image, ImageDraw
        x0, y0, x1, y1 = self.bbox
        W, H = int((x1 - x0 + 2 * pad) * scale), int((y1 - y0 + 2 * pad) * scale)
        im = Image.new("RGB", (max(W, 10), max(H, 10)), bg)
        dr = ImageDraw.Draw(im)
        allv = np.concatenate([s.dense_v for s in self.segs if s.down and s.dense_v is not None])
        vref = float(np.percentile(allv, 90)) or 1.0
        for s in self.segs:
            if not s.down or s.dense is None:
                continue
            P = (s.dense - [x0 - pad, y0 - pad]) * scale
            w = _smooth(width * scale * (1.25 - .6 * np.clip(s.dense_v / vref, 0, 1)), 9)
            for i in range(len(P) - 1):
                dr.line([tuple(P[i]), tuple(P[i + 1])], fill=ink, width=max(1, int(round(w[i]))))
                r = w[i] / 2
                dr.ellipse([P[i][0] - r, P[i][1] - r, P[i][0] + r, P[i][1] + r], fill=ink)
        im.save(path)
        return path


# ---------------------------------------------------------------------------
# Part A (inserted after the glyph tables): real-stroke bank + math symbols
# ---------------------------------------------------------------------------
import base64 as _b64, zlib as _zlib, json as _json

_BANK_B64 = "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"


def _load_bank():
    d = _json.loads(_zlib.decompress(_b64.b64decode(_BANK_B64)))
    return {k: [dict(w=g["w"], s=[np.array(s, float) for s in g["s"]]) for g in v] for k, v in d.items()}


BANK = _load_bank()      # strokes traced from the author's own handwriting scan (x-height = 1)
# math glyph aliases into the bank: char -> (bank key, y offset in x-heights)
BANK_ALIAS = {"×": ("x_small", .12), "÷": ("DIV", .1), "✓": ("CHECK", 0.0)}

_G = lambda strokes, adv: Glyph(strokes=strokes, adv=adv, ys=1.0)  # noqa: E731  (math symbols are in absolute x-heights)
MSYM = {
    "-": _G([[(0, .6), (.72, .6)]], .85), "−": _G([[(0, .6), (.72, .6)]], .85),
    "=": _G([[(0, .4), (.8, .4)], [(0, .8), (.8, .8)]], .95),
    "≠": _G([[(0, .4), (.8, .4)], [(0, .8), (.8, .8)], [(.2, .05), (.62, 1.15)]], .95),
    "≈": _G([[(0, .75), (.2, .9), (.4, .75), (.6, .6), (.8, .75)], [(0, .35), (.2, .5), (.4, .35), (.6, .2), (.8, .35)]], .95),
    "±": _G([[(0, .75), (.8, .75)], [(.4, 1.15), (.4, .35)], [(0, .0), (.8, .0)]], .95),
    "∓": _G([[(0, .35), (.8, .35)], [(.4, .75), (.4, -.05)], [(0, 1.0), (.8, 1.0)]], .95),
    "≤": _G([[(.75, 1.15), (0, .7), (.75, .25)], [(0, -.05), (.75, -.05)]], .9),
    "≥": _G([[(0, 1.15), (.75, .7), (0, .25)], [(0, -.05), (.75, -.05)]], .9),
    "·": _G([[(.05, .6), (.1, .66), (.14, .6)]], .3),
    "∞": _G([[(.5, .6), (.75, .95), (1.0, .6), (.75, .25), (.5, .6), (.25, .95), (0, .6), (.25, .25), (.5, .6)]], 1.1),
    "→": _G([[(0, .6), (1.0, .6)], [(.7, .95), (1.0, .6), (.7, .25)]], 1.1),
    "⇒": _G([[(0, .45), (1.0, .45)], [(0, .78), (1.0, .78)], [(.7, 1.1), (1.05, .62), (.7, .1)]], 1.15),
    "∂": _G([[(.7, 1.4), (.3, 1.65), (.1, 1.2), (.1, .6), (.35, .1), (.65, .3), (.7, .8), (.5, 1.0), (.2, .9)]], .85),
    "∇": _G([[(0, 1.8), (.5, 0), (1, 1.8), (0, 1.8)]], 1.1),
    "°": _G([[(.3, .5), (.08, .32), (.3, .1), (.52, .32), (.3, .5)]], .55),
    "′": _G([[(.12, .55), (0, .05)]], .3),
    "∠": _G([[(.8, 1.0), (0, 0), (.8, 0)]], .95),
    "⊥": _G([[(0, 0), (.9, 0)], [(.45, 0), (.45, 1.3)]], 1.0),
    "∥": _G([[(.1, 0), (.1, 1.5)], [(.4, 0), (.4, 1.5)]], .6),
    "∈": _G([[(.8, 1.0), (.3, .9), (.1, .5), (.3, .1), (.8, 0)], [(.1, .5), (.7, .5)]], .95),
    "∝": _G([[(.0, .8), (.2, .3), (.5, .7), (.75, 1.0), (.95, .6), (.75, .3), (.5, .7)]], 1.05),
    "∴": _G([[(.3, 1.0), (.33, 1.04)], [(.0, .2), (.03, .24)], [(.6, .2), (.63, .24)]], .9),
    "…": _G([[(0, .05), (.03, .09)], [(.4, .05), (.43, .09)], [(.8, .05), (.83, .09)]], 1.0),
    "∼": _G([[(0, .6), (.25, .85), (.5, .6), (.75, .35), (1.0, .6)]], 1.1),
    "≡": _G([[(0, .2), (.8, .2)], [(0, .6), (.8, .6)], [(0, 1.0), (.8, 1.0)]], .95),
    "∑": Glyph(strokes=[[(1.0, 1.9), (.1, 1.9), (.65, .85), (.1, -.2), (1.0, -.2)]], adv=1.2),
    "∫": Glyph(strokes=[[(.75, 2.0), (.55, 2.15), (.4, 1.9), (.35, .9), (.3, -.2), (.1, -.3), (0, -.12)]], adv=.8),
    "∏": Glyph(strokes=[[(0, 1.9), (1.1, 1.9)], [(.2, 1.9), (.2, -.2)], [(.9, 1.9), (.9, -.2)]], adv=1.3),
    # Greek (x-height = 1)
    "α": _G([[(.72, .85), (.45, 1.0), (.15, .8), (.1, .4), (.3, .05), (.6, .25), (.75, .85), (.72, .5), (.8, .05)]], .9),
    "β": _G([[(.15, -.7), (.15, 1.4), (.45, 1.65), (.65, 1.3), (.4, .95), (.15, .9)], [(.4, .95), (.78, .7), (.72, .2), (.4, .0), (.15, .1)]], .9),
    "γ": _G([[(.05, 1.0), (.3, .6), (.45, .0), (.45, -.6)], [(.85, 1.0), (.45, .0)]], .9),
    "δ": _G([[(.5, 1.55), (.25, 1.38), (.45, 1.1), (.7, .75), (.6, .2), (.3, .05), (.1, .4), (.3, .9), (.6, 1.0)]], .85),
    "ε": _G([[(.7, .9), (.4, 1.0), (.15, .8), (.4, .55), (.15, .3), (.4, .05), (.7, .12)]], .8),
    "θ": _G([[(.45, 1.7), (.15, 1.4), (.1, .6), (.4, .05), (.7, .5), (.75, 1.2), (.45, 1.7)], [(.2, .9), (.65, .9)]], .9),
    "λ": _G([[(.15, 1.7), (.35, 1.0), (.75, .0)], [(.3, .5), (.05, .0)]], .9),
    "μ": _G([[(.1, 1.0), (.1, -.6)], [(.1, .4), (.3, .05), (.55, .15), (.65, .95)], [(.65, .95), (.7, .1), (.9, .05)]], .95),
    "ν": _G([[(.05, 1.0), (.3, .0), (.75, 1.0)]], .85),
    "π": _G([[(0, 1.0), (.95, 1.0)], [(.3, 1.0), (.25, .0)], [(.65, 1.0), (.68, .08), (.85, .1)]], 1.05),
    "ρ": _G([[(.15, -.7), (.15, .9)], [(.15, .5), (.4, .95), (.7, .8), (.7, .3), (.4, .05), (.15, .3)]], .85),
    "σ": _G([[(.85, 1.0), (.45, 1.0), (.15, .75), (.12, .35), (.35, .05), (.65, .2), (.7, .6), (.5, .95)]], .9),
    "τ": _G([[(0, .95), (.8, .95)], [(.4, .95), (.4, .1), (.62, .02)]], .85),
    "φ": _G([[(.55, 1.55), (.5, -.5)], [(.5, 1.0), (.2, .9), (.1, .5), (.3, .1), (.6, .1), (.85, .5), (.7, .9), (.5, 1.0)]], 1.0),
    "ω": _G([[(.05, .95), (.1, .4), (.3, .05), (.5, .4), (.6, .05), (.85, .35), (.9, .95)]], 1.0),
    "Δ": Glyph(strokes=[[(.5, 1.85), (0, 0), (1.0, 0), (.5, 1.85)]], adv=1.15),
    "Σ": Glyph(strokes=[[(.95, 1.85), (.1, 1.85), (.6, .9), (.1, 0), (.95, 0)]], adv=1.1),
    "Ω": Glyph(strokes=[[(.15, 0), (.35, 0), (.1, .5), (.1, 1.2), (.45, 1.85), (.8, 1.2), (.8, .5), (.55, 0), (.8, 0)]], adv=1.05),
    "Π": Glyph(strokes=[[(0, 1.85), (1.0, 1.85)], [(.2, 1.85), (.2, 0)], [(.8, 1.85), (.8, 0)]], adv=1.15),
    "Γ": Glyph(strokes=[[(.1, 0), (.1, 1.85), (.8, 1.85)]], adv=.95),
    "Λ": Glyph(strokes=[[(0, 0), (.5, 1.85), (1, 0)]], adv=1.15),
    "Θ": Glyph(strokes=[[(.5, 1.85), (.1, 1.3), (.05, .5), (.5, 0), (.95, .5), (.9, 1.3), (.5, 1.85)], [(.3, .9), (.7, .9)]], adv=1.1),
}
GREEK = {"alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ", "epsilon": "ε", "varepsilon": "ε", "theta": "θ",
         "lambda": "λ", "mu": "μ", "nu": "ν", "pi": "π", "rho": "ρ", "sigma": "σ", "tau": "τ", "phi": "φ",
         "varphi": "φ", "omega": "ω", "Delta": "Δ", "Sigma": "Σ", "Omega": "Ω", "Pi": "Π", "Gamma": "Γ",
         "Lambda": "Λ", "Theta": "Θ"}
SYMCMD = {"times": "×", "div": "÷", "pm": "±", "mp": "∓", "cdot": "·", "leq": "≤", "le": "≤", "geq": "≥", "ge": "≥",
          "neq": "≠", "ne": "≠", "approx": "≈", "infty": "∞", "to": "→", "rightarrow": "→", "Rightarrow": "⇒",
          "implies": "⇒", "partial": "∂", "nabla": "∇", "angle": "∠", "perp": "⊥", "parallel": "∥", "in": "∈",
          "therefore": "∴", "degree": "°", "circ": "°", "prime": "′", "checkmark": "✓", "sum": "∑", "int": "∫",
          "prod": "∏", "propto": "∝", "ldots": "…", "cdots": "…", "dots": "…", "sim": "∼", "equiv": "≡"}
U2L = {"×": r"\times", "÷": r"\div", "±": r"\pm", "≤": r"\leq", "≥": r"\geq", "≠": r"\neq", "≈": r"\approx",
       "∞": r"\infty", "→": r"\to", "⇒": r"\Rightarrow", "∂": r"\partial", "∇": r"\nabla", "∠": r"\angle",
       "°": r"\degree", "′": r"\prime", "∑": r"\sum", "∫": r"\int", "✓": r"\checkmark", "²": "^2", "³": "^3",
       "¹": "^1", "⁴": "^4", "·": r"\cdot", "−": "-", "∈": r"\in", "∴": r"\therefore"}
U2L.update({v: "\\" + k for k, v in GREEK.items() if k not in ("varepsilon", "varphi")})
_RELS = set("=≠≈≤≥<>→⇒∈∝≡∼")
_BINS = set("+-−×÷±∓·")
_OPNAMES = {"sin", "cos", "tan", "cot", "sec", "csc", "log", "ln", "exp", "lim", "min", "max", "det", "arcsin", "arccos", "arctan"}


def _normalize(text: str) -> str:
    """Turn bare unicode math (√x, x², ≤, π ...) into $...$ segments the math layout understands."""
    import re
    text = re.sub(r"√\s*(\([^)]*\)|[A-Za-z0-9.]+)", lambda m: "$\\sqrt{%s}$" % m.group(1).strip("()"), text)
    out, ins = [], False
    for ch in text:
        if ch == "$":
            ins = not ins
        if not ins and ch in U2L:
            out.append("$" + U2L[ch] + "$")
        else:
            out.append(ch)
    return "".join(out)


def _warp(P, w):
    amp, f1, f2, p1, p2 = w
    dx = amp * (np.sin(f1 * P[:, 1] * 2 + p1) + .5 * np.sin(f2 * P[:, 0] * 3 + p2))
    dy = amp * (np.sin(f2 * P[:, 0] * 2 + p2) + .5 * np.sin(f1 * P[:, 1] * 3 + p1))
    return P + np.stack([dx, dy], 1)


def _resample(s, n):
    d = np.r_[0, np.cumsum(np.hypot(*np.diff(s, axis=0).T))]
    if d[-1] < 1e-9:
        return np.repeat(s[:1], n, 0)
    u = np.linspace(0, d[-1], n)
    return np.stack([np.interp(u, d, s[:, 0]), np.interp(u, d, s[:, 1])], 1)


def _blend(a, b, t):
    """Morph two real samples of the same letter (same stroke count, compatible starts)."""
    if len(a["s"]) != len(b["s"]):
        return None
    out = []
    for sa, sb in zip(a["s"], b["s"]):
        if np.hypot(*(sa[0] - sb[0])) > .7:
            return None
        n = int(np.clip(max(_plen(sa), _plen(sb)) / .05, 3, 60))
        out.append(_resample(sa, n) * (1 - t) + _resample(sb, n) * t)
    return dict(w=a["w"] * (1 - t) + b["w"] * t, s=out)


def _plen(s):
    return float(np.hypot(*np.diff(s, axis=0).T).sum()) if len(s) > 1 else 0.0


class _Box:
    __slots__ = ("w", "asc", "desc", "items")

    def __init__(self, w=0.0, asc=0.0, desc=0.0, items=None):
        self.w, self.asc, self.desc, self.items = w, asc, desc, items or []


class _Ctx:
    def __init__(self, rng, st, n):
        self.rng, self.st, self.n, self.prog, self.g = rng, st, n, 0.0, 0
        self.loc, self.lift, self.fat, self.ms, self.c = 1.0, 0.0, .5, 1.0, None
        self.wmul, self.wslant, self.wslope, self.wx0 = 1.0, 0.0, 0.0, 0.0
        self.D = dict(size=_Drift(rng, n, 7, st.size_drift), slant=_Drift(rng, n, 9, st.slant_drift_deg),
                      base=_Drift(rng, n, 5, st.baseline_drift), width=_Drift(rng, n, 8, st.width_drift),
                      speed=_Drift(rng, n, 10, st.speed_drift), space=_Drift(rng, n, 12, .15))


class _MathParser:
    """LaTeX subset -> tree.  Nodes: ('sym',c) ('grp',[n]) ('ss',base,sub,sup) ('frac',n,d) ('sqrt',idx,body)
    ('acc',kind,body) ('lr',open,[n],close) ('text',str) ('op',name) ('space',w)"""

    def __init__(self, s):
        self.s, self.i = s, 0

    def _ws(self):
        while self.i < len(self.s) and self.s[self.i] in " \t":
            self.i += 1

    def seq(self, stop=None):
        out = []
        while True:
            self._ws()
            if self.i >= len(self.s) or (stop and self.s.startswith(stop, self.i)) or (stop == "}" and self.s[self.i] == "}"):
                return out
            out.append(self.script())

    def script(self):
        base = self.atom()
        sub = sup = None
        while True:
            self._ws()
            c = self.s[self.i] if self.i < len(self.s) else ""
            if c == "^":
                self.i += 1; sup = self.atom()
            elif c == "_":
                self.i += 1; sub = self.atom()
            elif c == "'":
                self.i += 1; sup = ("sym", "′")
            else:
                break
        return ("ss", base, sub, sup) if (sub or sup) else base

    def group(self):
        self._ws()
        if self.i < len(self.s) and self.s[self.i] == "{":
            self.i += 1
            body = self.seq("}")
            self.i += 1
            return ("grp", body)
        return self.atom()

    def raw_group(self):
        self._ws()
        if self.i < len(self.s) and self.s[self.i] == "{":
            j = self.s.find("}", self.i)
            j = len(self.s) if j < 0 else j
            t = self.s[self.i + 1:j]; self.i = j + 1
            return t
        return ""

    def atom(self):
        self._ws()
        if self.i >= len(self.s):
            return ("grp", [])
        c = self.s[self.i]
        if c == "{":
            return self.group()
        if c in "^_":
            return ("grp", [])
        if c != "\\":
            self.i += 1
            return ("sym", c)
        self.i += 1
        j = self.i
        while j < len(self.s) and self.s[j].isalpha():
            j += 1
        name = self.s[self.i:j] if j > self.i else self.s[self.i:self.i + 1]
        self.i = max(j, self.i + 1)
        if name in ("frac", "dfrac", "tfrac"):
            return ("frac", self.group(), self.group())
        if name == "sqrt":
            self._ws(); idx = None
            if self.i < len(self.s) and self.s[self.i] == "[":
                j = self.s.find("]", self.i); idx = ("sym", self.s[self.i + 1:j]); self.i = j + 1
            return ("sqrt", idx, self.group())
        if name in ("vec", "hat", "bar", "overline", "dot", "tilde", "underline"):
            return ("acc", name, self.group())
        if name in ("text", "mathrm", "mathbf", "textbf", "operatorname"):
            return ("text", self.raw_group())
        if name == "left":
            op = self.s[self.i] if self.i < len(self.s) else "("; self.i += 1
            body = self.seq("\\right")
            cl = "."
            if self.s.startswith("\\right", self.i):
                self.i += 6; cl = self.s[self.i] if self.i < len(self.s) else ")"; self.i += 1
            return ("lr", op, body, cl)
        if name in _OPNAMES:
            return ("text", name)
        if name in GREEK:
            return ("sym", GREEK[name])
        if name in SYMCMD:
            return ("sym", SYMCMD[name])
        if name in (",", ";", ":", "quad", "qquad", " ", "!"):
            return ("space", {"quad": 1.2, "qquad": 2.4, "!": 0.0}.get(name, .5))
        if name in ("%", "{", "}", "&", "#", "_", "$"):
            return ("sym", {"{": "(", "}": ")"}.get(name, name))
        return ("text", name)


# =============================================================================
# 5. The handwriter: text -> Plan -> execution on a Cua computer
# =============================================================================


class Handwriter:
    def __init__(self, style: Optional[WriterStyle] = None, seed: Optional[int] = None, hand: str = "bank"):
        self.style = style or WriterStyle.from_seed(seed, hand)
        self.glyphs = CURSIVE if self.style.hand == "cursive" else CASUAL
        self.use_bank = self.style.hand == "bank"
        self._habit_rng = np.random.default_rng(seed)
        self._habit: dict = {}
        self._met: dict = {}

    # ---- glyph access ---------------------------------------------------------
    def _bank_glyph(self, key, rng, yoff=0.0):
        insts = BANK.get(key)
        if not insts:
            return None
        n = len(insts)
        w = self._habit.get(key)
        if w is None:      # this writer's habits: favours some of their own samples, but not exclusively
            w = self._habit_rng.dirichlet(np.ones(n) * 1.6)
            self._habit[key] = w
        i = int(rng.choice(n, p=w))
        g = insts[i]
        if n > 1 and rng.random() < .6:
            j = int(rng.choice([k for k in range(n) if k != i]))
            b = _blend(g, insts[j], float(rng.uniform(.2, .8)))
            if b:
                g = b
        sb = .22 if key.islower() else .3
        return Glyph(strokes=[s + [0, yoff] for s in g["s"]], adv=g["w"] + sb, bank=True)

    def _glyph(self, ch, rng):
        if ch in BANK_ALIAS:
            k, yo = BANK_ALIAS[ch]
            g = self._bank_glyph(k, rng, yo)
            if g:
                g.adv = max(g.adv, .75)
                return g
        if ch in MSYM:
            return MSYM[ch]
        if self.use_bank and ch in BANK and rng.random() < self.style.bank_ratio:
            return self._bank_glyph(ch, rng)
        return self.glyphs.get(ch)

    def _metrics(self, ch):
        """(advance, ascent, descent) in x-heights for layout."""
        if ch in self._met:
            return self._met[ch]
        st = self.style
        sets, adv = None, None
        if ch in BANK_ALIAS:
            k, yo = BANK_ALIAS[ch]
            sets = [[s + [0, yo] for s in g["s"]] for g in BANK.get(k, [])]
            adv = float(np.mean([g["w"] for g in BANK[k]])) + .3 if k in BANK else None
        elif ch not in MSYM and self.use_bank and ch in BANK:
            sets = [g["s"] for g in BANK[ch]]
            adv = float(np.mean([g["w"] for g in BANK[ch]])) + (.22 if ch.islower() else .3)
        if not sets:
            gl = MSYM.get(ch) or self.glyphs.get(ch)
            if gl is None:
                self._met[ch] = (.8, 1.0, 0.0)
                return self._met[ch]
            ss = [gl.pts] if gl.pts is not None else gl.strokes
            sets = [[np.asarray(s, float) * [1, gl.ys] for s in ss]] if ss else [[np.zeros((1, 2))]]
            adv = gl.adv
        ys = np.concatenate([np.concatenate([s[:, 1] for s in g]) for g in sets])
        m = (adv * st.width_scale, float(max(ys.max(), 0)), float(max(-ys.min(), 0)))
        self._met[ch] = m
        return m

    # ---- per-letter variation ---------------------------------------------------
    def _tf_for(self, gl, x, by, xh, g, ctx):
        st, r = ctx.st, ctx.rng
        k = ctx.loc
        s = (1 + k * (ctx.D["size"](g) + r.normal(0, st.size_jit))) * (1 - .08 * ctx.fat * ctx.prog) * ctx.wmul
        wsc = st.width_scale * (1 + k * (ctx.D["width"](g) + r.normal(0, .05)))
        sh = math.tan(math.radians(st.slant_deg + ctx.wslant + k * ctx.D["slant"](g) +
                                   r.normal(0, (2.5 if st.hand != "cursive" else 1.2) * ctx.ms * k)))
        Lu = st.loop_up * (1 + r.normal(0, .07))
        Ld = st.loop_down * (1 + r.normal(0, .08))
        if gl.bank:
            Lu, Ld = 1 + (Lu - 1) * .5, 1 + (Ld - 1) * .5
        warp = (st.shape_sigma * 1.3 * k, r.uniform(.8, 2.2), r.uniform(.8, 2.2), r.uniform(0, 6.28), r.uniform(0, 6.28))

        def tf(p, perturb=True):
            p = np.asarray(p, float).copy()
            if gl.bank:
                p = _warp(p, warp)
            elif perturb and len(p) > 2:
                off = r.normal(0, st.shape_sigma, p.shape)
                off = (off + np.roll(off, 1, 0) + np.roll(off, -1, 0)) / 3
                off[0] *= .3
                off[-1] *= .3
                p += off
            y = p[:, 1].copy()
            y = np.where(y > 1.0, 1.0 + (y - 1.0) * Lu, np.where(y < 0, y * Ld, y))
            X = p[:, 0] * wsc * s
            Y = y * s * gl.ys
            X = X + Y * sh
            return np.stack([x + X * xh, by - Y * xh], 1)

        adv = gl.adv * wsc * s * xh + (ctx.D["space"](g) + r.normal(0, .03)) * xh * .35 * k
        return tf, adv, s, wsc, sh

    # ---- math layout ------------------------------------------------------------
    @staticmethod
    def _shift(items, dx, dy):
        out = []
        for it in items:
            if it[0] in ("g",):
                out.append(("g", it[1], it[2] + dx, it[3] + dy, it[4]))
            elif it[0] == "line":
                out.append(("line", [(a + dx, b + dy) for a, b in it[1]]))
            elif it[0] == "paren":
                out.append(("paren", it[1], it[2] + dx, it[3] + dy, it[4] + dy))
            elif it[0] == "rad":
                out.append(("rad", it[1] + dx, it[2] + dy, it[3] + dy, it[4]))
        return out

    def _lay_seq(self, nodes, sc):
        items, x, asc, desc = [], 0.0, 0.0, 0.0
        for i, nd in enumerate(nodes):
            b = self._lay(nd, sc)
            ch = nd[1] if nd[0] == "sym" else None
            pl = pr = 0.0
            prev_op = i > 0 and nodes[i - 1][0] == "sym" and (nodes[i - 1][1] in _RELS or nodes[i - 1][1] in _BINS)
            if ch in _RELS:
                pl = pr = .42 * sc
            elif ch in _BINS and i > 0 and not prev_op:
                pl = pr = .3 * sc
            elif nd[0] == "text" and len(nd[1]) > 1:
                pr = .12 * sc
            elif nd[0] == "frac":
                pl = pr = .12 * sc
            items += self._shift(b.items, x + pl, 0)
            x += pl + b.w + pr
            asc, desc = max(asc, b.asc), max(desc, b.desc)
        return _Box(x, asc, desc, items)

    def _lay(self, nd, sc):
        k = nd[0]
        if k == "sym":
            ch = nd[1]
            adv, asc, desc = self._metrics(ch)
            if ch in "°′":
                up = (1.05 if ch == "°" else 1.2) * sc
                return _Box(adv * sc, asc * sc + up, 0, [("g", ch, 0, up, sc)])
            if ch in "∑∏∫":
                s2 = sc * 1.3
                return _Box(adv * s2, asc * s2 - .25 * sc, desc * s2 + .25 * sc, [("g", ch, 0, -.25 * sc, s2)])
            return _Box(adv * sc, asc * sc, desc * sc, [("g", ch, 0, 0, sc)])
        if k == "grp":
            return self._lay_seq(nd[1], sc)
        if k == "space":
            return _Box(nd[1] * sc, 0, 0)
        if k == "text":
            return self._lay_seq([("sym", c) for c in nd[1] if c != " "], sc)
        if k == "ss":
            _, base, sub, sup = nd
            B = self._lay(base, sc) if base else _Box(0, 1.0 * sc, 0)
            ssc = sc * (.9 if (sup and sup[0] == "sym" and sup[1] in "°′") else .52)
            S = self._lay(sup, ssc) if sup else None
            U = self._lay(sub, ssc) if sub else None
            if base and base[0] == "sym" and base[1] in "∑∏":
                w = max(B.w, S.w if S else 0, U.w if U else 0)
                items = self._shift(B.items, (w - B.w) / 2, 0)
                asc, desc = B.asc, B.desc
                if S:
                    y = B.asc + .12 * sc + S.desc
                    items += self._shift(S.items, (w - S.w) / 2, y); asc = y + S.asc
                if U:
                    y = -(B.desc + .12 * sc + U.asc)
                    items += self._shift(U.items, (w - U.w) / 2, y); desc = -y + U.desc
                return _Box(w, asc, desc, items)
            x = B.w + .05 * sc
            items, asc, desc, wmax = list(B.items), B.asc, B.desc, 0.0
            if S:
                ys_ = .62 * B.asc + .46 * sc
                if sup[0] == "sym" and sup[1] in "°′":
                    ys_ = max(0.0, B.asc - 1.15 * sc)
                items += self._shift(S.items, x, ys_); asc = max(asc, ys_ + S.asc); wmax = max(wmax, S.w)
            if U:
                yu = -(.32 * sc + (.2 * sc if B.desc > .3 * sc else 0))
                items += self._shift(U.items, x, yu); desc = max(desc, -yu + U.desc); wmax = max(wmax, U.w)
            return _Box(x + wmax, asc, desc, items)
        if k == "frac":
            N, D = self._lay(nd[1], sc * .9), self._lay(nd[2], sc * .9)
            w = max(N.w, D.w) + .4 * sc
            axis = .55 * sc
            ny = axis + .42 * sc + N.desc
            dy = axis - .5 * sc - D.asc
            items = [("line", [(0, axis), (w, axis)])]
            items += self._shift(N.items, (w - N.w) / 2, ny) + self._shift(D.items, (w - D.w) / 2, dy)
            return _Box(w, ny + N.asc, -(dy - D.desc), items)
        if k == "sqrt":
            Bd = self._lay(nd[2], sc)
            hook = .7 * sc
            top, bot = Bd.asc + .2 * sc, -Bd.desc - .1 * sc
            w = hook + Bd.w + .25 * sc
            items = [("rad", 0, bot, top, w)] + self._shift(Bd.items, hook, 0)
            if nd[1]:
                I = self._lay(nd[1], sc * .5)
                items += self._shift(I.items, 0, top - I.asc - .1 * sc)
            return _Box(w, top + .1 * sc, -bot, items)
        if k == "acc":
            Bd = self._lay(nd[2], sc)
            w, y = Bd.w, Bd.asc + .22 * sc
            kind = nd[1]
            if kind == "vec":
                ex = [("line", [(.0, y), (w, y)]), ("line", [(w - .28 * sc, y + .2 * sc), (w, y), (w - .28 * sc, y - .16 * sc)])]
            elif kind == "hat":
                ex = [("line", [(w * .15, y), (w * .5, y + .4 * sc), (w * .85, y)])]
            elif kind == "dot":
                ex = [("line", [(w * .5, y + .1 * sc), (w * .5 + .05 * sc, y + .02 * sc)])]
            elif kind == "tilde":
                ex = [("line", [(0, y), (w * .3, y + .2 * sc), (w * .6, y - .05 * sc), (w, y + .15 * sc)])]
            elif kind == "underline":
                ex = [("line", [(0, -Bd.desc - .15 * sc), (w, -Bd.desc - .15 * sc)])]
            else:
                ex = [("line", [(0, y), (w, y)])]
            return _Box(w, Bd.asc + .55 * sc, Bd.desc, Bd.items + ex)
        if k == "lr":
            Bd = self._lay_seq(nd[2], sc)
            y1, y0 = Bd.asc + .12 * sc, -Bd.desc - .12 * sc
            items, x = [], 0.0
            if nd[1] != ".":
                items.append(("paren", nd[1], 0, y0, y1)); x = .5 * sc
            items += self._shift(Bd.items, x, 0)
            x += Bd.w
            if nd[3] != ".":
                items.append(("paren", nd[3], x + .05 * sc, y0, y1)); x += .5 * sc
            return _Box(x, y1, -y0, items)
        return _Box(0, 0, 0)

    def _lay_math(self, s):
        return self._lay_seq(_MathParser(s).seq(), 1.0)

    # ---- emitting ---------------------------------------------------------------
    def _place(self, gl, x, by, xh, g, ctx):
        tf = self._tf_for(gl, x, by, xh, g, ctx)[0]
        if gl.cursive:
            return [tf(gl.pts)]
        return [tf(s) for s in gl.strokes if len(s) >= 2]

    def _radical(self, x, by, xh, x0, ybot, ytop, wu, rng):
        inst = BANK[("RAD_A", "RAD_B", "RAD_C")[int(rng.integers(3))]][0]
        strokes = [s.copy() for s in inst["s"]]
        im = max(range(len(strokes)), key=lambda q: _plen(strokes[q]))
        main = strokes[im]
        others = [t for q, t in enumerate(strokes) if q != im]
        if main[0, 0] > main[-1, 0]:
            main = main[::-1]
        allp = np.vstack(strokes)
        xmin, ymin, ymax = allp[:, 0].min(), allp[:, 1].min(), allp[:, 1].max()
        ci = int(np.argmax(main[:, 1] >= ymax - .12 * (ymax - ymin)))
        ci = max(1, min(ci, len(main) - 2))
        xc, xend = main[ci, 0], max(main[:, 0].max(), main[ci, 0] + 1e-3)
        sy = (ytop - ybot) / max(ymax - ymin, 1e-3)
        sx = min(.8, sy * .7)

        def rise(P):
            return np.stack([x0 + (P[:, 0] - xmin) * sx, ybot + (P[:, 1] - ymin) * sy], 1)
        left = rise(main[:ci + 1])
        bar = main[ci:]
        bx = left[-1, 0] + (bar[:, 0] - xc) / (xend - xc) * max(x0 + wu - left[-1, 0], .3)
        by_ = ytop + (bar[:, 1] - ymax) * .8
        mp = np.vstack([left, np.stack([bx, by_], 1)[1:]])
        out = [mp] + [rise(o) for o in others]
        return [np.stack([x + P[:, 0] * xh, by - P[:, 1] * xh], 1) for P in out]

    def _emit_box(self, box, x, by, xh, ctx, items, skipped):
        rng, st = ctx.rng, ctx.st
        for it in box.items:
            k = it[0]
            pause = float(rng.lognormal(math.log(.06), .45))
            if k == "g":
                _, ch, ax, ay, sc = it
                gl = self._glyph(ch, rng)
                if gl is None:
                    skipped.append(ch)
                    continue
                ctx.g += 1
                ctx.prog = ctx.g / ctx.n
                yb = by - ay * xh + rng.normal(0, st.letter_base * .6) * xh * sc
                for j, s in enumerate(self._place(gl, x + ax * xh, yb, xh * sc, ctx.g, ctx)):
                    items.append(dict(pts=s, kind="ink", pause=pause if j == 0 else float(rng.uniform(.02, .06)),
                                      vmul=1.0, line_start=False))
            elif k == "line":
                P = np.array([[x + a * xh, by - b * xh] for a, b in it[1]])
                if len(P) == 2:
                    d = P[1] - P[0]
                    P = np.array([P[0] - d * rng.uniform(0, .04), P[0] + d / 2, P[1] + d * rng.uniform(0, .05)])
                    P = _bow(P, rng, .03)
                else:
                    P = _bow(P, rng, .02)
                items.append(dict(pts=P, kind="ink", pause=pause, vmul=1.15, line_start=False))
            elif k == "paren":
                _, ch, ax, y0, y1 = it
                h = y1 - y0
                mid = (y0 + y1) / 2
                if ch in "([{":
                    pts = [(.34, y1), (.12, y1 - .27 * h), (.05, mid), (.12, y0 + .27 * h), (.34, y0)]
                    if ch == "[":
                        pts = [(.3, y1), (.08, y1), (.08, y0), (.3, y0)]
                elif ch in ")]}":
                    pts = [(.06, y1), (.28, y1 - .27 * h), (.35, mid), (.28, y0 + .27 * h), (.06, y0)]
                    if ch == "]":
                        pts = [(.1, y1), (.32, y1), (.32, y0), (.1, y0)]
                else:
                    pts = [(.15, y1), (.15, y0)]
                P = np.array([[x + (ax + a) * xh, by - b * xh] for a, b in pts])
                items.append(dict(pts=_bow(P, rng, .02), kind="ink", pause=pause, vmul=1.1, line_start=False))
            elif k == "rad":
                _, x0, ybot, ytop, wu = it
                for j, s in enumerate(self._radical(x, by, xh, x0, ybot, ytop, wu, rng)):
                    items.append(dict(pts=s, kind="ink", pause=pause if j == 0 else .03, vmul=1.0, line_start=False))

    # ---- text + math composition -------------------------------------------------
    _TAGRE = None

    @staticmethod
    def _tokenize(para):
        """Words / math tokens with inline-markup attributes.
        Tags: <big> <small> <neat> <messy> <fast> <slow> <u> <s> <circle> <slant=N> <size=F> <lift=F>"""
        import re
        tag = re.compile(r"<(/?)(big|small|u|s|circle|neat|messy|fast|slow|slant|size|lift)(?:=(-?[\d.]+))?>$")
        toks, sp, stack, did = [], True, [], [0]

        def attrs():
            a = dict(size=1.0, messy=1.0, slant=0.0, lift=0.0, speed=1.0, decos=[])
            for name, val, i in stack:
                if name == "big": a["size"] *= val or 1.35
                elif name == "small": a["size"] *= val or .7
                elif name == "size": a["size"] *= val or 1.0
                elif name == "neat": a["messy"] *= .35
                elif name == "messy": a["messy"] *= 1.8
                elif name == "fast": a["speed"] *= 1.5
                elif name == "slow": a["speed"] *= .65
                elif name == "slant": a["slant"] += val or 0.0
                elif name == "lift": a["lift"] += val or 0.0
                elif name in ("u", "s", "circle"): a["decos"].append((name, i))
            return a
        for part in re.split(r"(\$\$.+?\$\$|\$[^$]+\$)", para, flags=re.S):
            if not part:
                continue
            if part.startswith("$"):
                toks.append(dict(kind="m", s=part.strip("$"), sp=sp, a=attrs())); sp = False
                continue
            for piece in re.split(r"(<[^<>]+>)", part):
                m = tag.match(piece) if piece.startswith("<") else None
                if m:
                    if m.group(1):
                        for q in range(len(stack) - 1, -1, -1):
                            if stack[q][0] == m.group(2):
                                stack.pop(q); break
                    else:
                        did[0] += 1
                        stack.append((m.group(2), float(m.group(3)) if m.group(3) else None, did[0]))
                    continue
                for mm in re.finditer(r"(\s+)|(\S+)", piece):
                    if mm.group(1):
                        sp = True
                    else:
                        toks.append(dict(kind="w", s=mm.group(2), sp=sp, a=attrs())); sp = False
        return toks

    @staticmethod
    def _typo(word, rng):
        k = int(rng.integers(3))
        i = int(rng.integers(1, len(word) - 1))
        if k == 0:
            return word[:i] + word[i + 1] + word[i] + word[i + 2:]
        if k == 1:
            return word[:i] + word[i + 1:]
        return word[:i] + word[i] + word[i:]

    @staticmethod
    def _flick(P, xh, rng):
        if len(P) < 4:
            return P
        d = P[-1] - P[-4]
        n = np.hypot(*d)
        if n < 1e-6:
            return P
        d = d / n
        L = rng.uniform(.1, .24) * xh
        curl = np.array([-d[1], d[0]]) * rng.normal(0, .25) * L
        return np.vstack([P, P[-1] + d * L * .5 + curl * .5, P[-1] + d * L + curl])

    def _deco_stroke(self, kind, x0, x1, ytop, ybot, base_y, xh, rng, messy):
        w = max(x1 - x0, .3 * xh)
        if kind == "u":
            y = base_y + (.28 + rng.normal(0, .04)) * xh
            n = max(3, int(w / (1.2 * xh)) + 2)
            xs = np.linspace(x0 - .1 * xh, x1 + rng.uniform(.0, .25) * xh, n)
            ys = y + np.cumsum(rng.normal(0, .02 * xh * messy, n)) + np.linspace(0, rng.normal(0, .06) * xh, n)
            pts = np.stack([xs, ys], 1)
        elif kind == "s":
            y = base_y - (.42 + rng.normal(0, .05)) * xh
            n = max(3, int(w / (1.2 * xh)) + 2)
            xs = np.linspace(x0 - .1 * xh, x1 + .15 * xh, n)
            ys = y + np.linspace(0, rng.normal(0, .12) * xh, n) + rng.normal(0, .02 * xh * messy, n)
            pts = np.stack([xs, ys], 1)
        else:   # circle
            cx, cy = (x0 + x1) / 2, (ytop + ybot) / 2
            rx, ry = w / 2 + .45 * xh, (ybot - ytop) / 2 + .35 * xh
            a0 = rng.uniform(-2.4, -1.4)
            tt = np.linspace(0, 2 * math.pi * rng.uniform(1.08, 1.2), 16)
            grow = 1 + .06 * tt / (2 * math.pi) * rng.choice([-1, 1])
            ang = a0 - tt
            pts = np.stack([cx + rx * grow * np.cos(ang), cy + ry * grow * np.sin(ang)], 1)
            pts += rng.normal(0, .015 * xh * messy, pts.shape)
        return dict(pts=pts, kind="ink", pause=float(rng.uniform(.08, .22)), vmul=1.3, line_start=False)

    def _scribble(self, x0, x1, ytop, ybot, xh, rng):
        mid = (ytop + ybot) / 2
        n = max(3, int((x1 - x0) / (.45 * xh)) + 2)
        xs = np.linspace(x0 - .1 * xh, x1 + .1 * xh, 2 * n)
        ys = mid + np.where(np.arange(2 * n) % 2 == 0, -1, 1) * (.28 + rng.normal(0, .05, 2 * n)) * xh
        ys = ys + np.linspace(rng.normal(0, .08) * xh, rng.normal(0, .08) * xh, 2 * n)
        return dict(pts=np.stack([xs, ys], 1), kind="ink", pause=float(rng.uniform(.15, .35)), vmul=1.6, line_start=False)

    def _compose(self, text, x0, y0, xh, max_width, line_height, rng, st, c):
        text = _normalize(text)
        ctx = _Ctx(rng, st, len(text) + 16)
        ctx.c, ctx.fat = c, c.fatigue
        ctx.ms = c.messiness * c.slant_variation
        ctx.wmul, ctx.wslant, ctx.wslope, ctx.wx0, ctx.loc, ctx.lift = 1.0, 0.0, 0.0, 0.0, 1.0, 0.0
        m_, sv, pv = c.messiness, c.messiness * c.size_variation, c.messiness * c.position_variation
        items, skipped = [], []
        ycur = y0
        widest = 0.0
        paras = text.split("\n")
        for pi, para in enumerate(paras):
            toks = self._tokenize(para)
            for t in toks:
                sz = t["a"]["size"]
                if t["kind"] == "m":
                    t["box"] = self._lay_math(t["s"])
                    t["w"] = t["box"].w * xh * sz
                else:
                    t["w"] = sum(self._metrics(ch)[0] for ch in t["s"] if ch in BANK or ch in MSYM or ch in self.glyphs) * xh * sz
            chunks = []
            for t in toks:
                if t["sp"] or not chunks:
                    chunks.append([t])
                else:
                    chunks[-1].append(t)
            lines, cur_l, cw = [], [], 0.0
            gap_est = (st.gap_min + .5 * st.gap_var) * xh
            for ch_ in chunks:
                w = sum(t["w"] for t in ch_)
                if cur_l and max_width and cw + gap_est + w > max_width:
                    lines.append((cur_l, cw)); cur_l, cw = [], 0.0
                cw += (gap_est if cur_l else 0) + w
                cur_l.append(ch_)
            lines.append((cur_l, cw))
            for li, (line, lw) in enumerate(lines):
                flat = [t for ch_ in line for t in ch_]
                smax = max([1.0] + [t["a"]["size"] for t in flat])
                asc_u = max([2.0 * smax] + [t["box"].asc * t["a"]["size"] + .1 for t in flat if t["kind"] == "m"])
                desc_u = max([.8 * smax] + [t["box"].desc * t["a"]["size"] + .1 for t in flat if t["kind"] == "m"])
                slope = math.tan(math.radians((st.line_slope_deg if c.line_slope is None else c.line_slope) + rng.normal(0, .5 * pv)))
                base0 = ycur + asc_u * xh
                boxw = max_width or 0
                if c.align == "center":
                    x = x0 + ((boxw - lw) / 2 if boxw else -lw / 2)
                elif c.align == "right":
                    x = x0 + ((boxw - lw) if boxw else -lw)
                else:
                    x = x0
                if li or pi:
                    x += rng.normal(0, .25 * pv) * xh
                xl = x
                widest = max(widest, lw)
                state = dict(cur=None, prev=None, marks=[], first=True)
                decos = {}

                def baseline(xx, gg, xh_=xh):
                    return (base0 - slope * (xx - xl) - ctx.wslope * (xx - ctx.wx0)
                            - ctx.D["base"](gg) * xh_ * ctx.loc - ctx.lift * xh_)

                def flush(pause):
                    if state["cur"] is not None and len(state["cur"]) >= 2:
                        items.append(dict(pts=np.array(state["cur"]), kind="ink", pause=pause, vmul=1.0,
                                          line_start=state["first"], letter=state.get("lastch")))
                        state["first"] = False
                        pause = 0.0
                    state["cur"], state["prev"] = None, None
                    order = list(range(len(state["marks"])))
                    if rng.random() < .25:
                        rng.shuffle(order)
                    for kk in order:
                        items.append(state["marks"][kk] | dict(pause=float(rng.uniform(.02, .07))))
                    state["marks"].clear()

                def put_word(word, x, join_in, xh_):
                    prevch = None
                    for ci, ch in enumerate(word):
                        gl = self._glyph(ch, rng)
                        if gl is None:
                            skipped.append(ch)
                            continue
                        last = ci == len(word) - 1
                        ctx.g += 1
                        ctx.prog = ctx.g / ctx.n
                        g = ctx.g
                        by = baseline(x, g, xh_) + rng.normal(0, st.letter_base * (1 + ctx.fat * ctx.prog) * ctx.loc) * xh_
                        tf, adv, s_, wsc, sh = self._tf_for(gl, x, by, xh_, g, ctx)
                        if gl.cursive:
                            for m in gl.marks:
                                if rng.random() < st.miss_rate:
                                    continue
                                q = tf([[m[1], m[2]]], perturb=False)[0]
                                if m[0] == "dot":
                                    state["marks"].append(dict(kind="dot", pos=q, xh=xh_, sh=sh, fwd=st.dot_forward))
                                else:
                                    state["marks"].append(dict(kind="bar", pos=q, xh=xh_, sh=sh, hw=m[3] * wsc * s_))
                            prev = state["prev"]
                            joined = (state["cur"] is not None and prev is not None and prev.jo and gl.ji and
                                      ((ci == 0 and join_in) or (ci > 0 and rng.random() < st.connectivity)))
                            pts = tf(gl.pts)
                            if joined:
                                if prev.hi and gl.skip:
                                    pts = pts[gl.skip:]
                                state["cur"].extend(pts.tolist())
                            else:
                                if state["cur"] is not None:
                                    flush(float(rng.uniform(.02, .07)))
                                if ci == 0 and rng.random() < .3:
                                    pts = np.vstack([[pts[0][0] - .28 * xh_, pts[0][1] + .02 * xh_], pts])
                                state["cur"] = pts.tolist()
                            state["prev"] = gl
                            state["lastch"] = ch
                        else:
                            if state["cur"] is not None:
                                flush(float(rng.uniform(.03, .09)))
                            sp_ = [tf(_bow(s, rng) if (len(s) <= 3 and not gl.bank) else s) for s in gl.strokes if len(s) >= 2]
                            if gl.bank and sp_ and ch.isalpha() and rng.random() < (c.flourish if last else .25 * c.flourish):
                                k_ = max(range(len(sp_)), key=lambda q: _plen(sp_[q]))
                                if _plen(sp_[k_]) > .5 * xh_:
                                    sp_[k_] = self._flick(sp_[k_], xh_, rng)
                            ligate = (gl.bank and ci > 0 and sp_ and prevch is not None and ch.islower() and prevch.islower()
                                      and items and items[-1].get("letter") == prevch
                                      and _plen(items[-1]["pts"]) > .6 * xh_
                                      and np.hypot(*(items[-1]["pts"][-1] - sp_[0][0])) < 1.1 * xh_
                                      and rng.random() < st.connectivity * .6)
                            for si, s in enumerate(sp_):
                                if si == 0 and ligate:
                                    items[-1]["pts"] = np.vstack([items[-1]["pts"], s])
                                    items[-1]["letter"] = ch
                                    continue
                                items.append(dict(pts=s, kind="ink", pause=float(rng.uniform(.02, .08)), vmul=1.0,
                                                  line_start=state["first"], letter=ch if si == len(sp_) - 1 else None))
                                state["first"] = False
                            state["prev"] = None
                        prevch = ch
                        x += adv
                    return x

                for idx, tok in enumerate(flat):
                    a = tok["a"]
                    xh_t = xh * a["size"]
                    ctx.loc, ctx.lift = a["messy"], a["lift"]
                    n0 = len(items)
                    x_in = x
                    if tok["kind"] == "m":
                        flush(float(rng.uniform(.05, .15)))
                        by = baseline(x, ctx.g, xh_t)
                        self._emit_box(tok["box"], x, by, xh_t, ctx, items, skipped)
                        x += tok["w"]
                    else:
                        word = tok["s"]
                        ctx.wmul = float(np.exp(rng.normal(0, .045 * sv)))
                        ctx.wslant = float(rng.normal(0, 2.0 * m_ * c.slant_variation)) + a["slant"]
                        ctx.wslope = math.tan(math.radians(rng.normal(0, 1.4 * pv)))
                        ctx.wx0 = x
                        join_in = state["cur"] is not None
                        if c.mistakes > 0 and len(word) >= 4 and word.isalpha() and rng.random() < c.mistakes:
                            n1 = len(items)
                            xs0 = x
                            x = put_word(self._typo(word, rng), x, join_in, xh_t)
                            flush(float(rng.uniform(.3, .7)))
                            seg = [q["pts"] for q in items[n1:] if q.get("pts") is not None]
                            if seg:
                                A = np.vstack(seg)
                                items.append(self._scribble(xs0, A[:, 0].max(), A[:, 1].min(), A[:, 1].max(), xh_t, rng))
                            x += (.45 + .3 * rng.random()) * xh_t
                            ctx.wx0 = x
                            join_in = False
                        x = put_word(word, x, join_in, xh_t)
                        can_join = False
                        if state["cur"] is not None and state["prev"] is not None and idx + 1 < len(flat):
                            nt = flat[idx + 1]
                            if nt["kind"] == "w" and nt["sp"]:
                                ng = self.glyphs.get(nt["s"][0])
                                can_join = bool(ng and ng.cursive and ng.ji and state["prev"].jo and rng.random() < st.word_join)
                        gap = (st.gap_min + st.gap_var * rng.random()) * xh_t
                        if can_join:
                            state["cur"].append([x + gap * .5, baseline(x + gap * .5, ctx.g, xh_t) - rng.uniform(0, .05) * xh_t])
                        else:
                            has_next = idx + 1 < len(flat)
                            pz = float(rng.lognormal(math.log(.11), .45)) if has_next else 0.0
                            if rng.random() < .04:
                                pz += float(rng.uniform(.25, .6))
                            flush(pz)
                    # speed multiplier for this token
                    if a["speed"] != 1.0:
                        for q in items[n0:]:
                            q["vmul"] = q.get("vmul", 1.0) * a["speed"]
                    # decorations (underline / strike / circle), possibly spanning several tokens
                    seg = [q["pts"] for q in items[n0:] if q.get("pts") is not None and q["kind"] == "ink"]
                    for kind_, did_ in a["decos"]:
                        d = decos.setdefault(did_, dict(kind=kind_, x0=x_in, x1=x_in, ytop=1e9, ybot=-1e9, base=0.0, n=0, xh=xh_t))
                        d["x1"] = x
                        d["base"] = (d["base"] * d["n"] + baseline(x_in, ctx.g, xh_t)) / (d["n"] + 1)
                        d["n"] += 1
                        if seg:
                            A = np.vstack(seg)
                            d["ytop"], d["ybot"] = min(d["ytop"], A[:, 1].min()), max(d["ybot"], A[:, 1].max())
                    nxt_ids = {i for _, i in flat[idx + 1]["a"]["decos"]} if idx + 1 < len(flat) else set()
                    for did_ in [k_ for k_ in decos if k_ not in nxt_ids]:
                        d = decos.pop(did_)
                        flush(0.0)
                        if d["ybot"] > d["ytop"]:
                            items.append(self._deco_stroke(d["kind"], d["x0"], d["x1"], d["ytop"], d["ybot"], d["base"], d["xh"], rng, a["messy"]))
                    if idx + 1 < len(flat):
                        if flat[idx + 1]["sp"]:
                            x += (st.gap_min + st.gap_var * rng.random()) * xh_t
                flush(0.0)
                if items:
                    items.append(dict(kind="newline", pause=float(rng.uniform(.18, .45))))
                ycur = base0 + desc_u * xh + .8 * xh * line_height
        return items, skipped, ycur, widest

    @staticmethod
    def _mark_stroke(m, rng):
        p, xh = m["pos"], m["xh"]
        if m["kind"] == "dot":
            p = p + np.array([m["fwd"] * xh + rng.normal(0, .05 * xh), -rng.uniform(0, .08) * xh])
            d = np.array([rng.uniform(.09, .2), rng.uniform(.04, .12)]) * xh
            pts = np.array([p, p + d * .5 + [0, -.01 * xh], p + d])
            vmul = .55
        else:
            hw = m["hw"] * xh
            rise = rng.uniform(-.04, .12) * xh
            a = p + np.array([-hw * rng.uniform(.8, 1.2), 0]) + [0, rng.normal(0, .03 * xh)]
            b = p + np.array([hw * rng.uniform(.9, 1.5), -rise])
            mid = (a + b) / 2 + np.array([0, -rng.uniform(.0, .05) * xh])
            pts = np.array([a, mid, b])
            vmul = 1.5
        return dict(pts=pts, kind="mark", vmul=vmul, line_start=False)

    # ---- plan --------------------------------------------------------------
    # ---- planning ----------------------------------------------------------------
    def _eff(self, c):
        st = self.style
        m = c.messiness
        sv, pv = m * c.size_variation, m * c.position_variation
        return dataclasses.replace(
            st, size_drift=st.size_drift * sv, size_jit=st.size_jit * sv, width_drift=st.width_drift * sv,
            slant_drift_deg=st.slant_drift_deg * m * c.slant_variation, baseline_drift=st.baseline_drift * pv,
            letter_base=st.letter_base * pv, speed_drift=st.speed_drift * m * c.speed_variation,
            shape_sigma=st.shape_sigma * m,
            slant_deg=st.slant_deg if c.slant is None else c.slant,
            connectivity=st.connectivity if c.connectivity is None else c.connectivity)

    def plan(self, text, x, y, size=24, max_width=0, line_height=1.0, speed=1.0, rng=None, tol=.5,
             controls=None, max_height=0, min_size=12, **kw) -> Plan:
        """Compose `text` into a Plan. Extra keyword args are Controls fields (messiness, align, ...).
        max_height > 0 shrinks the size (down to min_size) until the text fits the box."""
        rng = rng or self._habit_rng
        c = controls or Controls.from_kwargs(**kw)
        st = self._eff(c)
        xh = float(size)
        seed = int(rng.integers(2 ** 31))
        while True:
            r2 = np.random.default_rng(seed)
            items, skipped, end_y, widest = self._compose(text, x, y, xh, max_width, line_height, r2, st, c)
            if not max_height or end_y - y <= max_height or xh <= min_size:
                break
            xh = max(min_size, xh * .93)
        p = self._plan_items(items, skipped, end_y, text, x, y, xh, speed, r2, tol, st, c)
        p.requested_size = float(size)
        return p

    def plan_sketch(self, kind, x0, y0, x1, y1, size=24, speed=1.0, rng=None, controls=None, **kw) -> Plan:
        rng = rng or self._habit_rng
        c = controls or Controls.from_kwargs(**kw)
        st = self._eff(c)
        A, B = np.array([x0, y0], float), np.array([x1, y1], float)
        items = self._shape_items(kind, A, B, float(size), rng, c.messiness)
        p = self._plan_items(items, [], max(y0, y1), f"<{kind}>", x0, y0, float(size), speed, rng, .5, st, c)
        p.requested_size = float(size)
        return p

    def _shape_items(self, kind, A, B, size, rng, m):
        def stroke(P, vmul=1.1, pause=None):
            return dict(pts=np.asarray(P, float), kind="ink", vmul=vmul, line_start=False,
                        pause=float(rng.uniform(.08, .2)) if pause is None else pause)

        def wavy(a, b, amp=.02):
            d = b - a
            L = float(np.hypot(*d))
            if L < 2:
                return np.array([a, b])
            n = np.array([-d[1], d[0]]) / L
            k = max(3, int(L / (1.6 * size)) + 2)
            t = np.linspace(0, 1, k)
            env = np.sin(np.pi * t)
            off = rng.normal(0, .03) * L * m * env + np.cumsum(rng.normal(0, amp * size * m, k)) * env
            return a + np.outer(t, d) + np.outer(off, n)

        x0, y0 = np.minimum(A, B)
        x1, y1 = np.maximum(A, B)
        w, h = x1 - x0, y1 - y0
        if kind in ("line", "underline"):
            return [stroke(wavy(A, B, .03 if kind == "underline" else .02))]
        if kind == "arrow":
            shaft = wavy(A, B)
            L = float(np.hypot(*(B - A)))
            u = (A - B) / max(L, 1e-6)
            hl = float(np.clip(.2 * L, .6 * size, 1.6 * size))
            def rot(v, a): return np.array([v[0] * math.cos(a) - v[1] * math.sin(a), v[0] * math.sin(a) + v[1] * math.cos(a)])
            a1, a2 = math.radians(rng.uniform(20, 34)), math.radians(rng.uniform(20, 34))
            l1, r1 = B + rot(u, a1) * hl * rng.uniform(.85, 1.1), B + rot(u, -a2) * hl * rng.uniform(.85, 1.1)
            if rng.random() < .6:
                return [stroke(shaft), stroke([l1, B, r1], 1.3)]
            return [stroke(shaft), stroke([B, l1], 1.3), stroke([B, r1], 1.3)]
        if kind == "box":
            j = lambda: rng.normal(0, .02 * size * m, 2)  # noqa: E731
            TL, TR, BR, BL = np.array([x0, y0]), np.array([x1, y0]), np.array([x1, y1]), np.array([x0, y1])
            pts, corners = [], [TL, TR, BR, BL, TL + np.array([.05 * w, 0])]
            for a, b in zip(corners[:-1], corners[1:]):
                pts.append(a + j()); pts.append((a + b) / 2 + j() + np.array([-(b - a)[1], (b - a)[0]]) / max(np.hypot(*(b - a)), 1) * rng.normal(0, .007) * np.hypot(*(b - a)) * m)
            pts.append(corners[-1] + j())
            return [stroke(pts, 1.2)]
        if kind in ("circle", "ellipse"):
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            rx, ry = max(w / 2, size * .5), max(h / 2, size * .5)
            tt = np.linspace(0, 2 * math.pi * rng.uniform(1.08, 1.2), 18)
            ang = rng.uniform(-2.5, -1.2) - tt
            grow = 1 + rng.choice([-1, 1]) * .07 * tt / (2 * math.pi)
            P = np.stack([cx + rx * grow * np.cos(ang), cy + ry * grow * np.sin(ang)], 1)
            return [stroke(P + rng.normal(0, .012 * size * m, P.shape), 1.3)]
        if kind == "check":
            P = [(x0, y0 + .58 * h), (x0 + .16 * w, y0 + .7 * h), (x0 + .34 * w, y1), (x0 + .62 * w, y0 + .45 * h), (x1, y0)]
            return [stroke(np.array(P) + rng.normal(0, .012 * size * m, (5, 2)), 1.3)]
        if kind == "cross":
            return [stroke(wavy(np.array([x0, y0]), np.array([x1, y1]))), stroke(wavy(np.array([x1, y0]), np.array([x0, y1])))]
        if kind == "scribble":
            q = self._scribble(x0, x1, y0, y1, max(size, h * 1.1), rng)
            q["pts"][:, 1] = np.clip(q["pts"][:, 1], y0, y1)
            return [q]
        raise ValueError(f"unknown sketch kind {kind!r}; use line, underline, arrow, box, circle, check, cross, scribble")

    def _plan_items(self, items, skipped, end_y, text, x, y, xh, speed, rng, tol, st, c) -> Plan:
        segs, pen = [], None
        all_pts = []
        carry = 0.0
        for it in items:
            if it["kind"] == "newline":
                carry += it["pause"]
                continue
            if it["kind"] in ("dot", "bar") or it.get("pos") is not None:
                it = self._mark_stroke(it, rng) | {"pause": it.get("pause", .03)}
            dense = catmull_rom(it["pts"], .6)
            vbase = st.speed * xh * speed * it.get("vmul", 1.0) * float(np.exp(rng.normal(0, .14 * c.messiness * c.speed_variation)))
            t = stroke_times(dense, vbase, xh)
            pause = it.get("pause", 0.0) + carry
            carry = 0.0
            if it.get("line_start"):
                pause += float(rng.uniform(.0, .08))
            if pen is not None:
                dist = float(np.hypot(*(dense[0] - pen)))
                if dist > 2.5:
                    dur = float(np.clip(.08 + dist / 1700 + rng.normal(0, .015), .06, .7))
                    mxy, mt = min_jerk_path(pen, dense[0], dur, rng)
                    segs.append(Seg(False, np.round(mxy), mt, pause=pause * .5 + float(rng.uniform(.01, .03)),
                                    corner=np.zeros(len(mxy), bool)))
                    pause = pause * .5 + float(rng.uniform(.015, .04))
                else:
                    pause = max(pause, .015)
            idx = rdp_indices(dense, tol)
            wp = np.round(dense[idx])
            wt = t[idx]
            keep = np.ones(len(wp), bool)
            keep[:-1] = np.any(np.diff(wp, axis=0) != 0, axis=1)
            if len(wp) > 1 and not keep.any():
                keep[-1] = True
            wp, wt = wp[keep], wt[keep]
            if len(wp) > 1:
                step = max(5.0, .3 * xh)
                P, T = [wp[0]], [wt[0]]
                for k in range(1, len(wp)):
                    dd = float(np.hypot(*(wp[k] - wp[k - 1])))
                    m = int(dd // step)
                    for j in range(1, m + 1):
                        f = j / (m + 1)
                        P.append(np.round(wp[k - 1] + (wp[k] - wp[k - 1]) * f))
                        T.append(wt[k - 1] + (wt[k] - wt[k - 1]) * f)
                    P.append(wp[k])
                    T.append(wt[k])
                wp, wt = np.array(P), np.array(T)
            corner = np.zeros(len(wp), bool)
            if len(wp) > 2:
                v1, v2 = wp[1:-1] - wp[:-2], wp[2:] - wp[1:-1]
                cos = np.sum(v1 * v2, 1) / (np.hypot(*v1.T) * np.hypot(*v2.T) + 1e-9)
                corner[1:-1] = cos < .75
            dv = np.hypot(*np.diff(dense, axis=0).T) / np.maximum(np.diff(t), 1e-4) if len(dense) > 1 else np.ones(1)
            dv = np.append(dv, dv[-1])
            segs.append(Seg(True, wp, wt, pause=pause, dense=dense, corner=corner, dense_v=dv))
            all_pts.append(dense)
            pen = dense[-1]
        if all_pts:
            A = np.vstack(all_pts)
            bbox = (float(A[:, 0].min()), float(A[:, 1].min()), float(A[:, 0].max()), float(A[:, 1].max()))
        else:
            bbox = (x, y, x, y)
        return Plan(segs, bbox, sorted(set(skipped)), text, end_y, xh)

    # ---- execution -----------------------------------------------------------
    async def write(self, target, text, x, y, size=24, max_width=0, line_height=1.0, speed=1.0,
                    max_lag=.12, rng=None, dry_run=False, **kw) -> "WriteResult":
        """`target` is a Cua `Computer` (uses .interface) or any object with
        async move_cursor/mouse_down/mouse_up(x, y).  Extra kwargs: see Controls (+ max_height)."""
        plan = self.plan(text, x, y, size, max_width, line_height, speed, rng, **kw)
        return await self._run(target, plan, max_lag, dry_run)

    async def draw(self, target, kind, x0, y0, x1, y1, size=24, speed=1.0, max_lag=.12, rng=None,
                   dry_run=False, **kw) -> "WriteResult":
        """Hand-drawn annotation: line, underline, arrow, box, circle, check, cross, scribble."""
        plan = self.plan_sketch(kind, x0, y0, x1, y1, size, speed, rng, **kw)
        return await self._run(target, plan, max_lag, dry_run)

    async def _run(self, target, plan, max_lag, dry_run):
        be = target if isinstance(target, _Backend) else CuaBackend(target)
        t_start = time.monotonic()
        if not dry_run:
            await self._execute(plan, be, max_lag)
        return WriteResult(plan, time.monotonic() - t_start, dry_run)

    async def _execute(self, plan: Plan, be: "_Backend", max_lag: float):
        down = False
        try:
            for seg in plan.segs:
                if seg.pause > 0:
                    await asyncio.sleep(seg.pause)
                xy, t = seg.xy, seg.t
                if len(xy) == 0:
                    continue
                x0, y0 = int(xy[0][0]), int(xy[0][1])
                await be.move(x0, y0)
                if seg.down:
                    await be.down(x0, y0)
                    down = True
                t0 = time.monotonic()
                i = 1
                last = (x0, y0)
                while i < len(xy):
                    lag = (time.monotonic() - t0) - t[i]
                    if lag < 0:
                        await asyncio.sleep(-lag)
                    elif lag > max_lag:       # behind schedule: skip non-corner waypoints
                        cap = .45 * plan.size      # never leap more than this far while catching up
                        while (i < len(xy) - 1 and not seg.corner[i]
                               and (time.monotonic() - t0) - t[i] > max_lag * .5
                               and np.hypot(xy[i + 1][0] - last[0], xy[i + 1][1] - last[1]) < cap):
                            i += 1
                    p = (int(xy[i][0]), int(xy[i][1]))
                    if p != last:
                        await be.move(*p)
                        last = p
                    i += 1
                if seg.down:
                    await be.up(*last)
                    down = False
        finally:
            if down:          # never leave the button held
                await be.release()


@dataclass
class WriteResult:
    plan: Plan
    elapsed: float
    dry_run: bool = False

    def summary(self) -> str:
        p = self.plan
        x0, y0, x1, y1 = p.bbox
        t = p.duration if self.dry_run else self.elapsed
        box = f"Ink bounding box: x {x0:.0f}-{x1:.0f}, y {y0:.0f}-{y1:.0f}."
        if p.text.startswith("<") and p.text.endswith(">"):
            return f"{'Would draw' if self.dry_run else 'Drew'} {p.text[1:-1]} in {p.n_strokes} pen strokes ({t:.1f}s). {box}"
        s = (f"{'Would write' if self.dry_run else 'Wrote'} {len(p.text)} chars in {p.n_strokes} pen strokes "
             f"({t:.1f}s). {box} Next line would start at y={p.next_y:.0f}.")
        if abs(p.size - p.requested_size) > .5:
            s += f" Text was shrunk to size {p.size:.0f} (requested {p.requested_size:.0f}) to fit max_height."
        if p.skipped:
            s += f" Unsupported characters skipped: {''.join(p.skipped)!r}."
        return s


# =============================================================================
# 6. Backends (Cua + a recording fake for tests)
# =============================================================================


class _Backend:
    async def move(self, x, y): ...
    async def down(self, x, y): ...
    async def up(self, x, y): ...
    async def release(self): ...


class CuaBackend(_Backend):
    """Thin adapter over `computer.interface` (cua-computer)."""

    def __init__(self, computer, button="left"):
        self.i = getattr(computer, "interface", computer)
        self.button = button
        self._last = (0, 0)

    async def _call(self, fn, *a, **k):
        try:
            if "delay" in inspect.signature(fn).parameters:
                k.setdefault("delay", 0)
        except (TypeError, ValueError):
            pass
        return await fn(*a, **k)

    async def move(self, x, y):
        self._last = (x, y)
        await self._call(self.i.move_cursor, x, y)

    async def down(self, x, y):
        self._last = (x, y)
        await self._call(self.i.mouse_down, x, y, button=self.button)

    async def up(self, x, y):
        await self._call(self.i.mouse_up, x, y, button=self.button)

    async def release(self):
        await self.up(*self._last)


class RecordingBackend(_Backend):
    def __init__(self, latency=0.0):
        self.events, self.latency, self._t0 = [], latency, time.monotonic()

    async def _rec(self, kind, x=None, y=None):
        if self.latency:
            await asyncio.sleep(self.latency)
        self.events.append((time.monotonic() - self._t0, kind, x, y))

    async def move(self, x, y): await self._rec("move", x, y)
    async def down(self, x, y): await self._rec("down", x, y)
    async def up(self, x, y): await self._rec("up", x, y)
    async def release(self): await self._rec("up")


class NullBackend(_Backend):
    """Swallows every event. Same code path as a real draw, zero side effects."""
    async def move(self, x, y): ...
    async def down(self, x, y): ...
    async def up(self, x, y): ...
    async def release(self): ...


class Win32Backend(_Backend):
    """Real pointer input on this Windows desktop via SendInput - works in any app, no Cua.

    x, y are physical screen pixels (per-monitor DPI aware), y down, virtual desktop origin.
    The caller owns the pointer while a stroke runs; nothing is restored afterwards.
    """

    _MOVE, _ABSOLUTE, _VIRTUALDESK, _DOWN, _UP = 0x0001, 0x8000, 0x4000, 0x0002, 0x0004

    def __init__(self):
        import ctypes
        from ctypes import wintypes

        self._ct = ctypes
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)      # per-monitor v2: px == screen px
        except Exception:
            try:
                ctypes.windll.user32.SetProcessDPIAware()
            except Exception:
                pass
        u = self._u = ctypes.windll.user32
        self._vx, self._vy = u.GetSystemMetrics(76), u.GetSystemMetrics(77)
        self._vw, self._vh = max(1, u.GetSystemMetrics(78)), max(1, u.GetSystemMetrics(79))
        self._last = (0, 0)

        class MOUSEINPUT(ctypes.Structure):
            _fields_ = [("dx", wintypes.LONG), ("dy", wintypes.LONG), ("mouseData", wintypes.DWORD),
                        ("dwFlags", wintypes.DWORD), ("time", wintypes.DWORD),
                        ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong))]

        class INPUT(ctypes.Structure):
            class _U(ctypes.Union):
                _fields_ = [("mi", MOUSEINPUT)]
            _anonymous_ = ("u",)
            _fields_ = [("type", wintypes.DWORD), ("u", _U)]

        self._INPUT, self._MOUSEINPUT = INPUT, MOUSEINPUT

    def _abs(self, x, y):
        ax = int(round((x - self._vx) / self._vw * 65535))
        ay = int(round((y - self._vy) / self._vh * 65535))
        return max(0, min(65535, ax)), max(0, min(65535, ay))

    def _send(self, x, y, button=0):
        ax, ay = self._abs(x, y)
        mi = self._MOUSEINPUT(ax, ay, 0, self._MOVE | self._ABSOLUTE | self._VIRTUALDESK | button, 0, None)
        inp = self._INPUT()
        inp.type = 0                                             # INPUT_MOUSE
        inp.mi = mi
        self._u.SendInput(1, self._ct.byref(inp), self._ct.sizeof(inp))
        if button & (self._DOWN | self._UP):
            self._last = (x, y)

    async def move(self, x, y):
        self._send(x, y)
        await asyncio.sleep(0)                                   # let the loop pace the stroke

    async def down(self, x, y):
        self._send(x, y, self._DOWN)                             # one batch: arrive + press
        await asyncio.sleep(0)

    async def up(self, x, y):
        self._send(x, y, self._UP)
        await asyncio.sleep(0)

    async def release(self):
        self._send(*self._last, button=self._UP)


def get_backend(name: str | _Backend | None = None) -> _Backend:
    """'win32' | 'none' | 'rec' | None/auto (win32 on Windows). Env: HUMANPEN_BACKEND.
    An existing backend instance is returned unchanged."""
    if isinstance(name, _Backend):
        return name
    name = (name or os.environ.get("HUMANPEN_BACKEND") or "auto").lower()

    if name in ("none", "null", "dry", "preview"):
        return NullBackend()
    if name in ("rec", "record"):
        return RecordingBackend()
    if name in ("win32", "auto"):
        if os.name != "nt":
            raise SystemExit("humanpen: no automatic backend on this OS - use the Cua backend")
        return Win32Backend()
    raise SystemExit(f"humanpen: unknown backend {name!r} (win32 | none | rec)")


# =============================================================================
# 7. Agent-facing tools
# =============================================================================

_KNOBS = {
    "messiness": {"type": "number", "description": "Global multiplier on all human variation. 0.3 neat, 1.0 natural (default), 1.8 hurried scrawl."},
    "size_variation": {"type": "number", "description": "Multiplier on letter/word size wobble (default 1.0)."},
    "position_variation": {"type": "number", "description": "Multiplier on baseline wander, letter spacing noise and per-word rise/fall (default 1.0)."},
    "slant": {"type": "number", "description": "Forward lean in degrees (default: this writer's own)."},
    "connectivity": {"type": "number", "description": "0..1 chance neighbouring letters join (default: this writer's own)."},
    "align": {"type": "string", "enum": ["left", "center", "right"], "description": "Alignment inside max_width (or around x when max_width=0)."},
    "mistakes": {"type": "number", "description": "0..1 chance per word of a typo that is struck out and rewritten. Leaves scribbles on the canvas: only for casual notes (default 0)."},
    "line_height": {"type": "number", "description": "Line spacing multiplier (default 1.0)."},
    "max_height": {"type": "integer", "description": "If > 0, shrink the text until it fits in this many px of height."},
    "dry_run": {"type": "boolean", "description": "Plan only: report bounding box / duration without touching the mouse."},
    "seed": {"type": "integer", "description": "Writer identity. Keep the SAME seed for a whole task; change it for a different person."},
    "hand": {"type": "string", "enum": ["bank", "casual", "cursive"], "description": "Letterforms: bank (default, traced from a real scan), casual, cursive."},
}

TOOL_SCHEMA = {
    "name": "handwrite",
    "description": (
        "Draw handwritten text with the mouse (real-handwriting strokes, human-like irregularity). Only for drawing "
        "surfaces: whiteboards, signature pads, canvas/drawing boxes - select the pen tool and a visible colour first. "
        "(x, y) is the top-left of the text box in screen px; size = height of lowercase letters in px (>= 16). "
        "Inline markup: <big>..</big> <small>..</small> <u>..</u> <s>..</s> <circle>..</circle> <neat>..</neat> "
        "<messy>..</messy> <fast>..</fast> <slow>..</slow> <slant=8> <lift=0.3>. "
        "Math: wrap LaTeX in $...$, e.g. 'Area $A=\\pi r^2$, $x=\\frac{-b\\pm\\sqrt{b^2-4ac}}{2a}$'. "
        "Never types into text fields - use type_text for those."),
    "input_schema": {
        "type": "object",
        "properties": {
            "text": {"type": "string", "description": "Text to write. Use \\n for new lines."},
            "x": {"type": "integer"}, "y": {"type": "integer"},
            "size": {"type": "integer", "description": "x-height in px (default 24)."},
            "max_width": {"type": "integer", "description": "Wrap lines at this pixel width (0 = no wrap)."},
            "speed": {"type": "number", "description": "Pen speed multiplier (default 1.0)."},
            **_KNOBS,
        },
        "required": ["text", "x", "y"],
    },
}

DRAW_SCHEMA = {
    "name": "handdraw",
    "description": ("Hand-drawn annotation with the mouse: line, underline, arrow (x0,y0 -> x1,y1), box / circle / check / "
                    "cross / scribble (inside the box x0,y0 - x1,y1). Select the pen tool first."),
    "input_schema": {
        "type": "object",
        "properties": {
            "kind": {"type": "string", "enum": ["line", "underline", "arrow", "box", "circle", "check", "cross", "scribble"]},
            "x0": {"type": "integer"}, "y0": {"type": "integer"}, "x1": {"type": "integer"}, "y1": {"type": "integer"},
            "size": {"type": "integer", "description": "Pen scale in px (default 24)."},
            "speed": {"type": "number"}, "messiness": _KNOBS["messiness"], "dry_run": _KNOBS["dry_run"],
            "seed": _KNOBS["seed"], "hand": _KNOBS["hand"],
        },
        "required": ["kind", "x0", "y0", "x1", "y1"],
    },
}
TOOL_SCHEMAS = [TOOL_SCHEMA, DRAW_SCHEMA]


def make_agent_tool(computer, seed: Optional[int] = None, style: Optional[WriterStyle] = None,
                    hand: str = "bank", with_draw: bool = False):
    """Returns an async `handwrite(...)` function for Cua's ComputerAgent(tools=[...]).
    One Handwriter per tool => the same 'person' writes every call.
    with_draw=True returns (handwrite, handdraw)."""
    writer = Handwriter(style=style, seed=seed, hand=hand)
    backend = CuaBackend(computer)

    async def handwrite(text: str, x: int, y: int, size: int = 24, max_width: int = 0, speed: float = 1.0,
                        messiness: float = 1.0, size_variation: float = 1.0, position_variation: float = 1.0,
                        slant: Optional[float] = None, connectivity: Optional[float] = None, align: str = "left",
                        mistakes: float = 0.0, line_height: float = 1.0, max_height: int = 0,
                        dry_run: bool = False) -> str:
        """Draw handwritten text with the mouse on a drawing surface (whiteboard, signature pad, canvas).
        Select the pen tool first. x,y = top-left of the text box (screen px); size = lowercase letter height
        in px; max_width wraps lines; max_height shrinks text to fit. Humanisation knobs: messiness (0.3 neat ..
        1.8 scrawl), size_variation, position_variation, slant, connectivity, align, mistakes. Inline markup:
        <big> <small> <u> <s> <circle> <neat> <messy> <fast> <slow> <slant=N> <lift=F>. Math: wrap LaTeX in $...$.
        dry_run=True only measures. Returns the ink bounding box and the y where the next line would start."""
        res = await writer.write(backend, text, x, y, size=size, max_width=max_width, speed=speed,
                                 messiness=messiness, size_variation=size_variation,
                                 position_variation=position_variation, slant=slant, connectivity=connectivity,
                                 align=align, mistakes=mistakes, line_height=line_height, max_height=max_height,
                                 dry_run=dry_run)
        return res.summary()

    async def handdraw(kind: str, x0: int, y0: int, x1: int, y1: int, size: int = 24, speed: float = 1.0,
                       messiness: float = 1.0, dry_run: bool = False) -> str:
        """Hand-drawn annotation: line, underline, arrow (from x0,y0 to x1,y1), box, circle, check, cross, scribble
        (inside the box x0,y0-x1,y1). Select the pen tool first."""
        res = await writer.draw(backend, kind, x0, y0, x1, y1, size=size, speed=speed, messiness=messiness, dry_run=dry_run)
        return res.summary()

    return (handwrite, handdraw) if with_draw else handwrite


_WRITERS: dict = {}


def _writer(key, seed: Optional[int] = None, hand: str = "bank") -> "Handwriter":
    """One Handwriter per (key, seed, hand) so a task keeps the same 'person'."""
    k = (key, seed, hand)
    if k not in _WRITERS:
        _WRITERS[k] = Handwriter(seed=seed, hand=hand)
    return _WRITERS[k]


async def handle_tool_call(target, tool_input: dict, name: str = "handwrite", seed: Optional[int] = None) -> str:
    """Framework-agnostic dispatcher for TOOL_SCHEMAS.

    `target` is a Cua Computer, any object with async move_cursor/mouse_down/mouse_up, or a
    `_Backend` (see `get_backend`).
    """
    a = dict(tool_input)
    s = a.pop("seed", seed)
    w = _writer(id(target), s, a.pop("hand", "bank"))
    if name == "handdraw":
        res = await w.draw(target, a.pop("kind"), a.pop("x0"), a.pop("y0"), a.pop("x1"), a.pop("y1"), **a)
        return res.summary()
    res = await w.write(target, a.pop("text"), a.pop("x"), a.pop("y"), **a)
    return res.summary()


def run_tool(name: str, tool_input: dict, backend: Optional[str] = None) -> str:
    """Sync entry point for plugin / MCP hosts: plans, draws through the backend, returns the summary."""
    a = dict(tool_input)
    be = get_backend(a.pop("backend", backend))
    return asyncio.run(handle_tool_call(be, a, name, seed=a.get("seed")))



# =============================================================================
# 8. MCP stdio server (Claude, Codex, Hermes, any MCP client)
# =============================================================================

PROTOCOLS = ("2024-11-05", "2025-03-26", "2025-06-18")


def _mcp_send(msg: dict) -> None:
    sys.stdout.write(json.dumps(msg) + "\n")
    sys.stdout.flush()


def _mcp_error(id_, code, message):
    _mcp_send({"jsonrpc": "2.0", "id": id_, "error": {"code": code, "message": message}})


def serve_mcp() -> None:
    """Line-delimited JSON-RPC on stdin/stdout. stdout is the protocol channel - log to stderr."""
    print(f"humanpen mcp {__version__} ready (backend={os.environ.get('HUMANPEN_BACKEND') or 'auto'})", file=sys.stderr)
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except Exception:
            _mcp_error(None, -32700, "parse error")
            continue
        method, id_ = req.get("method"), req.get("id")
        params = req.get("params") or {}
        try:
            if method == "initialize":
                want = params.get("protocolVersion")
                _mcp_send({"jsonrpc": "2.0", "id": id_, "result": {
                    "protocolVersion": want if want in PROTOCOLS else PROTOCOLS[-1],
                    "capabilities": {"tools": {"listChanged": False}},
                    "serverInfo": {"name": "human-handwriting", "version": __version__},
                    "instructions": "handwrite/handdraw move the REAL mouse. Check for a pen tool and a "
                                    "focused canvas first; use dry_run to measure.",
                }})
            elif method == "tools/list":
                tools = [{"name": t["name"], "description": t["description"],
                          "inputSchema": t["input_schema"]} for t in TOOL_SCHEMAS]
                _mcp_send({"jsonrpc": "2.0", "id": id_, "result": {"tools": tools}})
            elif method == "tools/call":
                got = run_tool(params.get("name", ""), params.get("arguments") or {})
                _mcp_send({"jsonrpc": "2.0", "id": id_,
                           "result": {"content": [{"type": "text", "text": got}], "isError": False}})
            elif method == "ping":
                _mcp_send({"jsonrpc": "2.0", "id": id_, "result": {}})
            elif id_ is not None:
                _mcp_error(id_, -32601, f"unknown method {method}")
        except Exception as exc:
            _mcp_error(id_, -32603, f"{type(exc).__name__}: {exc}")


# =============================================================================
# 9. CLI
# =============================================================================


def _add_knobs(p):
    p.add_argument("--messiness", type=float, default=1.0)
    p.add_argument("--size-variation", type=float, default=1.0)
    p.add_argument("--position-variation", type=float, default=1.0)
    p.add_argument("--slant", type=float, default=None)
    p.add_argument("--connectivity", type=float, default=None)
    p.add_argument("--align", default="left", choices=["left", "center", "right"])
    p.add_argument("--mistakes", type=float, default=0.0)
    p.add_argument("--line-height", type=float, default=1.0)
    p.add_argument("--max-height", type=int, default=0)
    p.add_argument("--speed", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=None, help="writer personality (keep it stable per task)")
    p.add_argument("--hand", default="bank", choices=["bank", "casual", "cursive"])
    p.add_argument("--backend", default=None, help="win32 | none | rec (default: $HUMANPEN_BACKEND or win32)")


def _doctor() -> int:
    ok = True
    print(f"humanpen {__version__}  python {sys.version.split()[0]}  platform {sys.platform}")
    for mod in ("numpy", "PIL"):
        try:
            m = __import__(mod)
            print(f"  {mod:<6} {getattr(m, '__version__', '?')}")
        except Exception as exc:
            ok = ok and mod != "numpy"
            print(f"  {mod:<6} MISSING ({exc.__class__.__name__})" + ("  <- required" if mod == "numpy" else "  (previews only)"))
    if os.name == "nt":
        try:
            be = Win32Backend()
            print(f"  win32  virtual desktop {be._vw}x{be._vh} at ({be._vx},{be._vy}) - SendInput OK")
        except Exception as exc:
            ok = False
            print(f"  win32  FAILED: {exc}")
    else:
        print("  win32  n/a (not Windows) - use the Cua backend")
    plan = Handwriter(seed=1).plan("Doctor check $x^2$", 100, 100, size=24)
    print(f"  plan   {plan.n_strokes} strokes, {plan.duration:.1f}s, bbox {tuple(round(v) for v in plan.bbox)}")
    print("  verdict", "OK" if ok else "PROBLEMS")
    return 0 if ok else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="humanpen", description="Human-like handwriting for agents (mouse strokes).")
    ap.add_argument("--version", action="version", version=__version__)
    sub = ap.add_subparsers(dest="cmd")

    pv = sub.add_parser("preview", help="render a PNG, never touches the pointer")
    pv.add_argument("text")
    pv.add_argument("--png", default="handwriting_preview.png")
    pv.add_argument("--size", type=int, default=30)
    pv.add_argument("--width", type=int, default=0)
    pv.add_argument("--draw", nargs=5, metavar=("KIND", "X0", "Y0", "X1", "Y1"))
    _add_knobs(pv)

    wr = sub.add_parser("write", help="draw text on the screen")
    wr.add_argument("text")
    wr.add_argument("--x", type=int, required=True)
    wr.add_argument("--y", type=int, required=True)
    wr.add_argument("--size", type=int, default=24)
    wr.add_argument("--width", type=int, default=0, help="wrap width in px")
    wr.add_argument("--dry-run", action="store_true")
    _add_knobs(wr)

    dr = sub.add_parser("draw", help="draw an annotation")
    dr.add_argument("kind", choices=["line", "underline", "arrow", "box", "circle", "check", "cross", "scribble"])
    dr.add_argument("--x0", type=int, required=True)
    dr.add_argument("--y0", type=int, required=True)
    dr.add_argument("--x1", type=int, required=True)
    dr.add_argument("--y1", type=int, required=True)
    dr.add_argument("--size", type=int, default=24)
    dr.add_argument("--dry-run", action="store_true")
    _add_knobs(dr)

    sub.add_parser("doctor", help="check interpreter, deps and pointer backend")
    sub.add_parser("mcp", help="run the MCP stdio server")

    a = ap.parse_args(argv)
    cmd = a.cmd or "preview"
    if cmd == "mcp":
        serve_mcp()
        return 0
    if cmd == "doctor":
        return _doctor()
    if cmd == "preview":
        hw = Handwriter(seed=a.seed, hand=a.hand)
        if a.draw:
            k, *xy = a.draw
            p = hw.plan_sketch(k, *map(float, xy), size=a.size, messiness=a.messiness)
        else:
            p = hw.plan(a.text.replace("\\n", "\n"), 0, 0, size=a.size, max_width=a.width, messiness=a.messiness,
                        size_variation=a.size_variation, position_variation=a.position_variation, slant=a.slant,
                        align=a.align, mistakes=a.mistakes, max_height=a.max_height)
        p.render_png(a.png)
        print(f"{a.png}: {p.n_strokes} strokes, ~{p.duration:.1f}s, skipped={p.skipped}")
        return 0

    kw = dict(size=a.size, speed=a.speed, messiness=a.messiness, size_variation=a.size_variation,
              position_variation=a.position_variation, slant=a.slant, connectivity=a.connectivity,
              align=a.align, mistakes=a.mistakes, line_height=a.line_height, max_height=a.max_height,
              hand=a.hand, seed=a.seed, backend=a.backend, dry_run=a.dry_run)
    if cmd == "write":
        print(run_tool("handwrite", dict(text=a.text, x=a.x, y=a.y, max_width=a.width, **kw)))
    else:
        print(run_tool("handdraw", dict(kind=a.kind, x0=a.x0, y0=a.y0, x1=a.x1, y1=a.y1, **kw)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

