# dsh-handwriting

Human handwriting on the real pointer for DeepSeek Harness. The harness face of the same engine the
Hermes, Claude and Codex packages ship.

## What it adds

| Tool | Does |
| --- | --- |
| `handwrite` | Text (and inline markup and LaTeX maths) drawn as natural pen strokes: drifting size, slant and baseline, ligatures, fatigue, optional struck-out corrections. Returns the ink bbox and the `y` for the next line. |
| `handdraw` | `line, underline, arrow, box, circle, check, cross, scribble`. Same writer seed as the text beside it. |

Both accept `dry_run` (plan + bbox, pointer untouched) and `seed` (one seed = one hand for the whole task).

Use them on drawing surfaces only - a whiteboard, a canvas, a "write your answer by hand" box, a signature pad.
For an `<input>` or a document body, type. The pointer really moves through Win32 `SendInput`, so the target
window must be foreground with its pen tool selected, and nothing else should be competing for focus.

The `human-handwriting` skill is installed alongside (`~/.dsh/skills/human-handwriting`) and carries the doctrine:
imperfection budget per surface, markup and maths reference, and the failure checklist. The tools work without
it; the output looks like a person only when it is read.

## Setup

- **Python with numpy.** Interpreter order: `$HUMANPEN_PYTHON`, DSH's bundled runtime
  (`$DSH_HOME/dsh-runtimes/*/dependencies/python`,`python.exe`), then `python` / `py` on `PATH`. The first one
  that imports numpy wins; the choice is cached for the process. `pillow` is only needed for PNG previews.
- **Windows** for the pointer backend. On another OS the engine's Cua backend is the route, and this plugin's
  CLI wiring does not cover it.

Check it end to end without moving the pointer:

```bash
python dsh-handwriting/humanpen.py doctor
python dsh-handwriting/humanpen.py preview "hello $x^2$" --png /tmp/hand.png
```

## Config

| Key | Default | |
| --- | --- | --- |
| `timeoutMs` | `180000` | per draw call; a long paragraph is genuinely slow |
| `backend` | `win32` | pointer backend passed to the engine |
| `probeMs` | `20000` | interpreter probe budget |

## Install

```json
"dependencies": { "dsh-handwriting": "link:<path to this checkout>" },
"dsh": { "profile": { "bundles": ["...", "dsh-handwriting"] } }
```

then `pnpm install` in the profile directory. Restart DSH to mount the bundle.
