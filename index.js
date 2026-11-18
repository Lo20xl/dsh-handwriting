/**
 * dsh-handwriting: human pen strokes on the real pointer.
 *
 * Registers `handwrite` (text, inline markup and LaTeX maths turned into natural pen paths: drifting size,
 * slant and baseline, ligatures, fatigue, optional struck-out corrections) and `handdraw` (line, underline,
 * arrow, box, circle, check, cross, scribble). Letters come from a bank of scanned handwriting, morphed on
 * every use, so the result reads as written rather than typeset.
 *
 * The engine is `humanpen.py` next to this file - one self-contained Python module (numpy) that either plans
 * (dry_run) or drives the real cursor through Win32 SendInput. It is spawned per call: no daemon, no state
 * to leak, and a hung draw dies with its timeout.
 *
 * Drawing surfaces only. For an <input>, <textarea> or document body use the normal typing tool - a pen
 * cannot type. The pointer really moves, so the target window must be foreground with its pen tool selected;
 * dry_run measures without touching anything.
 *
 * @module dsh-handwriting
 */
import { execFile } from 'node:child_process'
import { existsSync } from 'node:fs'
import { homedir } from 'node:os'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'
import { defineTool } from '@deepseek-ai/dsh-tools'

export const name = 'dsh-handwriting'
export const inject = ['tools']

const HERE = dirname(fileURLToPath(import.meta.url))
const ENGINE = join(HERE, 'humanpen.py')

const DEFAULTS = Object.freeze({ timeoutMs: 180_000, probeMs: 20_000, backend: 'win32' })

/** One bounded child process; failures become data, never exceptions. */
function run(exe, args, timeoutMs) {
  return new Promise(resolve => {
    execFile(exe, args, { timeout: timeoutMs, windowsHide: true, maxBuffer: 4 * 1024 * 1024 }, (error, stdout, stderr) => {
      if (error) {
        const code = error.killed ? `timed out after ${timeoutMs} ms` : String(error.message).slice(0, 400)
        resolve({ ok: false, out: String(stdout || '').trim(), error: code, stderr: String(stderr || '').trim().slice(0, 800) })
        return
      }
      resolve({ ok: true, out: String(stdout || '').trim(), stderr: String(stderr || '').trim().slice(0, 800) })
    })
  })
}

/**
 * Interpreter order: explicit override, DSH's bundled Python, then PATH. Probed once per process because
 * the engine needs numpy and a bare `python` is not always the one that has it.
 */
function candidates() {
  const list = []
  if (process.env.HUMANPEN_PYTHON) list.push(process.env.HUMANPEN_PYTHON)
  const dshHome = process.env.DSH_HOME || join(homedir(), '.dsh')
  list.push(join(dshHome, 'dsh-runtimes', 'dsh-next-primary-runtime', 'dependencies', 'python', 'python.exe'))
  list.push('python', 'py')
  return list
}

let python

async function pythonFor(settings) {
  if (python) return python
  for (const exe of candidates()) {
    const isPath = exe.includes('\\') || exe.includes('/')
    if (isPath && !existsSync(exe)) continue
    const probe = await run(exe, ['-c', 'import numpy, sys; print(sys.executable)'], settings.probeMs)
    if (probe.ok) { python = exe; return exe }
  }
  return null
}

/** argv for the engine CLI. Numbers are formatted here so `execFile` never sees an unvalidated value. */
function writeArgs(a, settings) {
  const argv = ['write', String(a.text), '--x', String(Math.round(Number(a.x))), '--y', String(Math.round(Number(a.y)))]
  argv.push('--size', String(Math.round(Number(a.size) || 24)))
  if (a.max_width) argv.push('--width', String(Math.round(Number(a.max_width))))
  if (a.align) argv.push('--align', String(a.align))
  if (a.messiness != null) argv.push('--messiness', String(Number(a.messiness)))
  if (a.speed != null) argv.push('--speed', String(Number(a.speed)))
  if (a.mistakes != null) argv.push('--mistakes', String(Number(a.mistakes)))
  if (a.seed != null) argv.push('--seed', String(Math.round(Number(a.seed))))
  argv.push('--backend', settings.backend)
  if (a.dry_run) argv.push('--dry-run')
  return argv
}

function drawArgs(a, settings) {
  const argv = ['draw', String(a.kind),
    '--x0', String(Math.round(Number(a.x0))), '--y0', String(Math.round(Number(a.y0))),
    '--x1', String(Math.round(Number(a.x1))), '--y1', String(Math.round(Number(a.y1)))]
  if (a.size) argv.push('--size', String(Math.round(Number(a.size))))
  if (a.messiness != null) argv.push('--messiness', String(Number(a.messiness)))
  if (a.seed != null) argv.push('--seed', String(Math.round(Number(a.seed))))
  argv.push('--backend', settings.backend)
  if (a.dry_run) argv.push('--dry-run')
  return argv
}

async function invoke(argv, settings) {
  const exe = await pythonFor(settings)
  if (!exe) return { text: JSON.stringify({ ok: false, error: 'no Python with numpy found - set HUMANPEN_PYTHON to one that has it' }, null, 2) }
  const result = await run(exe, [ENGINE, ...argv], settings.timeoutMs)
  return { text: JSON.stringify({ ok: result.ok, ...(result.ok ? { summary: result.out } : { error: result.error }), ...(result.stderr ? { log: result.stderr } : {}) }, null, 2) }
}

const DEVICE =
  'The real cursor moves and draws into whatever window is under it: foreground the target and select its ' +
  'pen/brush tool with a visible colour before the first call. dry_run measures without touching the pointer. ' +
  'Keep ONE seed for the whole task; change it for a different writer. Stack lines from the returned next_y.'

const NUM = { type: 'number' }
const OPTIONAL_NUM = { type: 'number', required: false }

/** Registers `handwrite` and `handdraw` over the bundled engine. */
export function apply(ctx, config) {
  const settings = { ...DEFAULTS, ...config }
  ctx.logger.info('dsh-handwriting: handwrite/handdraw registered (engine %s, backend %s)', ENGINE, settings.backend)

  ctx.tools.register(defineTool({
    name: 'handwrite',
    description:
      'Write handwritten text with real mouse strokes on a drawing surface (whiteboard, canvas, signature pad, ' +
      '"write your answer by hand" box). NOT for text inputs - a pen cannot type. x,y = top-left of the block in ' +
      'screen px, size = lowercase letter height (>= 24 reads best, never < 16), max_width wraps. Vary x/y/size/' +
      'messiness slightly per line and break before max_width so lines stay ragged. Inline markup <big> <small> ' +
      '<u> <s> <circle> <neat> <messy> <fast> <slow> <slant=N> <lift=F>; maths in $...$. ' + DEVICE,
    parameters: {
      text: { type: 'string', required: true, description: 'The text to write. Use \\n for a line break, $...$ for maths.' },
      x: { ...NUM, required: true, description: 'Left edge of the text block, screen px.' },
      y: { ...NUM, required: true, description: 'Top edge of the text block, screen px.' },
      size: { ...NUM, description: 'Lowercase letter height in px. 22-32 reads well at 1080p; default 24.' },
      max_width: { ...OPTIONAL_NUM, description: 'Wrap width in px. 0 or omitted = no wrapping.' },
      align: { type: 'string', required: false, enum: ['left', 'center', 'right'], description: 'Alignment inside max_width.' },
      messiness: { ...OPTIONAL_NUM, description: 'Multiplies every kind of variation: 0.4 careful label, 1.0 natural, 1.5 hurried.' },
      speed: { ...OPTIONAL_NUM, description: 'Pen speed multiplier: <1 careful, >1 hurried.' },
      mistakes: { ...OPTIONAL_NUM, description: 'Chance per word of typo -> scribble out -> rewrite. 0.05-0.1 for casual notes, never where it must be clean.' },
      seed: { ...OPTIONAL_NUM, description: 'Writer identity. Keep one value for the whole task; change it for a different person.' },
      dry_run: { type: 'boolean', required: false, description: 'Plan only: return ink bbox, duration and next_y without moving the pointer.' },
    },
    output: {
      schema: { type: 'object', additionalProperties: false, properties: { text: { type: 'string' } }, required: ['text'] },
      render: (_args, value) => [{ type: 'text', text: value.text }],
    },
    async execute(args) { return invoke(writeArgs(args, settings), settings) },
    presentCall: args => ({ card: 'generic', title: `Handwrite: ${String(args.text).slice(0, 60)}`, kind: 'other' }),
  }))

  ctx.tools.register(defineTool({
    name: 'handdraw',
    description:
      'Hand-drawn annotation on a drawing surface: line, underline, arrow (x0,y0 -> x1,y1), box, circle, check, ' +
      'cross, or scribble inside the box x0,y0-x1,y1. Overshoot underlines 6-12 px past the word and stop arrows a ' +
      'few px short of the target - dead-on looks plotted. Select the pen tool first. ' + DEVICE,
    parameters: {
      kind: { type: 'string', required: true, enum: ['line', 'underline', 'arrow', 'box', 'circle', 'check', 'cross', 'scribble'], description: 'Shape to draw.' },
      x0: { ...NUM, required: true }, y0: { ...NUM, required: true },
      x1: { ...NUM, required: true }, y1: { ...NUM, required: true, description: 'Second corner / target point, screen px.' },
      size: { ...NUM, description: 'Scale reference (stroke wobble), default 24.' },
      messiness: { ...OPTIONAL_NUM, description: 'Wobble multiplier, 1.0 default.' },
      seed: { ...OPTIONAL_NUM, description: 'Keep the same seed as the writing beside it - one hand.' },
      dry_run: { type: 'boolean', required: false, description: 'Plan only: no pointer movement.' },
    },
    output: {
      schema: { type: 'object', additionalProperties: false, properties: { text: { type: 'string' } }, required: ['text'] },
      render: (_args, value) => [{ type: 'text', text: value.text }],
    },
    async execute(args) { return invoke(drawArgs(args, settings), settings) },
    presentCall: args => ({ card: 'generic', title: `Handdraw: ${String(args.kind)}`, kind: 'other' }),
  }))
}
