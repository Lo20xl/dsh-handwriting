/**
 * Self-check for the plugin: registers into a mock context and plans one write and one draw with dry_run,
 * so nothing touches the pointer. Run `node check.mjs` from this directory.
 */
import assert from 'node:assert/strict'
import { apply } from './index.js'

const tools = new Map()
const ctx = {
  logger: { info: () => {}, warn: (...a) => console.error(...a) },
  tools: { register: t => tools.set(t.name, t) },
}

apply(ctx, {})
assert.deepEqual([...tools.keys()].sort(), ['handdraw', 'handwrite'])

const write = await tools.get('handwrite').execute({ text: 'check $x^2$', x: 100, y: 100, size: 26, dry_run: true })
const w = JSON.parse(write.text)
assert.equal(w.ok, true, `handwrite failed: ${write.text}`)
assert.match(w.summary, /stroke/i)

const draw = await tools.get('handdraw').execute({ kind: 'circle', x0: 300, y0: 300, x1: 420, y1: 360, dry_run: true })
const d = JSON.parse(draw.text)
assert.equal(d.ok, true, `handdraw failed: ${draw.text}`)

console.log('PASS')
console.log('write:', w.summary)
console.log('draw :', d.summary)
