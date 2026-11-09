// Writes dist/profile.html: a plain-HTML copy of the portfolio content,
// so tools that can't run JavaScript (e.g. Copilot's web reader) can read it.
// Run after `vite build` (see the "build" script in package.json).
import { writeFileSync, mkdirSync } from 'node:fs'
import timeline from '../src/data/timeline.js'
import projects from '../src/data/projects.js'
import publications from '../src/data/publications.js'
import hobbies from '../src/data/hobbies.js'

const esc = (s = '') => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
const link = (url, text) => (url ? ` <a href="${esc(url)}">${esc(text)}</a>` : '')

const html = `<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Marvin Wu (Wu Hung Mao) – Profile</title>
<style>body{font-family:system-ui,sans-serif;max-width:820px;margin:2rem auto;padding:0 1rem;line-height:1.5}h2{margin-top:2rem}li{margin-bottom:.6rem}</style>
</head><body>
<h1>Marvin Wu (Wu Hung Mao)</h1>
<p>CS Specialist @ UofT · Ex-Ericsson vDU Intern · Medical Video Understanding Research</p>
<p>Interactive portfolio: <a href="./">wuhungmao.github.io/academic-work</a></p>

<h2>Timeline</h2>
<ul>
${timeline.map((t) => `<li><strong>${esc(t.date)} – ${esc(t.title)}</strong><br>${esc(t.description)}</li>`).join('\n')}
</ul>

<h2>Projects</h2>
<ul>
${projects.map((p) => `<li><strong>${esc(p.name)}</strong> (${esc(p.category)}, ${esc(p.date)})<br>${esc(p.description)}${p.tags ? `<br><em>${esc(p.tags.join(', '))}</em>` : ''}${link(p.link, 'Link')}</li>`).join('\n')}
</ul>

<h2>Publications</h2>
<ul>
${publications.map((p) => `<li><strong>${esc(p.title)}</strong><br>${esc(p.authors)}. ${esc(p.venue)} (${esc(p.status)})<br>${esc(p.description)}${link(p.pdf, 'Paper')}</li>`).join('\n')}
</ul>

<h2>Hobbies</h2>
<ul>
${hobbies.map((h) => `<li><strong>${esc(h.name)}</strong>: ${esc(h.summary)}</li>`).join('\n')}
</ul>
</body></html>
`
mkdirSync('dist', { recursive: true })
writeFileSync('dist/profile.html', html)
console.log('wrote dist/profile.html')
