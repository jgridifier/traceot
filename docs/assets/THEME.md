# Theme: `brief.css`

A shared "daily brief" look for research pages, built because the reader finds
editorial daily-brief sites easy to scan. We reproduce the visual style only:
no logo, name or text from any reference site.

**Look**
- Warm paper background (`#f2e8d5` with a lighter radial top), near-black ink `#1c150d`.
- Headlines: Bricolage Grotesque 800, tight letter-spacing. Body: Newsreader (serif), ~17px, 1.58 line height.
  Kickers, tags, meta and table headers: JetBrains Mono, uppercase, wide tracking.
- Cards: `#fffdf7`, 3px ink border, hard 8px offset shadow in an accent colour
  (amber `#e8902e`, teal `#1f9e8e`, green `#54b441`, red `#df5236`). Dark "take" panels use `#1f1912`.
- Section heads: mono number label, big heading, rule line filling the rest of the row.
- Structure: masthead → ticker of one-liners → "short version" box → detailed sections.
  Summaries go first and detail goes below; long tables sit inside `<details>`.

**Loading**
```html
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,400..800&family=Newsreader:ital,opsz,wght@0,6..72,400..600;1,6..72,400..500&family=JetBrains+Mono:wght@400;500;700&display=swap">
<link rel="stylesheet" href="../assets/brief.css">   <!-- adjust depth -->
<!-- optional math -->
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/contrib/auto-render.min.js"
  onload="renderMathInElement(document.body,{delimiters:[{left:'$$',right:'$$',display:true},{left:'\\(',right:'\\)',display:false}]})"></script>
```

**Plain HTML works unchanged.** `main`, `h1`, `h2`, top-level `main > ul > li` (rendered as cards), `details`
(dark panel), `.note`, `.status`, `.meta`, `.lede`, `table`, `pre`, `code`, `footer` are all styled, so an
existing page can switch theme by swapping its `<style>` block for the two `<link>` tags.

**Components:** `.mast` (`.date`, `h1`, `.dek`, `nav`), `.ticker`, `.tldr`, `.sec-head` (`.n`, `h2`, `.line`),
`.card` (+ `.teal` `.green` `.red` `.blue` `.dark`, inner `.card-top`, `.foot`), `.tag`, `.verdict.adopt|interesting|skip`,
`.grid`, `.math-block`, `.table-wrap`, `td.num`, `figure`/`figcaption`.

The same file lives in `jgridifier/research-lab` and `jgridifier/traceot` at `docs/assets/brief.css`. Keep them in sync.
