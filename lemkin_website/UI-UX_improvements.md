# Lemkin UI — Desktop‑Only Redesign Spec (OpenAI/Palantir)

**Scope:** Desktop web app **only** (≥1280px). No mobile or tablet variants. Optimize for large displays and dense analytical workflows. This replaces prior guidance that referenced mobile/command bars.

---

## 1) Design Principles (Desktop‑Only)

* **Operational clarity:** Everything should read like mission software, not marketing.
* **Predictable hierarchy:** One elevation story, one border token, tight type rhythm.
* **Dense but legible:** Compact spacing by default; easy scan of high‑info rows.
* **Two precise themes:** Light and dark with AA contrast for all text on all surfaces.
* **Keyboard first:** Full focus rings and tab order; large hit targets despite desktop.

---

## 2) Layout Grid & Spatial System

* **Page width:** Use a centered container with `max-width: 1440px` by default, `1600px` for data‑dense views (e.g., Models table). Side gutters `48px`.
* **Grid:** 12 columns, 24px gutters. Align table columns to grid where feasible.
* **Baseline:** 8px baseline. Spacing primitives: 8, 12, 16, 20, 24, 32, 40.
* **Sections:** Page header → Controls (filters/search/actions) → Primary content → Secondary panels.

---

## 3) Theme Tokens (Light/Dark) — Desktop‑Tuned

Use CSS variables to ensure consistent borders/elevation/contrast. Tokens are optimized for large screens to prevent washout (light) and muddiness (dark).

```css
:root {
  /* Surfaces */
  --bg:        #FFFFFF;   /* page */
  --surface:   #F7F8FA;   /* section */
  --elevated:  #F1F3F7;   /* cards/panels */

  /* Text */
  --ink:       #0E1116;   /* primary */
  --muted:     #3F4752;   /* body on surface */
  --subtle:    #6B7280;   /* captions */

  /* Accent */
  --accent:    #2563EB;   /* AA on white */
  --accent-ink:#0B3C9C;   

  /* Lines */
  --line:      #D9DFEA;   /* hard rule */
  --line-soft: #E8EDF5;   /* inner separators */

  /* States */
  --success:#16A34A; --warning:#D97706; --danger:#DC2626; --info:#0EA5E9;

  /* Focus & Shadows */
  --focus: 0 0 0 4px rgba(37, 99, 235, .20);
  --shadow-sm: 0 1px 2px rgba(14,17,22,.06);
  --shadow-md: 0 2px 8px rgba(14,17,22,.10);
  --shadow-lg: 0 8px 24px rgba(14,17,22,.12);
}

:root.dark {
  --bg:        #0B0F14;
  --surface:   #10151B;
  --elevated:  #121923;

  --ink:       #E6EAF0;
  --muted:     #B8C1CC;
  --subtle:    #8C96A3;

  --accent:    #5DA1FF;   /* tuned for dark */
  --accent-ink:#CFE2FF;

  --line:      #243041;
  --line-soft: #1A2432;

  --focus: 0 0 0 4px rgba(93,161,255,.28);
  --shadow-sm: 0 1px 2px rgba(0,0,0,.35);
  --shadow-md: 0 2px 8px rgba(0,0,0,.45);
  --shadow-lg: 0 8px 24px rgba(0,0,0,.55);
}

/* Globals */
html,body { height:100%; }
body { color: var(--ink); background: var(--bg); font: 400 16px/1.5 system-ui, -apple-system, Segoe UI, Roboto, Inter, "Helvetica Neue", Arial, "Apple Color Emoji", "Segoe UI Emoji"; }
* { box-sizing: border-box; }
::selection { background: rgba(37,99,235,.18); }

hr, .divider { border-top:1px solid var(--line); }
.card { background: var(--surface); border:1px solid var(--line); box-shadow: var(--shadow-sm); border-radius: 16px; }
```

---

## 4) Typography (Desktop Rhythm)

* Scale: `12, 14, 16, 18, 20, 24, 32, 48`.
* Headings: `h1 48/1.2 700`, `h2 32/1.25 700`, `h3 24/1.3 600`.
* Body: 16/24 (`--muted` on surfaces). Captions: 14/20 (`--subtle`).
* Max text width: hero title 22ch, body 72ch.
* Tighten letter‑spacing for large titles (desktop often appears airy otherwise).

---

## 5) Header (Persistent & Condensing)

* **Always visible**; condenses on scroll (padding 24→12). Border‑bottom uses `--line`.
* Left: logo; Center: primary sections (segmented control); Right: light/dark toggle + GitHub.
* No hamburger, no mobile states.

```tsx
// Header.tsx (desktop‑only)
import { useEffect, useState } from 'react';
import { cn } from './utils/cn';

export default function Header() {
  const [condensed, setCondensed] = useState(false);
  useEffect(() => {
    const onScroll = () => setCondensed(window.scrollY > 24);
    onScroll();
    window.addEventListener('scroll', onScroll);
    return () => window.removeEventListener('scroll', onScroll);
  }, []);

  return (
    <header
      className={cn(
        'sticky top-0 z-50 w-full border-b border-[var(--line)] backdrop-blur',
        'transition-all duration-300',
        condensed ? 'py-3 shadow-sm' : 'py-6'
      )}
    >
      <div className="mx-auto" style={{maxWidth:1600, paddingInline:48}}>
        <div className="flex items-center gap-32">
          <Logo />
          <NavTabs />
          <div className="ml-auto flex items-center gap-12">
            <ThemeToggle />
            <a className="btn-outline" href="https://github.com/...">GitHub</a>
          </div>
        </div>
      </div>
    </header>
  );
}
```

**Segmented Tabs (desktop)**

```tsx
function NavTabs(){
  const tabs = ['Home','Overview','Models','Docs','Articles','Ecosystem','About'];
  const active = 'Models';
  return (
    <nav className="flex gap-8">
      {tabs.map(t => (
        <button
          key={t}
          className={cn(
            'px-18 py-10 rounded-xl border',
            'border-[var(--line)] text-[var(--muted)] hover:text-[var(--ink)]',
            t===active && 'text-[var(--ink)] shadow-sm ring-1 ring-[var(--accent)]/40'
          )}
        >{t}</button>
      ))}
    </nav>
  );
}
```

---

## 6) Theme Toggle (Desktop, System‑Aware)

```tsx
// useTheme.ts
import { useEffect, useState } from 'react';
export function useTheme(){
  const [theme, setTheme] = useState<'light'|'dark'>(() =>
    (localStorage.getItem('theme') as 'light'|'dark'|null) ??
    (window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light')
  );
  useEffect(() => {
    document.documentElement.classList.toggle('dark', theme==='dark');
    localStorage.setItem('theme', theme);
  }, [theme]);
  return { theme, setTheme };
}

// ThemeToggle.tsx
import { useTheme } from './useTheme';
export function ThemeToggle(){
  const { theme, setTheme } = useTheme();
  return (
    <button
      className="btn-outline"
      onClick={() => setTheme(theme==='dark' ? 'light' : 'dark')}
      aria-label="Toggle theme"
    >{theme==='dark' ? 'Light' : 'Dark'}</button>
  );
}
```

---

## 7) Hero & Status Pill (Desktop)

* Background is **solid token** (`--bg`), not a heavy gradient.
* Status pill: `--surface` fill, `--line` border; icon + label; sits above H1.
* Button row uses 12px gaps; icons aligned baseline.

```tsx
<section className="mx-auto" style={{maxWidth:1440, paddingInline:48, paddingBlock:56}}>
  <div className="inline-flex items-center gap-8 px-12 py-6 rounded-full border border-[var(--line)] bg-[var(--surface)] text-sm text-[var(--muted)]">
    <span className="inline-block w-8 h-8 rounded-full bg-[var(--accent)]"></span>
    <span>System operational · 12 active models</span>
  </div>
  <h1 className="mt-24" style={{fontSize:48, lineHeight:1.2, fontWeight:700, maxWidth:'22ch'}}>Evidence‑grade AI for <span style={{color:'var(--accent)'}}>International Justice</span></h1>
  <p className="mt-16 text-[var(--muted)]" style={{maxWidth:'72ch'}}>Open‑source machine learning models rigorously validated for legal proceedings…</p>
  <div className="mt-24 flex gap-12">
    <a className="btn-primary">Explore Models</a>
    <a className="btn-outline">Documentation</a>
  </div>
</section>
```

---

## 8) “Join the Mission” Cards (Desktop Equal Height)

* 16px radius, 1px border `--line`, `shadow-sm`, padding 24.
* Title 18/600; meta 14/`--subtle`; body 14/`--muted`.
* All three cards grid with equal heights; CTA is link style `Get started →`.

```tsx
<section className="mx-auto" style={{maxWidth:1440, paddingInline:48, paddingBlock:56}}>
  <h2 className="mb-24" style={{fontSize:32,fontWeight:700}}>Join the Mission</h2>
  <div className="grid" style={{gridTemplateColumns:'repeat(3, 1fr)', gap:24}}>
    {cards.map(c => (
      <article key={c.title} className="card h-full p-24">
        <div className="flex items-start gap-12">
          <div className="p-10 rounded-xl border border-[var(--line)]"></div>
          <div className="flex-1">
            <h3 className="text-lg font-semibold">{c.title}</h3>
            <p className="text-sm text-[var(--subtle)] mt-4">{c.meta}</p>
            <p className="text-sm text-[var(--muted)] mt-10">{c.body}</p>
            <a className="link inline-flex items-center mt-14">Get started<span className="ml-6">→</span></a>
          </div>
        </div>
      </article>
    ))}
  </div>
</section>
```

---

## 9) Practitioners’ Brief — Robust States (No Visual “Empty”)

Ensure this section never collapses or appears broken on desktop.

```tsx
export function PractitionersBrief({state, data}:{state:'loading'|'empty'|'ready', data?:Brief}){
  return (
    <section className="mx-auto" style={{maxWidth:1440, paddingInline:48, paddingBlock:56}}>
      <div className="card p-24" style={{minHeight:240}}>
        <header className="flex items-center justify-between">
          <h2 className="text-xl font-semibold">Practitioners’ Brief</h2>
          <a className="btn-outline" href="/docs/methodology">View methodology</a>
        </header>
        {state==='loading' && <SkeletonBrief />}
        {state==='empty' && (
          <Empty
            title="No brief available yet"
            body="We’re preparing concise, deployable guidance for legal workflows."
            action={{label:'Browse docs', href:'/docs'}}
          />
        )}
        {state==='ready' && data && <BriefContent data={data} />}
      </div>
    </section>
  )
}
```

* **Skeleton**: 3 lines at 72%, 64%, 48% width + pill placeholder.
* **Empty**: subdued illustration, not playful; same card recipe.

---

## 10) Models — Enterprise Table (Default) & Grid (Optional)

**Table Rules (desktop‑only):**

* Sticky first and last columns. Numeric columns right‑aligned. Status/version centered.
* Row height **Compact 44px** (default), **Comfortable 56px** (toggle).
* Head stays visible when scrolling long lists (within table container).
* “View” is a quiet primary (accent ring on hover/focus) to avoid noise.
* Consistent units/labels (e.g., Performance shows Accuracy; if multiple metrics, show composite in inspector).

```tsx
// ModelsTable.tsx
export function ModelsTable({rows}:{rows:Row[]}){
  return (
    <div className="mx-auto" style={{maxWidth:1600, paddingInline:48}}>
      <div className="card p-0 overflow-auto" style={{maxHeight:'70vh'}}>
        <table className="min-w-full">
          <thead className="bg-[var(--surface)] text-[var(--muted)] sticky top-0 z-10">
            <tr>
              <Th sticky="left">Model</Th>
              <Th align="right">Performance</Th>
              <Th align="center">Status</Th>
              <Th align="center">Version</Th>
              <Th align="right">Downloads</Th>
              <Th sticky="right" align="center">Actions</Th>
            </tr>
          </thead>
          <tbody>
            {rows.map(r => (
              <tr key={r.id} className="border-t border-[var(--line-soft)] hover:bg-[var(--elevated)]/40">
                <Td sticky="left"><ModelCell m={r} /></Td>
                <Td align="right">{r.accuracy.toFixed(1)}%</Td>
                <Td align="center"><StatusTag s={r.status} /></Td>
                <Td align="center">{r.version}</Td>
                <Td align="right">{r.downloads.toLocaleString()}</Td>
                <Td sticky="right" align="center"><ViewButton id={r.id} /></Td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
```

**Inspector Drawer** (right side, 480px): `shadow-lg`, dimmed scrim, metrics grid 2×2, copy‑to‑clipboard with toast `aria-live="polite"`.

**Grid View** (optional): 3‑4 columns, consistent card recipe; use only when preview imagery benefits (vision models), otherwise prefer table.

---

## 11) Controls Row (Search/Filters/Layout)

* Desktop row under page header: Search (left, 420px), chips row (middle), status dropdown (right), and layout toggle (Table/Grid) at far right.
* Chips are neutral: `background: var(--bg)`, `border: var(--line)`, text `--muted`. Selected uses faint accent ring.

---

## 12) Components: Buttons, Inputs, Links (Desktop)

```css
.btn-primary { display:inline-flex; align-items:center; gap:8px; padding:10px 16px; border-radius:12px; background:var(--accent); color:#fff; border:1px solid transparent; box-shadow: var(--shadow-sm); }
.btn-primary:hover { filter:brightness(.98); }
.btn-primary:focus-visible { box-shadow: var(--focus); }

.btn-outline { display:inline-flex; align-items:center; gap:8px; padding:10px 16px; border-radius:12px; background:var(--bg); color:var(--ink); border:1px solid var(--line); }
.btn-outline:hover { background: var(--surface); }

.link { color: var(--accent); text-underline-offset: 2px; }
.link:hover { text-decoration: underline; }

.input { height:40px; padding:0 12px; border-radius:12px; border:1px solid var(--line); background:var(--bg); color:var(--ink); }
.input:focus-visible { box-shadow: var(--focus); outline:none; }
```

---

## 13) Empty/Loading/Errored States (Desktop)

* **Skeletons** for lists and cards; animated shimmer optional (respect reduced‑motion).
* **Empty** has a one‑line heading + 1‑2 sentence body + single primary action.
* **Error** shows a short cause + retry; logs to console for dev.
* All states maintain **section min-height** to avoid layout jumps.

---

## 14) Motion & Interaction (Desktop)

* Hover 180–220ms; drawers/panels 260–300ms.
* Table row hover: subtle bg change + slight elevation (no aggressive scaling).
* Respect `prefers-reduced-motion`.

---

## 15) Accessibility (AA on Desktop)

* WCAG AA for all text in both themes.
* Focus ring `--focus` everywhere; keyboard can reach every interactive element in reading order.
* Click targets ≥ 40×40 even on desktop to reduce precision fatigue.

---

## 16) Tailwind Extension (If using Tailwind)

```js
// tailwind.config.js (excerpt)
export default {
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        bg: 'var(--bg)', surface: 'var(--surface)', elevated:'var(--elevated)',
        ink:'var(--ink)', muted:'var(--muted)', subtle:'var(--subtle)',
        line:'var(--line)', accent:'var(--accent)'
      },
      boxShadow: {
        sm:'var(--shadow-sm)', md:'var(--shadow-md)', lg:'var(--shadow-lg)'
      },
      borderRadius: { xl:'16px' }
    }
  }
}
```

---

## 17) Desktop QA Checklist

* [ ] Header condenses on scroll; border uses `--line`; no layout shift.
* [ ] All hero and “Join the Mission” text passes AA (light & dark) on their surfaces.
* [ ] Practitioners’ Brief never collapses: shows skeleton → empty → content.
* [ ] Models table headers sticky; first & last columns sticky; numeric columns right‑aligned.
* [ ] Density toggle switches 44px ↔ 56px row heights.
* [ ] Theme persists (localStorage) and respects system preference on first load.
* [ ] Buttons/links/inputs follow one recipe; focus rings visible.
* [ ] Shadows are opacity‑based in dark; borders never pure black/white.

---

## 18) Implementation Order (Desktop‑only)

1. **Tokens & global styles** (drop in `tokens.css`).
2. **Header** (condensing behavior) + **Theme toggle**.
3. **Hero** + **status pill**.
4. **Join the Mission** (equalized cards; consistent CTAs).
5. **Practitioners’ Brief** (states + min-height).
6. **Models Table** (sticky columns, density toggle, inspector drawer).
7. Sweep for **empty/loading/error states** and **contrast fixes**.

---

## 19) Notes on Your Screenshots (Applied Here)

* **Light mode**: previously washed-out hover/CTAs → corrected with `--accent` and stronger borders.
* **Dark mode**: text on `--surface` changed to `--muted` (not `--subtle`) for body; captions use `--subtle` only.
* **Header**: now condenses; no mobile artifacts.
* **Practitioners’ Brief**: displays skeleton/empty correctly; never visually empty.
* **Models list**: numeric alignment + sticky columns + quieter “View” buttons; grid kept clean and de‑emphasized.

---

### Appendix: Minimal Utility Helpers

```tsx
// utils/cn.ts
export function cn(...xs:(string|false|undefined)[]){ return xs.filter(Boolean).join(' '); }

// Table cell helpers
export function Th({children, align='left', sticky}:{children:any, align?:'left'|'right'|'center', sticky?:'left'|'right'}){
  return (
    <th className={cn(
      'text-sm font-medium px-16 py-12 border-b border-[var(--line)]',
      align==='right' && 'text-right', align==='center' && 'text-center',
      sticky==='left' && 'sticky left-0 bg-[var(--surface)] z-10',
      sticky==='right' && 'sticky right-0 bg-[var(--surface)] z-10'
    )}>{children}</th>
  );
}
export function Td({children, align='left', sticky}:{children:any, align?:'left'|'right'|'center', sticky?:'left'|'right'}){
  return (
    <td className={cn(
      'px-16 py-10 text-[var(--muted)]',
      align==='right' && 'text-right', align==='center' && 'text-center',
      sticky==='left' && 'sticky left-0 bg-[var(--bg)]',
      sticky==='right' && 'sticky right-0 bg-[var(--bg)]'
    )}>{children}</td>
  );
}
```
