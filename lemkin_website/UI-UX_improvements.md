import React, { useEffect, useMemo, useRef, useState, createContext, useContext } from 'react';
</section>
);
};


/***********************
* PAGE CHROME
***********************/
const PageHeader: React.FC<{ title: string; description?: string; actions?: React.ReactNode }> = ({ title, description, actions }) => (
<div className="mb-6 flex flex-col md:flex-row md:items-end md:justify-between gap-3">
<div>
<h1 className="text-2xl font-semibold tracking-tight">{title}</h1>
{description && <p className="text-[var(--color-text-secondary)] mt-1">{description}</p>}
</div>
{actions}
</div>
);


const Footer: React.FC = () => (
<footer className="border-t mt-16">
<div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-10 text-sm">
<div className="text-center mb-6">
<Badge tone="info" className="uppercase"><Shield className="w-3 h-3"/> Trust & Transparency Center</Badge>
<p className="text-[var(--color-text-tertiary)] mt-3">Security practices, evaluation methodologies, and ethical guidelines.</p>
</div>
<div className="grid grid-cols-2 md:grid-cols-4 gap-6">
<div><div className="font-semibold mb-2 flex items-center gap-2"><Eye className="w-4 h-4"/>Transparency</div><ul className="space-y-1 text-[var(--color-text-secondary)]"><li><a href="#" className="hover:underline">Changelog</a></li><li><a href="#" className="hover:underline">Eval Methodology</a></li><li><a href="#" className="hover:underline">Data Provenance</a></li></ul></div>
<div><div className="font-semibold mb-2 flex items-center gap-2"><Shield className="w-4 h-4"/>Security</div><ul className="space-y-1 text-[var(--color-text-secondary)]"><li><a href="#" className="hover:underline">Disclosure Policy</a></li><li><a href="#" className="hover:underline">SBOM</a></li><li><a href="#" className="hover:underline">Incident Response</a></li></ul></div>
<div><div className="font-semibold mb-2 flex items-center gap-2"><Gavel className="w-4 h-4"/>Legal</div><ul className="space-y-1 text-[var(--color-text-secondary)]"><li><a href="#" className="hover:underline">Licenses</a></li><li><a href="#" className="hover:underline">Privacy Policy</a></li><li><a href="#" className="hover:underline">Terms</a></li></ul></div>
<div><div className="font-semibold mb-2 flex items-center gap-2"><Github className="w-4 h-4"/>Community</div><ul className="space-y-1 text-[var(--color-text-secondary)]"><li><a href="https://github.com/lemkin-ai" className="inline-flex items-center gap-1 hover:underline">GitHub <ExternalLink className="w-3 h-3"/></a></li><li><a href="#" className="hover:underline">Governance</a></li><li><a href="#" className="hover:underline">Code of Conduct</a></li></ul></div>
</div>
<div className="mt-8 text-[var(--color-text-secondary)] text-center">© 2025 Lemkin AI. Open source licensed.</div>
</div>
</footer>
);


/***********************
* ROOT APP
***********************/
const RouteSwitch: React.FC = () => {
const { path } = useRouter();
if (path === '/models') return <ModelsPage/>;
// Stubs for other routes
if (path === '/docs') return <Stub title="Docs"/>;
if (path === '/articles') return <Stub title="Articles"/>;
if (path === '/ecosystem') return <Stub title="Ecosystem"/>;
if (path === '/about') return <Stub title="About"/>;
return <HomePage/>;
};


const Stub: React.FC<{ title: string }> = ({ title }) => (
<section id="main" className="pt-24 pb-16 max-w-5xl mx-auto px-4">
<PageHeader title={title} description="Coming soon." />
<Card>
<p className="text-[var(--color-text-secondary)]">This area is scaffolded. Replace with your real content.</p>
</Card>
</section>
);


const AppShell: React.FC<{children: React.ReactNode}> = ({ children }) => (
<div className="min-h-screen bg-[var(--color-bg-primary)] text-[var(--color-text-primary)]">
<TopNav/>
<main className="pt-16">{children}</main>
<Footer/>
</div>
);


export default function LemkinApp() {
return (
<ThemeProvider>
<Router>
<AppShell>
<RouteSwitch/>
</AppShell>
</Router>
</ThemeProvider>
);
}

Lemkin UI Redesign – AppShell + Models Workspace (single-file .tsx)
(ready to paste in for a drop-in replacement)

What changed (at a glance)

AppShell & Chrome

Fixed, minimal TopNav with active-state tabs, theme toggle, skip link, and mobile menu.

Command Palette (⌘/Ctrl + K) for fast nav to Models/Docs/Articles/Trust pages; ESC closes; focus management built-in.

Models Workspace

Token-driven list view (enterprise default) with quick-scan columns: name/desc, status, version, accuracy, actions.

Slide-over inspector with evaluation/provenance cards and action buttons (copy spec / open repo).

Inline search that filters by name/tags.

Home (mission-critical posture)

Tighter hero copy; neutral CTA pair; compact “Upload / Analyze / Export” capability tiles.

Featured models in a readable, dense list (not glossy cards), matching high-trust tools.

A11y & Interaction Quality

Skip link, focus rings, ARIA roles on dialogs/menus, keyboard shortcuts, and ESC to close overlays.

All buttons/inputs sized to enterprise Fitts’ Law defaults (≥40px tap targets).

Why this meets OpenAI × Palantir expectations

Neutral, tokenized palette with restrained elevation and crisp borders for clarity in high-stakes contexts.

Power-first information architecture (table/list density, secondary text, compact tags) optimized for expert scanning.

Transparent model handling (status pill + accuracy + evaluator/provenance links) to signal reliability and legal rigor.

Integration steps (copy/paste)

Replace your current page component with the canvas file (it exports default function LemkinApp()).

Keep your tokens (your index.css variable system is solid) and add tiny utilities for consistent elevation (below).

CSS patch (add to index.css)

Your TSX previously used several shadow classes that weren’t defined in your tokens. Add these neutral, readable elevations to keep the look consistent across light/dark:

@layer utilities {
  .shadow-elevation-1 { box-shadow: 0 1px 2px rgba(14,17,22,.05); }
  .shadow-elevation-2 { box-shadow: 0 4px 12px rgba(14,17,22,.07); }
  .shadow-elevation-3 { box-shadow: 0 8px 24px rgba(14,17,22,.08); }
  .shadow-elevation-4 { box-shadow: 0 14px 40px rgba(14,17,22,.10); }
}

Important cleanups I made (and why)

Removed non-token classes like bg-neural-950, text-neural-300, bg-neural-net, accent-* that aren’t defined in your tokens—these drift visual language and can fail silently in Tailwind builds. Everything now uses CSS variables and Tailwind primitives from your system for consistency and easier auditing. 

LemkinAIWebsite

Kept and relied on your design tokens (ink/neutral, --color-bg-*, --color-text-*, borders, focus rings). Your base theme and dark-mode variables already match the neutral, institutional aesthetic—we’re just applying them more strictly across UI primitives and pages. 

index

What you now have

Single-file, production-style scaffold you can splice into your codebase now (keeps your mock data shape).

Consistent, accessible primitives: Button, Badge, Card, MetricTile, StatusPill.

Enterprise-grade patterns: command bar, slide-over inspector, dense list/table defaults, focused IA.

If you want, I can follow up with:

A real DataTable (sortable columns, column density controls).

Evaluation panel that reads from a JSON spec and renders methodologies, datasets, and bias checks.

A Documented UI kit (Props tables + Storybook-style examples) that mirrors this token system.