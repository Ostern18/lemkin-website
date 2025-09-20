# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a React-based website for Lemkin AI, an open-source initiative providing machine learning models and tools for international criminal justice, human rights investigation, and legal technology applications. The website is built with TypeScript and features a sophisticated design system with full dark/light mode support.

## Tech Stack

- **React 18** with TypeScript: Core framework using hooks (useState, useEffect, useContext)
- **Tailwind CSS 3.3**: Utility-first CSS framework with custom design tokens
- **Lucide React**: Icon library providing 20+ icons throughout the UI
- **Custom Routing**: Built-in router context without external dependencies

## Development Commands

```bash
# Start development server
npm start

# Build for production
npm run build

# Run tests
npm test

# Eject (not recommended)
npm run eject
```

## Architecture

### Application Structure

The entire application is contained in a single file (`src/LemkinAIWebsite.tsx` - ~3000 lines) with 14 page components and a sophisticated component hierarchy:

1. **Context Providers** (Root level)
   - `ThemeProvider`: Manages dark/light mode with DOM class manipulation
   - `Router`: Custom client-side routing with state-based navigation

2. **Reusable UI Components**
   - `Button`: Enhanced with 5 variants (primary, secondary, tertiary, ghost, danger) and 4 sizes
   - `Badge`: Status indicators with dynamic color variants
   - `Card`: Container with hover states and 4 variants (default, elevated, outlined, filled)
   - `ModelCard`: Complex component with performance indicators and evaluation transparency
   - `CodeBlock`: Interactive code display with copy functionality

3. **Layout Components**
   - `Navigation`: Fixed header with mobile menu, theme toggle, and animated nav indicators
   - `Footer`: Site footer (implementation varies by page)

4. **Page Components** (14 total)
   - `HomePage`: Hero section, featured models, practitioner briefs, contribution CTAs
   - `ModelsPage`: Model grid with search, filtering, and comparison features
   - Other pages: About, Articles, Contact, Contribute, Docs, Governance, Legal, Overview, Resources

### Custom Design System

The project includes an extensive custom design system spanning both `tailwind.config.js` and `src/index.css`:

**Tailwind Config (`tailwind.config.js`):**
- **Custom Colors**: Primary (blue), neutral (grays), accent (purple/pink/cyan/emerald/orange)
- **Typography Scale**: Professional scale with display sizes (2xl-sm) with optimized line heights and letter spacing
- **Shadow System**: Sophisticated elevation system (elevation-1 through elevation-5, plus glow and soft variants)
- **Animations**: 14 custom animations including gradient effects, glow pulse, shimmer, and float
- **Background Patterns**: Advanced gradients (mesh, grid, dot, noise patterns)

**CSS Variables System (`src/index.css`):**
- Professional Palantir/OpenAI-inspired design tokens with CSS custom properties
- Complete dark/light mode support with semantic color naming (--ink, --muted, --subtle, --surface, --elevated)
- Enhanced shadow and focus ring systems
- Professional button components with ripple effects
- Comprehensive hover states and micro-interactions

### Data Architecture

Mock data is structured for:
- **Models**: Complex objects with performance metrics, tags, status, and metadata
- **Practitioner Briefs**: Role-based filtering (Investigators/Prosecutors/Researchers) with peer review status
- **Resources**: Documentation and reference materials

## Key Implementation Patterns

### Custom Routing System
- Navigation through `RouterContext` with `currentPath` state
- `navigate()` function handles route changes and auto-scrolls to top
- Conditional rendering based on path matching
- No external router dependencies

### Theme System Architecture
- Tailwind's `class` dark mode strategy
- Theme state persisted via `useEffect` with DOM manipulation
- All components support both themes with `dark:` prefixes
- Theme toggle in navigation with smooth transitions

### Component Design Patterns
- Extensive use of compound components (Button with variants/sizes)
- Hover states with transform and shadow changes
- Mobile-first responsive design
- Component composition over inheritance

### Performance Considerations
- Single file architecture reduces bundle complexity
- Lazy-loaded features through conditional rendering
- Optimized Tailwind classes with custom utilities
- Mock data prevents external API dependencies during development

## Development Notes

### When Modifying Routing
- Update both the `navItems` array in Navigation component
- Add corresponding case in the main App component's route rendering
- Ensure mobile menu includes new routes

### Design System Usage
- Use custom elevation shadow classes (elevation-1 through elevation-5, plus soft and glow variants) instead of default Tailwind shadows
- Follow the CSS custom property system: `var(--ink)` for primary text, `var(--muted)` for body text, `var(--surface)` for backgrounds
- Use the professional button classes: `.btn-primary`, `.btn-outline` with built-in hover states and ripple effects
- Maintain consistency with the button hierarchy and professional color palette

### Component Styling
- Dark mode variants are required for all new components - the system uses CSS custom properties that automatically adapt
- Use the established animation classes from Tailwind config (fade-in, fade-up, slide-in, etc.)
- Follow the responsive breakpoint strategy: mobile-first with sm/md/lg/xl
- Leverage CSS utility classes like `.card`, `.skeleton`, `.glass`, `.hover-lift` for consistent styling

### Logo Assets
The project references specific logo files that should be placed in the public directory:
- `Lemkin Logo Black_Shape_clear.png`: Black version for light mode
- `Lemkin Logo (shape only).png`: White version for dark mode

## Testing and Quality

The project uses Create React App's default testing setup:
- Test runner: Jest with React Testing Library
- No additional linting setup beyond React App's ESLint configuration
- No separate prettier or formatting configuration

## Important Development Notes

### Single File Architecture
- The entire UI is in `src/LemkinAIWebsite.tsx` (~3000 lines) - this is intentional for this project
- When adding new components, add them within this file following the established patterns
- Mock data is defined at the top of the file and used throughout pages

### Component Location Reference
Key components are located at these approximate line numbers in `LemkinAIWebsite.tsx`:
- Theme/Router contexts: lines 1-100
- UI components (Button, Badge, Card, etc.): lines 400-700
- Navigation component: line ~830
- Page components: lines 1050+ (HomePage, ModelsPage, ArticlesPage, etc.)
- Main App routing logic: line ~2910