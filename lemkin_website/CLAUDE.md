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

The entire application is contained in a single file (`src/LemkinAIWebsite.tsx`) with approximately 14 page components and a sophisticated component hierarchy:

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

The project includes an extensive custom design system in `tailwind.config.js`:

- **Custom Colors**: Primary (blue), accent (purple/pink/cyan/emerald/orange), neural (grays), authority (navy/charcoal/steel)
- **Typography Scale**: Display sizes (4xl-lg) with optimized line heights and letter spacing
- **Shadow System**: Neural shadows (neural, neural-md, neural-lg, neural-xl) for depth hierarchy
- **Animations**: 12 custom animations including gradient effects, glow pulse, and float
- **Background Patterns**: Neural-net and mesh-gradient patterns for visual depth

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
- Use custom shadow classes (neural, neural-md, neural-lg, neural-xl) instead of default Tailwind shadows
- Follow the established color hierarchy: slate for backgrounds, blue for primary actions
- Maintain consistency with the button hierarchy (primary/secondary/tertiary/ghost)

### Component Styling
- Dark mode variants are required for all new components
- Use the established animation classes for consistency
- Follow the responsive breakpoint strategy: mobile-first with sm/md/lg/xl

### Logo Assets
The project references specific logo files that should be placed in the public directory:
- `Lemkin Logo Black_Shape_clear.png`: Black version for light mode
- `Lemkin Logo (shape only).png`: White version for dark mode