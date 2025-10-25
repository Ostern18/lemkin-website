# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a React-based website for Lemkin AI, an open-source initiative providing machine learning models and tools for international criminal justice, human rights investigation, and legal technology applications. The website is built with TypeScript and features a sophisticated design system with full dark/light mode support and OpenAI-inspired UI/UX patterns.

## Tech Stack

- **React 18** with TypeScript: Core framework using hooks (useState, useEffect, useContext)
- **Tailwind CSS 3.3**: Utility-first CSS framework with custom design tokens
- **Lucide React**: Icon library providing 20+ icons throughout the UI
- **Framer Motion**: Animation library for smooth transitions and micro-interactions
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

### Multi-File Application Structure

The application has evolved from a single-file architecture to a more modular approach:

**Core Files:**
- `src/App.tsx`: Application entry point that renders LemkinAIWebsite
- `src/LemkinAIWebsite.tsx`: Main application with 14 page components (~5100 lines)
- `src/ModelsPageRevised.tsx`: Dedicated AI Models & Tools page with advanced features (~1300 lines)
- `src/ArticlesPage.tsx`: Dedicated articles page with search, filtering, and article reader (~300 lines)
- `src/ArticleReader.tsx`: Component for reading individual articles with related articles
- `src/modelsData.ts`: Comprehensive data models and mock data for tools and models
- `src/articlesData.ts`: Article metadata and structure for the articles system
- `src/index.css`: Custom design system with CSS variables and OpenAI-inspired patterns

**Key Architecture Components:**

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
   - `ModelsPageRevised`: Advanced model/tool grid with search, filtering, and comparison features
   - `ArticlesPage`: Standalone articles page with search, filtering by category/tags, and article reader integration
   - Other pages: About, Contact, Contribute, Docs, Governance, Legal, Overview, Resources

### Custom Design System

The project includes an extensive OpenAI-inspired design system spanning both `tailwind.config.js` and `src/index.css`:

**Tailwind Config (`tailwind.config.js`):**
- **Custom Colors**: Primary (blue), neutral (grays), accent (purple/pink/cyan/emerald/orange)
- **Typography Scale**: Professional scale with display sizes (2xl-sm) with optimized line heights and letter spacing
- **Shadow System**: Sophisticated elevation system (elevation-1 through elevation-5, plus glow and soft variants)
- **Animations**: 14 custom animations including gradient effects, glow pulse, shimmer, and float
- **Background Patterns**: Advanced gradients (mesh, grid, dot, noise patterns)

**CSS Variables System (`src/index.css`):**
- Professional Palantir/OpenAI-inspired design tokens with CSS custom properties
- Complete dark/light mode support with semantic color naming (--ink, --muted, --subtle, --surface, --elevated, --accent)
- Enhanced shadow and focus ring systems
- Professional button components with ripple effects
- Comprehensive hover states and micro-interactions

### Data Architecture

**Models Data (`src/modelsData.ts`):**
Complex TypeScript interfaces and mock data structured for:
- **Models**: AI/ML models with performance metrics, technical specifications, and evaluation data
- **Tools**: 18+ renamed tools (no longer "Lemkin-X" pattern) with specific functional names
- **Model Metrics**: Accuracy, F1 scores, inference speed, model size, parameters
- **Capabilities**: Real-world impact examples and use case categorization
- **Module Types**: Differentiation between 'model' and 'module' (tool) types

**Articles Data (`src/articlesData.ts`):**
- **Article Interface**: Comprehensive article metadata including title, excerpt, category, tags, author, date, read time
- **Content Loading**: Articles load content from markdown files in `/public/articles/` directory
- **Categorization**: Articles organized by categories (AI/ML, Legal Tech, Investigation Methods, Technical, Policy)
- **Tagging System**: Rich tagging for search and filtering functionality

## Key Implementation Patterns

### Custom Routing System
- Navigation through `RouterContext` with `currentPath` state
- `navigate()` function handles route changes and auto-scrolls to top
- Conditional rendering based on path matching
- No external router dependencies

### Theme System Architecture
- Tailwind's `class` dark mode strategy with CSS variables
- Theme state persisted via `useEffect` with DOM manipulation
- All components support both themes using semantic CSS variables
- Theme toggle in navigation with smooth transitions

### Component Design Patterns
- Extensive use of compound components (Button with variants/sizes)
- Hover states with transform and shadow changes using CSS custom properties
- Mobile-first responsive design
- Component composition over inheritance
- Tool-specific icon mappings and color differentiation

### Performance Considerations
- Modular file architecture for better code organization
- Lazy-loaded features through conditional rendering
- Optimized Tailwind classes with custom utilities
- Mock data prevents external API dependencies during development

## Development Notes

### When Modifying Routing
- Update both the `navItems` array in Navigation component
- Add corresponding case in the main App component's route rendering
- Ensure mobile menu includes new routes
- Consider whether new pages should use `ModelsPageRevised.tsx` pattern

### Design System Usage - OpenAI Inspired Patterns
- **Colors**: Use semantic CSS variables (`var(--accent)`, `var(--ink)`, `var(--muted)`) instead of hard-coded Tailwind colors
- **Shadows**: Use custom elevation classes (elevation-1 through elevation-5, plus soft and glow variants) instead of default Tailwind shadows
- **Buttons**: Use professional button classes (`.btn-primary`, `.btn-outline`) with built-in hover states and ripple effects
- **Icons**: Maintain consistency with tool-specific icons using lucide-react library
- **Gradients**: Use subtle gradients with low opacity for professional appearance (e.g., `from-[var(--accent)]/8 to-[var(--accent)]/3`)

### Component Styling Best Practices
- **Dark mode variants**: Required for all new components using CSS custom properties that automatically adapt
- **Animation classes**: Use established animation classes from Tailwind config (fade-in, fade-up, slide-in, etc.)
- **Responsive design**: Follow mobile-first strategy with sm/md/lg/xl breakpoints
- **CSS utilities**: Leverage utility classes like `.card`, `.skeleton`, `.glass`, `.hover-lift` for consistent styling

### Models & Tools Page (`src/ModelsPageRevised.tsx`)
- **Tool naming**: Uses descriptive names instead of generic "Lemkin-X" pattern
- **Icon consistency**: Tool-specific icons that match functionality (Shield, Clock, BarChart3, etc.)
- **Color scheme**: Unified color system using `var(--accent)` for cohesive appearance
- **No tier labels**: Removed tier categorization as requested
- **Differentiated styling**: Models vs tools use same accent color but different semantic meanings

### Logo Assets
The project references specific logo files that should be placed in the public directory:
- `Lemkin Logo Black_Shape_clear.png`: Black version for light mode
- `Lemkin Logo (shape only).png`: White version for dark mode

## Testing and Quality

The project uses Create React App's default testing setup:
- Test runner: Jest with React Testing Library
- No additional linting setup beyond React App's ESLint configuration
- No separate prettier or formatting configuration
- Build warnings are acceptable (mainly unused imports), but no build errors

## Deployment

The project is configured for AWS Amplify deployment via `amplify.yml`:
- Build process: `npm install` → `npm run build`
- Output directory: `build/`
- Caching: `node_modules` cached between builds

## Important Development Notes

### Architecture Evolution
- Originally single-file architecture, now evolved (`src/LemkinAIWebsite.tsx` ~5100 lines)
- Modular with dedicated components like `ModelsPageRevised.tsx` and `ArticlesPage.tsx`
- When adding new complex pages, consider creating dedicated files following the `ModelsPageRevised.tsx` or `ArticlesPage.tsx` patterns
- Mock data is centralized in `src/modelsData.ts` and `src/articlesData.ts` with comprehensive TypeScript interfaces
- Articles system loads content dynamically from markdown files in `/public/articles/` directory

### Component Location Reference
Key components in `LemkinAIWebsite.tsx`:
- Theme/Router contexts: lines 1-100
- UI components (Button, Badge, Card, etc.): lines 400-700
- Navigation component: line ~830
- Page components: lines 1050+ (HomePage, ArticlesPage, etc.)
- Main App routing logic: line ~2910

### Icon Management
- Use only valid lucide-react icons (verify imports exist)
- Tool-specific icons should be semantic and professional
- Maintain consistency between icon choice and functionality
- Use `var(--accent)` for icon colors instead of hard-coded values

### Data Management
- Tool data in `src/modelsData.ts` uses descriptive names
- Models have detailed metrics and technical specifications
- All mock data includes realistic performance indicators
- Article data in `src/articlesData.ts` includes comprehensive metadata for filtering and search
- Articles content is stored as markdown files in `/public/articles/` directory
- Maintain TypeScript interface consistency when adding new data

### Articles System
- **Content Structure**: Markdown files in `/public/articles/` with frontmatter-style metadata
- **Dynamic Loading**: Articles loaded via fetch API with error handling
- **Search & Filter**: Full-text search across title, excerpt, and tags with category and tag filtering
- **Related Articles**: Automatically generated based on shared tags
- **Performance**: Client-side filtering with memoized computations for responsive UI