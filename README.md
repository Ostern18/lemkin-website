# Lemkin AI Website

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![React](https://img.shields.io/badge/React-18.2-61dafb.svg)](https://reactjs.org/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.0-blue.svg)](https://www.typescriptlang.org/)
[![Tailwind CSS](https://img.shields.io/badge/Tailwind_CSS-3.3-38bdf8.svg)](https://tailwindcss.com/)

## Overview

Official website for Lemkin AI - an open-source initiative providing machine learning models and tools for international criminal justice and human rights investigation. The website showcases our AI models, analysis tools, and resources for legal professionals and human rights investigators.

## 🚀 Features

- **AI Models Gallery**: Interactive catalog of 15+ ML models with performance metrics
- **Tools Showcase**: 18+ specialized analysis tools for investigations
- **Dark/Light Mode**: Full theme support with OpenAI-inspired design
- **Responsive Design**: Mobile-first approach with professional UI/UX
- **Performance Dashboards**: Transparent model evaluation and benchmarks
- **Resource Center**: Documentation, guides, and case studies

## 🛠️ Tech Stack

- **React 18** with TypeScript
- **Tailwind CSS 3.3** with custom design system
- **Lucide React** icons
- **Framer Motion** animations
- **Custom routing** (no external dependencies)

## 📦 Quick Start

### Prerequisites

- Node.js 16+
- npm 7+

### Installation

```bash
# Clone the repository
git clone https://github.com/Ostern18/lemkin-website.git
cd lemkin-website

# Install dependencies
npm install

# Start development server
npm start
```

Visit `http://localhost:3000` to see the website.

### Build for Production

```bash
npm run build
```

## 🏗️ Project Structure

```
lemkin-website/
├── public/
│   ├── index.html
│   ├── Lemkin Logo Black_Shape_clear.png
│   └── Lemkin Logo (shape only).png
├── src/
│   ├── LemkinAIWebsite.tsx      # Main app (14 pages)
│   ├── ModelsPageRevised.tsx    # Models & tools page
│   ├── modelsData.ts            # Data and interfaces
│   ├── index.css                # Design system
│   └── index.tsx                # Entry point
├── tailwind.config.js           # Tailwind config
├── package.json
└── CLAUDE.md                    # Development notes
```

## 🎨 Pages

The website includes 14 pages:

- **Home** - Hero section with featured models
- **Models & Tools** - Interactive gallery with search/filter
- **About** - Organization mission and team
- **Resources** - Documentation and guides
- **Articles** - Blog and research papers
- **Docs** - Technical documentation
- **Overview** - Platform capabilities
- **Contribute** - How to get involved
- **Governance** - Project governance
- **Legal** - Terms and privacy
- **Contact** - Get in touch

## 💅 Design System

OpenAI/Palantir-inspired design featuring:

- Custom CSS variables for semantic theming
- 5-level elevation system with shadows
- 14 custom animations (gradients, shimmer, glow)
- Professional component library
- Full dark/light mode support

## 📊 Featured Models & Tools

### AI Models
- Document Classification
- Entity Recognition
- Sentiment Analysis
- Image Analysis
- Audio Transcription
- Video Analysis
- And more...

### Analysis Tools
- Evidence Processor
- Timeline Constructor
- Document Analyzer
- Communication Mapper
- Geographic Analyzer
- Pattern Detector
- And more...

## 🧪 Development

```bash
# Run tests
npm test

# Lint code
npm run lint

# Type checking
npm run type-check
```

## 🚀 Deployment

The site can be deployed to any static hosting service:

- Vercel
- Netlify
- GitHub Pages
- AWS S3
- Cloudflare Pages

## 📝 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

- Website: [lemkin.ai](https://lemkin.ai)
- Email: contact@lemkin.ai
- GitHub: [@Ostern18](https://github.com/Ostern18)

---

Built with ❤️ for justice and human rights