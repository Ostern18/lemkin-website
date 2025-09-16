import React, { useState, useEffect, createContext, useContext } from 'react';
import { Menu, X, Sun, Moon, Search, Calendar, Clock, AlertCircle, CheckCircle, Book, Code, Users, Mail, ExternalLink, Github, Twitter, FileText, Download, ArrowRight, ArrowLeft, Copy, Check, Scale, Shield, Eye, Gavel, Terminal } from 'lucide-react';

// Theme Context
interface ThemeContextType {
  theme: string;
  toggleTheme: () => void;
}

const ThemeContext = createContext<ThemeContextType | undefined>(undefined);

const useTheme = () => {
  const context = useContext(ThemeContext);
  if (!context) throw new Error('useTheme must be used within ThemeProvider');
  return context;
};

const ThemeProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [theme, setTheme] = useState<'light'|'dark'>(() => {
    if (typeof window === "undefined") return 'light';
    return (localStorage.getItem("theme") as 'light'|'dark')
      ?? (window.matchMedia("(prefers-color-scheme: dark)").matches ? 'dark' : 'light');
  });

  useEffect(() => {
    const root = document.documentElement;
    if (theme === 'dark') {
      root.classList.add('dark');
    } else {
      root.classList.remove('dark');
    }
    localStorage.setItem("theme", theme);
  }, [theme]);

  const toggleTheme = () => setTheme(prev => prev === 'light' ? 'dark' : 'light');

  return (
    <ThemeContext.Provider value={{ theme, toggleTheme }}>
      {children}
    </ThemeContext.Provider>
  );
};

// Router Context
interface RouterContextType {
  currentPath: string;
  navigate: (path: string) => void;
}

const RouterContext = createContext<RouterContextType | undefined>(undefined);

const useRouter = () => {
  const context = useContext(RouterContext);
  if (!context) throw new Error('useRouter must be used within Router');
  return context;
};

const Router: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [currentPath, setCurrentPath] = useState('/');
  
  const navigate = (path: string) => {
    setCurrentPath(path);
    window.scrollTo(0, 0);
  };

  return (
    <RouterContext.Provider value={{ currentPath, navigate }}>
      {children}
    </RouterContext.Provider>
  );
};

// Mock Data
const mockModels = [
  {
    id: 'whisper-legal-v2',
    name: 'Whisper Legal v2',
    description: 'Fine-tuned speech recognition model optimized for legal proceedings and testimony transcription.',
    tags: ['audio', 'transcription', 'legal'],
    status: 'stable',
    version: '2.1.0',
    license: 'Apache 2.0',
    lastUpdated: '2025-01-10',
    downloads: 15420,
    accuracy: 94.7
  },
  {
    id: 'doc-analyzer-xl',
    name: 'Document Analyzer XL',
    description: 'Multi-modal model for analyzing legal documents, evidence photos, and case materials.',
    tags: ['vision', 'nlp', 'multimodal'],
    status: 'beta',
    version: '1.0.0-beta.3',
    license: 'MIT',
    lastUpdated: '2025-01-08',
    downloads: 8930,
    accuracy: 91.2
  },
  {
    id: 'testimony-classifier',
    name: 'Testimony Classifier',
    description: 'NLP model for categorizing and analyzing witness testimonies and statements.',
    tags: ['nlp', 'classification', 'legal'],
    status: 'stable',
    version: '3.2.1',
    license: 'Apache 2.0',
    lastUpdated: '2024-12-20',
    downloads: 22105,
    accuracy: 89.5
  }
];

const mockPractitionerBriefs = [
  {
    id: 'ethical-ai-tribunals',
    title: 'Ethical Guidelines for AI in International Tribunals',
    excerpt: 'Comprehensive framework for responsible deployment of AI tools in international criminal proceedings.',
    author: 'Dr. Sarah Chen',
    date: '2025-01-12',
    lastReviewed: 'Jan 2025',
    readTime: '12 min',
    category: 'Guidelines',
    tags: ['ethics', 'guidelines', 'tribunals'],
    roles: ['Prosecutors', 'Researchers'],
    peerReviewed: true
  },
  {
    id: 'osint-workflows',
    title: 'OSINT Workflows for War Crimes Investigation',
    excerpt: 'Best practices for integrating open-source intelligence tools with machine learning models.',
    author: 'Marcus Rodriguez',
    date: '2025-01-05',
    lastReviewed: 'Jan 2025',
    readTime: '8 min',
    category: 'Methodology',
    tags: ['osint', 'investigation', 'workflow'],
    roles: ['Investigators'],
    peerReviewed: true
  },
  {
    id: 'model-evaluation',
    title: 'Evaluating Model Performance in Legal Contexts',
    excerpt: 'Metrics and methodologies for assessing AI model reliability in high-stakes legal applications.',
    author: 'Dr. Elena Petrov',
    date: '2024-12-28',
    lastReviewed: 'Dec 2024',
    readTime: '15 min',
    category: 'Evaluation',
    tags: ['evaluation', 'metrics', 'methodology'],
    roles: ['Researchers', 'Prosecutors'],
    peerReviewed: true
  },
  {
    id: 'digital-evidence-analysis',
    title: 'Digital Evidence Analysis with ML Models',
    excerpt: 'Systematic approaches to analyzing digital evidence using machine learning for criminal investigations.',
    author: 'Dr. Ahmed Hassan',
    date: '2024-12-15',
    lastReviewed: 'Dec 2024',
    readTime: '18 min',
    category: 'Technical',
    tags: ['digital-evidence', 'analysis', 'investigation'],
    roles: ['Investigators'],
    peerReviewed: true
  },
  {
    id: 'courtroom-ai-presentation',
    title: 'Presenting AI Evidence in Court',
    excerpt: 'Legal standards and best practices for introducing AI-generated evidence in judicial proceedings.',
    author: 'Prof. Maria Santos',
    date: '2024-12-08',
    lastReviewed: 'Dec 2024',
    readTime: '22 min',
    category: 'Legal Practice',
    tags: ['courtroom', 'evidence', 'presentation'],
    roles: ['Prosecutors'],
    peerReviewed: true
  },
  {
    id: 'bias-detection-methods',
    title: 'Bias Detection in AI Models for Justice',
    excerpt: 'Research methodologies for identifying and mitigating bias in AI systems used for legal applications.',
    author: 'Dr. Jennifer Liu',
    date: '2024-11-28',
    lastReviewed: 'Nov 2024',
    readTime: '14 min',
    category: 'Research',
    tags: ['bias', 'fairness', 'methodology'],
    roles: ['Researchers'],
    peerReviewed: true
  }
];

// Keep mockArticles for backward compatibility in other components
const mockArticles = mockPractitionerBriefs;

const mockResources = [
  {
    id: 'quickstart',
    title: 'Quick Start Guide',
    description: 'Get up and running with Lemkin AI models in under 5 minutes.',
    icon: 'book',
    link: '/docs/quickstart'
  },
  {
    id: 'api-reference',
    title: 'API Reference',
    description: 'Complete API documentation for all models and endpoints.',
    icon: 'code',
    link: '/docs/api'
  },
  {
    id: 'best-practices',
    title: 'Best Practices',
    description: 'Guidelines for responsible and effective model deployment.',
    icon: 'check-circle',
    link: '/docs/best-practices'
  }
];

// Enhanced Button Component from homepage_improvements.md
interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: 'primary' | 'secondary' | 'tertiary' | 'ghost' | 'danger';
  size?: 'sm' | 'md' | 'lg' | 'xl';
  loading?: boolean;
  icon?: React.ReactNode;
  children: React.ReactNode;
  className?: string;
}

const Button: React.FC<ButtonProps> = ({
  variant = 'primary',
  size = 'md',
  loading = false,
  icon,
  children,
  className = '',
  ...props
}) => {
  const variants = {
    /* Light: ink on paper; Dark: paper on ink */
    primary: 'bg-[var(--color-accent-cta)] text-[var(--color-fg-inverse)] hover:opacity-90 active:opacity-80',
    /* Subtle neutral borders */
    secondary: 'bg-[var(--color-bg-elevated)] text-[var(--color-fg-primary)] border border-[var(--color-border-default)] hover:bg-[var(--color-bg-surface)]',
    /* Tertiary variant */
    tertiary: 'bg-[var(--color-bg-surface)] text-[var(--color-fg-primary)] border border-[var(--color-border-default)] hover:bg-[var(--color-bg-elevated)]',
    /* Ghost variant */
    ghost: 'bg-transparent text-[var(--color-fg-primary)] hover:bg-[color-mix(in_oklab,var(--color-fg-primary)_6%,transparent)]',
    /* Danger keeps semantic color */
    danger: 'bg-[var(--color-status-danger)] text-white hover:opacity-90'
  };

  const sizes = {
    sm: 'h-7 px-3 text-xs rounded-md gap-1',
    md: 'h-9 px-4 text-sm rounded-md gap-1.5',
    lg: 'h-11 px-6 text-base rounded-md gap-2',
    xl: 'h-12 px-8 text-lg rounded-md gap-2'
  };

  return (
    <button
      className={`
        inline-flex items-center justify-center font-medium
        transition-all duration-200
        active:scale-[0.97]
        focus-visible:outline-none focus-visible:ring-2
        focus-visible:ring-[var(--color-border-focus)]
        ring-offset-2 ring-offset-[var(--color-bg-surface)]
        disabled:opacity-50 disabled:cursor-not-allowed
        ${variants[variant]}
        ${sizes[size]}
        ${className}
      `}
      disabled={loading}
      {...props}
    >
      {loading && (
        <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24">
          <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
          <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
        </svg>
      )}
      {icon && !loading && icon}
      {children}
    </button>
  );
};

interface BadgeProps {
  variant?: 'default' | 'stable' | 'beta' | 'deprecated' | string;
  children: React.ReactNode;
  className?: string;
}

const Badge: React.FC<BadgeProps> = ({ variant = 'default', children, className = '' }) => {
  const variants: Record<string, string> = {
    default: 'bg-[var(--color-bg-elevated)] text-[var(--color-fg-muted)] border border-[var(--color-border-default)]',
    stable: 'bg-[var(--color-status-success)] text-white',
    beta: 'bg-[var(--color-status-warning)] text-white',
    deprecated: 'bg-[var(--color-status-danger)] text-white'
  };

  const icons = {
    stable: <CheckCircle className="w-3 h-3" />,
    beta: <AlertCircle className="w-3 h-3" />,
    deprecated: <X className="w-3 h-3" />
  };

  return (
    <span className={`inline-flex items-center gap-1 px-2.5 py-1 rounded-md text-xs font-medium ${variants[variant] || variants.default} ${className}`}>
      {variant !== 'default' && icons[variant as keyof typeof icons]}
      {children}
    </span>
  );
};

interface CardProps {
  children: React.ReactNode;
  variant?: 'default' | 'elevated' | 'outlined' | 'filled';
  hover?: boolean;
  className?: string;
}

const Card: React.FC<CardProps> = ({
  children,
  variant = 'default',
  hover = false,
  className = ''
}) => {
  const variants = {
    default: 'bg-[var(--color-bg-default)] border border-[var(--color-border-default)]',
    elevated: 'bg-[var(--color-bg-elevated)] border border-[var(--color-border-default)]',
    outlined: 'bg-[var(--color-bg-default)] border-2 border-[var(--color-border-default)]',
    filled: 'bg-[var(--color-bg-surface)] border border-[var(--color-border-default)]',
  };

  const hoverClasses = hover
    ? 'hover:transform hover:-translate-y-1 hover:[box-shadow:var(--shadow-interactive)] dark:hover:[box-shadow:var(--shadow-glow)] cursor-pointer transition-all duration-200'
    : '';

  return (
    <div className={`
      ${variants[variant]}
      ${hoverClasses}
      rounded-lg p-6 transition-colors duration-200
      ${className}
    `}>
      {children}
    </div>
  );
};

// Logo Component with theme-aware switching
const LemkinLogo: React.FC<{ className?: string }> = ({ className = "w-8 h-8" }) => {
  const { theme } = useTheme();

  // Use black logo for light mode, white logo for dark mode
  const logoSrc = theme === 'light'
    ? '/lemkin-logo-black-shape.png'
    : '/lemkin-logo-shape-white.png';

  return (
    <img
      src={logoSrc}
      alt="Lemkin AI Logo"
      className={className}
    />
  );
};



// Enhanced Model Card with improved information architecture
interface ModelCardProps {
  model: any;
}

const ModelCard: React.FC<ModelCardProps> = ({ model }) => {
  const { navigate } = useRouter();

  // Quick-scan performance indicators
  const getPerformanceLevel = (accuracy: number) => {
    if (accuracy >= 95) return { label: 'Excellent', color: 'text-[var(--color-fg-primary)]' };
    if (accuracy >= 90) return { label: 'Very Good', color: 'text-[var(--color-fg-primary)]' };
    if (accuracy >= 85) return { label: 'Good', color: 'text-[var(--color-fg-muted)]' };
    return { label: 'Experimental', color: 'text-[var(--color-fg-subtle)]' };
  };

  const performance = getPerformanceLevel(model.accuracy);

  return (
    <Card hover className="group relative overflow-hidden">
      {/* Quick-scan badge */}
      <div className="absolute top-4 right-4">
        <div className={`text-xs font-semibold ${performance.color}`}>
          {performance.label}
        </div>
      </div>

      <div className="flex items-start gap-3 mb-4">
        <div className="w-12 h-12 rounded-xl flex items-center justify-center
                     bg-[var(--color-bg-elevated)] border border-[var(--color-border-default)]">
          <Scale className="w-6 h-6 text-[var(--color-fg-primary)]" />
        </div>
        <div className="flex-1">
          <h3 className="font-semibold text-lg text-slate-900 dark:text-white group-hover:text-[var(--color-fg-primary)] transition-colors">
            {model.name}
          </h3>
          <div className="flex items-center gap-2">
            <Badge variant={model.status}>{model.status}</Badge>
            <span className="text-sm text-slate-500 dark:text-gray-400">v{model.version}</span>
          </div>
        </div>
      </div>

      <p className="text-slate-600 dark:text-gray-300 mb-4 line-clamp-2">{model.description}</p>

      {/* Clean Performance metrics */}
      <div className="grid grid-cols-2 gap-3 mb-4">
        <div className="bg-[var(--color-bg-elevated)] border border-[var(--color-border-default)] rounded-md p-3">
          <div className="text-lg font-semibold text-[var(--color-fg-primary)]">{model.accuracy}%</div>
          <div className="text-xs text-[var(--color-fg-subtle)]">accuracy score</div>
        </div>
        <div className="bg-[var(--color-bg-elevated)] border border-[var(--color-border-default)] rounded-md p-3">
          <div className="text-lg font-semibold text-[var(--color-fg-primary)]">{model.downloads.toLocaleString()}</div>
          <div className="text-xs text-[var(--color-fg-subtle)]">downloads</div>
        </div>
      </div>

      {/* Deployment complexity signal */}
      <div className="flex items-center gap-2 p-2 bg-[var(--color-status-warning)] rounded-md mb-4">
        <AlertCircle className="w-4 h-4 text-white" />
        <span className="text-xs text-white">GPU required for optimal performance</span>
      </div>

      {/* Tags */}
      <div className="flex flex-wrap gap-1 mb-4">
        {model.tags.slice(0, 3).map((tag: string) => (
          <span key={tag} className="px-2 py-1 text-xs bg-[var(--color-bg-elevated)] text-[var(--color-fg-muted)] border border-[var(--color-border-default)] rounded-md">
            {tag}
          </span>
        ))}
      </div>

      {/* Enhanced Actions with evaluation transparency */}
      <div className="space-y-2">
        <div className="flex gap-2">
          <Button size="sm" className="flex-1">View Details</Button>
          <Button variant="ghost" size="sm">
            <Github className="w-4 h-4" />
          </Button>
        </div>
        <a
          href="/docs/evaluation"
          onClick={(e)=>{e.preventDefault(); navigate('/docs/evaluation');}}
          className="text-xs text-[var(--color-fg-primary)] underline underline-offset-[3px] hover:opacity-90 font-medium inline-flex items-center gap-1 transition-opacity"
        >
          <Shield className="w-3 h-3" />
          View evaluation report
        </a>
      </div>
    </Card>
  );
};

// Model Comparison Component
const ModelComparison: React.FC = () => {
  const { navigate } = useRouter();
  const [selectedModels, setSelectedModels] = useState<any[]>([]);
  const [showComparison, setShowComparison] = useState(false);


  const ComparisonTable = () => (
    <div className="fixed inset-0 bg-black/30 backdrop-blur-md z-50 flex items-center justify-center p-4">
      <div className="bg-neural-900 border border-neural-700 rounded-2xl p-8 max-w-4xl w-full max-h-[80vh] overflow-auto">
        <div className="flex items-center justify-between mb-6">
          <h3 className="text-2xl font-bold text-white">Model Comparison</h3>
          <button
            onClick={() => setShowComparison(false)}
            className="text-neural-400 hover:text-white transition-colors"
          >
            <X className="w-6 h-6" />
          </button>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-neural-700">
                <th className="text-left py-4 text-neural-300 font-medium">Specification</th>
                {selectedModels.map(model => (
                  <th key={model.id} className="text-left py-4 text-white font-medium">{model.name}</th>
                ))}
              </tr>
            </thead>
            <tbody className="text-neural-300">
              <tr className="border-b border-neural-800">
                <td className="py-3 font-medium">Primary Metric</td>
                {selectedModels.map(model => (
                  <td key={model.id} className="py-3 text-[var(--color-fg-primary)] font-medium">{model.accuracy}%</td>
                ))}
              </tr>
              <tr className="border-b border-neural-800">
                <td className="py-3 font-medium">License</td>
                {selectedModels.map(model => (
                  <td key={model.id} className="py-3">{model.license}</td>
                ))}
              </tr>
              <tr className="border-b border-neural-800">
                <td className="py-3 font-medium">Version</td>
                {selectedModels.map(model => (
                  <td key={model.id} className="py-3">{model.version}</td>
                ))}
              </tr>
              <tr className="border-b border-neural-800">
                <td className="py-3 font-medium">Downloads</td>
                {selectedModels.map(model => (
                  <td key={model.id} className="py-3">{model.downloads.toLocaleString()}</td>
                ))}
              </tr>
              <tr className="border-b border-neural-800">
                <td className="py-3 font-medium">Last Updated</td>
                {selectedModels.map(model => (
                  <td key={model.id} className="py-3">{model.lastUpdated}</td>
                ))}
              </tr>
              <tr>
                <td className="py-3 font-medium">Dataset Provenance</td>
                {selectedModels.map(model => (
                  <td key={model.id} className="py-3">
                    <a href="/docs/provenance"
                       onClick={(e)=>{e.preventDefault(); navigate('/docs/provenance');}}
                       className="underline underline-offset-[3px] text-[var(--color-fg-primary)] hover:opacity-90 transition-opacity">
                      View source
                    </a>
                  </td>
                ))}
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );

  return (
    <>
      <div className="space-y-8">
        {/* Selection controls */}
        {selectedModels.length > 0 && (
          <div className="flex items-center justify-between p-4 bg-neural-800/30 border border-neural-700/50 rounded-xl">
            <div className="flex items-center gap-4">
              <span className="text-neural-300 text-sm">
                {selectedModels.length} model{selectedModels.length > 1 ? 's' : ''} selected
              </span>
              <div className="flex gap-2">
                {selectedModels.map(model => (
                  <span key={model.id} className="px-2 py-1 rounded text-xs
                    bg-[var(--color-bg-elevated)] text-[var(--color-fg-primary)] border border-[var(--color-border-default)]">
                    {model.name}
                  </span>
                ))}
              </div>
            </div>
            <div className="flex gap-3">
              <button
                onClick={() => setSelectedModels([])}
                className="text-neural-400 hover:text-white transition-colors text-sm"
              >
                Clear
              </button>
              {selectedModels.length >= 2 && (
                <Button size="sm" onClick={() => setShowComparison(true)} className="px-4">
                  Compare
                </Button>
              )}
            </div>
          </div>
        )}

        {/* Featured Models Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {mockModels.map(model => (
            <ModelCard key={model.id} model={model} />
          ))}
        </div>
      </div>

      {showComparison && <ComparisonTable />}
    </>
  );
};

// Navigation Component
const Navigation = () => {
  const { currentPath, navigate } = useRouter();
  const { theme, toggleTheme } = useTheme();
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  const navItems = [
    { path: '/', label: 'Home' },
    { path: '/overview', label: 'Overview' },
    { path: '/models', label: 'Models' },
    { path: '/docs', label: 'Docs' },
    { path: '/articles', label: 'Articles' },
    { path: '/ecosystem', label: 'Ecosystem' },
    { path: '/about', label: 'About' }
  ];

  return (
    <nav className="fixed top-0 left-0 right-0 z-50 bg-white/80 dark:bg-neutral-950/70 backdrop-blur-xl border-b border-slate-200/50 dark:border-white/10">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          {/* Enhanced logo with better presence */}
          <div className="flex items-center">
            <button onClick={() => navigate('/')} className="flex items-center gap-3 group">
              <div className="w-10 h-10 bg-[var(--color-accent-cta)] rounded-lg flex items-center justify-center">
                <div className="w-6 h-6 bg-[var(--color-fg-inverse)] rounded-sm"></div>
              </div>
              <div>
                <span className="font-semibold text-xl text-[var(--color-fg-primary)]">Lemkin AI</span>
                <div className="text-xs text-[var(--color-fg-muted)] font-medium">Evidence-Grade AI</div>
              </div>
            </button>

            {/* Enhanced navigation items with better visual hierarchy */}
            <div className="hidden md:flex items-center ml-10 space-x-1">
              {navItems.map(item => (
                <button
                  key={item.path}
                  onClick={() => navigate(item.path)}
                  className={[
                    "relative px-4 py-2 text-sm font-medium rounded-lg transition-colors duration-200",
                    currentPath === item.path
                      ? "text-[var(--color-fg-primary)] underline underline-offset-[6px]"
                      : "text-[var(--color-fg-muted)] hover:text-[var(--color-fg-primary)]"
                  ].join(" ")}
                >
                  {item.label}
                </button>
              ))}
            </div>
          </div>

          <div className="flex items-center gap-4">
            <button
              onClick={toggleTheme}
              className="p-2 text-slate-600 dark:text-neural-400 hover:text-slate-900 dark:hover:text-white transition-colors rounded-lg hover:bg-slate-100 dark:hover:bg-white/10"
              aria-label="Toggle theme"
            >
              {theme === 'light' ? <Moon className="w-5 h-5" /> : <Sun className="w-5 h-5" />}
            </button>

            <a
              href="https://github.com/lemkin-ai"
              className="hidden md:flex items-center gap-2 px-4 py-2 bg-[var(--color-accent-cta)] text-[var(--color-fg-inverse)] rounded-md hover:opacity-90 transition-opacity duration-200"
            >
              <Github className="w-4 h-4" />
              <span className="text-sm font-medium">GitHub</span>
            </a>

            <button
              onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
              className="md:hidden p-2 text-slate-600 dark:text-neural-400 hover:text-slate-900 dark:hover:text-white rounded-lg hover:bg-slate-100 dark:hover:bg-white/10 transition-colors"
            >
              {mobileMenuOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
            </button>
          </div>
        </div>
      </div>

      {/* Enhanced mobile menu */}
      {mobileMenuOpen && (
        <div className="md:hidden bg-[var(--color-bg-default)] border-t border-[var(--color-border-default)]">
          <div className="px-4 py-4 space-y-2">
            {navItems.map(item => (
              <button
                key={item.path}
                onClick={() => {
                  navigate(item.path);
                  setMobileMenuOpen(false);
                }}
                className={[
                  "block w-full text-left px-4 py-3 rounded-md text-sm font-medium transition-colors duration-200",
                  currentPath === item.path
                    ? "bg-[var(--color-accent-cta)] text-[var(--color-fg-inverse)]"
                    : "text-[var(--color-fg-muted)] hover:text-[var(--color-fg-primary)] hover:bg-[var(--color-bg-elevated)]"
                ].join(" ")}
              >
                {item.label}
              </button>
            ))}
            <a
              href="https://github.com/lemkin-ai"
              className="flex items-center gap-2 px-4 py-3 text-sm font-medium text-[var(--color-fg-muted)] hover:text-[var(--color-fg-primary)] hover:bg-[var(--color-bg-elevated)] rounded-md transition-colors duration-200"
            >
              <Github className="w-4 h-4" />
              GitHub
            </a>
          </div>
        </div>
      )}
    </nav>
  );
};

// Enhanced Footer with Trust Center from homepage_improvements.md
const Footer = () => {
  const { navigate } = useRouter();

  return (
    <footer className="bg-[var(--color-bg-default)] dark:bg-[var(--color-bg-surface)]">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-16">
        {/* Trust center highlight */}
        <div className="text-center mb-12 pb-8 border-b border-slate-700">
          <div className="inline-flex items-center gap-2 px-4 py-2 bg-slate-800 rounded-full mb-4">
            <Shield className="w-4 h-4 text-green-400" />
            <span className="text-sm font-medium text-green-400">Trust & Transparency Center</span>
          </div>
          <p className="text-slate-300 max-w-2xl mx-auto">
            Comprehensive documentation of our security practices, evaluation methodologies,
            and ethical guidelines for responsible AI development.
          </p>
        </div>

        {/* Enhanced grid with better visual hierarchy */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-8 mb-8">
          {/* Transparency */}
          <div>
            <h3 className="text-sm font-semibold text-white mb-4 flex items-center gap-2">
              <Eye className="w-4 h-4 text-white" />
              Transparency
            </h3>
            <ul className="space-y-3 text-sm">
              <li><button onClick={() => navigate('/docs/changelog')} className="text-slate-400 hover:text-slate-200 transition-colors text-left block">Changelog</button></li>
              <li><button onClick={() => navigate('/docs/evaluation')} className="text-slate-400 hover:text-slate-200 transition-colors text-left block">Eval Methodology</button></li>
              <li><button onClick={() => navigate('/docs/provenance')} className="text-slate-400 hover:text-slate-200 transition-colors text-left block">Data Provenance</button></li>
              <li><button onClick={() => navigate('/docs/audits')} className="text-slate-400 hover:text-slate-200 transition-colors text-left block">Audit Reports</button></li>
              <li><button onClick={() => navigate('/docs/performance')} className="text-slate-400 hover:text-slate-200 transition-colors text-left block">Performance Metrics</button></li>
            </ul>
          </div>

          {/* Security */}
          <div>
            <h3 className="text-sm font-semibold text-white mb-4 flex items-center gap-2">
              <Shield className="w-4 h-4 text-white" />
              Security
            </h3>
            <ul className="space-y-3 text-sm">
              <li><button onClick={() => navigate('/legal/responsible-use')} className="text-slate-400 hover:text-slate-200 transition-colors text-left block">Responsible Use</button></li>
              <li><button onClick={() => navigate('/security/disclosure')} className="text-slate-400 hover:text-slate-200 transition-colors text-left block">Disclosure Policy</button></li>
              <li><button onClick={() => navigate('/security/sbom')} className="text-slate-400 hover:text-slate-200 transition-colors text-left block">SBOM</button></li>
              <li><button onClick={() => navigate('/security/compliance')} className="text-slate-400 hover:text-slate-200 transition-colors text-left block">Compliance</button></li>
              <li><button onClick={() => navigate('/security/incident')} className="text-slate-400 hover:text-slate-200 transition-colors text-left block">Incident Response</button></li>
            </ul>
          </div>

          {/* Legal */}
          <div>
            <h3 className="text-sm font-semibold text-white mb-4 flex items-center gap-2">
              <Gavel className="w-4 h-4 text-white" />
              Legal
            </h3>
            <ul className="space-y-3 text-sm">
              <li><button onClick={() => navigate('/legal/licensing')} className="text-slate-400 hover:text-slate-200 transition-colors text-left block">Licenses</button></li>
              <li><button onClick={() => navigate('/legal/privacy')} className="text-slate-400 hover:text-slate-200 transition-colors text-left block">Privacy Policy</button></li>
              <li><button onClick={() => navigate('/legal/terms')} className="text-slate-400 hover:text-slate-200 transition-colors text-left block">Terms of Use</button></li>
              <li><button onClick={() => navigate('/legal/copyright')} className="text-slate-400 hover:text-slate-200 transition-colors text-left block">Copyright</button></li>
              <li><button onClick={() => navigate('/legal/dmca')} className="text-slate-400 hover:text-slate-200 transition-colors text-left block">DMCA Policy</button></li>
            </ul>
          </div>

          {/* Community */}
          <div>
            <h3 className="text-sm font-semibold text-white mb-4 flex items-center gap-2">
              <Users className="w-4 h-4 text-white" />
              Community
            </h3>
            <ul className="space-y-3 text-sm">
              <li><button onClick={() => navigate('/contribute')} className="text-slate-400 hover:text-slate-200 transition-colors text-left block">Contribute</button></li>
              <li><button onClick={() => navigate('/governance')} className="text-slate-400 hover:text-slate-200 transition-colors text-left block">Governance</button></li>
              <li><a href="https://github.com/lemkin-ai" className="text-slate-400 hover:text-slate-200 transition-colors inline-flex items-center gap-1">
                GitHub <ExternalLink className="w-3 h-3" />
              </a></li>
              <li><a href="https://discord.gg/lemkin-ai" className="text-slate-400 hover:text-slate-200 transition-colors block">Discord</a></li>
              <li><button onClick={() => navigate('/code-of-conduct')} className="text-slate-400 hover:text-slate-200 transition-colors text-left block">Code of Conduct</button></li>
            </ul>
          </div>
        </div>

        {/* Bottom Section */}
        <div className="border-t border-slate-700 pt-8">
          <div className="flex flex-col lg:flex-row justify-between items-start lg:items-center gap-6">
            {/* Brand */}
            <div className="flex items-center gap-3">
              <LemkinLogo className="w-8 h-8" />
              <div>
                <span className="font-semibold text-lg text-white">Lemkin AI</span>
                <p className="text-sm text-slate-400 mt-1">Evidence-grade AI for international justice</p>
              </div>
            </div>

            {/* Social Links */}
            <div className="flex items-center gap-4">
              <a href="https://github.com/lemkin-ai" className="text-slate-400 hover:text-slate-200 transition-colors">
                <Github className="w-5 h-5" />
              </a>
              <a href="https://twitter.com/lemkin-ai" className="text-slate-400 hover:text-slate-200 transition-colors">
                <Twitter className="w-5 h-5" />
              </a>
              <a href="mailto:contact@lemkin.ai" className="text-slate-400 hover:text-slate-200 transition-colors">
                <Mail className="w-5 h-5" />
              </a>
            </div>

            {/* Copyright */}
            <div className="text-sm text-slate-400">
              &copy; 2025 Lemkin AI. Open source licensed.
            </div>
          </div>
        </div>
      </div>
    </footer>
  );
};

// Page Components
const HomePage = () => {
  const { navigate } = useRouter();
  const [activeBriefTab, setActiveBriefTab] = useState('Investigators');

  const getFilteredBriefs = () => {
    return mockPractitionerBriefs.filter(brief =>
      brief.roles.includes(activeBriefTab)
    );
  };

  return (
    <div className="relative min-h-screen bg-[var(--color-bg-default)]">
      {/* Enhanced dot grid pattern */}
      <div className="absolute inset-0 bg-[radial-gradient(circle_at_1px_1px,var(--color-border-default)_1px,transparent_0)] [background-size:32px_32px] animate-pan-grid" />

      {/* Clean Hero Section */}
      <section className="relative pt-24 pb-16 px-4 sm:px-6 lg:px-8 bg-[var(--color-bg-default)]">

        <div className="relative max-w-7xl mx-auto">
          {/* Clean Trust Indicator */}
          <div className="flex justify-center mb-8">
            <div className="inline-flex items-center gap-2 px-4 py-2 bg-[var(--color-bg-elevated)] border border-[var(--color-border-default)] rounded-md">
              <CheckCircle className="w-4 h-4 text-[var(--color-fg-muted)]" />
              <span className="text-sm text-[var(--color-fg-primary)]">Verified by 15+ international organizations</span>
            </div>
          </div>

          {/* Clean typography */}
          <div className="text-center mb-12">
            <h1 className="text-heading-xl md:text-5xl font-semibold text-[var(--color-fg-primary)] mb-6 max-w-4xl mx-auto leading-tight">
              Evidence-Grade AI for International Justice
            </h1>

            <p className="text-body-lg text-[var(--color-fg-muted)] max-w-3xl mx-auto mb-8 leading-relaxed">
              Open-source machine learning models and tools designed for war crimes investigation,
              human rights documentation, and international criminal proceedings.
            </p>

            {/* Clean CTA buttons */}
            <div className="flex flex-col sm:flex-row gap-4 justify-center items-center">
              <Button
                size="lg"
                className="min-w-[200px]"
                icon={<ArrowRight className="w-5 h-5" />}
                onClick={() => navigate('/models')}
              >
                Explore Models
              </Button>
              <Button
                variant="secondary"
                size="lg"
                onClick={() => navigate('/docs')}
              >
                View Documentation
              </Button>
            </div>
          </div>

          {/* Clean metrics */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-8 max-w-3xl mx-auto">
            <div className="text-center">
              <div className="text-heading-lg font-semibold text-[var(--color-fg-primary)]">12</div>
              <div className="text-sm font-medium text-[var(--color-fg-muted)]">Active Models</div>
            </div>
            <div className="text-center">
              <div className="text-heading-lg font-semibold text-[var(--color-fg-primary)]">46K+</div>
              <div className="text-sm font-medium text-[var(--color-fg-muted)]">Downloads</div>
            </div>
            <div className="text-center">
              <div className="text-heading-lg font-semibold text-[var(--color-fg-primary)]">8</div>
              <div className="text-sm font-medium text-[var(--color-fg-muted)]">Languages</div>
            </div>
            <div className="text-center">
              <div className="text-heading-lg font-semibold text-[var(--color-fg-primary)]">15+</div>
              <div className="text-sm font-medium text-[var(--color-fg-muted)]">Organizations</div>
            </div>
          </div>
        </div>
      </section>

      {/* Evidence-Grade Trust Slice */}
      <section className="py-12 px-4 sm:px-6 lg:px-8 bg-neural-900/30 border-y border-neural-800/50">
        <div className="max-w-6xl mx-auto">
          <div className="grid grid-cols-1 md:grid-cols-4 gap-8 text-center">
            <div className="space-y-2">
              <h3 className="text-sm font-semibold text-neural-200 uppercase tracking-wide">Who Reviews</h3>
              <p className="text-neural-400 text-sm leading-relaxed">
                Tribunals, NGOs, Universities
              </p>
              <a href="/reviewers" className="text-xs text-[var(--color-fg-primary)] underline underline-offset-[3px] hover:opacity-90">View details</a>
            </div>
            <div className="space-y-2">
              <h3 className="text-sm font-semibold text-neural-200 uppercase tracking-wide">How We Evaluate</h3>
              <p className="text-neural-400 text-sm leading-relaxed">
                Bias testing, Legal accuracy, Chain of custody
              </p>
              <a href="/methodology" className="text-xs text-[var(--color-fg-primary)] underline underline-offset-[3px] hover:opacity-90">View methodology</a>
            </div>
            <div className="space-y-2">
              <h3 className="text-sm font-semibold text-neural-200 uppercase tracking-wide">Update Cadence</h3>
              <p className="text-neural-400 text-sm leading-relaxed">
                Monthly security, Quarterly evaluation
              </p>
              <a href="/changelog" className="text-xs text-[var(--color-fg-primary)] underline underline-offset-[3px] hover:opacity-90">View changelog</a>
            </div>
            <div className="space-y-2">
              <h3 className="text-sm font-semibold text-neural-200 uppercase tracking-wide">Misuse Reporting</h3>
              <p className="text-neural-400 text-sm leading-relaxed">
                24h response, Public disclosure
              </p>
              <a href="/report" className="text-xs text-[var(--color-fg-primary)] underline underline-offset-[3px] hover:opacity-90">Report issue</a>
            </div>
          </div>
        </div>
      </section>

      {/* Trust & Credibility Section */}
      <section className="relative py-24 px-4 sm:px-6 lg:px-8 bg-gradient-to-b from-neural-950 to-neural-900">
        <div className="absolute inset-0 bg-neural-net opacity-20"></div>
        <div className="relative max-w-7xl mx-auto">
          <div className="text-center mb-16">
            <div className="inline-flex items-center gap-2 px-4 py-2 backdrop-blur-xl bg-white/5 border border-white/10 rounded-full mb-8">
              <div className="w-2 h-2 bg-white rounded-full animate-pulse"></div>
              <span className="text-neural-300 text-sm">Developed with practitioners from international tribunals and NGOs</span>
            </div>

            <div className="flex justify-center items-center gap-8 mb-12 flex-wrap">
              <div className="flex items-center gap-3 px-6 py-3 backdrop-blur-xl bg-white/5 border border-white/10 rounded-2xl group hover:bg-white/10 transition-all duration-300">
                <CheckCircle className="w-5 h-5 text-white group-hover:text-white transition-colors" />
                <span className="text-white font-medium">Rigorously Validated</span>
              </div>
              <div className="flex items-center gap-3 px-6 py-3 backdrop-blur-xl bg-white/5 border border-white/10 rounded-2xl group hover:bg-white/10 transition-all duration-300">
                <Scale className="w-5 h-5 text-white group-hover:text-white transition-colors" />
                <span className="text-white font-medium">Legally Aware</span>
              </div>
              <div className="flex items-center gap-3 px-6 py-3 backdrop-blur-xl bg-white/5 border border-white/10 rounded-2xl group hover:bg-white/10 transition-all duration-300">
                <Users className="w-5 h-5 text-white group-hover:text-white transition-colors" />
                <span className="text-white font-medium">Community-Driven</span>
              </div>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            <div className="group relative backdrop-blur-xl bg-white/5 border border-white/10 rounded-3xl p-8 shadow-neural hover:shadow-glow transition-all duration-500 hover:scale-105">
              <div className="absolute inset-0 bg-gradient-to-br from-accent-emerald/10 to-transparent rounded-3xl opacity-0 group-hover:opacity-100 transition-opacity duration-500"></div>
              <div className="relative text-center">
                <div className="inline-flex items-center justify-center w-16 h-16 rounded-2xl mb-6
                                 bg-[var(--color-bg-elevated)] border border-[var(--color-border-default)]">
                  <Shield className="w-8 h-8 text-white" />
                </div>
                <h3 className="font-display text-xl font-bold text-white mb-4">
                  Vetted & Validated
                </h3>
                <p className="text-neural-300 leading-relaxed mb-6">
                  All models undergo rigorous testing for accuracy, bias, and reliability in legal contexts with transparent evaluation metrics.
                </p>
                <button
                  onClick={() => navigate('/docs/evaluation')}
                  className="inline-flex items-center gap-2 text-white/90 hover:text-white font-medium underline underline-offset-[3px]"
                >
                  View evaluation process
                  <ArrowRight className="w-4 h-4 group-hover:translate-x-1 transition-transform" />
                </button>
              </div>
            </div>

            <div className="group relative backdrop-blur-xl bg-white/5 border border-white/10 rounded-3xl p-8 shadow-neural hover:shadow-glow transition-all duration-500 hover:scale-105">
              <div className="absolute inset-0 bg-gradient-to-br from-accent-cyan/10 to-transparent rounded-3xl opacity-0 group-hover:opacity-100 transition-opacity duration-500"></div>
              <div className="relative text-center">
                <div className="inline-flex items-center justify-center w-16 h-16 rounded-2xl mb-6
                                 bg-[var(--color-bg-elevated)] border border-[var(--color-border-default)]">
                  <Gavel className="w-8 h-8 text-white" />
                </div>
                <h3 className="font-display text-xl font-bold text-white mb-4">
                  Legally Aware
                </h3>
                <p className="text-neural-300 leading-relaxed mb-6">
                  Built with deep understanding of legal standards, evidence requirements, and chain of custody protocols.
                </p>
                <button
                  onClick={() => navigate('/governance')}
                  className="inline-flex items-center gap-2 text-white/90 hover:text-white font-medium underline underline-offset-[3px]"
                >
                  See governance
                  <ArrowRight className="w-4 h-4 group-hover:translate-x-1 transition-transform" />
                </button>
              </div>
            </div>

            <div className="group relative backdrop-blur-xl bg-white/5 border border-white/10 rounded-3xl p-8 shadow-neural hover:shadow-glow transition-all duration-500 hover:scale-105">
              <div className="absolute inset-0 bg-gradient-to-br from-accent-purple/10 to-transparent rounded-3xl opacity-0 group-hover:opacity-100 transition-opacity duration-500"></div>
              <div className="relative text-center">
                <div className="inline-flex items-center justify-center w-16 h-16 rounded-2xl mb-6
                                 bg-[var(--color-bg-elevated)] border border-[var(--color-border-default)]">
                  <Eye className="w-8 h-8 text-white" />
                </div>
                <h3 className="font-display text-xl font-bold text-white mb-4">
                  Community-Driven
                </h3>
                <p className="text-neural-300 leading-relaxed mb-6">
                  Open development with full transparency, peer review, and collaborative governance from the global community.
                </p>
                <button
                  onClick={() => navigate('/contribute')}
                  className="inline-flex items-center gap-2 text-white/90 hover:text-white font-medium underline underline-offset-[3px]"
                >
                  Join community
                  <ArrowRight className="w-4 h-4 group-hover:translate-x-1 transition-transform" />
                </button>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Featured Models */}
      <section className="py-16 px-4 sm:px-6 lg:px-8 bg-[var(--color-bg-surface)]">
        <div className="max-w-7xl mx-auto">
          <div className="flex items-center justify-between mb-12">
            <div>
              <h2 className="text-heading-lg font-semibold text-[var(--color-fg-primary)] mb-2">Featured Models</h2>
              <p className="text-[var(--color-fg-muted)]">Production-ready AI models with full evaluation transparency</p>
            </div>
            <button
              onClick={() => navigate('/models')}
              className="text-[var(--color-fg-primary)] underline underline-offset-[3px] hover:opacity-90 font-medium inline-flex items-center gap-2 transition-opacity"
            >
              View All Models
              <ArrowRight className="w-4 h-4" />
            </button>
          </div>

          <ModelComparison />
        </div>
      </section>

      {/* Enhanced Practitioner Briefs Section */}
      <section className="py-20 px-4 sm:px-6 lg:px-8 bg-gradient-to-br from-slate-50 via-white to-slate-50 dark:bg-gray-800/30">
        <div className="max-w-7xl mx-auto">
          {/* Enhanced Header */}
          <div className="text-center mb-16">
            <div className="inline-flex items-center gap-2 px-4 py-2 bg-white/80 dark:bg-gray-800/80 backdrop-blur-sm border border-slate-200/60 dark:border-gray-700/50 rounded-full shadow-sm mb-6">
              <CheckCircle className="w-4 h-4 text-emerald-500" />
              <span className="text-sm font-medium text-slate-700 dark:text-gray-300">Expert-Reviewed Content</span>
            </div>

            <h2 className="text-4xl md:text-5xl font-bold text-slate-900 dark:text-white mb-4">Practitioner Briefs</h2>
            <p className="text-xl text-slate-600 dark:text-gray-400 max-w-3xl mx-auto leading-relaxed">Expert insights and methodologies for international justice professionals</p>
          </div>

          {/* Enhanced Role-Based Tabs */}
          <div className="flex justify-center mb-12">
            <div className="inline-flex items-center p-1.5 bg-white/90 dark:bg-gray-800/90 backdrop-blur-xl border border-slate-200/60 dark:border-gray-700/50 rounded-2xl shadow-lg">
              {['Investigators', 'Prosecutors', 'Researchers'].map((role) => {
                const roleIcons = {
                  Investigators: <Search className="w-4 h-4" />,
                  Prosecutors: <Scale className="w-4 h-4" />,
                  Researchers: <Users className="w-4 h-4" />
                };
                return (
                  <button
                    key={role}
                    onClick={() => setActiveBriefTab(role)}
                    className={`
                      relative px-6 py-3 text-sm font-semibold rounded-xl transition-all duration-300
                      inline-flex items-center gap-2
                      ${activeBriefTab === role
                        ? 'bg-[var(--color-accent-cta)] text-[var(--color-fg-inverse)] shadow-lg'
                        : 'text-[var(--color-fg-muted)] hover:text-[var(--color-fg-primary)]'
                      }
                    `}
                  >
                    {roleIcons[role as keyof typeof roleIcons]}
                    Briefs for {role}
                    {activeBriefTab === role && (
                      <div className="absolute inset-0 bg-[var(--color-accent-cta)] rounded-xl shadow-lg -z-10"></div>
                    )}
                  </button>
                );
              })}
            </div>
          </div>

          {/* Enhanced Brief Cards */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
            {getFilteredBriefs().map((brief, index) => (
              <div
                key={brief.id}
                style={{ animationDelay: `${index * 75}ms` }}
                className="group bg-white dark:bg-gray-800 border border-slate-200 dark:border-gray-700 rounded-2xl p-6 shadow-lg hover:shadow-xl hover:border-slate-300 dark:hover:border-gray-600 transition-all duration-300 cursor-pointer animate-fade-in-up opacity-0"
              >
                {/* Card Header */}
                <div className="flex items-start justify-between mb-4">
                  <div className="flex items-center gap-3">
                    <div className="px-3 py-1 bg-slate-100 dark:bg-gray-700 text-slate-700 dark:text-gray-300 rounded-lg text-xs font-medium">
                      {brief.category}
                    </div>
                    {brief.peerReviewed && (
                      <div className="flex items-center gap-1 px-2 py-1 bg-emerald-50 dark:bg-emerald-900/30 text-emerald-700 dark:text-emerald-400 rounded-md text-xs font-medium">
                        <CheckCircle className="w-3 h-3" />
                        Peer-Reviewed
                      </div>
                    )}
                  </div>
                  <span className="text-xs text-slate-500 dark:text-gray-400 font-medium">
                    {brief.readTime}
                  </span>
                </div>

                {/* Title */}
                <h3 className="text-xl font-bold text-slate-900 dark:text-white mb-3 leading-tight group-hover:text-[var(--color-fg-primary)] transition-colors">
                  {brief.title}
                </h3>

                {/* Excerpt */}
                <p className="text-slate-600 dark:text-gray-400 leading-relaxed mb-4 line-clamp-3">
                  {brief.excerpt}
                </p>

                {/* Tags */}
                <div className="flex flex-wrap gap-1.5 mb-4">
                  {brief.tags.map(tag => (
                    <span
                      key={tag}
                      className="px-2 py-1 text-xs bg-slate-100 dark:bg-gray-700 text-slate-600 dark:text-gray-400 rounded-md"
                    >
                      {tag}
                    </span>
                  ))}
                </div>

                {/* Footer */}
                <div className="flex items-center justify-between pt-4 border-t border-slate-100 dark:border-gray-700">
                  <div>
                    <div className="font-semibold text-slate-900 dark:text-white text-sm">
                      {brief.author}
                    </div>
                    <div className="text-xs text-slate-500 dark:text-gray-500 mt-0.5">
                      Last reviewed: {brief.lastReviewed}
                    </div>
                  </div>
                  <div className="flex items-center gap-1 text-xs text-slate-500 dark:text-gray-500">
                    <Calendar className="w-3 h-3" />
                    {new Date(brief.date).toLocaleDateString('en-US', { month: 'short', day: 'numeric' })}
                  </div>
                </div>

                {/* Hover Arrow */}
                <div className="flex items-center justify-center mt-4 opacity-0 group-hover:opacity-100 transition-opacity">
                  <ArrowRight className="w-4 h-4 text-[var(--color-fg-primary)] group-hover:translate-x-1 transition-transform" />
                </div>
              </div>
            ))}
          </div>

          {/* Enhanced CTA */}
          <div className="text-center mt-12">
            <Button
              variant="secondary"
              size="lg"
              onClick={() => navigate('/articles')}
              className="inline-flex items-center gap-2"
            >
              View All Practitioner Briefs
              <ArrowRight className="w-5 h-5" />
            </Button>
          </div>
        </div>
      </section>

      {/* Join the Mission */}
      <section className="py-20 px-4 sm:px-6 lg:px-8 bg-gradient-to-r from-slate-900 to-slate-800 dark:from-slate-800 dark:to-slate-900">
        <div className="max-w-6xl mx-auto">
          <div className="text-center mb-12">
            <h2 className="font-serif text-display-lg font-bold text-white mb-6">Join the Mission</h2>
            <p className="text-xl text-white/80 mb-10 leading-relaxed max-w-4xl mx-auto">
              Help build the future of ethical AI for international justice.
              Contribute models, share expertise, or join our governance community.
            </p>
          </div>

          {/* Actionable First-Contribution Tasks */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-12">
            <div className="bg-white/10 backdrop-blur-sm rounded-lg p-6 text-white">
              <div className="flex items-center gap-3 mb-4">
                <div className="w-10 h-10 bg-white/20 rounded-lg flex items-center justify-center">
                  <CheckCircle className="w-5 h-5" />
                </div>
                <div>
                  <h3 className="font-serif text-lg font-semibold">Improve Model Evaluation</h3>
                  <span className="text-white/70 text-sm">10–15 min</span>
                </div>
              </div>
              <p className="text-white/75 text-sm mb-4">Help expand our evaluation datasets with legal domain expertise and bias testing protocols.</p>
              <button className="text-white hover:opacity-85 text-sm font-medium inline-flex items-center gap-1 underline underline-offset-[3px]">
                Get started <ArrowRight className="w-3 h-3" />
              </button>
            </div>

            <div className="bg-white/10 backdrop-blur-sm rounded-lg p-6 text-white">
              <div className="flex items-center gap-3 mb-4">
                <div className="w-10 h-10 bg-white/20 rounded-lg flex items-center justify-center">
                  <FileText className="w-5 h-5" />
                </div>
                <div>
                  <h3 className="font-serif text-lg font-semibold">Write Dataset Cards</h3>
                  <span className="text-white/70 text-sm">20–30 min</span>
                </div>
              </div>
              <p className="text-white/75 text-sm mb-4">Document training data sources, ethical considerations, and usage guidelines for transparency.</p>
              <button className="text-white hover:opacity-85 text-sm font-medium inline-flex items-center gap-1 underline underline-offset-[3px]">
                Get started <ArrowRight className="w-3 h-3" />
              </button>
            </div>

            <div className="bg-white/10 backdrop-blur-sm rounded-lg p-6 text-white">
              <div className="flex items-center gap-3 mb-4">
                <div className="w-10 h-10 bg-white/20 rounded-lg flex items-center justify-center">
                  <Code className="w-5 h-5" />
                </div>
                <div>
                  <h3 className="font-serif text-lg font-semibold">Add Unit Tests</h3>
                  <span className="text-white/70 text-sm">15–25 min</span>
                </div>
              </div>
              <p className="text-white/75 text-sm mb-4">Enhance model reliability with edge case testing and performance validation scripts.</p>
              <button className="text-white hover:opacity-85 text-sm font-medium inline-flex items-center gap-1 underline underline-offset-[3px]">
                Get started <ArrowRight className="w-3 h-3" />
              </button>
            </div>
          </div>

          <div className="text-center">
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <Button
                onClick={() => navigate('/contribute')}
                className="bg-white text-[var(--color-fg-primary)] hover:bg-slate-50 shadow-lift"
              >
                <Users className="w-5 h-5" />
                Start Contributing
              </Button>
              <Button
                variant="secondary"
                onClick={() => navigate('/governance')}
                className="bg-[var(--color-accent-cta)] hover:bg-slate-700 text-[var(--color-fg-inverse)] border-slate-500"
              >
                Learn About Governance
              </Button>
            </div>
          </div>
        </div>
      </section>
    </div>
  );
};

const ModelsPage = () => {
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedTags, setSelectedTags] = useState<string[]>([]);
  const [selectedStatus, setSelectedStatus] = useState('all');

  const allTags = [...new Set(mockModels.flatMap(m => m.tags))];
  
  const filteredModels = mockModels.filter(model => {
    const matchesSearch = model.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
                          model.description.toLowerCase().includes(searchQuery.toLowerCase());
    const matchesTags = selectedTags.length === 0 || selectedTags.some(tag => model.tags.includes(tag));
    const matchesStatus = selectedStatus === 'all' || model.status === selectedStatus;
    return matchesSearch && matchesTags && matchesStatus;
  });

  return (
    <div className="min-h-screen py-12 px-4 sm:px-6 lg:px-8">
      <div className="max-w-7xl mx-auto">
        <div className="mb-8">
          <h1 className="text-4xl font-bold text-gray-900 dark:text-white mb-4">Models</h1>
          <p className="text-lg text-gray-600 dark:text-gray-400">
            Open-source models vetted for use in international criminal justice and human rights investigation.
            All models undergo rigorous evaluation for accuracy, bias, and ethical considerations.
          </p>
        </div>

        {/* Filters */}
        <div className="mb-8 space-y-4">
          <div className="flex flex-col sm:flex-row gap-4">
            <div className="flex-1">
              <div className="relative">
                <label htmlFor="model-search" className="sr-only">Search models</label>
                <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-5 h-5" />
                <input
                  id="model-search"
                  type="search"
                  placeholder="Search models…"
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="w-full pl-10 pr-4 py-2 border border-gray-200 dark:border-gray-700 rounded-lg bg-white dark:bg-gray-800 text-gray-900 dark:text-white focus:outline-none focus:ring-2 focus:ring-[var(--color-border-focus)]"
                />
              </div>
            </div>
            
            <select
              value={selectedStatus}
              onChange={(e) => setSelectedStatus(e.target.value)}
              className="px-4 py-2 border border-gray-200 dark:border-gray-700 rounded-lg bg-white dark:bg-gray-800 text-gray-900 dark:text-white focus:outline-none focus:ring-2 focus:ring-[var(--color-border-focus)]"
            >
              <option value="all">All Status</option>
              <option value="stable">Stable</option>
              <option value="beta">Beta</option>
              <option value="deprecated">Deprecated</option>
            </select>
          </div>

          <div className="flex flex-wrap gap-2">
            {allTags.map(tag => (
              <button
                key={tag}
                onClick={() => setSelectedTags(prev => 
                  prev.includes(tag) ? prev.filter(t => t !== tag) : [...prev, tag]
                )}
                className={`px-3 py-1 rounded-full text-sm transition-colors ${
                  selectedTags.includes(tag)
                    ? 'bg-[var(--color-bg-elevated)] text-[var(--color-fg-primary)] border border-[var(--color-border-default)]'
                    : 'bg-gray-100 text-gray-600 dark:bg-gray-800 dark:text-gray-400 hover:bg-gray-200 dark:hover:bg-gray-700'
                }`}
              >
                {tag}
              </button>
            ))}
          </div>
        </div>

        {/* Models Grid */}
        {filteredModels.length > 0 ? (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {filteredModels.map((model, index) => (
              <div
                key={model.id}
                style={{ animationDelay: `${index * 75}ms` }}
                className="animate-fade-in-up opacity-0"
              >
                <ModelCard model={model} />
              </div>
            ))}
          </div>
        ) : (
          <div className="text-center py-12">
            <AlertCircle className="w-12 h-12 text-gray-400 mx-auto mb-4" />
            <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-2">No models found</h3>
            <p className="text-gray-600 dark:text-gray-400">
              Try adjusting your search or filters
            </p>
          </div>
        )}
      </div>
    </div>
  );
};



const ModelDetailPage = () => {
  const [activeTab, setActiveTab] = useState('overview');
  const [copied, setCopied] = useState(false);
  const model = mockModels[0]; // For demo purposes

  const handleCopy = (text: string) => {
    navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const tabs = [
    { id: 'overview', label: 'Overview' },
    { id: 'usage', label: 'Usage' },
    { id: 'evaluation', label: 'Evaluation' },
    { id: 'changelog', label: 'Changelog' },
    { id: 'responsible-ai', label: 'Responsible AI' }
  ];

  return (
    <div className="min-h-screen py-12 px-4 sm:px-6 lg:px-8">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="mb-8">
          <div className="flex items-center gap-2 mb-4">
            <Button variant="ghost" size="sm" onClick={() => window.history.back()}>
              <ArrowLeft className="w-4 h-4" />
              Back
            </Button>
          </div>
          
          <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
            <div>
              <div className="flex items-center gap-3 mb-2">
                <h1 className="text-3xl font-bold text-gray-900 dark:text-white">{model.name}</h1>
                <Badge variant={model.status}>{model.status}</Badge>
              </div>
              <p className="text-lg text-gray-600 dark:text-gray-400">{model.description}</p>
            </div>
            
            <div className="flex gap-3">
              <Button variant="secondary">
                <Github className="w-4 h-4" />
                Repository
              </Button>
              <Button>
                <Download className="w-4 h-4" />
                Download
              </Button>
            </div>
          </div>
        </div>

        {/* Metadata */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 mb-8">
          <Card className="p-4">
            <div className="text-sm text-gray-500 dark:text-gray-400">Version</div>
            <div className="font-semibold text-gray-900 dark:text-white">{model.version}</div>
          </Card>
          <Card className="p-4">
            <div className="text-sm text-gray-500 dark:text-gray-400">License</div>
            <div className="font-semibold text-gray-900 dark:text-white">{model.license}</div>
          </Card>
          <Card className="p-4">
            <div className="text-sm text-gray-500 dark:text-gray-400">Downloads</div>
            <div className="font-semibold text-gray-900 dark:text-white">{model.downloads.toLocaleString()}</div>
          </Card>
          <Card className="p-4">
            <div className="text-sm text-gray-500 dark:text-gray-400">Accuracy</div>
            <div className="font-semibold text-gray-900 dark:text-white">{model.accuracy}%</div>
          </Card>
        </div>

        {/* Tabs */}
        <div className="border-b border-gray-200 dark:border-gray-700 mb-8">
          <div className="flex gap-8 overflow-x-auto">
            {tabs.map(tab => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`pb-4 px-1 text-sm font-medium whitespace-nowrap transition-colors ${
                  activeTab === tab.id
                    ? 'text-[var(--color-fg-primary)] border-b-2 border-[var(--color-accent-cta)]'
                    : 'text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white'
                }`}
              >
                {tab.label}
              </button>
            ))}
          </div>
        </div>

        {/* Tab Content */}
        <div className="prose prose-gray dark:prose-invert max-w-none">
          {activeTab === 'overview' && (
            <div>
              <h2 className="text-2xl font-bold mb-4">Overview</h2>
              <p className="text-gray-600 dark:text-gray-400 mb-6">
                Whisper Legal v2 is a state-of-the-art speech recognition model specifically fine-tuned for legal proceedings 
                and testimony transcription. Built upon OpenAI's Whisper architecture, this model has been enhanced with 
                extensive training on international court proceedings, witness testimonies, and legal terminology across 
                multiple languages.
              </p>
              
              <h3 className="text-xl font-semibold mt-8 mb-4">Key Features</h3>
              <ul className="space-y-2 text-gray-600 dark:text-gray-400">
                <li>• Multi-language support for 15+ languages commonly used in international proceedings</li>
                <li>• Enhanced accuracy for legal terminology and proper nouns</li>
                <li>• Speaker diarization capabilities for multi-party conversations</li>
                <li>• Timestamp alignment for evidence synchronization</li>
                <li>• Privacy-preserving processing with on-premise deployment options</li>
              </ul>

              <h3 className="text-xl font-semibold mt-8 mb-4">Use Cases</h3>
              <ul className="space-y-2 text-gray-600 dark:text-gray-400">
                <li>• Transcribing witness testimonies and victim statements</li>
                <li>• Processing intercepted communications as evidence</li>
                <li>• Creating searchable archives of court proceedings</li>
                <li>• Real-time transcription for remote hearings</li>
              </ul>
            </div>
          )}

          {activeTab === 'usage' && (
            <div>
              <h2 className="text-2xl font-bold mb-4">Usage</h2>
              
              <h3 className="text-xl font-semibold mb-4">Installation</h3>
              <div className="bg-gray-900 dark:bg-gray-950 rounded-lg p-4 mb-6">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-xs text-gray-400">bash</span>
                  <button
                    onClick={() => handleCopy('pip install lemkin-whisper-legal')}
                    className="text-gray-400 hover:text-white transition-colors"
                  >
                    {copied ? <Check className="w-4 h-4" /> : <Copy className="w-4 h-4" />}
                  </button>
                </div>
                <code className="text-sm text-gray-300">pip install lemkin-whisper-legal</code>
              </div>

              <h3 className="text-xl font-semibold mb-4">Quick Start</h3>
              <div className="bg-gray-900 dark:bg-gray-950 rounded-lg p-4 mb-6">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-xs text-gray-400">python</span>
                  <button
                    onClick={() => handleCopy(`from lemkin import WhisperLegal\n\nmodel = WhisperLegal.from_pretrained("whisper-legal-v2")\ntranscription = model.transcribe("testimony.wav")`)}
                    className="text-gray-400 hover:text-white transition-colors"
                  >
                    {copied ? <Check className="w-4 h-4" /> : <Copy className="w-4 h-4" />}
                  </button>
                </div>
                <pre className="text-sm text-gray-300">
{`from lemkin import WhisperLegal

model = WhisperLegal.from_pretrained("whisper-legal-v2")
transcription = model.transcribe("testimony.wav")
print(transcription.text)`}
                </pre>
              </div>

              <h3 className="text-xl font-semibold mb-4">Advanced Configuration</h3>
              <p className="text-gray-600 dark:text-gray-400 mb-4">
                For production deployments, we recommend using the following configuration for optimal performance:
              </p>
              <div className="bg-gray-900 dark:bg-gray-950 rounded-lg p-4">
                <pre className="text-sm text-gray-300">
{`model = WhisperLegal.from_pretrained(
    "whisper-legal-v2",
    device="cuda",
    compute_type="float16",
    enable_diarization=True,
    language_detection=True
)`}
                </pre>
              </div>
            </div>
          )}

          {activeTab === 'evaluation' && (
            <div>
              <h2 className="text-2xl font-bold mb-4">Evaluation</h2>
              
              <h3 className="text-xl font-semibold mb-4">Performance Metrics</h3>
              <div className="overflow-x-auto">
                <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-700">
                  <thead>
                    <tr>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Dataset</th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">WER (%)</th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">CER (%)</th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">F1 Score</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-gray-200 dark:divide-gray-700">
                    <tr>
                      <td className="px-6 py-4 whitespace-nowrap text-sm">ICC Proceedings</td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm">5.3</td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm">1.8</td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm">0.947</td>
                    </tr>
                    <tr>
                      <td className="px-6 py-4 whitespace-nowrap text-sm">ICTY Archive</td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm">6.1</td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm">2.2</td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm">0.938</td>
                    </tr>
                    <tr>
                      <td className="px-6 py-4 whitespace-nowrap text-sm">Multi-language Legal</td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm">7.8</td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm">3.1</td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm">0.921</td>
                    </tr>
                  </tbody>
                </table>
              </div>

              <h3 className="text-xl font-semibold mt-8 mb-4">Bias Evaluation</h3>
              <p className="text-gray-600 dark:text-gray-400">
                The model has been evaluated for bias across different demographics and linguistic groups. 
                Detailed bias cards and fairness metrics are available in the technical documentation.
              </p>
            </div>
          )}

          {activeTab === 'changelog' && (
            <div>
              <h2 className="text-2xl font-bold mb-4">Changelog</h2>
              
              <div className="space-y-6">
                <div className="border-l-4 border-blue-500 pl-4">
                  <div className="flex items-center gap-2 mb-2">
                    <h3 className="font-semibold">v2.1.0</h3>
                    <Badge variant="stable">Current</Badge>
                    <span className="text-sm text-gray-500">January 10, 2025</span>
                  </div>
                  <ul className="text-sm text-gray-600 dark:text-gray-400 space-y-1">
                    <li>• Improved accuracy for non-native English speakers</li>
                    <li>• Added support for 3 additional languages</li>
                    <li>• Performance optimizations reducing inference time by 15%</li>
                    <li>• Fixed edge cases in speaker diarization</li>
                  </ul>
                </div>

                <div className="border-l-4 border-gray-300 pl-4">
                  <div className="flex items-center gap-2 mb-2">
                    <h3 className="font-semibold">v2.0.0</h3>
                    <span className="text-sm text-gray-500">December 1, 2024</span>
                  </div>
                  <ul className="text-sm text-gray-600 dark:text-gray-400 space-y-1">
                    <li>• Major architecture update based on Whisper v3</li>
                    <li>• Complete retraining on expanded legal corpus</li>
                    <li>• Breaking API changes for improved consistency</li>
                  </ul>
                </div>
              </div>
            </div>
          )}

          {activeTab === 'responsible-ai' && (
            <div>
              <h2 className="text-2xl font-bold mb-4">Responsible AI</h2>
              
              <div className="bg-yellow-50 dark:bg-yellow-900/20 border border-yellow-200 dark:border-yellow-800 rounded-lg p-4 mb-6">
                <div className="flex gap-3">
                  <AlertCircle className="w-5 h-5 text-yellow-600 dark:text-yellow-400 flex-shrink-0 mt-0.5" />
                  <div>
                    <h3 className="font-semibold text-yellow-800 dark:text-yellow-300 mb-1">Important Notice</h3>
                    <p className="text-sm text-yellow-700 dark:text-yellow-400">
                      This model is designed to assist, not replace, human judgment in legal proceedings. 
                      All outputs should be reviewed by qualified legal professionals.
                    </p>
                  </div>
                </div>
              </div>

              <h3 className="text-xl font-semibold mb-4">Ethical Considerations</h3>
              <ul className="space-y-2 text-gray-600 dark:text-gray-400">
                <li>• The model has been trained exclusively on publicly available or ethically sourced data</li>
                <li>• Personal information detection and redaction capabilities are built-in</li>
                <li>• Regular audits are conducted to identify and mitigate biases</li>
                <li>• Transparency reports are published quarterly</li>
              </ul>

              <h3 className="text-xl font-semibold mt-8 mb-4">Limitations</h3>
              <ul className="space-y-2 text-gray-600 dark:text-gray-400">
                <li>• Accuracy may decrease for heavily accented speech or poor audio quality</li>
                <li>• Technical legal terminology in rare languages may be transcribed incorrectly</li>
                <li>• Not suitable for real-time translation between languages</li>
              </ul>

              <h3 className="text-xl font-semibold mt-8 mb-4">Recommended Use</h3>
              <p className="text-gray-600 dark:text-gray-400">
                This model should be used as part of a comprehensive evidence processing workflow, 
                with appropriate human oversight and validation. It is particularly suited for 
                initial processing and indexing of large audio archives.
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

const ArticlesPage = () => {
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedTags, setSelectedTags] = useState<string[]>([]);
  
  const allTags = [...new Set(mockArticles.flatMap(a => a.tags))];
  
  const filteredArticles = mockArticles.filter(article => {
    const matchesSearch = article.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
                          article.excerpt.toLowerCase().includes(searchQuery.toLowerCase());
    const matchesTags = selectedTags.length === 0 || selectedTags.some(tag => article.tags.includes(tag));
    return matchesSearch && matchesTags;
  });

  return (
    <div className="min-h-screen py-12 px-4 sm:px-6 lg:px-8">
      <div className="max-w-7xl mx-auto">
        <div className="mb-8">
          <h1 className="text-4xl font-bold text-gray-900 dark:text-white mb-4">Articles</h1>
          <p className="text-lg text-gray-600 dark:text-gray-400">
            Guides, best practices, and insights from the community on using AI for international justice.
          </p>
        </div>

        {/* Search and filters */}
        <div className="mb-8 space-y-4">
          <div className="relative">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-5 h-5" />
            <input
              type="text"
              placeholder="Search articles..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="w-full pl-10 pr-4 py-2 border border-gray-200 dark:border-gray-700 rounded-lg bg-white dark:bg-gray-800 text-gray-900 dark:text-white focus:outline-none focus:ring-2 focus:ring-[var(--color-border-focus)]"
            />
          </div>

          <div className="flex flex-wrap gap-2">
            {allTags.map(tag => (
              <button
                key={tag}
                onClick={() => setSelectedTags(prev => 
                  prev.includes(tag) ? prev.filter(t => t !== tag) : [...prev, tag]
                )}
                className={`px-3 py-1 rounded-full text-sm transition-colors ${
                  selectedTags.includes(tag)
                    ? 'bg-[var(--color-bg-elevated)] text-[var(--color-fg-primary)] border border-[var(--color-border-default)]'
                    : 'bg-gray-100 text-gray-600 dark:bg-gray-800 dark:text-gray-400 hover:bg-gray-200 dark:hover:bg-gray-700'
                }`}
              >
                {tag}
              </button>
            ))}
          </div>
        </div>

        {/* Articles list */}
        <div className="space-y-6">
          {filteredArticles.map(article => (
            <ArticleCard key={article.id} article={article} />
          ))}
        </div>
      </div>
    </div>
  );
};

interface Article {
  id: string;
  title: string;
  excerpt: string;
  author: string;
  date: string;
  readTime: string;
  tags: string[];
}

const ArticleCard: React.FC<{ article: Article }> = ({ article }) => {
  const { navigate } = useRouter();
  
  return (
    <Card hover>
      <article onClick={() => navigate(`/articles/${article.id}`)}>
        <h2 className="text-xl font-semibold text-gray-900 dark:text-white mb-2">
          {article.title}
        </h2>
        <p className="text-gray-600 dark:text-gray-400 mb-4">
          {article.excerpt}
        </p>
        <div className="flex flex-wrap items-center gap-4 text-sm text-gray-500 dark:text-gray-400">
          <span>{article.author}</span>
          <span className="flex items-center gap-1">
            <Calendar className="w-4 h-4" />
            {article.date}
          </span>
          <span className="flex items-center gap-1">
            <Clock className="w-4 h-4" />
            {article.readTime}
          </span>
          <div className="flex gap-2">
            {article.tags.map(tag => (
              <span key={tag} className="text-xs px-2 py-1 bg-gray-100 dark:bg-gray-700 text-gray-600 dark:text-gray-300 rounded">
                {tag}
              </span>
            ))}
          </div>
        </div>
      </article>
    </Card>
  );
};

const ResourcesPage = () => {
  const { navigate } = useRouter();
  
  return (
    <div className="min-h-screen py-12 px-4 sm:px-6 lg:px-8">
      <div className="max-w-7xl mx-auto">
        <div className="mb-8">
          <h1 className="text-4xl font-bold text-gray-900 dark:text-white mb-4">Resources</h1>
          <p className="text-lg text-gray-600 dark:text-gray-400">
            Tools, workflows, and documentation to help you get started with Lemkin AI.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {mockResources.map(resource => (
            <Card key={resource.id} hover>
              <div onClick={() => navigate(resource.link)}>
                <div className="flex items-start gap-4">
                  <div className="p-2 bg-slate-100 dark:bg-slate-800 text-[var(--color-fg-primary)] rounded-lg">
                    {resource.icon === 'book' && <Book className="w-6 h-6" />}
                    {resource.icon === 'code' && <Code className="w-6 h-6" />}
                    {resource.icon === 'check-circle' && <CheckCircle className="w-6 h-6" />}
                  </div>
                  <div className="flex-1">
                    <h3 className="font-semibold text-lg text-gray-900 dark:text-white mb-2">
                      {resource.title}
                    </h3>
                    <p className="text-gray-600 dark:text-gray-400">
                      {resource.description}
                    </p>
                  </div>
                </div>
              </div>
            </Card>
          ))}
        </div>

        <div className="mt-12">
          <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6">Additional Resources</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <Card>
              <h3 className="font-semibold text-lg text-gray-900 dark:text-white mb-2">Training Data</h3>
              <p className="text-gray-600 dark:text-gray-400 mb-4">
                Access curated datasets for training and fine-tuning models for legal applications.
              </p>
              <Button variant="secondary" size="sm">
                Browse Datasets
                <ExternalLink className="w-4 h-4" />
              </Button>
            </Card>
            
            <Card>
              <h3 className="font-semibold text-lg text-gray-900 dark:text-white mb-2">Research Papers</h3>
              <p className="text-gray-600 dark:text-gray-400 mb-4">
                Read the latest research on AI applications in international criminal justice.
              </p>
              <Button variant="secondary" size="sm">
                View Papers
                <FileText className="w-4 h-4" />
              </Button>
            </Card>
          </div>
        </div>
      </div>
    </div>
  );
};

const DocsPage = () => {
  const [activeSection, setActiveSection] = useState('getting-started');
  // activeSection can be used for highlighting current section
  console.log(activeSection); // Remove this when implementing section navigation
  
  const sections = [
    {
      id: 'getting-started',
      title: 'Getting Started',
      items: ['Introduction', 'Installation', 'Quick Start', 'Configuration']
    },
    {
      id: 'models',
      title: 'Models',
      items: ['Overview', 'Whisper Legal', 'Document Analyzer', 'Testimony Classifier']
    },
    {
      id: 'api',
      title: 'API Reference',
      items: ['Authentication', 'Endpoints', 'Rate Limits', 'Errors']
    },
    {
      id: 'guides',
      title: 'Guides',
      items: ['Best Practices', 'Security', 'Deployment', 'Monitoring']
    }
  ];

  return (
    <div className="min-h-screen py-12 px-4 sm:px-6 lg:px-8">
      <div className="max-w-7xl mx-auto">
        <div className="flex gap-8">
          {/* Sidebar */}
          <aside className="hidden lg:block w-64 flex-shrink-0">
            <nav className="sticky top-24">
              {sections.map(section => (
                <div key={section.id} className="mb-6">
                  <h3 className="font-semibold text-gray-900 dark:text-white mb-2">
                    {section.title}
                  </h3>
                  <ul className="space-y-1">
                    {section.items.map(item => (
                      <li key={item}>
                        <button
                          onClick={() => setActiveSection(section.id)}
                          className="block w-full text-left px-3 py-1.5 text-sm text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white hover:bg-gray-100 dark:hover:bg-gray-800 rounded"
                        >
                          {item}
                        </button>
                      </li>
                    ))}
                  </ul>
                </div>
              ))}
            </nav>
          </aside>

          {/* Content */}
          <main className="flex-1 max-w-4xl">
            <h1 className="text-4xl font-bold text-gray-900 dark:text-white mb-8">Documentation</h1>
            
            <div className="prose prose-gray dark:prose-invert max-w-none">
              <h2 className="text-2xl font-bold mb-4">Getting Started with Lemkin AI</h2>
              
              <p className="text-gray-600 dark:text-gray-400 mb-6">
                Welcome to the Lemkin AI documentation. This guide will help you get started with our 
                open-source models and tools for international criminal justice applications.
              </p>

              <div className="bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-lg p-4 mb-6">
                <div className="flex gap-3">
                  <AlertCircle className="w-5 h-5 text-[var(--color-fg-primary)] flex-shrink-0 mt-0.5" />
                  <div>
                    <h3 className="font-semibold text-blue-800 dark:text-blue-300 mb-1">Note</h3>
                    <p className="text-sm text-blue-700 dark:text-blue-400">
                      All models require explicit acceptance of our ethical use policy before deployment.
                    </p>
                  </div>
                </div>
              </div>

              <h3 className="text-xl font-semibold mb-4">Prerequisites</h3>
              <ul className="space-y-2 text-gray-600 dark:text-gray-400 mb-6">
                <li>• Python 3.8 or higher</li>
                <li>• CUDA-capable GPU (recommended for optimal performance)</li>
                <li>• Minimum 16GB RAM</li>
                <li>• Active internet connection for model downloads</li>
              </ul>

              <h3 className="text-xl font-semibold mb-4">Installation</h3>
              <p className="text-gray-600 dark:text-gray-400 mb-4">
                Install the Lemkin AI SDK using pip:
              </p>
              
              <div className="bg-gray-900 dark:bg-gray-950 rounded-lg p-4 mb-6">
                <code className="text-sm text-gray-300">pip install lemkin-ai</code>
              </div>

              <h3 className="text-xl font-semibold mb-4">First Steps</h3>
              <p className="text-gray-600 dark:text-gray-400 mb-4">
                After installation, you can verify everything is working correctly:
              </p>
              
              <div className="bg-gray-900 dark:bg-gray-950 rounded-lg p-4 mb-6">
                <pre className="text-sm text-gray-300">
{`import lemkin

# Check version
print(lemkin.__version__)

# List available models
models = lemkin.list_models()
for model in models:
    print(f"- {model.name}: {model.description}")`}
                </pre>
              </div>

              <h3 className="text-xl font-semibold mb-4">Next Steps</h3>
              <p className="text-gray-600 dark:text-gray-400">
                Now that you have Lemkin AI installed, explore our model catalog to find the right 
                tools for your use case, or dive into our guides for best practices on deployment 
                and integration.
              </p>
            </div>
          </main>
        </div>
      </div>
    </div>
  );
};

const AboutPage = () => {
  return (
    <div className="min-h-screen py-12 px-4 sm:px-6 lg:px-8">
      <div className="max-w-4xl mx-auto">
        <h1 className="text-4xl font-bold text-gray-900 dark:text-white mb-8">About Lemkin AI</h1>
        
        <div className="prose prose-gray dark:prose-invert max-w-none">
          <p className="text-lg text-gray-600 dark:text-gray-400 mb-6">
            Lemkin AI is an open-source initiative dedicated to developing and maintaining machine learning 
            models and tools specifically designed for international criminal justice, human rights 
            investigation, and legal technology applications.
          </p>

          <h2 className="text-2xl font-bold mt-8 mb-4">Our Mission</h2>
          <p className="text-gray-600 dark:text-gray-400 mb-6">
            We believe that advanced AI capabilities should be accessible to organizations working to 
            document war crimes, investigate human rights violations, and pursue international justice. 
            Our mission is to provide reliable, ethical, and transparent AI tools that enhance the 
            capacity of investigators, prosecutors, and human rights defenders worldwide.
          </p>

          <h2 className="text-2xl font-bold mt-8 mb-4">Named After Raphael Lemkin</h2>
          <p className="text-gray-600 dark:text-gray-400 mb-6">
            Our project is named in honor of Raphael Lemkin, the Polish lawyer who coined the term 
            "genocide" and drafted the initial version of the Genocide Convention. His tireless work 
            to establish international legal frameworks for preventing mass atrocities inspires our 
            commitment to leveraging technology for justice.
          </p>

          <h2 className="text-2xl font-bold mt-8 mb-4">Core Principles</h2>
          <ul className="space-y-3 text-gray-600 dark:text-gray-400 mb-6">
            <li>
              <strong className="text-gray-900 dark:text-white">Transparency:</strong> All our models 
              are open-source with published training data sources and evaluation metrics.
            </li>
            <li>
              <strong className="text-gray-900 dark:text-white">Accountability:</strong> We maintain 
              detailed documentation of model limitations and potential biases.
            </li>
            <li>
              <strong className="text-gray-900 dark:text-white">Privacy:</strong> Our tools are designed 
              with privacy-by-design principles to protect sensitive information.
            </li>
            <li>
              <strong className="text-gray-900 dark:text-white">Accessibility:</strong> We ensure our 
              tools can be deployed in resource-constrained environments.
            </li>
          </ul>

          <h2 className="text-2xl font-bold mt-8 mb-4">Partners & Supporters</h2>
          <p className="text-gray-600 dark:text-gray-400 mb-6">
            Lemkin AI is supported by a coalition of international organizations, academic institutions, 
            and technology partners committed to advancing justice through responsible AI development.
          </p>

          <h2 className="text-2xl font-bold mt-8 mb-4">Get Involved</h2>
          <p className="text-gray-600 dark:text-gray-400 mb-6">
            We welcome contributions from developers, legal professionals, researchers, and human rights 
            practitioners. Whether through code contributions, model evaluation, documentation improvements, 
            or field testing, your expertise can help advance our mission.
          </p>
        </div>
      </div>
    </div>
  );
};

const ContributePage = () => {
  const { navigate } = useRouter();
  
  return (
    <div className="min-h-screen py-12 px-4 sm:px-6 lg:px-8">
      <div className="max-w-4xl mx-auto">
        <h1 className="text-4xl font-bold text-gray-900 dark:text-white mb-8">Contribute</h1>
        
        <div className="prose prose-gray dark:prose-invert max-w-none">
          <p className="text-lg text-gray-600 dark:text-gray-400 mb-6">
            Lemkin AI is a community-driven project. We welcome contributions from developers, 
            researchers, legal professionals, and human rights practitioners worldwide.
          </p>

          <h2 className="text-2xl font-bold mt-8 mb-4">Ways to Contribute</h2>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
            <Card>
              <Code className="w-8 h-8 text-[var(--color-fg-primary)] mb-3" />
              <h3 className="font-semibold text-lg mb-2">Code Contributions</h3>
              <p className="text-gray-600 dark:text-gray-400 text-sm">
                Submit pull requests for bug fixes, new features, or model improvements.
              </p>
            </Card>
            
            <Card>
              <FileText className="w-8 h-8 text-[var(--color-fg-primary)] mb-3" />
              <h3 className="font-semibold text-lg mb-2">Documentation</h3>
              <p className="text-gray-600 dark:text-gray-400 text-sm">
                Help improve our guides, API documentation, and tutorials.
              </p>
            </Card>
            
            <Card>
              <AlertCircle className="w-8 h-8 text-[var(--color-fg-primary)] mb-3" />
              <h3 className="font-semibold text-lg mb-2">Bug Reports</h3>
              <p className="text-gray-600 dark:text-gray-400 text-sm">
                Report issues and help us improve the reliability of our tools.
              </p>
            </Card>
            
            <Card>
              <Users className="w-8 h-8 text-[var(--color-fg-primary)] mb-3" />
              <h3 className="font-semibold text-lg mb-2">Community Support</h3>
              <p className="text-gray-600 dark:text-gray-400 text-sm">
                Help other users in our forums and discussion channels.
              </p>
            </Card>
          </div>

          <h2 className="text-2xl font-bold mt-8 mb-4">Getting Started</h2>
          
          <ol className="space-y-3 text-gray-600 dark:text-gray-400 mb-6">
            <li>1. Fork the repository on GitHub</li>
            <li>2. Create a feature branch for your contribution</li>
            <li>3. Make your changes following our coding standards</li>
            <li>4. Write tests for any new functionality</li>
            <li>5. Submit a pull request with a clear description</li>
          </ol>

          <h2 className="text-2xl font-bold mt-8 mb-4">Code of Conduct</h2>
          <p className="text-gray-600 dark:text-gray-400 mb-6">
            All contributors are expected to adhere to our Code of Conduct. We are committed to 
            providing a welcoming and inclusive environment for everyone, regardless of background, 
            identity, or experience level.
          </p>

          <h2 className="text-2xl font-bold mt-8 mb-4">Recognition</h2>
          <p className="text-gray-600 dark:text-gray-400 mb-6">
            We value all contributions and maintain a contributors list recognizing everyone who 
            helps advance the project. Significant contributors may be invited to join our core 
            maintainers team.
          </p>

          <div className="mt-8 flex gap-4">
            <Button onClick={() => window.open('https://github.com/lemkin-ai')}>
              <Github className="w-5 h-5" />
              View on GitHub
            </Button>
            <Button variant="secondary" onClick={() => navigate('/governance')}>
              Learn About Governance
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
};

const GovernancePage = () => {
  return (
    <div className="min-h-screen py-12 px-4 sm:px-6 lg:px-8">
      <div className="max-w-4xl mx-auto">
        <h1 className="text-4xl font-bold text-gray-900 dark:text-white mb-8">Governance</h1>
        
        <div className="prose prose-gray dark:prose-invert max-w-none">
          <p className="text-lg text-gray-600 dark:text-gray-400 mb-6">
            Lemkin AI operates under a transparent governance model designed to ensure the project 
            remains aligned with its mission while incorporating diverse perspectives from the 
            international justice community.
          </p>

          <h2 className="text-2xl font-bold mt-8 mb-4">Governance Structure</h2>
          
          <h3 className="text-xl font-semibold mt-6 mb-3">Steering Committee</h3>
          <p className="text-gray-600 dark:text-gray-400 mb-6">
            The Steering Committee provides strategic direction and ensures the project adheres to 
            its ethical principles. Members include representatives from international tribunals, 
            human rights organizations, and technical experts.
          </p>

          <h3 className="text-xl font-semibold mt-6 mb-3">Technical Advisory Board</h3>
          <p className="text-gray-600 dark:text-gray-400 mb-6">
            The Technical Advisory Board reviews model architectures, evaluation metrics, and 
            deployment guidelines to ensure technical excellence and responsible AI practices.
          </p>

          <h3 className="text-xl font-semibold mt-6 mb-3">Core Maintainers</h3>
          <p className="text-gray-600 dark:text-gray-400 mb-6">
            Core maintainers are responsible for day-to-day project management, code review, 
            and release coordination. They are selected based on sustained contributions and 
            technical expertise.
          </p>

          <h2 className="text-2xl font-bold mt-8 mb-4">Decision Making Process</h2>
          <ul className="space-y-3 text-gray-600 dark:text-gray-400 mb-6">
            <li>
              <strong className="text-gray-900 dark:text-white">Minor Changes:</strong> Bug fixes 
              and small improvements can be approved by any two core maintainers.
            </li>
            <li>
              <strong className="text-gray-900 dark:text-white">Major Features:</strong> New models 
              or significant features require review by the Technical Advisory Board.
            </li>
            <li>
              <strong className="text-gray-900 dark:text-white">Strategic Decisions:</strong> Changes 
              to project direction or governance require Steering Committee approval.
            </li>
          </ul>

          <h2 className="text-2xl font-bold mt-8 mb-4">Ethical Review Process</h2>
          <p className="text-gray-600 dark:text-gray-400 mb-6">
            All models undergo ethical review before release, evaluating potential misuse, bias, 
            privacy implications, and alignment with international human rights standards.
          </p>

          <h2 className="text-2xl font-bold mt-8 mb-4">Transparency Reports</h2>
          <p className="text-gray-600 dark:text-gray-400 mb-6">
            We publish quarterly transparency reports detailing project activities, funding sources, 
            model deployments, and any ethical concerns raised by the community.
          </p>

          <h2 className="text-2xl font-bold mt-8 mb-4">Community Participation</h2>
          <p className="text-gray-600 dark:text-gray-400">
            Community members can participate in governance through our RFC (Request for Comments) 
            process for proposing changes, monthly community calls, and annual contributor summits.
          </p>
        </div>
      </div>
    </div>
  );
};

const ContactPage = () => {
  return (
    <div className="min-h-screen py-12 px-4 sm:px-6 lg:px-8">
      <div className="max-w-4xl mx-auto">
        <h1 className="text-4xl font-bold text-gray-900 dark:text-white mb-8">Contact</h1>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-8 mb-12">
          <Card>
            <Mail className="w-8 h-8 text-[var(--color-fg-primary)] mb-3" />
            <h3 className="font-semibold text-lg mb-2">General Inquiries</h3>
            <p className="text-gray-600 dark:text-gray-400 mb-4">
              For general questions about the project
            </p>
            <a href="mailto:info@lemkin.ai" className="text-[var(--color-fg-primary)] hover:underline">
              info@lemkin.ai
            </a>
          </Card>
          
          <Card>
            <Github className="w-8 h-8 text-[var(--color-fg-primary)] mb-3" />
            <h3 className="font-semibold text-lg mb-2">Technical Support</h3>
            <p className="text-gray-600 dark:text-gray-400 mb-4">
              For bug reports and technical issues
            </p>
            <a href="https://github.com/lemkin-ai/issues" className="text-[var(--color-fg-primary)] hover:underline">
              GitHub Issues
            </a>
          </Card>
          
          <Card>
            <Users className="w-8 h-8 text-[var(--color-fg-primary)] mb-3" />
            <h3 className="font-semibold text-lg mb-2">Partnerships</h3>
            <p className="text-gray-600 dark:text-gray-400 mb-4">
              For collaboration and partnership inquiries
            </p>
            <a href="mailto:partnerships@lemkin.ai" className="text-[var(--color-fg-primary)] hover:underline">
              partnerships@lemkin.ai
            </a>
          </Card>
          
          <Card>
            <AlertCircle className="w-8 h-8 text-[var(--color-fg-primary)] mb-3" />
            <h3 className="font-semibold text-lg mb-2">Security</h3>
            <p className="text-gray-600 dark:text-gray-400 mb-4">
              For reporting security vulnerabilities
            </p>
            <a href="mailto:security@lemkin.ai" className="text-[var(--color-fg-primary)] hover:underline">
              security@lemkin.ai
            </a>
          </Card>
        </div>

        <div className="prose prose-gray dark:prose-invert max-w-none">
          <h2 className="text-2xl font-bold mb-4">Response Times</h2>
          <p className="text-gray-600 dark:text-gray-400 mb-6">
            We aim to respond to all inquiries within 48 hours during business days. Security 
            issues are prioritized and addressed immediately.
          </p>

          <h2 className="text-2xl font-bold mb-4">Community Channels</h2>
          <p className="text-gray-600 dark:text-gray-400 mb-4">
            Join our community discussions:
          </p>
          <ul className="space-y-2 text-gray-600 dark:text-gray-400">
            <li>• Discord: Community chat and support</li>
            <li>• GitHub Discussions: Technical discussions and RFCs</li>
            <li>• Twitter: @lemkin_ai for updates and announcements</li>
            <li>• Monthly Community Calls: Second Tuesday of each month</li>
          </ul>
        </div>
      </div>
    </div>
  );
};

const LegalPage = () => {
  const [activeTab, setActiveTab] = useState('privacy');
  
  return (
    <div className="min-h-screen py-12 px-4 sm:px-6 lg:px-8">
      <div className="max-w-4xl mx-auto">
        <h1 className="text-4xl font-bold text-gray-900 dark:text-white mb-8">Legal</h1>
        
        <div className="flex gap-4 mb-8 border-b border-gray-200 dark:border-gray-700">
          <button
            onClick={() => setActiveTab('privacy')}
            className={`pb-4 px-1 font-medium transition-colors ${
              activeTab === 'privacy'
                ? 'text-[var(--color-fg-primary)] border-b-2 border-[var(--color-accent-cta)]'
                : 'text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white'
            }`}
          >
            Privacy Policy
          </button>
          <button
            onClick={() => setActiveTab('terms')}
            className={`pb-4 px-1 font-medium transition-colors ${
              activeTab === 'terms'
                ? 'text-[var(--color-fg-primary)] border-b-2 border-[var(--color-accent-cta)]'
                : 'text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white'
            }`}
          >
            Terms of Service
          </button>
        </div>

        <div className="prose prose-gray dark:prose-invert max-w-none">
          {activeTab === 'privacy' && (
            <>
              <h2 className="text-2xl font-bold mb-4">Privacy Policy</h2>
              <p className="text-gray-600 dark:text-gray-400 mb-4">
                Last updated: January 15, 2025
              </p>
              
              <h3 className="text-xl font-semibold mt-6 mb-3">Data Collection</h3>
              <p className="text-gray-600 dark:text-gray-400 mb-4">
                Lemkin AI collects minimal data necessary for providing our services. We do not sell, 
                trade, or otherwise transfer your information to third parties.
              </p>

              <h3 className="text-xl font-semibold mt-6 mb-3">Usage Analytics</h3>
              <p className="text-gray-600 dark:text-gray-400 mb-4">
                We collect anonymous usage statistics to improve our models and services. This data 
                does not contain personally identifiable information.
              </p>

              <h3 className="text-xl font-semibold mt-6 mb-3">Model Training</h3>
              <p className="text-gray-600 dark:text-gray-400 mb-4">
                Our models are trained exclusively on publicly available or ethically sourced data. 
                We do not use user-submitted data for training without explicit consent.
              </p>

              <h3 className="text-xl font-semibold mt-6 mb-3">Data Security</h3>
              <p className="text-gray-600 dark:text-gray-400 mb-4">
                We implement industry-standard security measures to protect your data. All data 
                transmission is encrypted using TLS 1.3 or higher.
              </p>

              <h3 className="text-xl font-semibold mt-6 mb-3">Your Rights</h3>
              <p className="text-gray-600 dark:text-gray-400">
                You have the right to access, correct, or delete your personal information. Contact 
                us at privacy@lemkin.ai to exercise these rights.
              </p>
            </>
          )}

          {activeTab === 'terms' && (
            <>
              <h2 className="text-2xl font-bold mb-4">Terms of Service</h2>
              <p className="text-gray-600 dark:text-gray-400 mb-4">
                Last updated: January 15, 2025
              </p>

              <h3 className="text-xl font-semibold mt-6 mb-3">Acceptance of Terms</h3>
              <p className="text-gray-600 dark:text-gray-400 mb-4">
                By accessing and using Lemkin AI services, you agree to be bound by these Terms of 
                Service and all applicable laws and regulations.
              </p>

              <h3 className="text-xl font-semibold mt-6 mb-3">Use License</h3>
              <p className="text-gray-600 dark:text-gray-400 mb-4">
                Our models and software are provided under open-source licenses specified in each 
                repository. Commercial use requires compliance with respective license terms.
              </p>

              <h3 className="text-xl font-semibold mt-6 mb-3">Ethical Use Policy</h3>
              <p className="text-gray-600 dark:text-gray-400 mb-4">
                Users must comply with our Ethical Use Policy, which prohibits use of our tools for 
                harassment, discrimination, surveillance of protected groups, or any activity that 
                violates human rights.
              </p>

              <h3 className="text-xl font-semibold mt-6 mb-3">Disclaimer</h3>
              <p className="text-gray-600 dark:text-gray-400 mb-4">
                Our services are provided "as is" without warranties of any kind. We are not liable 
                for any damages arising from the use of our services.
              </p>

              <h3 className="text-xl font-semibold mt-6 mb-3">Indemnification</h3>
              <p className="text-gray-600 dark:text-gray-400">
                You agree to indemnify and hold harmless Lemkin AI and its contributors from any 
                claims arising from your use of our services.
              </p>
            </>
          )}
        </div>
      </div>
    </div>
  );
};

const OverviewPage = () => {
  return (
    <div className="min-h-screen py-12 px-4 sm:px-6 lg:px-8">
      <div className="max-w-7xl mx-auto">
        <h1 className="text-4xl font-bold text-gray-900 dark:text-white mb-8">Project Overview</h1>
        
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8 mb-12">
          <Card className="lg:col-span-2">
            <h2 className="text-2xl font-bold mb-4">What is Lemkin AI?</h2>
            <p className="text-gray-600 dark:text-gray-400 mb-4">
              Lemkin AI is a comprehensive open-source initiative providing machine learning models 
              and tools specifically designed for international criminal justice applications. Our 
              platform enables investigators, prosecutors, and human rights organizations to leverage 
              AI technology in their pursuit of justice.
            </p>
            <p className="text-gray-600 dark:text-gray-400">
              From transcribing witness testimonies to analyzing vast archives of evidence, our models 
              are rigorously tested and ethically developed to meet the unique requirements of 
              international legal proceedings.
            </p>
          </Card>
          
          <Card>
            <h3 className="font-semibold text-lg mb-4">Key Statistics</h3>
            <div className="space-y-3">
              <div className="flex justify-between">
                <span className="text-gray-600 dark:text-gray-400">Active Models</span>
                <span className="font-semibold">12</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600 dark:text-gray-400">Contributors</span>
                <span className="font-semibold">247</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600 dark:text-gray-400">Organizations</span>
                <span className="font-semibold">38</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600 dark:text-gray-400">Languages Supported</span>
                <span className="font-semibold">23</span>
              </div>
            </div>
          </Card>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 mb-12">
          <Card>
            <CheckCircle className="w-8 h-8 text-green-600 dark:text-green-400 mb-3" />
            <h3 className="font-semibold text-lg mb-2">Vetted & Validated</h3>
            <p className="text-gray-600 dark:text-gray-400 text-sm">
              All models undergo rigorous evaluation by legal and technical experts before release.
            </p>
          </Card>
          
          <Card>
            <LemkinLogo className="w-8 h-8 mb-3" />
            <h3 className="font-semibold text-lg mb-2">Legally Aware</h3>
            <p className="text-gray-600 dark:text-gray-400 text-sm">
              Designed with understanding of international legal standards and evidentiary requirements.
            </p>
          </Card>
          
          <Card>
            <Users className="w-8 h-8 text-purple-600 dark:text-purple-400 mb-3" />
            <h3 className="font-semibold text-lg mb-2">Community Driven</h3>
            <p className="text-gray-600 dark:text-gray-400 text-sm">
              Developed in collaboration with practitioners from tribunals, NGOs, and research institutions.
            </p>
          </Card>
        </div>

        <div className="prose prose-gray dark:prose-invert max-w-none">
          <h2 className="text-2xl font-bold mb-4">Current Focus Areas</h2>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
            <div>
              <h3 className="text-xl font-semibold mb-3">Evidence Processing</h3>
              <ul className="space-y-2 text-gray-600 dark:text-gray-400">
                <li>• Audio transcription and translation</li>
                <li>• Document analysis and classification</li>
                <li>• Image and video verification</li>
                <li>• Metadata extraction and validation</li>
              </ul>
            </div>
            
            <div>
              <h3 className="text-xl font-semibold mb-3">Investigation Support</h3>
              <ul className="space-y-2 text-gray-600 dark:text-gray-400">
                <li>• Pattern recognition in testimony</li>
                <li>• Entity extraction and linking</li>
                <li>• Timeline reconstruction</li>
                <li>• Cross-reference verification</li>
              </ul>
            </div>
          </div>

          <h2 className="text-2xl font-bold mb-4">Roadmap</h2>
          <p className="text-gray-600 dark:text-gray-400 mb-6">
            Our development roadmap is guided by feedback from field practitioners and evolving 
            needs in international justice:
          </p>
          
          <div className="space-y-4">
            <div className="flex gap-4">
              <div className="w-2 h-2 rounded-full bg-green-500 mt-2"></div>
              <div>
                <h4 className="font-semibold">Q1 2025 - Enhanced Multilingual Support</h4>
                <p className="text-gray-600 dark:text-gray-400 text-sm">
                  Expanding language coverage for underserved regions
                </p>
              </div>
            </div>
            
            <div className="flex gap-4">
              <div className="w-2 h-2 rounded-full bg-blue-500 mt-2"></div>
              <div>
                <h4 className="font-semibold">Q2 2025 - Real-time Processing Pipeline</h4>
                <p className="text-gray-600 dark:text-gray-400 text-sm">
                  Enabling live transcription and analysis capabilities
                </p>
              </div>
            </div>
            
            <div className="flex gap-4">
              <div className="w-2 h-2 rounded-full bg-gray-400 mt-2"></div>
              <div>
                <h4 className="font-semibold">Q3 2025 - Advanced Verification Tools</h4>
                <p className="text-gray-600 dark:text-gray-400 text-sm">
                  Deepfake detection and chain-of-custody validation
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

const NotFoundPage = () => {
  const { navigate } = useRouter();

  return (
    <div className="min-h-screen flex items-center justify-center px-4">
      <div className="text-center">
        <div className="flex justify-center mb-6">
          <LemkinLogo className="w-12 h-12 opacity-50" />
        </div>
        <h1 className="text-9xl font-bold text-gray-200 dark:text-gray-800">404</h1>
        <h2 className="text-2xl font-semibold text-gray-900 dark:text-white mb-4">Page Not Found</h2>
        <p className="text-gray-600 dark:text-gray-400 mb-8">
          The page you're looking for doesn't exist or has been moved.
        </p>
        <Button onClick={() => navigate('/')}>
          Return Home
        </Button>
      </div>
    </div>
  );
};

// Main App Component
const App = () => {
  const { currentPath } = useRouter();

  // Route rendering logic
  const renderPage = () => {
    // Handle dynamic routes for models
    if (currentPath.startsWith('/models/')) {
      return <ModelDetailPage />;
    }

    // Handle dynamic routes for articles
    if (currentPath.startsWith('/articles/') && currentPath !== '/articles') {
      // Article detail page would go here
      return <ArticlesPage />; // Placeholder for now
    }

    // Static routes
    switch (currentPath) {
      case '/':
        return <HomePage />;
      case '/overview':
        return <OverviewPage />;
      case '/models':
        return <ModelsPage />;
      case '/articles':
        return <ArticlesPage />;
      case '/resources':
        return <ResourcesPage />;
      case '/docs':
      case '/docs/quickstart':
      case '/docs/api':
      case '/docs/best-practices':
        return <DocsPage />;
      case '/about':
        return <AboutPage />;
      case '/contribute':
        return <ContributePage />;
      case '/governance':
        return <GovernancePage />;
      case '/contact':
        return <ContactPage />;
      case '/legal':
        return <LegalPage />;
      default:
        return <NotFoundPage />;
    }
  };

  return (
    <div className="min-h-screen bg-white dark:bg-gray-900">
      <Navigation />
      <main className="flex-1">
        {renderPage()}
      </main>
      <Footer />
    </div>
  );
};

// Root Component with Providers
const LemkinAIWebsite = () => {
  return (
    <ThemeProvider>
      <GlobalStyles />
      <Router>
        <App />
      </Router>
    </ThemeProvider>
  );
};

// Global Styles Component with improved design system
const GlobalStyles: React.FC = () => (
  <style>{`
    :root {
      /* Surfaces */
      --color-bg-default: #FFFFFF;
      --color-bg-surface: #F9FAFB;
      --color-bg-elevated: #F3F4F6;

      /* Text - DARKER for better contrast */
      --color-fg-primary: #000000;
      --color-fg-muted: #404040;
      --color-fg-subtle: #666666;
      --color-fg-inverse: #FFFFFF;

      /* Neutral accent */
      --color-accent-cta: #000000;
      --color-border-default: #E5E7EB;
      --color-border-focus: #000000;

      /* Status (semantic only) */
      --color-status-success: #16A34A;
      --color-status-warning: #D97706;
      --color-status-danger: #DC2626;
    }

    .dark {
      --color-bg-default: #0A0A0A;
      --color-bg-surface: #111111;
      --color-bg-elevated: #1A1A1A;

      --color-fg-primary: #FFFFFF;
      --color-fg-muted: #B0B0B0;
      --color-fg-subtle: #808080;
      --color-fg-inverse: #0A0A0A;

      --color-accent-cta: #FFFFFF;
      --color-border-default: #262626;
      --color-border-focus: #FFFFFF;
    }

    /* Ensure proper focus rings */
    :focus-visible {
      outline: none;
      box-shadow: 0 0 0 2px var(--color-border-focus),
                  0 0 0 4px var(--color-bg-surface);
    }

    /* Typography weight fixes */
    h1, h2, h3 {
      font-weight: 600;
    }

    body {
      font-weight: 400;
    }
  `}</style>
);

// Export the main component
export default LemkinAIWebsite;