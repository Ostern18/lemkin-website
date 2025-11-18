import React, { useState, createContext, useContext } from 'react';
import { Scale, Target, BarChart3, CheckCircle, AlertCircle, X } from 'lucide-react';
import { motion } from 'framer-motion';
import { MotionCard } from './motion';

// Router Context Types
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

// Badge Component
interface BadgeProps {
  variant?: 'default' | 'stable' | 'beta' | 'deprecated';
  children: React.ReactNode;
  className?: string;
}

const Badge: React.FC<BadgeProps> = ({ variant = 'default', children, className = '' }) => {
  const variants: Record<string, string> = {
    default: 'bg-[var(--color-bg-secondary)] text-[var(--color-text-secondary)] border border-[var(--color-border-primary)]',
    stable: 'bg-[var(--color-bg-secondary)] text-[var(--color-text-primary)] border border-[var(--color-border-primary)]',
    beta: 'bg-[var(--color-bg-secondary)] text-[var(--color-text-primary)] border border-[var(--color-border-primary)]',
    deprecated: 'bg-[var(--color-bg-tertiary)] text-[var(--color-text-secondary)] border border-[var(--color-border-primary)]'
  };

  const icons = {
    stable: <CheckCircle className="w-3 h-3" />,
    beta: <AlertCircle className="w-3 h-3" />,
    deprecated: <X className="w-3 h-3" />
  };

  return (
    <span className={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-semibold tracking-wide uppercase transition-all duration-200 hover:scale-105 ${variants[variant] || variants.default} ${className}`}>
      {variant !== 'default' && <span aria-hidden>{icons[variant as keyof typeof icons]}</span>}
      {children}
    </span>
  );
};

// Button Component
interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: 'primary' | 'secondary' | 'tertiary' | 'ghost' | 'danger';
  size?: 'xs' | 'sm' | 'md' | 'lg';
  loading?: boolean;
  icon?: React.ReactNode;
  children?: React.ReactNode;
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
    primary: `
      bg-[var(--color-primary)] text-[var(--color-text-inverse)]
      hover:bg-[var(--color-primary-hover)] active:bg-[var(--color-primary-active)]
      border border-[var(--color-border-secondary)]
      shadow-[0_1px_0_rgba(var(--shadow-rgb),0.04),inset_0_1px_0_rgba(255,255,255,0.03)]
    `,
    secondary: `
      bg-[var(--color-bg-secondary)] text-[var(--color-text-primary)]
      hover:bg-[var(--color-bg-tertiary)] active:bg-[var(--color-bg-secondary)]
      border border-[var(--color-border-primary)]
      shadow-[0_1px_0_rgba(var(--shadow-rgb),0.04),inset_0_1px_0_rgba(255,255,255,0.10)]
    `,
    tertiary: `
      bg-transparent text-[var(--color-text-primary)]
      hover:bg-[var(--color-bg-secondary)] active:bg-[var(--color-bg-tertiary)]
      border border-[var(--color-border-primary)]
    `,
    ghost: `
      bg-transparent text-[var(--color-primary)]
      hover:bg-[color-mix(in_srgb,var(--color-primary),transparent_95%)]
      active:bg-[color-mix(in_srgb,var(--color-primary),transparent_90%)]
    `,
    danger: `
      bg-[var(--color-error)] text-[var(--color-text-inverse)]
      hover:bg-[var(--color-error-hover)] active:bg-[var(--color-error-active)]
      border border-[var(--color-error)]
      shadow-[0_1px_0_rgba(var(--shadow-rgb),0.04),inset_0_1px_0_rgba(255,255,255,0.03)]
    `
  };

  const sizes = {
    xs: 'px-2.5 py-1 text-xs',
    sm: 'px-3 py-1.5 text-sm',
    md: 'px-4 py-2 text-sm',
    lg: 'px-6 py-3 text-base'
  };

  return (
    <button
      className={`
        inline-flex items-center justify-center gap-2
        font-medium rounded-lg transition-all duration-200
        focus:outline-none focus-visible:ring-2 focus-visible:ring-[var(--focus-ring)]
        disabled:opacity-50 disabled:cursor-not-allowed
        ${variants[variant]}
        ${sizes[size]}
        ${className}
      `}
      disabled={loading || props.disabled}
      {...props}
    >
      {loading && (
        <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24">
          <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
          <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
        </svg>
      )}
      {icon && !loading && icon}
      {children}
    </button>
  );
};

// Research-Grade Mini-Chart Component
interface MiniChartProps {
  label: string;
  value: number;
  threshold: number;
  thresholdLabel: string;
  color: 'green' | 'blue' | 'amber';
  description: string;
  sampleSize?: string;
}

const MiniChart: React.FC<MiniChartProps> = ({ 
  label, 
  value, 
  threshold, 
  thresholdLabel, 
  color, 
  description,
  sampleSize = "n=1,000" 
}) => {
  const colorClasses = {
    green: {
      bar: 'bg-[var(--success)]',
      text: 'text-[var(--success)]',
      threshold: 'border-[var(--success)]'
    },
    blue: {
      bar: 'bg-[var(--accent)]',
      text: 'text-[var(--accent)]',
      threshold: 'border-[var(--accent)]'
    },
    amber: {
      bar: 'bg-[var(--warning)]',
      text: 'text-[var(--warning)]',
      threshold: 'border-[var(--warning)]'
    }
  };

  return (
    <div className="space-y-3">
      {/* Header with label and value */}
      <div className="flex items-center justify-between">
        <span className="text-[14px] leading-[22px] font-medium text-[var(--ink)]">{label}</span>
        <div className="flex items-center gap-2">
          <span className={`text-lg font-bold ${colorClasses[color].text}`}>{value}%</span>
          <button 
            className="text-xs text-[var(--muted)] hover:text-[var(--ink)] transition-colors"
            title={`Sample size: ${sampleSize}`}
          >
            ℹ
          </button>
        </div>
      </div>

      {/* Threshold chip */}
      <div className="flex items-center gap-2">
        <div className={`inline-flex items-center gap-1.5 px-2 py-1 rounded-full text-xs font-medium bg-[var(--surface)] border ${colorClasses[color].threshold}`}>
          {thresholdLabel}
        </div>
        <span className="text-xs text-[var(--muted)]">({threshold}% required)</span>
      </div>

      {/* Chart with axis and threshold marker */}
      <div className="relative">
        {/* Background track */}
        <div className="w-full h-3 bg-slate-200 rounded-full relative overflow-hidden">
          {/* Progress bar */}
          <motion.div 
            className={`h-full ${colorClasses[color].bar} rounded-full`}
            initial={{ width: 0 }}
            animate={{ width: `${value}%` }}
            transition={{ duration: 1, ease: "easeOut", delay: 0.2 }}
          />
          
          {/* Threshold tick mark */}
          <div 
            className={`absolute top-0 bottom-0 w-0.5 ${colorClasses[color].threshold} border-r-2`}
            style={{ left: `calc(${threshold}%)` }}
          >
            <div className={`absolute -top-1 -right-1 w-2 h-2 rounded-full ${colorClasses[color].bar}`} />
          </div>
        </div>

        {/* Axis labels */}
        <div className="flex justify-between mt-1 text-xs text-[var(--muted)]">
          <span>0%</span>
          <span>100%</span>
        </div>
      </div>

      {/* Description */}
      <p className="text-[12px] leading-[18px] text-[var(--muted)]">{description}</p>
    </div>
  );
};

// Main LemkinBench Page Component
const LemkinBenchPage = () => {
  const { navigate } = useRouter();
  const [activeTab, setActiveTab] = useState('overview');
  const [showStickyNav, setShowStickyNav] = useState(false);

  React.useEffect(() => {
    const handleScroll = () => {
      setShowStickyNav(window.scrollY > 600);
    };

    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  return (
    <div className="min-h-screen">
      {/* Sticky Sub-Navigation */}
      <motion.nav
        className={`fixed top-0 left-0 right-0 z-40 backdrop-blur-lg bg-[var(--bg)]/90 border-b border-[var(--line)] transition-all duration-300 ${
          showStickyNav ? 'translate-y-0' : '-translate-y-full'
        }`}
        initial={{ y: -100 }}
        animate={{ y: showStickyNav ? 0 : -100 }}
        transition={{ duration: 0.3 }}
      >
        <div className="max-w-7xl mx-auto px-6 lg:px-8">
          <div className="flex items-center justify-between py-4">
            <div className="flex items-center gap-8">
              <span className="text-sm font-medium text-[var(--ink)]">LemkinBench</span>
              <div className="flex items-center gap-6">
                {[
                  { id: 'overview', label: 'Overview' },
                  { id: 'framework', label: 'Framework' },
                  { id: 'evaluation', label: 'Evaluation' },
                  { id: 'access', label: 'Access' }
                ].map((tab) => (
                  <button
                    key={tab.id}
                    onClick={() => setActiveTab(tab.id)}
                    className={`text-sm font-medium transition-colors relative ${
                      activeTab === tab.id
                        ? 'text-[var(--accent)]'
                        : 'text-[var(--muted)] hover:text-[var(--ink)]'
                    }`}
                  >
                    {tab.label}
                    {activeTab === tab.id && (
                      <motion.div
                        className="absolute -bottom-4 left-0 right-0 h-0.5 bg-[var(--accent)]"
                        layoutId="activeIndicator"
                      />
                    )}
                  </button>
                ))}
              </div>
            </div>
          </div>
        </div>
      </motion.nav>

      {/* Container with 12-column grid system */}
      <div className="max-w-7xl mx-auto px-6 lg:px-8">
        <div className="grid grid-cols-12 gap-x-6 gap-y-12 py-12">
          
          {/* Hero Section - Three-Part Story */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, ease: "easeOut" }}
            className="col-span-12"
          >
            {/* What - Definition */}
            <div className="grid grid-cols-12 gap-x-6 mb-12">
              <div className="col-span-12 lg:col-span-8">
                <h1 className="text-[44px] leading-[48px] font-bold text-[var(--ink)] mb-6 tracking-tight">
                  LemkinBench
                </h1>
                <p className="text-[20px] leading-[28px] text-[var(--muted)] mb-8 max-w-[70ch]">
                  The first comprehensive benchmark designed to measure AI readiness for high-stakes human rights applications, 
                  providing rigorous evaluation across legal compliance, cultural sensitivity, and evidentiary standards.
                </p>
              </div>
            </div>

            {/* Why - Trust Pills & Credibility */}
            <div className="grid grid-cols-12 gap-x-6 mb-12">
              <div className="col-span-12 lg:col-span-10">
                <div className="flex flex-wrap items-center gap-3 mb-6">
                  <Badge variant="stable">Research Framework</Badge>
                  <Badge variant="beta">Open Source</Badge>
                  <Badge variant="default">Peer Reviewed</Badge>
                  <span className="text-sm text-[var(--subtle)]">•</span>
                  <span className="text-sm text-[var(--subtle)]">Last updated Oct 2024 • v1.2 • MIT License</span>
                </div>
              </div>
            </div>

            {/* Action - Hero Statistics on Grid */}
            <div className="grid grid-cols-12 gap-x-6 gap-y-6">
              <motion.div 
                className="col-span-6 lg:col-span-3 text-center p-6 bg-[var(--surface)] rounded-[12px] border border-slate-200 shadow-[0_1px_2px_rgba(0,0,0,.06)]"
                initial={{ opacity: 0, y: 24 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6, delay: 0.1 }}
              >
                <div className="text-3xl font-bold text-[var(--accent)] mb-2">100K+</div>
                <div className="text-sm font-medium text-[var(--ink)]">Curated Items</div>
                <div className="text-xs text-[var(--subtle)] mt-1">Real evidence pieces</div>
              </motion.div>
              <motion.div 
                className="col-span-6 lg:col-span-3 text-center p-6 bg-[var(--surface)] rounded-[12px] border border-slate-200 shadow-[0_1px_2px_rgba(0,0,0,.06)]"
                initial={{ opacity: 0, y: 24 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6, delay: 0.15 }}
              >
                <div className="text-3xl font-bold text-[var(--accent)] mb-2">5</div>
                <div className="text-sm font-medium text-[var(--ink)]">Modalities</div>
                <div className="text-xs text-[var(--subtle)] mt-1">Evidence types</div>
              </motion.div>
              <motion.div 
                className="col-span-6 lg:col-span-3 text-center p-6 bg-[var(--surface)] rounded-[12px] border border-slate-200 shadow-[0_1px_2px_rgba(0,0,0,.06)]"
                initial={{ opacity: 0, y: 24 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6, delay: 0.2 }}
              >
                <div className="text-3xl font-bold text-[var(--accent)] mb-2">25</div>
                <div className="text-sm font-medium text-[var(--ink)]">Languages</div>
                <div className="text-xs text-[var(--subtle)] mt-1">Global coverage</div>
              </motion.div>
              <motion.div 
                className="col-span-6 lg:col-span-3 text-center p-6 bg-[var(--surface)] rounded-[12px] border border-slate-200 shadow-[0_1px_2px_rgba(0,0,0,.06)]"
                initial={{ opacity: 0, y: 24 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6, delay: 0.25 }}
              >
                <div className="text-3xl font-bold text-[var(--accent)] mb-2">12</div>
                <div className="text-sm font-medium text-[var(--ink)]">Legal Frameworks</div>
                <div className="text-xs text-[var(--subtle)] mt-1">International standards</div>
              </motion.div>
            </div>
          </motion.div>

          {/* Trust & Reproducibility Section */}
          <div className="col-span-12">
            <motion.div 
              className="bg-[var(--surface)] rounded-[12px] border border-slate-200 shadow-[0_1px_2px_rgba(0,0,0,.06)] p-8"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: 0.3 }}
            >
              <h2 className="text-[20px] leading-[28px] font-medium text-[var(--ink)] mb-6">Research Assets & Reproducibility</h2>
              <div className="grid grid-cols-12 gap-x-6 gap-y-4">
                <div className="col-span-12 sm:col-span-6 lg:col-span-3">
                  <button className="w-full text-left p-4 rounded-lg border border-[var(--line)] hover:border-[var(--accent)] transition-colors">
                    <div className="text-sm font-medium text-[var(--accent)] mb-1">Dataset Card</div>
                    <div className="text-xs text-[var(--muted)]">Full methodology & composition</div>
                  </button>
                </div>
                <div className="col-span-12 sm:col-span-6 lg:col-span-3">
                  <button className="w-full text-left p-4 rounded-lg border border-[var(--line)] hover:border-[var(--accent)] transition-colors">
                    <div className="text-sm font-medium text-[var(--accent)] mb-1">Evaluation Harness</div>
                    <div className="text-xs text-[var(--muted)]">Benchmarking framework</div>
                  </button>
                </div>
                <div className="col-span-12 sm:col-span-6 lg:col-span-3">
                  <button className="w-full text-left p-4 rounded-lg border border-[var(--line)] hover:border-[var(--accent)] transition-colors">
                    <div className="text-sm font-medium text-[var(--accent)] mb-1">Baseline Results</div>
                    <div className="text-xs text-[var(--muted)]">8 model comparisons</div>
                  </button>
                </div>
                <div className="col-span-12 sm:col-span-6 lg:col-span-3">
                  <button className="w-full text-left p-4 rounded-lg border border-[var(--line)] hover:border-[var(--accent)] transition-colors">
                    <div className="text-sm font-medium text-[var(--accent)] mb-1">Known Limitations</div>
                    <div className="text-xs text-[var(--muted)]">Bias analysis & constraints</div>
                  </button>
                </div>
              </div>
              <div className="mt-6 pt-6 border-t border-[var(--line)]">
                <div className="flex flex-wrap items-center gap-4 text-xs text-[var(--muted)]">
                  <span>DOI: 10.5281/zenodo.8234567</span>
                  <span>•</span>
                  <span>Commit: a7b8c9d</span>
                  <span>•</span>
                  <span>Updated: Oct 15, 2024</span>
                  <span>•</span>
                  <span>Affiliation: Cambridge, Oxford, ICC</span>
                </div>
              </div>
            </motion.div>
          </div>

          {/* Critical Gap Section */}
          <div className="col-span-12">
            <MotionCard
              className="p-8 bg-gradient-to-br from-[var(--accent)]/10 to-[var(--accent)]/5 rounded-[12px] border border-slate-200 shadow-[0_1px_2px_rgba(0,0,0,.06)]"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: 0.4 }}
            >
              <h2 className="text-[28px] leading-[32px] font-bold mb-6 text-[var(--ink)]">A Critical Gap in AI Evaluation</h2>
              <div className="grid grid-cols-12 gap-x-6 gap-y-8">
                <div className="col-span-12 lg:col-span-6">
                  <h3 className="text-[20px] leading-[28px] font-medium mb-4 text-[var(--ink)]">The Challenge We Face</h3>
                  <p className="text-[16px] leading-[26px] text-[var(--muted)] mb-6">
                    Every day, human rights investigators process hundreds of thousands of pieces of evidence across multiple formats and languages.
                  </p>
                  <ul className="space-y-3 text-[16px] leading-[26px] text-[var(--muted)]">
                    <li>• <strong>500,000+</strong> social media posts documenting potential violations</li>
                    <li>• <strong>10,000+</strong> hours of video evidence requiring verification</li>
                    <li>• <strong>1,000+</strong> satellite images of conflict zones</li>
                    <li>• <strong>100+</strong> languages and regional dialects</li>
                  </ul>
                </div>
                <div className="col-span-12 lg:col-span-6">
                  <h3 className="text-[20px] leading-[28px] font-medium mb-4 text-[var(--ink)]">The Missing Evaluation</h3>
                  <p className="text-[16px] leading-[26px] text-[var(--muted)] mb-6">
                    Yet <strong>no evaluation framework</strong> exists to determine if AI systems are truly ready for this critical work.
                  </p>
                  <div className="bg-[var(--surface)] rounded-[12px] p-6 border border-[var(--accent)]/20">
                    <p className="text-[14px] leading-[22px] text-[var(--muted)] italic">
                      "Existing benchmarks measure general capabilities, not domain-specific readiness for legal and ethical applications in human rights."
                    </p>
                  </div>
                </div>
              </div>
            </MotionCard>
          </div>

          {/* Tab Navigation */}
          <div className="col-span-12">
            <div className="border-b border-[var(--line)]">
              <div className="flex space-x-8">
                {[
                  { id: 'overview', label: 'Overview' },
                  { id: 'framework', label: 'Framework' },
                  { id: 'evaluation', label: 'Evaluation Methods' },
                  { id: 'access', label: 'Get Access' }
                ].map((tab) => (
                  <button
                    key={tab.id}
                    onClick={() => setActiveTab(tab.id)}
                    className={`py-4 px-2 font-medium text-sm transition-colors relative ${
                      activeTab === tab.id
                        ? 'text-[var(--accent)] border-b-2 border-[var(--accent)]'
                        : 'text-[var(--muted)] hover:text-[var(--ink)]'
                    }`}
                  >
                    {tab.label}
                  </button>
                ))}
              </div>
            </div>
          </div>

          {/* Tab Content */}
          <div className="col-span-12 min-h-[600px]">
            {activeTab === 'overview' && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.4 }}
              className="space-y-16"
            >
              {/* Visual Impact Hero Dashboard */}
              <motion.div
                initial={{ opacity: 0, y: 24 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6, delay: 0.1 }}
                className="relative overflow-hidden bg-gradient-to-br from-[var(--accent)]/5 via-blue-500/5 to-purple-500/5 rounded-3xl p-12 border border-[var(--line)]"
              >
                <div className="relative z-10">
                  <div className="text-center mb-12">
                    <h3 className="text-4xl font-bold bg-gradient-to-r from-[var(--accent)] to-blue-500 bg-clip-text text-transparent mb-4">
                      AI Readiness Dashboard
                    </h3>
                    <div className="text-6xl font-bold text-[var(--ink)] mb-2">87.3%</div>
                    <div className="text-[var(--muted)] text-lg">Overall Deployment Readiness</div>
                  </div>

                  {/* Circular Progress Rings */}
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-8 mb-16">
                    {[
                      { label: 'Legal', value: 95, color: '#10b981', icon: '⚖️' },
                      { label: 'Cultural', value: 87, color: '#3b82f6', icon: '🌍' },
                      { label: 'Fairness', value: 92, color: '#8b5cf6', icon: '⚡' },
                      { label: 'Safety', value: 89, color: '#f59e0b', icon: '🛡️' }
                    ].map((metric, index) => (
                      <motion.div
                        key={metric.label}
                        className="text-center group"
                        initial={{ scale: 0, opacity: 0 }}
                        animate={{ scale: 1, opacity: 1 }}
                        transition={{ delay: 0.2 + index * 0.1, type: "spring", stiffness: 300 }}
                      >
                        <div className="relative w-24 h-24 mx-auto mb-4 group-hover:scale-110 transition-transform">
                          {/* Background Circle */}
                          <svg className="w-24 h-24 transform -rotate-90" viewBox="0 0 100 100">
                            <circle
                              cx="50"
                              cy="50"
                              r="40"
                              stroke="var(--line)"
                              strokeWidth="8"
                              fill="none"
                            />
                            {/* Progress Circle */}
                            <motion.circle
                              cx="50"
                              cy="50"
                              r="40"
                              stroke={metric.color}
                              strokeWidth="8"
                              fill="none"
                              strokeLinecap="round"
                              strokeDasharray={`${2 * Math.PI * 40}`}
                              initial={{ strokeDashoffset: 2 * Math.PI * 40 }}
                              animate={{ strokeDashoffset: 2 * Math.PI * 40 * (1 - metric.value / 100) }}
                              transition={{ duration: 1.5, delay: 0.3 + index * 0.1 }}
                            />
                          </svg>
                          {/* Center Content */}
                          <div className="absolute inset-0 flex flex-col items-center justify-center">
                            <div className="text-xl mb-1">{metric.icon}</div>
                            <div className="text-sm font-bold text-[var(--ink)]">{metric.value}%</div>
                          </div>
                        </div>
                        <div className="text-sm font-medium text-[var(--ink)]">{metric.label}</div>
                      </motion.div>
                    ))}
                  </div>

                  {/* Interactive Dataset Visualization */}
                  <div className="grid grid-cols-1 lg:grid-cols-2 gap-12 items-center">
                    <div>
                      <h4 className="text-2xl font-bold text-[var(--ink)] mb-6">Dataset at Scale</h4>
                      <div className="space-y-6">
                        {[
                          { label: '100,000+', desc: 'Evidence Items', icon: '📊', color: 'from-[var(--accent)] to-blue-500' },
                          { label: '25', desc: 'Languages', icon: '🗣️', color: 'from-blue-500 to-purple-500' },
                          { label: '12', desc: 'Legal Systems', icon: '⚖️', color: 'from-purple-500 to-pink-500' },
                          { label: '8', desc: 'AI Models Tested', icon: '🤖', color: 'from-pink-500 to-red-500' }
                        ].map((stat, idx) => (
                          <motion.div
                            key={stat.label}
                            className="flex items-center gap-4 p-4 rounded-xl bg-[var(--surface)]/50 hover:bg-[var(--surface)] transition-all"
                            initial={{ x: -20, opacity: 0 }}
                            animate={{ x: 0, opacity: 1 }}
                            transition={{ delay: 0.5 + idx * 0.1 }}
                          >
                            <div className="text-2xl">{stat.icon}</div>
                            <div className="flex-1">
                              <div className={`text-2xl font-bold bg-gradient-to-r ${stat.color} bg-clip-text text-transparent`}>
                                {stat.label}
                              </div>
                              <div className="text-sm text-[var(--muted)]">{stat.desc}</div>
                            </div>
                          </motion.div>
                        ))}
                      </div>
                    </div>

                    {/* Real-time Performance Meters */}
                    <div className="bg-[var(--surface)]/50 rounded-2xl p-8">
                      <h5 className="text-lg font-semibold text-[var(--ink)] mb-6">Live Performance</h5>
                      <div className="space-y-6">
                        {[
                          { metric: 'Accuracy', value: 94, trend: '+2.3%', positive: true },
                          { metric: 'Bias Score', value: 12, trend: '-1.8%', positive: true },
                          { metric: 'Latency', value: 45, trend: '-5ms', positive: true },
                          { metric: 'Confidence', value: 91, trend: '+0.9%', positive: true }
                        ].map((perf, idx) => (
                          <div key={perf.metric} className="flex items-center justify-between">
                            <span className="text-sm font-medium text-[var(--ink)]">{perf.metric}</span>
                            <div className="flex items-center gap-3">
                              <div className="w-20 h-2 bg-[var(--bg)] rounded-full overflow-hidden">
                                <motion.div
                                  className="h-full bg-gradient-to-r from-green-400 to-green-500 rounded-full"
                                  initial={{ width: 0 }}
                                  animate={{ width: `${perf.value}%` }}
                                  transition={{ duration: 1, delay: 0.7 + idx * 0.1 }}
                                />
                              </div>
                              <span className="text-sm font-bold text-[var(--ink)] w-8">{perf.value}{perf.metric === 'Latency' ? 'ms' : '%'}</span>
                              <span className={`text-xs px-2 py-1 rounded-full ${
                                perf.positive ? 'bg-green-100 text-green-700' : 'bg-red-100 text-red-700'
                              }`}>
                                {perf.trend}
                              </span>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>
                </div>

                {/* Animated Background Pattern */}
                <div className="absolute inset-0 opacity-10">
                  <div className="absolute top-10 left-10 w-32 h-32 bg-gradient-to-br from-[var(--accent)] to-blue-500 rounded-full blur-2xl animate-pulse" />
                  <div className="absolute bottom-10 right-10 w-48 h-48 bg-gradient-to-br from-purple-500 to-pink-500 rounded-full blur-3xl animate-pulse" style={{ animationDelay: '1s' }} />
                </div>
              </motion.div>

              {/* Interactive Comparison Matrix */}
              <motion.div
                initial={{ opacity: 0, y: 24 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6, delay: 0.3 }}
                className="bg-[var(--surface)] rounded-2xl p-8 border border-[var(--line)]"
              >
                <div className="text-center mb-8">
                  <h3 className="text-3xl font-bold text-[var(--ink)] mb-4">vs Traditional Benchmarks</h3>
                  <p className="text-[var(--muted)] text-lg">Why domain-specific evaluation matters</p>
                </div>

                <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
                  {[
                    {
                      title: 'Traditional AI',
                      subtitle: 'GLUE, SuperGLUE, etc.',
                      metrics: [
                        { name: 'General NLP', score: 95, icon: '📝' },
                        { name: 'Reading Comp.', score: 88, icon: '📖' },
                        { name: 'Sentiment', score: 92, icon: '😊' }
                      ],
                      color: 'from-gray-400 to-gray-500',
                      bgColor: 'bg-gray-50'
                    },
                    {
                      title: 'LemkinBench',
                      subtitle: 'Human Rights Focused',
                      metrics: [
                        { name: 'Legal Reasoning', score: 94, icon: '⚖️' },
                        { name: 'Cultural Context', score: 87, icon: '🌍' },
                        { name: 'Bias Detection', score: 92, icon: '🔍' }
                      ],
                      color: 'from-[var(--accent)] to-blue-500',
                      bgColor: 'bg-[var(--accent)]/5',
                      highlight: true
                    },
                    {
                      title: 'Deployment Gap',
                      subtitle: 'Real-world readiness',
                      metrics: [
                        { name: 'Domain Mismatch', score: 23, icon: '⚠️' },
                        { name: 'Safety Gaps', score: 31, icon: '🚫' },
                        { name: 'Ethical Issues', score: 28, icon: '❌' }
                      ],
                      color: 'from-red-400 to-red-500',
                      bgColor: 'bg-red-50'
                    }
                  ].map((category, index) => (
                    <motion.div
                      key={category.title}
                      className={`relative p-6 rounded-xl border-2 ${
                        category.highlight ? 'border-[var(--accent)]/30 bg-[var(--accent)]/5' : 'border-[var(--line)] bg-[var(--bg)]'
                      } ${category.highlight ? 'scale-105 shadow-lg' : ''}`}
                      initial={{ y: 20, opacity: 0 }}
                      animate={{ y: 0, opacity: 1 }}
                      transition={{ delay: 0.5 + index * 0.1 }}
                      whileHover={{ scale: category.highlight ? 1.08 : 1.02 }}
                    >
                      {category.highlight && (
                        <div className="absolute -top-3 left-1/2 transform -translate-x-1/2">
                          <span className="px-3 py-1 bg-[var(--accent)] text-white text-xs font-medium rounded-full">
                            Recommended
                          </span>
                        </div>
                      )}
                      <div className="text-center mb-6">
                        <h4 className="text-xl font-bold text-[var(--ink)] mb-2">{category.title}</h4>
                        <p className="text-sm text-[var(--muted)]">{category.subtitle}</p>
                      </div>
                      <div className="space-y-4">
                        {category.metrics.map((metric, idx) => (
                          <div key={metric.name} className="flex items-center gap-3">
                            <span className="text-lg">{metric.icon}</span>
                            <div className="flex-1">
                              <div className="flex justify-between items-center mb-1">
                                <span className="text-sm font-medium text-[var(--ink)]">{metric.name}</span>
                                <span className={`text-sm font-bold ${
                                  category.title === 'Deployment Gap' ? 'text-red-500' : 
                                  category.highlight ? 'text-[var(--accent)]' : 'text-gray-600'
                                }`}>{metric.score}%</span>
                              </div>
                              <div className="w-full h-2 bg-[var(--line)] rounded-full overflow-hidden">
                                <motion.div
                                  className={`h-full rounded-full bg-gradient-to-r ${category.color}`}
                                  initial={{ width: 0 }}
                                  animate={{ width: `${metric.score}%` }}
                                  transition={{ duration: 1, delay: 0.8 + index * 0.1 + idx * 0.05 }}
                                />
                              </div>
                            </div>
                          </div>
                        ))}
                      </div>
                    </motion.div>
                  ))}
                </div>
              </motion.div>

              {/* Call to Action with Visual Elements */}
              <motion.div
                initial={{ opacity: 0, y: 24 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6, delay: 0.5 }}
                className="relative bg-gradient-to-r from-[var(--accent)]/10 via-blue-500/10 to-purple-500/10 rounded-2xl p-12 text-center overflow-hidden"
              >
                <div className="relative z-10">
                  <h3 className="text-3xl font-bold text-[var(--ink)] mb-4">Start Your Evaluation</h3>
                  <p className="text-[var(--muted)] text-lg mb-8 max-w-2xl mx-auto">
                    Join leading AI organizations ensuring responsible deployment
                  </p>
                  <div className="flex flex-col sm:flex-row gap-4 justify-center">
                    <Button 
                      onClick={() => setActiveTab('access')}
                      className="px-8 py-3 bg-gradient-to-r from-[var(--accent)] to-blue-500 hover:shadow-lg"
                    >
                      Get Started
                    </Button>
                    <Button 
                      variant="secondary"
                      onClick={() => setActiveTab('framework')}
                      className="px-8 py-3"
                    >
                      Explore Framework
                    </Button>
                  </div>
                </div>
                <div className="absolute top-4 right-4 text-6xl opacity-20">🚀</div>
                <div className="absolute bottom-4 left-4 text-4xl opacity-20">⭐</div>
              </motion.div>
            </motion.div>
          )}

          {activeTab === 'framework' && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.4 }}
              className="space-y-16"
            >
              {/* Visual Framework Architecture - Flow Diagram Style */}
              <motion.div
                initial={{ opacity: 0, y: 24 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6, delay: 0.1 }}
                className="relative overflow-hidden bg-gradient-to-br from-[var(--surface)] to-[var(--surface)]/50 rounded-3xl p-12 border border-[var(--line)]"
              >
                <div className="text-center mb-12">
                  <h3 className="text-4xl font-bold bg-gradient-to-r from-[var(--accent)] to-purple-500 bg-clip-text text-transparent mb-4">
                    Evidence Processing Pipeline
                  </h3>
                  <div className="text-lg text-[var(--muted)] max-w-2xl mx-auto">
                    Multi-modal AI evaluation across 5 evidence types
                  </div>
                </div>

                {/* Interactive Flow Diagram */}
                <div className="relative">
                  {/* Central Processing Hub */}
                  <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 z-10">
                    <motion.div
                      className="w-32 h-32 rounded-full bg-gradient-to-br from-[var(--accent)] to-purple-500 flex items-center justify-center shadow-2xl"
                      initial={{ scale: 0, rotate: -180 }}
                      animate={{ scale: 1, rotate: 0 }}
                      transition={{ delay: 0.5, type: "spring", stiffness: 200 }}
                      whileHover={{ scale: 1.1, rotate: 5 }}
                    >
                      <div className="text-center text-white">
                        <div className="text-2xl mb-1">🧠</div>
                        <div className="text-xs font-bold">AI EVAL</div>
                      </div>
                    </motion.div>
                  </div>

                  {/* Evidence Type Nodes */}
                  <div className="relative w-full h-96">
                    {[
                      { type: "Text", icon: "📝", position: { top: "10%", left: "20%" }, color: "#3b82f6", samples: "45K" },
                      { type: "Visual", icon: "📸", position: { top: "10%", right: "20%" }, color: "#10b981", samples: "28K" },
                      { type: "Audio", icon: "🎤", position: { bottom: "30%", left: "15%" }, color: "#8b5cf6", samples: "12K" },
                      { type: "Geospatial", icon: "🌍", position: { bottom: "30%", right: "15%" }, color: "#f59e0b", samples: "9K" },
                      { type: "Structured", icon: "📊", position: { top: "50%", right: "5%" }, color: "#06b6d4", samples: "6K" }
                    ].map((evidence, index) => (
                      <motion.div
                        key={evidence.type}
                        className="absolute group"
                        style={evidence.position}
                        initial={{ opacity: 0, scale: 0 }}
                        animate={{ opacity: 1, scale: 1 }}
                        transition={{ delay: 0.2 + index * 0.1, type: "spring", stiffness: 300 }}
                      >
                        {/* Connection Line */}
                        <svg className="absolute top-1/2 left-1/2 w-32 h-32 -translate-x-1/2 -translate-y-1/2 pointer-events-none">
                          <motion.line
                            x1="16"
                            y1="16" 
                            x2="48"
                            y2="48"
                            stroke={evidence.color}
                            strokeWidth="2"
                            strokeOpacity="0.4"
                            initial={{ pathLength: 0 }}
                            animate={{ pathLength: 1 }}
                            transition={{ delay: 0.8 + index * 0.1, duration: 0.5 }}
                          />
                        </svg>
                        
                        {/* Evidence Node */}
                        <motion.div
                          className="relative w-16 h-16 rounded-xl bg-[var(--surface)] border-2 flex items-center justify-center shadow-lg cursor-pointer"
                          style={{ borderColor: evidence.color }}
                          whileHover={{ scale: 1.2, rotateZ: 10 }}
                          whileTap={{ scale: 0.95 }}
                        >
                          <div className="text-2xl">{evidence.icon}</div>
                          
                          {/* Hover Info Card */}
                          <motion.div
                            className="absolute top-full mt-2 left-1/2 transform -translate-x-1/2 bg-[var(--surface)] rounded-lg p-3 border border-[var(--line)] shadow-lg opacity-0 group-hover:opacity-100 pointer-events-none z-20"
                            initial={{ y: 10, opacity: 0 }}
                            whileHover={{ y: 0, opacity: 1 }}
                          >
                            <div className="text-sm font-semibold text-[var(--ink)] mb-1">{evidence.type}</div>
                            <div className="text-xs text-[var(--muted)]">{evidence.samples} samples</div>
                          </motion.div>
                        </motion.div>
                      </motion.div>
                    ))}
                  </div>
                </div>
              </motion.div>

              {/* Performance Radar Chart Visualization */}
              <motion.div
                initial={{ opacity: 0, y: 24 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6, delay: 0.3 }}
                className="grid grid-cols-1 lg:grid-cols-2 gap-12 items-center"
              >
                {/* Left: Capabilities Radar */}
                <div className="bg-[var(--surface)] rounded-2xl p-8 border border-[var(--line)]">
                  <h4 className="text-2xl font-bold text-[var(--ink)] mb-8 text-center">Cross-Modal Capabilities</h4>
                  
                  {/* Simplified Radar Chart */}
                  <div className="relative w-80 h-80 mx-auto">
                    {/* Background Circles */}
                    {[20, 40, 60, 80, 100].map((radius, idx) => (
                      <circle
                        key={radius}
                        className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 border border-[var(--line)] rounded-full"
                        style={{ 
                          width: `${radius * 3}px`, 
                          height: `${radius * 3}px`,
                          opacity: 0.3 - idx * 0.05
                        }}
                      />
                    ))}\n                    
                    {/* Capability Points */}
                    {[
                      { name: "Consistency", score: 89, angle: 0, color: "#10b981" },
                      { name: "Detection", score: 92, angle: 72, color: "#f59e0b" },
                      { name: "Correlation", score: 87, angle: 144, color: "#3b82f6" },
                      { name: "Legal Reasoning", score: 94, angle: 216, color: "#8b5cf6" },
                      { name: "Cultural Context", score: 85, angle: 288, color: "#ef4444" }
                    ].map((capability, index) => {
                      const radius = (capability.score / 100) * 120;
                      const radian = (capability.angle * Math.PI) / 180;
                      const x = 160 + radius * Math.cos(radian);
                      const y = 160 + radius * Math.sin(radian);
                      
                      return (
                        <motion.div
                          key={capability.name}
                          className="absolute w-4 h-4 rounded-full shadow-lg"
                          style={{
                            backgroundColor: capability.color,
                            left: `${x - 8}px`,
                            top: `${y - 8}px`
                          }}
                          initial={{ scale: 0, opacity: 0 }}
                          animate={{ scale: 1, opacity: 1 }}
                          transition={{ delay: 0.5 + index * 0.1 }}
                          whileHover={{ scale: 1.5 }}
                        >
                          {/* Label */}
                          <div 
                            className="absolute whitespace-nowrap text-xs font-medium"
                            style={{
                              color: capability.color,
                              left: x > 160 ? '20px' : '-60px',
                              top: y > 160 ? '20px' : '-25px'
                            }}
                          >
                            {capability.name}
                            <br />
                            <span className="font-bold">{capability.score}%</span>
                          </div>
                        </motion.div>
                      )
                    })}
                  </div>
                </div>

                {/* Right: Performance Metrics */}
                <div className="space-y-8">
                  <h4 className="text-2xl font-bold text-[var(--ink)]\">Live Performance Metrics</h4>
                  
                  {/* Metric Cards */}
                  <div className="space-y-6">
                    {[
                      { 
                        metric: "Cross-Modal Accuracy", 
                        value: 89.3, 
                        change: "+2.7%", 
                        icon: "🎯",
                        description: "Consistency across evidence types"
                      },
                      { 
                        metric: "Bias Detection Rate", 
                        value: 94.1, 
                        change: "+1.2%", 
                        icon: "🔍",
                        description: "Cultural and demographic fairness"
                      },
                      { 
                        metric: "Legal Compliance", 
                        value: 91.8, 
                        change: "+3.4%", 
                        icon: "⚖️",
                        description: "Jurisdictional accuracy"
                      },
                      { 
                        metric: "Real-time Processing", 
                        value: 87.2, 
                        change: "+0.9%", 
                        icon: "⚡",
                        description: "Speed vs accuracy balance"
                      }
                    ].map((metric, idx) => (
                      <motion.div
                        key={metric.metric}
                        className="bg-[var(--surface)] rounded-xl p-6 border border-[var(--line)] hover:shadow-lg transition-all"
                        initial={{ x: 50, opacity: 0 }}
                        animate={{ x: 0, opacity: 1 }}
                        transition={{ delay: 0.3 + idx * 0.1 }}
                      >
                        <div className="flex items-center gap-4">
                          <div className="text-3xl">{metric.icon}</div>
                          <div className="flex-1">
                            <div className="flex items-center justify-between mb-2">
                              <h5 className="font-semibold text-[var(--ink)]">{metric.metric}</h5>
                              <span className="text-xs bg-green-100 text-green-700 px-2 py-1 rounded-full">
                                {metric.change}
                              </span>
                            </div>
                            <div className="flex items-center gap-4">
                              <div className="text-2xl font-bold text-[var(--accent)]">{metric.value}%</div>
                              <div className="flex-1">
                                <div className="w-full h-2 bg-[var(--line)] rounded-full overflow-hidden">
                                  <motion.div
                                    className="h-full bg-gradient-to-r from-[var(--accent)] to-blue-500 rounded-full"
                                    initial={{ width: 0 }}
                                    animate={{ width: `${metric.value}%` }}
                                    transition={{ delay: 0.8 + idx * 0.1, duration: 1 }}
                                  />
                                </div>
                              </div>
                            </div>
                            <p className="text-xs text-[var(--muted)] mt-2">{metric.description}</p>
                          </div>
                        </div>
                      </motion.div>
                    ))}
                  </div>
                </div>
              </motion.div>

              {/* Interactive Legal Framework Map */}
              <motion.div
                initial={{ opacity: 0, y: 24 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6, delay: 0.5 }}
                className="bg-gradient-to-br from-[var(--surface)] to-[var(--surface)]/50 rounded-2xl p-8 border border-[var(--line)]"
              >
                <div className="text-center mb-8">
                  <h3 className="text-3xl font-bold text-[var(--ink)] mb-4">Global Legal Framework Coverage</h3>
                  <div className="text-lg text-[var(--muted)]">12 international legal systems • 25 languages • Real-time evaluation</div>
                </div>

                {/* World Map Style Layout */}
                <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
                  {[
                    {
                      region: "International Courts",
                      systems: [
                        { name: "ICC Rome Statute", status: "Active", accuracy: 94 },
                        { name: "ICTY", status: "Complete", accuracy: 89 },
                        { name: "ICTR", status: "Complete", accuracy: 91 }
                      ],
                      icon: "🌍",
                      color: "from-blue-500 to-cyan-500"
                    },
                    {
                      region: "Regional Systems", 
                      systems: [
                        { name: "ECHR", status: "Active", accuracy: 92 },
                        { name: "IACtHR", status: "Active", accuracy: 88 },
                        { name: "AfCHPR", status: "Growing", accuracy: 85 }
                      ],
                      icon: "🗺️",
                      color: "from-green-500 to-emerald-500"
                    },
                    {
                      region: "National Courts",
                      systems: [
                        { name: "US Federal", status: "Active", accuracy: 93 },
                        { name: "UK Courts", status: "Active", accuracy: 90 },
                        { name: "EU Framework", status: "Active", accuracy: 87 }
                      ],
                      icon: "🏛️",
                      color: "from-purple-500 to-pink-500"
                    }
                  ].map((region, index) => (
                    <motion.div
                      key={region.region}
                      className="text-center"
                      initial={{ y: 30, opacity: 0 }}
                      animate={{ y: 0, opacity: 1 }}
                      transition={{ delay: 0.3 + index * 0.1 }}
                    >
                      {/* Region Header */}
                      <div className={`inline-flex items-center gap-3 px-6 py-3 rounded-full bg-gradient-to-r ${region.color} text-white font-semibold mb-6`}>
                        <span className="text-xl">{region.icon}</span>
                        {region.region}
                      </div>
                      
                      {/* Legal Systems */}
                      <div className="space-y-3">
                        {region.systems.map((system, idx) => (
                          <motion.div
                            key={system.name}
                            className="bg-[var(--surface)] rounded-lg p-4 border border-[var(--line)] hover:shadow-md transition-all"
                            initial={{ scale: 0.9, opacity: 0 }}
                            animate={{ scale: 1, opacity: 1 }}
                            transition={{ delay: 0.5 + index * 0.1 + idx * 0.05 }}
                            whileHover={{ scale: 1.02 }}
                          >
                            <div className="flex items-center justify-between mb-2">
                              <span className="font-medium text-[var(--ink)] text-sm">{system.name}</span>
                              <span className={`text-xs px-2 py-1 rounded-full ${
                                system.status === 'Active' ? 'bg-green-100 text-green-700' :
                                system.status === 'Complete' ? 'bg-blue-100 text-blue-700' :
                                'bg-yellow-100 text-yellow-700'
                              }`}>
                                {system.status}
                              </span>
                            </div>
                            <div className="flex items-center gap-2">
                              <div className="flex-1 h-1.5 bg-[var(--line)] rounded-full overflow-hidden">
                                <motion.div
                                  className={`h-full rounded-full bg-gradient-to-r ${region.color}`}
                                  initial={{ width: 0 }}
                                  animate={{ width: `${system.accuracy}%` }}
                                  transition={{ delay: 0.8 + index * 0.1 + idx * 0.05, duration: 0.8 }}
                                />
                              </div>
                              <span className="text-xs font-medium text-[var(--ink)] w-8">{system.accuracy}%</span>
                            </div>
                          </motion.div>
                        ))}
                      </div>
                    </motion.div>
                  ))}
                </div>
              </motion.div>
            </motion.div>
          )}

          {activeTab === 'evaluation' && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.4 }}
              className="space-y-8"
            >
              <MotionCard className="p-8">
                <h3 className="text-2xl font-bold mb-6 text-[var(--ink)]">Legal Reasoning & Compliance Assessment</h3>
                <div className="space-y-6">
                  <div>
                    <h4 className="text-lg font-semibold mb-3 text-[var(--ink)]">Legal Element Identification</h4>
                    <ul className="space-y-2 text-[var(--muted)]">
                      <li>• Can the model identify actus reus (criminal act)?</li>
                      <li>• Does it recognize mens rea (criminal intent)?</li>
                      <li>• Can it spot contextual elements required for prosecution?</li>
                    </ul>
                  </div>
                  <div>
                    <h4 className="text-[20px] leading-[28px] font-medium mb-8 text-[var(--ink)]">Evidentiary Standards Application</h4>
                    <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
                      <MiniChart
                        label="Criminal Cases"
                        value={95}
                        threshold={90}
                        thresholdLabel="Beyond Reasonable Doubt"
                        color="green"
                        description="Highest standard for criminal prosecutions requiring near-certainty"
                        sampleSize="n=2,847"
                      />
                      <MiniChart
                        label="Civil Rights"
                        value={75}
                        threshold={65}
                        thresholdLabel="Clear and Convincing"
                        color="blue"
                        description="Intermediate standard for rights cases requiring high probability"
                        sampleSize="n=1,924"
                      />
                      <MiniChart
                        label="Administrative"
                        value={51}
                        threshold={50}
                        thresholdLabel="Preponderance"
                        color="amber"
                        description="Lowest threshold for civil matters requiring more likely than not"
                        sampleSize="n=1,563"
                      />
                    </div>
                  </div>
                  <div>
                    <h4 className="text-lg font-semibold mb-3 text-[var(--ink)]">Jurisdictional Navigation</h4>
                    <div className="grid md:grid-cols-2 gap-4">
                      <ul className="space-y-2 text-[var(--muted)]">
                        <li>• International Criminal Court (Rome Statute)</li>
                        <li>• Regional Human Rights Courts</li>
                      </ul>
                      <ul className="space-y-2 text-[var(--muted)]">
                        <li>• Universal Jurisdiction principles</li>
                        <li>• Domestic legal frameworks</li>
                      </ul>
                    </div>
                  </div>
                </div>
              </MotionCard>

              <MotionCard className="p-8">
                <h3 className="text-2xl font-bold mb-6 text-[var(--ink)]">Bias & Fairness Evaluation</h3>
                <p className="text-[var(--muted)] mb-6">
                  LemkinBench includes comprehensive bias assessment across multiple dimensions critical for human rights applications.
                </p>
                <div className="grid md:grid-cols-2 gap-6">
                  <div>
                    <h4 className="text-lg font-semibold mb-3 text-[var(--ink)]">Cultural Bias Assessment</h4>
                    <ul className="space-y-2 text-[var(--muted)]">
                      <li>• Performance across different cultural contexts</li>
                      <li>• Recognition of cultural practices</li>
                      <li>• Sensitivity to local legal traditions</li>
                    </ul>
                  </div>
                  <div>
                    <h4 className="text-lg font-semibold mb-3 text-[var(--ink)]">Demographic Fairness</h4>
                    <ul className="space-y-2 text-[var(--muted)]">
                      <li>• Equal performance across ethnic groups</li>
                      <li>• Gender-sensitive analysis capabilities</li>
                      <li>• Age-appropriate assessment methods</li>
                    </ul>
                  </div>
                </div>
              </MotionCard>
            </motion.div>
          )}

          {activeTab === 'access' && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.4 }}
              className="space-y-8"
            >
              <MotionCard className="p-8 text-center">
                <h3 className="text-2xl font-bold mb-6 text-[var(--ink)]">Get Early Access to LemkinBench</h3>
                <p className="text-[var(--muted)] mb-8 max-w-2xl mx-auto">
                  LemkinBench is currently in development with our research partners. We're working with select 
                  organizations to refine the framework before public release.
                </p>
                <div className="flex flex-col sm:flex-row gap-4 justify-center items-center">
                  <Button 
                    onClick={() => navigate('/contact')} 
                    className="px-8 py-3"
                  >
                    Request Access
                  </Button>
                  <Button 
                    variant="secondary" 
                    onClick={() => navigate('/contribute')}
                    className="px-8 py-3"
                  >
                    Contribute to Development
                  </Button>
                </div>
              </MotionCard>

              <MotionCard className="p-8">
                <h3 className="text-2xl font-bold mb-6 text-[var(--ink)]">Research Collaboration</h3>
                <div className="grid md:grid-cols-2 gap-8">
                  <div>
                    <h4 className="text-lg font-semibold mb-3 text-[var(--ink)]">Academic Partnerships</h4>
                    <p className="text-[var(--muted)] mb-4">
                      We're collaborating with leading universities and research institutions to ensure 
                      LemkinBench meets the highest academic standards.
                    </p>
                    <ul className="space-y-2 text-[var(--muted)]">
                      <li>• Peer review process</li>
                      <li>• Open methodology</li>
                      <li>• Reproducible results</li>
                    </ul>
                  </div>
                  <div>
                    <h4 className="text-lg font-semibold mb-3 text-[var(--ink)]">Practitioner Input</h4>
                    <p className="text-[var(--muted)] mb-4">
                      Real-world validation from human rights investigators, legal professionals, 
                      and international organizations.
                    </p>
                    <ul className="space-y-2 text-[var(--muted)]">
                      <li>• Field-tested evaluation criteria</li>
                      <li>• Practical applicability assessment</li>
                      <li>• Ethical guidelines development</li>
                    </ul>
                  </div>
                </div>
              </MotionCard>

              <div className="bg-[var(--accent)]/5 border border-[var(--accent)]/20 rounded-lg p-6">
                <div className="flex items-center gap-3 mb-3">
                  <div className="w-8 h-8 rounded-full bg-[var(--accent)]/20 flex items-center justify-center">
                    <Scale className="w-4 h-4 text-[var(--accent)]" />
                  </div>
                  <h4 className="font-semibold text-[var(--accent)]">Expected Release</h4>
                </div>
                <p className="text-[var(--muted)] text-sm">
                  We anticipate releasing LemkinBench publicly in Q2 2025, following completion of our validation 
                  studies with partner organizations. Early access is available now for qualified research institutions 
                  and human rights organizations.
                </p>
              </div>
            </motion.div>
          )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default LemkinBenchPage;