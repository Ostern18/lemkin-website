import React, { useState, useMemo, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Search, Filter, Github, Code2, Layers,
  Eye, Brain, FileText, Shield, Target, Scale,
  ChevronRight, Play, ArrowLeft,
  CheckCircle, AlertCircle, Database,
  Settings, Activity, TrendingUp, Users, Grid,
  Clock, MapPin, HardDrive, Video, Image, Headphones,
  Scan, BookOpen, MessageSquare, BarChart3, FileText as FileReport, Download, PenTool
} from 'lucide-react';
import { models, Model, modelCategories, getFeaturedModels } from './modelsData';

const ModelsPageRevised: React.FC = () => {
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedCategory, setSelectedCategory] = useState('All Models');
  const [selectedModel, setSelectedModel] = useState<Model | null>(null);
  const [activeSection, setActiveSection] = useState<'overview' | 'implementation' | 'technical'>('overview');
  const [currentCapabilityIndex, setCurrentCapabilityIndex] = useState(0);
  const [transitionState, setTransitionState] = useState<'entered' | 'exiting' | 'entering'>('entered');
  const [expandedCards, setExpandedCards] = useState<Set<string>>(new Set());
  const [isLoading, setIsLoading] = useState(true);

  // Filter models based on search and filters
  const filteredModels = useMemo(() => {
    return models.filter(model => {
      // Search filter
      if (searchQuery) {
        const query = searchQuery.toLowerCase();
        const matchesSearch =
          model.name.toLowerCase().includes(query) ||
          model.description.toLowerCase().includes(query) ||
          model.shortDescription.toLowerCase().includes(query) ||
          model.tags.some(tag => tag.toLowerCase().includes(query)) ||
          model.useCases.some(useCase => useCase.toLowerCase().includes(query));
        if (!matchesSearch) return false;
      }

      // Category filter
      if (selectedCategory !== 'All Models' && model.category !== selectedCategory) {
        return false;
      }

      return true;
    });
  }, [searchQuery, selectedCategory]);

  const featuredModels = getFeaturedModels();

  // Simulate loading for progressive data reveal
  useEffect(() => {
    const timer = setTimeout(() => setIsLoading(false), 800);
    return () => clearTimeout(timer);
  }, []);

  const toggleCardExpansion = (modelId: string) => {
    setExpandedCards(prev => {
      const newSet = new Set(prev);
      if (newSet.has(modelId)) {
        newSet.delete(modelId);
      } else {
        newSet.add(modelId);
      }
      return newSet;
    });
  };

  // Create rotating capability cards data
  const capabilityCards = useMemo(() => [
    {
      title: "Infrastructure Protection",
      description: "Identify hospitals, schools, and civilian facilities",
      icon: Shield,
      color: "text-[var(--accent)]"
    },
    {
      title: "Damage Assessment",
      description: "Analyze building damage from satellite imagery",
      icon: Target,
      color: "text-red-500"
    },
    {
      title: "Rights Monitoring",
      description: "Detect violations of international humanitarian law",
      icon: Scale,
      color: "text-[var(--accent)]"
    },
    {
      title: "Legal Analysis",
      description: "Extract entities from multilingual legal documents",
      icon: FileText,
      color: "text-green-500"
    },
    {
      title: "Document Generation",
      description: "Create professional legal narratives and reports",
      icon: Brain,
      color: "text-orange-500"
    },
    {
      title: "Real-time Processing",
      description: "Process satellite imagery and documents instantly",
      icon: Activity,
      color: "text-cyan-500"
    },
    {
      title: "Multi-language Support",
      description: "Analyze documents in English, French, Spanish, Arabic",
      icon: Users,
      color: "text-pink-500"
    },
    {
      title: "Production Ready",
      description: "Deploy models with comprehensive documentation",
      icon: Settings,
      color: "text-indigo-500"
    }
  ], []);

  // OpenAI-style professional capability rotation - longer intervals, subtle transitions
  useEffect(() => {
    const interval = setInterval(() => {
      // Start exit transition
      setTransitionState('exiting');

      setTimeout(() => {
        // Update content during opacity fade
        setCurrentCapabilityIndex((prevIndex) =>
          (prevIndex + 4) % capabilityCards.length
        );
        setTransitionState('entering');

        // Complete entrance
        setTimeout(() => {
          setTransitionState('entered');
        }, 50);
      }, 400); // Wait for exit transition
    }, 7000); // 7 seconds - professional, non-distracting interval

    return () => clearInterval(interval);
  }, [capabilityCards.length]);

  // Initial load animation
  useEffect(() => {
    const timer = setTimeout(() => {
      const cards = document.querySelectorAll('.capability-card');
      cards.forEach((card, index) => {
        setTimeout(() => {
          card.classList.add('visible');
        }, index * 80);
      });
    }, 100);

    return () => clearTimeout(timer);
  }, []);

  // Staggered card visibility for professional entrance effect
  useEffect(() => {
    if (transitionState === 'entering') {
      const cards = document.querySelectorAll('.capability-card');
      cards.forEach((card, index) => {
        setTimeout(() => {
          card.classList.add('visible');
        }, index * 80);
      });
    } else if (transitionState === 'exiting') {
      const cards = document.querySelectorAll('.capability-card');
      cards.forEach(card => {
        card.classList.remove('visible');
      });
    }
  }, [transitionState, currentCapabilityIndex]);

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'production':
        return 'bg-[var(--color-success)]/10 text-[var(--color-success)] border border-[var(--color-success)]/20';
      case 'development':
        return 'bg-[var(--color-warning)]/10 text-[var(--color-warning)] border border-[var(--color-warning)]/20';
      default:
        return 'bg-[var(--color-bg-secondary)] text-[var(--color-text-secondary)] border border-[var(--color-border-primary)]';
    }
  };

  const getSpecificToolIcon = (name: string) => {
    switch (name) {
      case 'Evidence Integrity Toolkit':
        return <Shield className="w-5 h-5 text-[var(--accent)]" />;
      case 'Automated Redaction Suite':
        return <Eye className="w-5 h-5 text-[var(--accent)]" />;
      case 'Document Classification Engine':
        return <FileText className="w-5 h-5 text-[var(--accent)]" />;
      case 'Named Entity Recognition Toolkit':
        return <Target className="w-5 h-5 text-[var(--accent)]" />;
      case 'Timeline Reconstruction Tool':
        return <Clock className="w-5 h-5 text-[var(--accent)]" />;
      case 'Legal Framework Analysis Tool':
        return <Scale className="w-5 h-5 text-[var(--accent)]" />;
      case 'Open Source Intelligence Collector':
        return <Search className="w-5 h-5 text-[var(--accent)]" />;
      case 'Geospatial Evidence Analyzer':
        return <MapPin className="w-5 h-5 text-[var(--accent)]" />;
      case 'Digital Forensics Toolkit':
        return <HardDrive className="w-5 h-5 text-[var(--accent)]" />;
      case 'Video Evidence Processor':
        return <Video className="w-5 h-5 text-[var(--accent)]" />;
      case 'Image Analysis Suite':
        return <Image className="w-5 h-5 text-[var(--accent)]" />;
      case 'Audio Processing Toolkit':
        return <Headphones className="w-5 h-5 text-[var(--accent)]" />;
      case 'Optical Character Recognition Engine':
        return <Scan className="w-5 h-5 text-[var(--accent)]" />;
      case 'Research Document Analyzer':
        return <BookOpen className="w-5 h-5 text-[var(--accent)]" />;
      case 'Communication Analysis Tool':
        return <MessageSquare className="w-5 h-5 text-[var(--accent)]" />;
      case 'Investigation Dashboard':
        return <BarChart3 className="w-5 h-5 text-[var(--accent)]" />;
      case 'Report Generation Suite':
        return <FileReport className="w-5 h-5 text-[var(--accent)]" />;
      case 'Evidence Export Manager':
        return <Download className="w-5 h-5 text-[var(--accent)]" />;
      case 'Public Interest Legal AI Prompts':
        return <PenTool className="w-5 h-5 text-[var(--accent)]" />;
      default:
        return <Settings className="w-5 h-5 text-[var(--accent)]" />;
    }
  };

  const getTypeIcon = (type: string, moduleType?: string, modelName?: string) => {
    if (moduleType === 'module' && modelName) {
      return getSpecificToolIcon(modelName);
    }

    switch (type) {
      case 'computer-vision':
        return <Eye className="w-5 h-5 text-[var(--color-primary)]" />;
      case 'nlp':
        return <FileText className="w-5 h-5 text-[var(--color-primary)]" />;
      case 'hybrid':
        return <Grid className="w-5 h-5 text-[var(--color-primary)]" />;
      case 'multimodal':
        return <Brain className="w-5 h-5 text-[var(--color-primary)]" />;
      default:
        return <Brain className="w-5 h-5 text-[var(--color-primary)]" />;
    }
  };

  const getCategoryIcon = (category: string) => {
    switch (category) {
      case 'Infrastructure Monitoring':
        return <Shield className="w-5 h-5" />;
      case 'Damage Assessment':
        return <Target className="w-5 h-5" />;
      case 'Rights Violations':
        return <Scale className="w-5 h-5" />;
      case 'Legal Analysis':
        return <FileText className="w-5 h-5" />;
      case 'Narrative Generation':
        return <Brain className="w-5 h-5" />;
      case 'Foundation & Safety':
        return <Shield className="w-5 h-5" />;
      case 'Core Analysis':
        return <Activity className="w-5 h-5" />;
      case 'Evidence Collection':
        return <Search className="w-5 h-5" />;
      case 'Media Analysis':
        return <Eye className="w-5 h-5" />;
      case 'Document Processing':
        return <FileText className="w-5 h-5" />;
      case 'Visualization & Reporting':
        return <TrendingUp className="w-5 h-5" />;
      default:
        return <Settings className="w-5 h-5" />;
    }
  };

  const getToolTypeDescription = (name: string, category: string) => {
    // Specific descriptions for each tool based on its name and function
    switch (name) {
      case 'Evidence Integrity Toolkit':
        return 'Cryptographic tool';
      case 'Automated Redaction Suite':
        return 'Privacy protection';
      case 'Document Classification Engine':
        return 'Classification system';
      case 'Named Entity Recognition Toolkit':
        return 'Entity extraction';
      case 'Timeline Reconstruction Tool':
        return 'Temporal analysis';
      case 'Legal Framework Analysis Tool':
        return 'Legal framework';
      case 'Open Source Intelligence Collector':
        return 'OSINT collection';
      case 'Geospatial Evidence Analyzer':
        return 'Geographic analysis';
      case 'Digital Forensics Toolkit':
        return 'Forensic analysis';
      case 'Video Evidence Processor':
        return 'Video processing';
      case 'Image Analysis Suite':
        return 'Image processing';
      case 'Audio Processing Toolkit':
        return 'Audio analysis';
      case 'Optical Character Recognition Engine':
        return 'Text extraction';
      case 'Research Document Analyzer':
        return 'Document research';
      case 'Communication Analysis Tool':
        return 'Communication data';
      case 'Investigation Dashboard':
        return 'Data visualization';
      case 'Report Generation Suite':
        return 'Report automation';
      case 'Evidence Export Manager':
        return 'Export utility';
      default:
        return 'Specialized tool';
    }
  };

  if (selectedModel) {
    return (
      <div className="min-h-screen bg-[var(--color-bg-primary)]">
        {/* Header */}
        <div className="bg-[var(--color-bg-surface)] border-b border-[var(--color-border-primary)]">
          <div className="max-w-6xl mx-auto px-6 py-8">
            <button
              onClick={() => setSelectedModel(null)}
              className="inline-flex items-center gap-2 text-[var(--color-text-secondary)] hover:text-[var(--color-text-primary)] transition-all duration-200 hover:gap-3 mb-6"
            >
              <ArrowLeft className="w-4 h-4" />
              <span className="text-sm font-medium">All Models</span>
            </button>

            {/* Model Header - Enhanced Professional Style */}
            <div className="flex items-start gap-6">
              <div className="flex-shrink-0 p-4 bg-gradient-to-br from-[var(--color-primary)]/10 to-[var(--color-primary)]/5 rounded-2xl border border-[var(--color-primary)]/20 shadow-sm">
                {getTypeIcon(selectedModel.type, selectedModel.moduleType, selectedModel.name)}
              </div>
              <div className="flex-1">
                <div className="flex items-center gap-3 mb-2">
                  <h1 className="text-4xl font-bold text-[var(--color-text-primary)] tracking-tight">
                    {selectedModel.name}
                  </h1>
                </div>
                <p className="text-xl text-[var(--color-text-secondary)] mb-4 leading-relaxed">
                  {selectedModel.publicSummary}
                </p>
                <div className="flex flex-wrap items-center gap-3">
                  <span className="px-4 py-2 bg-[var(--surface)] text-[var(--muted)] rounded-xl text-sm border border-[var(--line)] font-medium">
                    {selectedModel.category}
                  </span>
                </div>
              </div>

              {/* Quick Actions - Enhanced */}
              <div className="flex flex-col gap-3">
                {selectedModel.huggingFaceModel && (
                  <a href={`https://huggingface.co/${selectedModel.huggingFaceModel}`} target="_blank" rel="noopener noreferrer"
                     className="inline-flex items-center gap-2 px-6 py-3 bg-gradient-to-r from-[var(--color-primary)] to-[var(--color-primary)]/90 text-white rounded-xl hover:shadow-elevation-2 transition-all font-semibold group">
                    <Play className="w-4 h-4 group-hover:scale-110 transition-transform" />
                    Try Model
                  </a>
                )}
                <a href={selectedModel.githubRepo || `https://github.com/Lemkin-AI/${selectedModel.name.toLowerCase().replace(/\s+/g, '-')}`} target="_blank" rel="noopener noreferrer"
                   className="inline-flex items-center gap-2 px-6 py-3 border border-[var(--color-border-primary)] text-[var(--color-text-primary)] rounded-xl hover:bg-[var(--color-bg-secondary)] hover:shadow-sm transition-all font-semibold group">
                  <Github className="w-4 h-4 group-hover:scale-110 transition-transform" />
                  View Source
                </a>
              </div>
            </div>
          </div>
        </div>

        {/* Navigation Tabs - Enhanced Professional Style */}
        <div className="bg-[var(--color-bg-surface)] border-b border-[var(--color-border-primary)]">
          <div className="max-w-6xl mx-auto px-6">
            <nav className="flex">
              {[
                { id: 'overview', label: 'Overview', icon: Layers, description: 'Model capabilities and performance' },
                { id: 'implementation', label: 'Implementation', icon: Code2, description: 'Technical architecture and usage' },
                { id: 'technical', label: 'Specifications', icon: Settings, description: 'Detailed technical information' }
              ].map(({ id, label, icon: Icon, description }) => (
                <button
                  key={id}
                  onClick={() => setActiveSection(id as any)}
                  className={`flex-1 relative group px-6 py-6 text-left transition-all duration-200 ${
                    activeSection === id
                      ? 'bg-[var(--color-bg-elevated)] border-b-2 border-[var(--color-primary)]'
                      : 'border-b-2 border-transparent hover:bg-[var(--color-bg-secondary)]/50'
                  }`}
                >
                  <div className="flex items-start gap-3">
                    <div className={`p-2 rounded-lg transition-colors ${
                      activeSection === id
                        ? 'bg-[var(--color-primary)]/10 text-[var(--color-primary)]'
                        : 'bg-[var(--color-bg-secondary)] text-[var(--color-text-secondary)] group-hover:bg-[var(--color-primary)]/5 group-hover:text-[var(--color-primary)]'
                    }`}>
                      <Icon className="w-5 h-5" />
                    </div>
                    <div className="flex-1">
                      <div className={`font-semibold text-sm transition-colors ${
                        activeSection === id
                          ? 'text-[var(--color-primary)]'
                          : 'text-[var(--color-text-primary)] group-hover:text-[var(--color-primary)]'
                      }`}>
                        {label}
                      </div>
                      <div className="text-xs text-[var(--color-text-tertiary)] mt-1">
                        {description}
                      </div>
                    </div>
                  </div>
                </button>
              ))}
            </nav>
          </div>
        </div>

        {/* Content Sections */}
        <div className="max-w-6xl mx-auto px-6 py-12">
          {activeSection === 'overview' && (
            <div className="space-y-12">
              {/* Purpose & Description - Primary Focus */}
              <div className="max-w-4xl">
                <div className="mb-8">
                  <h2 className="text-3xl font-bold text-[var(--ink)] mb-6">
                    {selectedModel.moduleType === 'module' ? 'What this tool does' : 'What this model does'}
                  </h2>
                  <p className="text-xl text-[var(--muted)] leading-relaxed mb-6">
                    {selectedModel.description}
                  </p>
                  <p className="text-lg text-[var(--subtle)] leading-relaxed">
                    {selectedModel.publicSummary}
                  </p>
                </div>
              </div>

              {/* Core Capabilities - Secondary Focus */}
              <div className="bg-gradient-to-br from-[var(--surface)] to-[var(--elevated)] border border-[var(--line)] rounded-3xl p-8 shadow-elevation-1">
                <div className="flex items-center gap-3 mb-8">
                  <Target className="w-6 h-6 text-[var(--accent)]" />
                  <h2 className="text-2xl font-bold text-[var(--ink)]">Core Capabilities</h2>
                </div>
                <div className="grid md:grid-cols-2 gap-6">
                  {selectedModel.capabilities.map((capability, capIndex) => (
                    <div key={capIndex} className="bg-[var(--bg)] border border-[var(--line)] rounded-2xl p-6">
                      <h3 className="text-lg font-semibold text-[var(--ink)] mb-4">{capability.category}</h3>
                      <div className="space-y-3">
                        {capability.items.map((item, index) => (
                          <div key={index} className="flex items-start gap-3">
                            <div className="w-2 h-2 bg-[var(--accent)] rounded-full mt-2 flex-shrink-0"></div>
                            <span className="text-[var(--muted)] leading-relaxed">{item}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Real-World Impact */}
              {selectedModel.realWorldImpact && selectedModel.realWorldImpact.length > 0 && (
                <div className="bg-gradient-to-br from-[var(--surface)] to-[var(--elevated)] border border-[var(--line)] rounded-3xl p-8 shadow-elevation-1">
                  <div className="flex items-center gap-3 mb-8">
                    <Scale className="w-6 h-6 text-[var(--accent)]" />
                    <h2 className="text-2xl font-bold text-[var(--ink)]">Real-World Impact</h2>
                  </div>
                  <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
                    {selectedModel.realWorldImpact.map((impact, index) => (
                      <div key={index} className="bg-[var(--bg)] border border-[var(--line)] rounded-2xl p-6">
                        <h3 className="text-lg font-semibold text-[var(--ink)] mb-4">{impact.domain}</h3>
                        <div className="space-y-3">
                          {impact.examples.map((example, exIndex) => (
                            <div key={exIndex} className="flex items-start gap-3">
                              <CheckCircle className="w-4 h-4 text-[var(--success)] mt-1 flex-shrink-0" />
                              <span className="text-[var(--muted)] text-sm leading-relaxed">{example}</span>
                            </div>
                          ))}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Performance Metrics - Tertiary Focus */}
              <div className="bg-gradient-to-br from-[var(--surface)] to-[var(--elevated)] border border-[var(--line)] rounded-3xl p-8 shadow-elevation-1">
                <div className="flex items-center gap-3 mb-8">
                  <Activity className="w-6 h-6 text-[var(--accent)]" />
                  <h2 className="text-2xl font-bold text-[var(--ink)]">
                    {selectedModel.moduleType === 'module' ? 'Tool Performance' : 'Model Performance'}
                  </h2>
                </div>
                <div className="grid md:grid-cols-3 gap-6">
                  {selectedModel.metrics.accuracy && (
                    <div className="bg-[var(--bg)] border border-[var(--line)] rounded-2xl p-6 hover:shadow-elevation-1 transition-all">
                      <div className="text-2xl font-bold text-[var(--accent)] mb-2">{selectedModel.metrics.accuracy}</div>
                      <div className="text-sm font-medium text-[var(--ink)]">Accuracy</div>
                      <div className="text-xs text-[var(--subtle)] mt-1">Production validated</div>
                    </div>
                  )}
                  {selectedModel.metrics.f1Score && (
                    <div className="bg-[var(--bg)] border border-[var(--line)] rounded-2xl p-6 hover:shadow-elevation-1 transition-all">
                      <div className="text-2xl font-bold text-[var(--accent)] mb-2">{selectedModel.metrics.f1Score.split(',')[0]}</div>
                      <div className="text-sm font-medium text-[var(--ink)]">F1 Score</div>
                      <div className="text-xs text-[var(--subtle)] mt-1">Precision & Recall</div>
                    </div>
                  )}
                  {selectedModel.metrics.inferenceSpeed && (
                    <div className="bg-[var(--bg)] border border-[var(--line)] rounded-2xl p-6 hover:shadow-elevation-1 transition-all">
                      <div className="text-2xl font-bold text-[var(--accent)] mb-2">{selectedModel.metrics.inferenceSpeed}</div>
                      <div className="text-sm font-medium text-[var(--ink)]">Processing Speed</div>
                      <div className="text-xs text-[var(--subtle)] mt-1">Optimized performance</div>
                    </div>
                  )}
                  {/* Only show model size for actual AI models, not tools */}
                  {selectedModel.moduleType === 'model' && selectedModel.metrics.modelSize && (
                    <div className="bg-[var(--bg)] border border-[var(--line)] rounded-2xl p-6 hover:shadow-elevation-1 transition-all">
                      <div className="text-2xl font-bold text-[var(--accent)] mb-2">{selectedModel.metrics.modelSize}</div>
                      <div className="text-sm font-medium text-[var(--ink)]">Model Size</div>
                      <div className="text-xs text-[var(--subtle)] mt-1">Optimized for deployment</div>
                    </div>
                  )}
                  {/* Show additional relevant metrics for tools */}
                  {selectedModel.moduleType === 'module' && (
                    <>
                      {selectedModel.metrics.coverage && (
                        <div className="bg-[var(--bg)] border border-[var(--line)] rounded-2xl p-6 hover:shadow-elevation-1 transition-all">
                          <div className="text-2xl font-bold text-[var(--accent)] mb-2">{selectedModel.metrics.coverage}</div>
                          <div className="text-sm font-medium text-[var(--ink)]">Coverage</div>
                          <div className="text-xs text-[var(--subtle)] mt-1">Comprehensive support</div>
                        </div>
                      )}
                      {selectedModel.metrics.languages && (
                        <div className="bg-[var(--bg)] border border-[var(--line)] rounded-2xl p-6 hover:shadow-elevation-1 transition-all">
                          <div className="text-2xl font-bold text-[var(--accent)] mb-2">{selectedModel.metrics.languages}</div>
                          <div className="text-sm font-medium text-[var(--ink)]">Languages</div>
                          <div className="text-xs text-[var(--subtle)] mt-1">Multilingual support</div>
                        </div>
                      )}
                      {selectedModel.metrics.formats && (
                        <div className="bg-[var(--bg)] border border-[var(--line)] rounded-2xl p-6 hover:shadow-elevation-1 transition-all">
                          <div className="text-2xl font-bold text-[var(--accent)] mb-2">{selectedModel.metrics.formats}</div>
                          <div className="text-sm font-medium text-[var(--ink)]">Formats</div>
                          <div className="text-xs text-[var(--subtle)] mt-1">Universal compatibility</div>
                        </div>
                      )}
                      {selectedModel.metrics.sources && (
                        <div className="bg-[var(--bg)] border border-[var(--line)] rounded-2xl p-6 hover:shadow-elevation-1 transition-all">
                          <div className="text-2xl font-bold text-[var(--accent)] mb-2">{selectedModel.metrics.sources}</div>
                          <div className="text-sm font-medium text-[var(--ink)]">Sources</div>
                          <div className="text-xs text-[var(--subtle)] mt-1">Comprehensive reach</div>
                        </div>
                      )}
                      {selectedModel.metrics.entities && (
                        <div className="bg-[var(--bg)] border border-[var(--line)] rounded-2xl p-6 hover:shadow-elevation-1 transition-all">
                          <div className="text-2xl font-bold text-[var(--accent)] mb-2">{selectedModel.metrics.entities}</div>
                          <div className="text-sm font-medium text-[var(--ink)]">Entity Types</div>
                          <div className="text-xs text-[var(--subtle)] mt-1">Specialized recognition</div>
                        </div>
                      )}
                      {selectedModel.metrics.precision && (
                        <div className="bg-[var(--bg)] border border-[var(--line)] rounded-2xl p-6 hover:shadow-elevation-1 transition-all">
                          <div className="text-2xl font-bold text-[var(--accent)] mb-2">{selectedModel.metrics.precision}</div>
                          <div className="text-sm font-medium text-[var(--ink)]">Precision</div>
                          <div className="text-xs text-[var(--subtle)] mt-1">High accuracy timing</div>
                        </div>
                      )}
                      {selectedModel.metrics.compatibility && (
                        <div className="bg-[var(--bg)] border border-[var(--line)] rounded-2xl p-6 hover:shadow-elevation-1 transition-all">
                          <div className="text-2xl font-bold text-[var(--accent)] mb-2">{selectedModel.metrics.compatibility}</div>
                          <div className="text-sm font-medium text-[var(--ink)]">Compatibility</div>
                          <div className="text-xs text-[var(--subtle)] mt-1">Broad system support</div>
                        </div>
                      )}
                      {selectedModel.metrics.protection && (
                        <div className="bg-[var(--bg)] border border-[var(--line)] rounded-2xl p-6 hover:shadow-elevation-1 transition-all">
                          <div className="text-2xl font-bold text-[var(--accent)] mb-2">{selectedModel.metrics.protection}</div>
                          <div className="text-sm font-medium text-[var(--ink)]">Privacy Protection</div>
                          <div className="text-xs text-[var(--subtle)] mt-1">Compliance standards</div>
                        </div>
                      )}
                    </>
                  )}
                </div>
              </div>

              {/* Model Description */}
              <div className="grid lg:grid-cols-3 gap-8">
                <div className="lg:col-span-2 space-y-8">
                  <div className="bg-[var(--color-bg-surface)] border border-[var(--color-border-primary)] rounded-2xl p-8">
                    <div className="flex items-center gap-3 mb-6">
                      <Brain className="w-6 h-6 text-[var(--color-primary)]" />
                      <h2 className="text-2xl font-bold text-[var(--color-text-primary)]">
                        Overview
                      </h2>
                    </div>
                    <p className="text-[var(--color-text-secondary)] leading-relaxed text-lg mb-8">
                      {selectedModel.description}
                    </p>

                    {/* Real World Impact */}
                    {selectedModel.realWorldImpact && selectedModel.realWorldImpact.length > 0 && (
                      <div className="mb-8">
                        <h3 className="text-lg font-semibold text-[var(--color-text-primary)] mb-4 flex items-center gap-2">
                          <Scale className="w-5 h-5 text-[var(--color-primary)]" />
                          Real-World Impact
                        </h3>
                        <div className="grid gap-4">
                          {selectedModel.realWorldImpact.map((impact, index) => (
                            <div key={index} className="p-4 bg-[var(--color-bg-elevated)] border border-[var(--color-border-primary)] rounded-xl">
                              <h4 className="font-semibold text-[var(--color-text-primary)] mb-2">{impact.domain}</h4>
                              <ul className="text-sm text-[var(--color-text-secondary)] space-y-1">
                                {impact.examples.map((example, exIndex) => (
                                  <li key={exIndex} className="flex items-start gap-2">
                                    <span className="text-[var(--color-primary)] mt-1">•</span>
                                    {example}
                                  </li>
                                ))}
                              </ul>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}

                    {/* Key Capabilities - Enhanced Grid */}
                    <h3 className="text-lg font-semibold text-[var(--color-text-primary)] mb-6 flex items-center gap-2">
                      <Target className="w-5 h-5 text-[var(--color-primary)]" />
                      Core Capabilities
                    </h3>
                    <div className="grid gap-3">
                      {selectedModel.capabilities[0]?.items.slice(0, 6).map((item, index) => (
                        <div key={index} className="flex items-center gap-4 p-4 bg-[var(--color-bg-elevated)] border border-[var(--color-border-primary)] rounded-xl hover:shadow-sm transition-all group">
                          <div className="w-2 h-2 bg-[var(--color-primary)] rounded-full group-hover:scale-125 transition-transform"></div>
                          <span className="text-[var(--color-text-secondary)] flex-1">{item}</span>
                          <CheckCircle className="w-4 h-4 text-[var(--color-success)] opacity-60 group-hover:opacity-100 transition-opacity" />
                        </div>
                      ))}
                    </div>
                  </div>
                </div>

                {/* Quick Stats Sidebar */}
                <div className="space-y-6">
                  <div className="bg-[var(--color-bg-surface)] border border-[var(--color-border-primary)] rounded-2xl p-6">
                    <h3 className="text-lg font-semibold text-[var(--color-text-primary)] mb-4 flex items-center gap-2">
                      <TrendingUp className="w-5 h-5 text-[var(--color-primary)]" />
                      Quick Stats
                    </h3>
                    <div className="space-y-4">
                      <div className="flex justify-between items-center pb-3 border-b border-[var(--color-border-primary)]">
                        <span className="text-sm text-[var(--color-text-secondary)]">Category</span>
                        <span className="text-sm font-medium text-[var(--color-text-primary)]">{selectedModel.category}</span>
                      </div>
                      <div className="flex justify-between items-center pb-3 border-b border-[var(--color-border-primary)]">
                        <span className="text-sm text-[var(--color-text-secondary)]">Framework</span>
                        <span className="text-sm font-medium text-[var(--color-text-primary)] font-mono">{selectedModel.technicalSpecs.framework.split(' ')[0]}</span>
                      </div>
                      <div className="flex justify-between items-center pb-3 border-b border-[var(--color-border-primary)]">
                        <span className="text-sm text-[var(--color-text-secondary)]">Use Cases</span>
                        <span className="text-sm font-medium text-[var(--color-primary)]">{selectedModel.useCases.length}</span>
                      </div>
                      {selectedModel.tier && (
                        <div className="flex justify-between items-center">
                          <span className="text-sm text-[var(--color-text-secondary)]">Tier Level</span>
                          <span className="text-sm font-medium text-[var(--color-primary)]">{selectedModel.tier}</span>
                        </div>
                      )}
                    </div>
                  </div>

                  {/* Action Buttons */}
                  <div className="bg-[var(--color-bg-surface)] border border-[var(--color-border-primary)] rounded-2xl p-6">
                    <h3 className="text-lg font-semibold text-[var(--color-text-primary)] mb-4">Get Started</h3>
                    <div className="space-y-3">
                      {selectedModel.huggingFaceModel && (
                        <a href={`https://huggingface.co/${selectedModel.huggingFaceModel}`} target="_blank" rel="noopener noreferrer"
                           className="w-full inline-flex items-center justify-center gap-2 px-4 py-3 bg-[var(--color-primary)] text-white rounded-xl hover:bg-[var(--color-primary)]/90 transition-all font-medium group">
                          <Play className="w-4 h-4 group-hover:scale-110 transition-transform" />
                          Try Model
                        </a>
                      )}
                      <a href={selectedModel.githubRepo || `https://github.com/Lemkin-AI/${selectedModel.name.toLowerCase().replace(/\s+/g, '-')}`} target="_blank" rel="noopener noreferrer"
                         className="w-full inline-flex items-center justify-center gap-2 px-4 py-3 border border-[var(--color-border-primary)] text-[var(--color-text-primary)] rounded-xl hover:bg-[var(--color-bg-secondary)] transition-all font-medium group">
                        <Github className="w-4 h-4 group-hover:scale-110 transition-transform" />
                        View Source
                      </a>
                      <a href="/docs" className="w-full inline-flex items-center justify-center gap-2 px-4 py-3 border border-[var(--color-border-primary)] text-[var(--color-text-primary)] rounded-xl hover:bg-[var(--color-bg-secondary)] transition-all font-medium group">
                        <FileText className="w-4 h-4 group-hover:scale-110 transition-transform" />
                        Documentation
                      </a>
                    </div>
                  </div>
                </div>
              </div>

              {/* Use Cases - Enhanced Display */}
              <div className="bg-[var(--color-bg-surface)] border border-[var(--color-border-primary)] rounded-2xl p-8">
                <div className="flex items-center gap-3 mb-8">
                  <Users className="w-6 h-6 text-[var(--color-primary)]" />
                  <h2 className="text-2xl font-bold text-[var(--color-text-primary)]">Application Areas</h2>
                </div>
                <div className="grid md:grid-cols-2 gap-6">
                  {selectedModel.useCases.map((useCase, index) => (
                    <div key={index} className="group relative overflow-hidden bg-gradient-to-br from-[var(--color-bg-elevated)] to-[var(--color-bg-secondary)] border border-[var(--color-border-primary)] rounded-xl p-6 hover:shadow-elevation-2 transition-all duration-300">
                      <div className="absolute top-0 right-0 w-16 h-16 bg-gradient-to-br from-[var(--color-primary)]/5 to-transparent rounded-bl-full"></div>
                      <div className="flex items-start gap-4">
                        <div className="p-2 bg-[var(--color-primary)]/10 rounded-lg group-hover:bg-[var(--color-primary)]/20 transition-colors">
                          <Target className="w-5 h-5 text-[var(--color-primary)]" />
                        </div>
                        <div className="flex-1">
                          <span className="text-[var(--color-text-primary)] font-medium leading-relaxed">{useCase}</span>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}

          {activeSection === 'implementation' && selectedModel.howItWorks && (
            <div className="space-y-10">
              {/* Architecture Overview */}
              <div className="bg-gradient-to-br from-[var(--color-bg-surface)] to-[var(--color-bg-elevated)] border border-[var(--color-border-primary)] rounded-3xl p-8 shadow-elevation-1">
                <div className="flex items-center gap-3 mb-6">
                  <Layers className="w-6 h-6 text-[var(--color-primary)]" />
                  <h2 className="text-2xl font-bold text-[var(--color-text-primary)]">Architecture & Implementation</h2>
                </div>
                <p className="text-[var(--color-text-secondary)] leading-relaxed text-lg mb-8">
                  {selectedModel.howItWorks.overview}
                </p>
                <div className="grid md:grid-cols-2 gap-6">
                  <div className="bg-[var(--color-bg-primary)] border border-[var(--color-border-primary)] rounded-xl p-6">
                    <h3 className="font-semibold text-[var(--color-text-primary)] mb-3 flex items-center gap-2">
                      <Code2 className="w-5 h-5 text-[var(--color-primary)]" />
                      Architecture
                    </h3>
                    <p className="text-sm font-mono text-[var(--color-text-secondary)] bg-[var(--color-bg-secondary)] px-3 py-2 rounded">
                      {selectedModel.technicalSpecs.architecture}
                    </p>
                  </div>
                  <div className="bg-[var(--color-bg-primary)] border border-[var(--color-border-primary)] rounded-xl p-6">
                    <h3 className="font-semibold text-[var(--color-text-primary)] mb-3 flex items-center gap-2">
                      <Settings className="w-5 h-5 text-[var(--color-primary)]" />
                      Framework
                    </h3>
                    <p className="text-sm font-mono text-[var(--color-text-secondary)] bg-[var(--color-bg-secondary)] px-3 py-2 rounded">
                      {selectedModel.technicalSpecs.framework}
                    </p>
                  </div>
                </div>
              </div>

              {/* Implementation Pipeline */}
              <div className="space-y-8">
                <div className="flex items-center gap-3 mb-2">
                  <Activity className="w-6 h-6 text-[var(--color-primary)]" />
                  <h3 className="text-2xl font-bold text-[var(--color-text-primary)]">Processing Pipeline</h3>
                </div>
                <div className="relative">
                  {/* Connection Lines */}
                  <div className="absolute left-8 top-16 bottom-0 w-0.5 bg-gradient-to-b from-[var(--color-primary)] to-[var(--color-primary)]/20 hidden lg:block"></div>

                  <div className="space-y-6">
                    {selectedModel.howItWorks.steps.map((step, index) => (
                      <div key={index} className="relative flex gap-6 group">
                        <div className="flex-shrink-0 relative z-10">
                          <div className="w-16 h-16 bg-gradient-to-br from-[var(--color-primary)] to-[var(--color-primary)]/80 text-white rounded-2xl flex items-center justify-center font-bold text-lg shadow-lg group-hover:scale-110 transition-transform">
                            {index + 1}
                          </div>
                        </div>
                        <div className="flex-1 bg-[var(--color-bg-surface)] border border-[var(--color-border-primary)] rounded-2xl p-8 group-hover:shadow-elevation-2 transition-all">
                          <h4 className="text-xl font-bold text-[var(--color-text-primary)] mb-3 group-hover:text-[var(--color-primary)] transition-colors">{step.title}</h4>
                          <p className="text-[var(--color-text-secondary)] leading-relaxed text-lg">{step.description}</p>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>

              {/* Technical Deep Dive */}
              <div className="bg-gradient-to-br from-[var(--color-bg-surface)] to-[var(--color-bg-elevated)] border border-[var(--color-border-primary)] rounded-2xl p-8">
                <div className="flex items-center gap-3 mb-6">
                  <Database className="w-6 h-6 text-[var(--color-primary)]" />
                  <h3 className="text-2xl font-bold text-[var(--color-text-primary)]">Technical Implementation</h3>
                </div>
                <div className="bg-[var(--color-bg-primary)] border border-[var(--color-border-primary)] rounded-xl p-6">
                  <p className="text-[var(--color-text-secondary)] leading-relaxed text-lg">
                    {selectedModel.howItWorks.technicalDetails}
                  </p>
                </div>
              </div>
            </div>
          )}

          {activeSection === 'technical' && (
            <div className="space-y-10">
              {/* Technical Specifications - Enhanced */}
              <div className="grid lg:grid-cols-2 gap-8">
                <div className="bg-gradient-to-br from-[var(--color-bg-surface)] to-[var(--color-bg-elevated)] border border-[var(--color-border-primary)] rounded-3xl p-8 shadow-elevation-1">
                  <div className="flex items-center gap-3 mb-8">
                    <Settings className="w-6 h-6 text-[var(--color-primary)]" />
                    <h3 className="text-2xl font-bold text-[var(--color-text-primary)]">Technical Specifications</h3>
                  </div>
                  <div className="space-y-6">
                    {[
                      { label: 'Framework', value: selectedModel.technicalSpecs.framework },
                      { label: 'Architecture', value: selectedModel.technicalSpecs.architecture },
                      { label: 'Input Format', value: selectedModel.technicalSpecs.inputFormat },
                      { label: 'Output Format', value: selectedModel.technicalSpecs.outputFormat }
                    ].map((spec, index) => (
                      <div key={index} className="bg-[var(--color-bg-primary)] border border-[var(--color-border-primary)] rounded-xl p-6 hover:shadow-sm transition-all">
                        <div className="flex items-center justify-between mb-2">
                          <span className="text-[var(--color-text-secondary)] text-sm font-semibold uppercase tracking-wide">{spec.label}</span>
                          <div className="w-2 h-2 bg-[var(--color-primary)] rounded-full"></div>
                        </div>
                        <p className="text-[var(--color-text-primary)] font-mono text-sm bg-[var(--color-bg-secondary)] px-3 py-2 rounded">{spec.value}</p>
                      </div>
                    ))}
                  </div>
                </div>

                <div className="bg-gradient-to-br from-[var(--color-bg-surface)] to-[var(--color-bg-elevated)] border border-[var(--color-border-primary)] rounded-3xl p-8 shadow-elevation-1">
                  <div className="flex items-center gap-3 mb-8">
                    <Activity className="w-6 h-6 text-[var(--color-primary)]" />
                    <h3 className="text-2xl font-bold text-[var(--color-text-primary)]">Performance Metrics</h3>
                  </div>
                  <div className="space-y-4">
                    {Object.entries(selectedModel.metrics).map(([key, value], index) => (
                      <div key={key} className="flex justify-between items-center p-4 bg-[var(--color-bg-primary)] border border-[var(--color-border-primary)] rounded-xl hover:shadow-sm transition-all group">
                        <span className="text-[var(--color-text-secondary)] font-medium capitalize flex items-center gap-2">
                          <div className="w-2 h-2 bg-[var(--color-primary)] rounded-full group-hover:scale-125 transition-transform"></div>
                          {key.replace(/([A-Z])/g, ' $1').trim()}
                        </span>
                        <span className="font-bold text-[var(--color-text-primary)] text-lg">{value}</span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>

              {/* Limitations and Ethical Considerations - Enhanced */}
              {selectedModel.limitations && (
                <div className="bg-gradient-to-br from-[var(--color-bg-surface)] to-[var(--color-bg-elevated)] border border-[var(--color-border-primary)] rounded-3xl p-8 shadow-elevation-1">
                  <div className="flex items-center gap-3 mb-8">
                    <Shield className="w-6 h-6 text-[var(--color-primary)]" />
                    <h3 className="text-2xl font-bold text-[var(--color-text-primary)]">Limitations & Ethical Framework</h3>
                  </div>
                  <div className="grid lg:grid-cols-2 gap-8">
                    <div className="bg-[var(--color-bg-primary)] border border-[var(--color-border-primary)] rounded-2xl p-6">
                      <h4 className="text-lg font-bold text-[var(--color-text-primary)] mb-6 flex items-center gap-3">
                        <div className="p-2 bg-[var(--color-warning)]/10 rounded-lg">
                          <AlertCircle className="w-5 h-5 text-[var(--color-warning)]" />
                        </div>
                        Known Limitations
                      </h4>
                      <div className="space-y-4">
                        {selectedModel.limitations.map((limitation, index) => (
                          <div key={index} className="flex items-start gap-4 p-4 bg-[var(--color-bg-secondary)] rounded-xl border border-[var(--color-border-primary)] hover:shadow-sm transition-all">
                            <div className="w-1.5 h-1.5 bg-[var(--color-warning)] rounded-full mt-2 flex-shrink-0" />
                            <span className="text-[var(--color-text-secondary)] text-sm leading-relaxed">{limitation}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                    {selectedModel.ethicalConsiderations && (
                      <div className="bg-[var(--color-bg-primary)] border border-[var(--color-border-primary)] rounded-2xl p-6">
                        <h4 className="text-lg font-bold text-[var(--color-text-primary)] mb-6 flex items-center gap-3">
                          <div className="p-2 bg-[var(--color-success)]/10 rounded-lg">
                            <CheckCircle className="w-5 h-5 text-[var(--color-success)]" />
                          </div>
                          Ethical Safeguards
                        </h4>
                        <div className="space-y-4">
                          {selectedModel.ethicalConsiderations.map((consideration, index) => (
                            <div key={index} className="flex items-start gap-4 p-4 bg-[var(--color-bg-secondary)] rounded-xl border border-[var(--color-border-primary)] hover:shadow-sm transition-all">
                              <div className="w-1.5 h-1.5 bg-[var(--color-success)] rounded-full mt-2 flex-shrink-0" />
                              <span className="text-[var(--color-text-secondary)] text-sm leading-relaxed">{consideration}</span>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              )}

              {/* Installation & Deployment - Enhanced */}
              <div className="bg-gradient-to-br from-[var(--color-bg-surface)] to-[var(--color-bg-elevated)] border border-[var(--color-border-primary)] rounded-3xl p-8 shadow-elevation-1">
                <div className="flex items-center gap-3 mb-8">
                  <Code2 className="w-6 h-6 text-[var(--color-primary)]" />
                  <h3 className="text-2xl font-bold text-[var(--color-text-primary)]">Installation & Deployment</h3>
                </div>
                <div className="grid lg:grid-cols-2 gap-8">
                  {/* Installation */}
                  <div className="bg-[var(--color-bg-primary)] border border-[var(--color-border-primary)] rounded-2xl p-6">
                    <h4 className="font-bold text-[var(--color-text-primary)] mb-4 flex items-center gap-2">
                      <Github className="w-5 h-5 text-[var(--color-primary)]" />
                      Quick Start
                    </h4>
                    <div className="space-y-3 mb-4">
                      <div className="bg-[var(--color-bg-secondary)] border border-[var(--color-border-primary)] rounded-lg p-4">
                        <code className="text-sm text-[var(--color-text-secondary)]">
                          git clone {selectedModel.githubRepo || `https://github.com/Lemkin-AI/${selectedModel.name.toLowerCase().replace(/\s+/g, '-')}`}
                        </code>
                      </div>
                      <div className="bg-[var(--color-bg-secondary)] border border-[var(--color-border-primary)] rounded-lg p-4">
                        <code className="text-sm text-[var(--color-text-secondary)]">
                          cd {selectedModel.name.toLowerCase().replace(/\s+/g, '-')}
                        </code>
                      </div>
                      <div className="bg-[var(--color-bg-secondary)] border border-[var(--color-border-primary)] rounded-lg p-4">
                        <code className="text-sm text-[var(--color-text-secondary)]">
                          pip install -r requirements.txt
                        </code>
                      </div>
                    </div>
                    <a href={selectedModel.githubRepo || `https://github.com/Lemkin-AI/${selectedModel.name.toLowerCase().replace(/\s+/g, '-')}`} target="_blank" rel="noopener noreferrer"
                       className="inline-flex items-center gap-2 px-4 py-2 bg-[var(--color-primary)] text-white rounded-lg hover:bg-[var(--color-primary)]/90 transition-all font-medium text-sm">
                      <Github className="w-4 h-4" />
                      View Full Setup Guide
                    </a>
                  </div>

                  {/* Requirements */}
                  <div className="bg-[var(--color-bg-primary)] border border-[var(--color-border-primary)] rounded-2xl p-6">
                    <h4 className="font-bold text-[var(--color-text-primary)] mb-4 flex items-center gap-2">
                      <Settings className="w-5 h-5 text-[var(--color-primary)]" />
                      System Requirements
                    </h4>
                    <div className="space-y-3">
                      {selectedModel.deployment.requirements.map((req, index) => (
                        <div key={index} className="flex items-center gap-3 p-3 bg-[var(--color-bg-secondary)] rounded-lg border border-[var(--color-border-primary)] hover:shadow-sm transition-all">
                          <CheckCircle className="w-4 h-4 text-[var(--color-success)]" />
                          <span className="text-[var(--color-text-secondary)] text-sm">{req}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-[var(--color-bg-primary)]">
      {/* Hero Section - Enhanced Professional */}
      <section className="bg-gradient-to-b from-[var(--color-bg-surface)] to-[var(--color-bg-primary)] border-b border-[var(--color-border-primary)]">
        <div className="max-w-6xl mx-auto px-6 py-20">
          <div className="text-center max-w-4xl mx-auto">
            <h1 className="text-6xl font-bold text-[var(--color-text-primary)] mb-6 leading-[1.1] tracking-tight">
              AI Models & Tools for Justice
            </h1>
            <p className="text-xl text-[var(--color-text-secondary)] leading-relaxed max-w-3xl mx-auto mb-8">
              Production-ready AI models and technical tools specifically designed for human rights monitoring, legal analysis,
              and humanitarian applications. Built with transparency, accuracy, and ethical deployment in mind.
            </p>

            {/* Professional Capability Cards - OpenAI Style */}
            <div className="capability-cards-container mb-12">
              <div className={`capability-cards-set ${transitionState} grid grid-cols-2 md:grid-cols-4 gap-6`}>
                {capabilityCards.slice(currentCapabilityIndex, currentCapabilityIndex + 4).map((capability, index) => {
                  const IconComponent = capability.icon;
                  return (
                    <div
                      key={`${currentCapabilityIndex}-${index}`}
                      className={`capability-card text-center p-6 bg-[var(--color-bg-surface)] border border-[var(--color-border-primary)] rounded-2xl group ${
                        transitionState === 'entered' ? 'visible' : ''
                      }`}
                    >
                      <div className="flex justify-center mb-4">
                        <div className="p-3 rounded-full bg-gradient-to-br from-[var(--color-primary)]/8 to-[var(--color-primary)]/4 group-hover:from-[var(--color-primary)]/12 group-hover:to-[var(--color-primary)]/6 transition-all duration-300">
                          <IconComponent className={`w-5 h-5 ${capability.color} opacity-80 group-hover:opacity-100 transition-opacity`} />
                        </div>
                      </div>
                      <div className="text-base font-semibold text-[var(--color-text-primary)] mb-2 group-hover:text-[var(--color-primary)] transition-colors">
                        {capability.title}
                      </div>
                      <div className="text-sm text-[var(--color-text-secondary)] leading-relaxed">
                        {capability.description}
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>

            {/* Enhanced Search */}
            <motion.div
              className="max-w-2xl mx-auto"
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ duration: 0.4, delay: 0.2, ease: [0.16, 1, 0.3, 1] }}
            >
              <motion.div
                className="relative"
                whileHover={{ scale: 1.01 }}
                transition={{ duration: 0.2, ease: [0.16, 1, 0.3, 1] }}
              >
                <Search className="absolute left-4 top-1/2 transform -translate-y-1/2 text-[var(--subtle)] w-5 h-5 transition-colors" />
                <input
                  type="text"
                  placeholder="Search models by name, capabilities, or use cases..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="w-full pl-12 pr-12 py-4 border border-[var(--line)] rounded-2xl bg-[var(--elevated)] text-[var(--ink)] placeholder-[var(--subtle)] focus:outline-none focus:ring-2 focus:ring-[var(--accent)]/50 focus:border-[var(--accent)] transition-all text-lg shadow-sm"
                />
                {searchQuery && (
                  <motion.button
                    initial={{ opacity: 0, scale: 0.8 }}
                    animate={{ opacity: 1, scale: 1 }}
                    whileHover={{ scale: 1.1 }}
                    whileTap={{ scale: 0.9 }}
                    onClick={() => setSearchQuery('')}
                    className="absolute right-4 top-1/2 transform -translate-y-1/2 w-6 h-6 text-[var(--subtle)] hover:text-[var(--accent)] transition-colors flex items-center justify-center rounded-full hover:bg-[var(--surface)]"
                  >
                    ✕
                  </motion.button>
                )}
              </motion.div>
            </motion.div>
          </div>
        </div>
      </section>


      {/* Filters */}
      <section className="bg-[var(--color-bg-surface)] border-b border-[var(--color-border-primary)]">
        <div className="max-w-6xl mx-auto px-6 py-4">
          <div className="flex flex-wrap items-center gap-4">
            <div className="flex items-center gap-2">
              <Filter className="w-4 h-4 text-[var(--color-text-tertiary)]" />
              <span className="text-sm text-[var(--color-text-secondary)] font-medium">Category:</span>
            </div>

            {modelCategories.map((category) => (
              <button
                key={category}
                onClick={() => setSelectedCategory(category)}
                className={`px-4 py-2 rounded-full text-sm font-medium transition-all ${
                  selectedCategory === category
                    ? 'bg-[var(--color-primary)] text-white shadow-sm'
                    : 'bg-[var(--color-bg-secondary)] text-[var(--color-text-secondary)] hover:bg-[var(--color-bg-tertiary)] border border-[var(--color-border-primary)]'
                }`}
              >
                {category}
              </button>
            ))}
          </div>

          <div className="mt-4 text-sm text-[var(--color-text-secondary)]">
            <span className="font-medium text-[var(--color-text-primary)]">{filteredModels.length}</span> models & tools found
          </div>
        </div>
      </section>

      {/* All Models Grid - Enhanced */}
      <section className="py-12">
        <div className="max-w-6xl mx-auto px-6">
          <div className="grid md:grid-cols-2 gap-8">
            {isLoading ? (
              // Loading skeletons with staggered animation
              Array.from({ length: 6 }, (_, index) => (
                <motion.div
                  key={`skeleton-${index}`}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{
                    duration: 0.5,
                    delay: index * 0.05,
                    ease: [0.16, 1, 0.3, 1]
                  }}
                  className="bg-[var(--surface)] border border-[var(--line)] rounded-2xl overflow-hidden p-8"
                >
                  <div className="flex items-start gap-4 mb-6">
                    <div className="w-16 h-16 bg-[var(--elevated)] rounded-xl animate-pulse" />
                    <div className="flex-1">
                      <div className="h-6 bg-[var(--elevated)] rounded mb-2 animate-pulse" />
                      <div className="h-4 bg-[var(--elevated)] rounded w-2/3 animate-pulse" />
                    </div>
                  </div>
                  <div className="space-y-2 mb-6">
                    <div className="h-4 bg-[var(--elevated)] rounded animate-pulse" />
                    <div className="h-4 bg-[var(--elevated)] rounded w-5/6 animate-pulse" />
                    <div className="h-4 bg-[var(--elevated)] rounded w-4/6 animate-pulse" />
                  </div>
                  <div className="flex justify-between items-center">
                    <div className="h-4 bg-[var(--elevated)] rounded w-24 animate-pulse" />
                    <div className="h-4 bg-[var(--elevated)] rounded w-16 animate-pulse" />
                  </div>
                </motion.div>
              ))
            ) : (
              filteredModels.map((model, index) => (
                <motion.div
                  key={model.id}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{
                    duration: 0.5,
                    delay: index * 0.05,
                    ease: [0.16, 1, 0.3, 1]
                  }}
                  whileHover={{
                    y: -4,
                    scale: 1.01,
                    transition: { duration: 0.2, ease: [0.16, 1, 0.3, 1] }
                  }}
                  whileTap={{ scale: 0.99 }}
                  className="group bg-[var(--surface)] border border-[var(--line)] rounded-2xl overflow-hidden shadow-sm hover:shadow-elevation-3 transition-shadow duration-300"
                >
                <div className="p-8">
                  {/* Header */}
                  <div className="flex items-start gap-4 mb-6">
                    <div className="flex-shrink-0 p-3 rounded-xl border bg-gradient-to-br from-[var(--accent)]/8 to-[var(--accent)]/3 border-[var(--accent)]/15">
                      {getTypeIcon(model.type, model.moduleType, model.name)}
                    </div>
                    <div className="flex-1 min-w-0">
                      <h3 className="text-xl font-semibold text-[var(--ink)] group-hover:text-[var(--accent)] transition-colors line-clamp-2 mb-2">
                        {model.name}
                      </h3>
                      <span className="text-sm text-[var(--subtle)] block">{model.category}</span>
                    </div>
                  </div>

                  {/* Description */}
                  <p className="text-[var(--muted)] mb-6 leading-relaxed line-clamp-3">
                    {model.cardSummary}
                  </p>

                  {/* Collapsible Details Section */}
                  <AnimatePresence>
                    {expandedCards.has(model.id) && (
                      <motion.div
                        initial={{ opacity: 0, height: 0 }}
                        animate={{ opacity: 1, height: 'auto' }}
                        exit={{ opacity: 0, height: 0 }}
                        transition={{ duration: 0.3, ease: [0.16, 1, 0.3, 1] }}
                        className="overflow-hidden mb-4"
                      >
                        <div className="pt-4 border-t border-[var(--line)]">
                          <h4 className="text-sm font-semibold text-[var(--ink)] mb-3">Key Capabilities</h4>
                          <div className="space-y-2 mb-4">
                            {model.capabilities[0]?.items.slice(0, 3).map((capability, idx) => (
                              <div key={idx} className="flex items-start gap-2">
                                <div className="w-1.5 h-1.5 bg-[var(--accent)] rounded-full mt-2 flex-shrink-0" />
                                <span className="text-xs text-[var(--muted)] leading-relaxed">{capability}</span>
                              </div>
                            ))}
                          </div>
                          {model.technicalSpecs && (
                            <div>
                              <h4 className="text-sm font-semibold text-[var(--ink)] mb-2">Technical Details</h4>
                              <div className="grid grid-cols-2 gap-3 text-xs">
                                <div>
                                  <span className="text-[var(--subtle)]">Framework:</span>
                                  <span className="text-[var(--muted)] ml-1 font-mono">{model.technicalSpecs.framework.split(' ')[0]}</span>
                                </div>
                                <div>
                                  <span className="text-[var(--subtle)]">Architecture:</span>
                                  <span className="text-[var(--muted)] ml-1 font-mono">{model.technicalSpecs.architecture.split(' ')[0]}</span>
                                </div>
                              </div>
                            </div>
                          )}
                        </div>
                      </motion.div>
                    )}
                  </AnimatePresence>

                  {/* Footer */}
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      {getCategoryIcon(model.category)}
                      <span className="text-sm text-[var(--subtle)]">
                        {getToolTypeDescription(model.name, model.category)}
                      </span>
                    </div>
                    <div className="flex items-center gap-2">
                      <motion.button
                        onClick={(e) => {
                          e.stopPropagation();
                          toggleCardExpansion(model.id);
                        }}
                        whileHover={{ scale: 1.05 }}
                        whileTap={{ scale: 0.95 }}
                        className="text-xs text-[var(--subtle)] hover:text-[var(--accent)] transition-colors p-1 rounded"
                      >
                        {expandedCards.has(model.id) ? 'Less' : 'Details'}
                      </motion.button>
                      <div
                        className="flex items-center gap-2 text-[var(--accent)] group-hover:gap-3 transition-all cursor-pointer"
                        onClick={() => setSelectedModel(model)}
                      >
                        <span className="text-sm font-medium">
                          {model.moduleType === 'module' ? 'Explore tool' : 'Learn more'}
                        </span>
                        <ChevronRight className="w-4 h-4" />
                      </div>
                    </div>
                  </div>
                </div>
              </motion.div>
            )))
            }
          </div>

          {/* No Results */}
          {filteredModels.length === 0 && (
            <div className="text-center py-16">
              <Database className="w-16 h-16 text-[var(--color-text-tertiary)] mx-auto mb-4" />
              <h3 className="text-xl font-semibold text-[var(--color-text-primary)] mb-2">No models or tools found</h3>
              <p className="text-[var(--color-text-secondary)] mb-6">
                Try adjusting your search query or category filters
              </p>
              <button
                onClick={() => {
                  setSearchQuery('');
                  setSelectedCategory('All Models');
                }}
                className="px-6 py-3 bg-[var(--color-primary)] text-white rounded-lg hover:bg-[var(--color-primary)]/90 transition-colors font-medium"
              >
                Reset Filters
              </button>
            </div>
          )}
        </div>
      </section>

      {/* Call to Action */}
      <section className="py-16 bg-[var(--color-bg-surface)]">
        <div className="max-w-4xl mx-auto px-6 text-center">
          <h2 className="text-3xl font-bold text-[var(--color-text-primary)] mb-4">Ready to Deploy?</h2>
          <p className="text-[var(--color-text-secondary)] text-lg mb-8">
            All models and tools are production-ready with comprehensive documentation, API endpoints, and deployment guides.
          </p>
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <a
              href="https://github.com/Lemkin-AI"
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-2 px-6 py-3 bg-[var(--color-primary)] text-white rounded-lg hover:bg-[var(--color-primary)]/90 transition-colors font-medium"
            >
              <Github className="w-5 h-5" />
              Lemkin AI on GitHub
            </a>
            <a
              href="/docs"
              className="inline-flex items-center gap-2 px-6 py-3 border border-[var(--color-border-primary)] text-[var(--color-text-primary)] rounded-lg hover:bg-[var(--color-bg-secondary)] transition-colors font-medium"
            >
              <FileText className="w-5 h-5" />
              Documentation
            </a>
          </div>
        </div>
      </section>
    </div>
  );
};

export default ModelsPageRevised;