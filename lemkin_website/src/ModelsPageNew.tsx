import React, { useState, useMemo } from 'react';
import {
  Search, Filter, Download, ExternalLink, Github,
  Cpu, Database, Zap, Eye, Brain, FileText,
  Shield, Scale, Globe, Clock, Award, Target,
  ChevronDown, ChevronRight, Play, Star,
  BarChart3, TrendingUp, CheckCircle
} from 'lucide-react';
import { models, Model, modelCategories, modelTypes, modelStatuses, getFeaturedModels } from './modelsData';

const ModelsPageNew: React.FC = () => {
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedCategory, setSelectedCategory] = useState('All Models');
  const [selectedType, setSelectedType] = useState('All Types');
  const [selectedStatus, setSelectedStatus] = useState('All Status');
  const [selectedModel, setSelectedModel] = useState<Model | null>(null);
  const [expandedCards, setExpandedCards] = useState<Set<string>>(new Set());

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

      // Type filter
      if (selectedType !== 'All Types' && model.type !== selectedType) {
        return false;
      }

      // Status filter
      if (selectedStatus !== 'All Status' && model.status !== selectedStatus) {
        return false;
      }

      return true;
    });
  }, [searchQuery, selectedCategory, selectedType, selectedStatus]);

  const featuredModels = getFeaturedModels().slice(0, 3);

  const toggleCardExpansion = (modelId: string) => {
    const newExpanded = new Set(expandedCards);
    if (newExpanded.has(modelId)) {
      newExpanded.delete(modelId);
    } else {
      newExpanded.add(modelId);
    }
    setExpandedCards(newExpanded);
  };

  const getStatusBadge = (status: string) => {
    switch (status) {
      case 'production':
        return 'bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-300 border border-green-200 dark:border-green-800';
      case 'development':
        return 'bg-yellow-100 dark:bg-yellow-900/30 text-yellow-700 dark:text-yellow-300 border border-yellow-200 dark:border-yellow-800';
      case 'research':
        return 'bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300 border border-blue-200 dark:border-blue-800';
      default:
        return 'bg-[var(--color-bg-secondary)] text-[var(--color-text-secondary)] border border-[var(--color-border-primary)]';
    }
  };

  const getTypeIcon = (type: string) => {
    switch (type) {
      case 'computer-vision':
        return <Eye className="w-5 h-5" />;
      case 'nlp':
        return <FileText className="w-5 h-5" />;
      case 'multimodal':
        return <Brain className="w-5 h-5" />;
      case 'hybrid':
        return <Cpu className="w-5 h-5" />;
      default:
        return <Cpu className="w-5 h-5" />;
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
      default:
        return <Cpu className="w-5 h-5" />;
    }
  };

  if (selectedModel) {
    return (
      <div className="min-h-screen bg-[var(--color-bg-primary)]">
        {/* Model Detail View */}
        <div className="bg-[var(--color-bg-surface)] border-b border-[var(--color-border-primary)]">
          <div className="max-w-6xl mx-auto px-6 py-8">
            <button
              onClick={() => setSelectedModel(null)}
              className="inline-flex items-center gap-2 text-[var(--color-text-secondary)] hover:text-[var(--color-text-primary)] transition-all duration-200 hover:gap-3 mb-6"
            >
              <ChevronDown className="w-4 h-4 rotate-90" />
              <span className="text-sm font-medium">All Models</span>
            </button>

            {/* Model Header */}
            <div className="flex items-start gap-6 mb-8">
              <div className="flex-shrink-0 p-3 bg-[var(--color-primary)]/10 rounded-xl">
                {getTypeIcon(selectedModel.type)}
              </div>
              <div className="flex-1">
                <div className="flex items-center gap-3 mb-2">
                  <h1 className="text-4xl font-bold text-[var(--color-text-primary)]">{selectedModel.name}</h1>
                  {selectedModel.featured && <Star className="w-6 h-6 text-yellow-500 fill-current" />}
                </div>
                <p className="text-xl text-[var(--color-text-secondary)] mb-4">{selectedModel.shortDescription}</p>
                <div className="flex flex-wrap items-center gap-3">
                  <span className={`px-3 py-1 rounded-full text-sm font-medium ${getStatusBadge(selectedModel.status)}`}>
                    {selectedModel.status.charAt(0).toUpperCase() + selectedModel.status.slice(1)}
                  </span>
                  <span className="px-3 py-1 bg-[var(--color-bg-secondary)] text-[var(--color-text-secondary)] rounded-full text-sm">
                    {selectedModel.category}
                  </span>
                  <span className="px-3 py-1 bg-[var(--color-bg-secondary)] text-[var(--color-text-secondary)] rounded-full text-sm">
                    {selectedModel.type.replace('-', ' ').replace(/\b\w/g, l => l.toUpperCase())}
                  </span>
                </div>
              </div>
              <div className="flex gap-3">
                {selectedModel.githubRepo && (
                  <a href={selectedModel.githubRepo} target="_blank" rel="noopener noreferrer"
                     className="p-3 bg-[var(--color-bg-surface)] border border-[var(--color-border-primary)] rounded-lg hover:bg-[var(--color-bg-secondary)] transition-colors">
                    <Github className="w-5 h-5" />
                  </a>
                )}
                {selectedModel.huggingFaceModel && (
                  <a href={`https://huggingface.co/${selectedModel.huggingFaceModel}`} target="_blank" rel="noopener noreferrer"
                     className="p-3 bg-[var(--color-bg-surface)] border border-[var(--color-border-primary)] rounded-lg hover:bg-[var(--color-bg-secondary)] transition-colors">
                    <ExternalLink className="w-5 h-5" />
                  </a>
                )}
              </div>
            </div>
          </div>
        </div>

        {/* Model Details */}
        <div className="max-w-6xl mx-auto px-6 py-12">
          <div className="grid lg:grid-cols-3 gap-8">
            {/* Left Column - Description & Capabilities */}
            <div className="lg:col-span-2 space-y-8">
              {/* Description */}
              <div className="bg-[var(--color-bg-surface)] border border-[var(--color-border-primary)] rounded-2xl p-8">
                <h2 className="text-2xl font-semibold text-[var(--color-text-primary)] mb-4">Overview</h2>
                <p className="text-[var(--color-text-secondary)] leading-relaxed">{selectedModel.description}</p>
              </div>

              {/* Capabilities */}
              <div className="bg-[var(--color-bg-surface)] border border-[var(--color-border-primary)] rounded-2xl p-8">
                <h2 className="text-2xl font-semibold text-[var(--color-text-primary)] mb-6">Capabilities</h2>
                <div className="space-y-6">
                  {selectedModel.capabilities.map((capability, index) => (
                    <div key={index}>
                      <h3 className="text-lg font-medium text-[var(--color-text-primary)] mb-3">{capability.category}</h3>
                      <div className="grid md:grid-cols-2 gap-2">
                        {capability.items.map((item, itemIndex) => (
                          <div key={itemIndex} className="flex items-start gap-2">
                            <CheckCircle className="w-4 h-4 text-green-500 mt-0.5 flex-shrink-0" />
                            <span className="text-[var(--color-text-secondary)] text-sm">{item}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Use Cases */}
              <div className="bg-[var(--color-bg-surface)] border border-[var(--color-border-primary)] rounded-2xl p-8">
                <h2 className="text-2xl font-semibold text-[var(--color-text-primary)] mb-6">Use Cases</h2>
                <div className="grid md:grid-cols-2 gap-3">
                  {selectedModel.useCases.map((useCase, index) => (
                    <div key={index} className="flex items-start gap-2">
                      <Target className="w-4 h-4 text-[var(--color-primary)] mt-1 flex-shrink-0" />
                      <span className="text-[var(--color-text-secondary)]">{useCase}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            {/* Right Column - Metrics & Technical Specs */}
            <div className="space-y-6">
              {/* Performance Metrics */}
              <div className="bg-[var(--color-bg-surface)] border border-[var(--color-border-primary)] rounded-2xl p-6">
                <h3 className="text-lg font-semibold text-[var(--color-text-primary)] mb-4 flex items-center gap-2">
                  <BarChart3 className="w-5 h-5" />
                  Performance Metrics
                </h3>
                <div className="space-y-3">
                  {selectedModel.metrics.accuracy && (
                    <div className="flex justify-between">
                      <span className="text-[var(--color-text-secondary)]">Accuracy</span>
                      <span className="font-semibold text-[var(--color-text-primary)]">{selectedModel.metrics.accuracy}</span>
                    </div>
                  )}
                  {selectedModel.metrics.f1Score && (
                    <div className="flex justify-between">
                      <span className="text-[var(--color-text-secondary)]">F1 Score</span>
                      <span className="font-semibold text-[var(--color-text-primary)]">{selectedModel.metrics.f1Score}</span>
                    </div>
                  )}
                  {selectedModel.metrics.rougeL && (
                    <div className="flex justify-between">
                      <span className="text-[var(--color-text-secondary)]">ROUGE-L</span>
                      <span className="font-semibold text-[var(--color-text-primary)]">{selectedModel.metrics.rougeL}</span>
                    </div>
                  )}
                  {selectedModel.metrics.bleuScore && (
                    <div className="flex justify-between">
                      <span className="text-[var(--color-text-secondary)]">BLEU Score</span>
                      <span className="font-semibold text-[var(--color-text-primary)]">{selectedModel.metrics.bleuScore}</span>
                    </div>
                  )}
                  <div className="flex justify-between">
                    <span className="text-[var(--color-text-secondary)]">Model Size</span>
                    <span className="font-semibold text-[var(--color-text-primary)]">{selectedModel.metrics.modelSize}</span>
                  </div>
                  {selectedModel.metrics.inferenceSpeed && (
                    <div className="flex justify-between">
                      <span className="text-[var(--color-text-secondary)]">Speed</span>
                      <span className="font-semibold text-[var(--color-text-primary)]">{selectedModel.metrics.inferenceSpeed}</span>
                    </div>
                  )}
                  {selectedModel.metrics.parameters && (
                    <div className="flex justify-between">
                      <span className="text-[var(--color-text-secondary)]">Parameters</span>
                      <span className="font-semibold text-[var(--color-text-primary)]">{selectedModel.metrics.parameters}</span>
                    </div>
                  )}
                </div>
              </div>

              {/* Technical Specifications */}
              <div className="bg-[var(--color-bg-surface)] border border-[var(--color-border-primary)] rounded-2xl p-6">
                <h3 className="text-lg font-semibold text-[var(--color-text-primary)] mb-4 flex items-center gap-2">
                  <Cpu className="w-5 h-5" />
                  Technical Specs
                </h3>
                <div className="space-y-3">
                  <div>
                    <span className="text-[var(--color-text-secondary)] text-sm">Framework</span>
                    <p className="font-medium text-[var(--color-text-primary)]">{selectedModel.technicalSpecs.framework}</p>
                  </div>
                  <div>
                    <span className="text-[var(--color-text-secondary)] text-sm">Architecture</span>
                    <p className="font-medium text-[var(--color-text-primary)]">{selectedModel.technicalSpecs.architecture}</p>
                  </div>
                  <div>
                    <span className="text-[var(--color-text-secondary)] text-sm">Input</span>
                    <p className="font-medium text-[var(--color-text-primary)]">{selectedModel.technicalSpecs.inputFormat}</p>
                  </div>
                  <div>
                    <span className="text-[var(--color-text-secondary)] text-sm">Output</span>
                    <p className="font-medium text-[var(--color-text-primary)]">{selectedModel.technicalSpecs.outputFormat}</p>
                  </div>
                </div>
              </div>

              {/* Deployment */}
              <div className="bg-[var(--color-bg-surface)] border border-[var(--color-border-primary)] rounded-2xl p-6">
                <h3 className="text-lg font-semibold text-[var(--color-text-primary)] mb-4 flex items-center gap-2">
                  <Zap className="w-5 h-5" />
                  Deployment
                </h3>
                <div className="space-y-2">
                  {selectedModel.deployment.requirements.map((req, index) => (
                    <div key={index} className="flex items-center gap-2">
                      <CheckCircle className="w-4 h-4 text-green-500" />
                      <span className="text-[var(--color-text-secondary)] text-sm">{req}</span>
                    </div>
                  ))}
                </div>
                {selectedModel.deployment.apiEndpoint && (
                  <div className="mt-4 p-3 bg-[var(--color-bg-secondary)] rounded-lg">
                    <p className="text-xs text-[var(--color-text-tertiary)] mb-1">API Endpoint</p>
                    <code className="text-[var(--color-text-primary)] text-sm">{selectedModel.deployment.apiEndpoint}</code>
                  </div>
                )}
              </div>

              {/* Quick Actions */}
              <div className="bg-[var(--color-bg-surface)] border border-[var(--color-border-primary)] rounded-2xl p-6">
                <h3 className="text-lg font-semibold text-[var(--color-text-primary)] mb-4">Quick Actions</h3>
                <div className="space-y-3">
                  {selectedModel.huggingFaceModel && (
                    <a href={`https://huggingface.co/${selectedModel.huggingFaceModel}`} target="_blank" rel="noopener noreferrer"
                       className="w-full flex items-center justify-center gap-2 px-4 py-2 bg-[var(--color-primary)] text-white rounded-lg hover:bg-[var(--color-primary)]/90 transition-colors">
                      <Play className="w-4 h-4" />
                      Try on Hugging Face
                    </a>
                  )}
                  {selectedModel.githubRepo && (
                    <a href={selectedModel.githubRepo} target="_blank" rel="noopener noreferrer"
                       className="w-full flex items-center justify-center gap-2 px-4 py-2 border border-[var(--color-border-primary)] text-[var(--color-text-primary)] rounded-lg hover:bg-[var(--color-bg-secondary)] transition-colors">
                      <Github className="w-4 h-4" />
                      View Source
                    </a>
                  )}
                  <button className="w-full flex items-center justify-center gap-2 px-4 py-2 border border-[var(--color-border-primary)] text-[var(--color-text-primary)] rounded-lg hover:bg-[var(--color-bg-secondary)] transition-colors">
                    <Download className="w-4 h-4" />
                    Download Model
                  </button>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-[var(--color-bg-primary)]">
      {/* Hero Section */}
      <section className="bg-gradient-to-b from-[var(--color-bg-surface)] to-[var(--color-bg-primary)] border-b border-[var(--color-border-primary)]">
        <div className="max-w-6xl mx-auto px-6 py-20">
          <div className="text-center max-w-4xl mx-auto">
            <h1 className="text-6xl font-bold text-[var(--color-text-primary)] mb-6 leading-[1.1] tracking-tight">
              AI Models for Justice
            </h1>
            <p className="text-xl text-[var(--color-text-secondary)] leading-relaxed max-w-3xl mx-auto mb-8">
              Production-ready AI models specifically designed for human rights monitoring, legal analysis,
              and humanitarian applications. Optimized for accuracy, efficiency, and ethical deployment.
            </p>

            {/* Stats */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-6 mb-12">
              <div className="text-center">
                <div className="text-3xl font-bold text-[var(--color-primary)] mb-1">{models.length}</div>
                <div className="text-sm text-[var(--color-text-secondary)]">Total Models</div>
              </div>
              <div className="text-center">
                <div className="text-3xl font-bold text-[var(--color-primary)] mb-1">{models.filter(m => m.status === 'production').length}</div>
                <div className="text-sm text-[var(--color-text-secondary)]">Production Ready</div>
              </div>
              <div className="text-center">
                <div className="text-3xl font-bold text-[var(--color-primary)] mb-1">4</div>
                <div className="text-sm text-[var(--color-text-secondary)]">Languages</div>
              </div>
              <div className="text-center">
                <div className="text-3xl font-bold text-[var(--color-primary)] mb-1">95%</div>
                <div className="text-sm text-[var(--color-text-secondary)]">Avg Accuracy</div>
              </div>
            </div>

            {/* Search Bar */}
            <div className="max-w-2xl mx-auto">
              <div className="relative">
                <Search className="absolute left-4 top-1/2 transform -translate-y-1/2 text-[var(--color-text-tertiary)] w-5 h-5" />
                <input
                  type="text"
                  placeholder="Search models by name, capabilities, or use cases..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="w-full pl-12 pr-6 py-4 border border-[var(--color-border-primary)] rounded-2xl bg-[var(--color-bg-elevated)] text-[var(--color-text-primary)] placeholder-[var(--color-text-tertiary)] focus:outline-none focus:ring-2 focus:ring-[var(--color-primary)] focus:border-transparent transition-all text-lg shadow-sm"
                />
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Featured Models */}
      {searchQuery === '' && selectedCategory === 'All Models' && selectedType === 'All Types' && selectedStatus === 'All Status' && (
        <section className="py-16 bg-[var(--color-bg-surface)]">
          <div className="max-w-6xl mx-auto px-6">
            <div className="text-center mb-12">
              <h2 className="text-3xl font-bold text-[var(--color-text-primary)] mb-4">Featured Models</h2>
              <p className="text-[var(--color-text-secondary)] max-w-2xl mx-auto">
                Our flagship AI models powering the next generation of human rights monitoring and legal technology.
              </p>
            </div>
            <div className="grid md:grid-cols-3 gap-8">
              {featuredModels.map((model) => (
                <div
                  key={model.id}
                  className="group cursor-pointer bg-[var(--color-bg-primary)] border border-[var(--color-border-primary)] rounded-2xl p-6 hover:shadow-elevation-3 hover:border-[var(--color-border-secondary)] hover:scale-[1.02] transition-all duration-300"
                  onClick={() => setSelectedModel(model)}
                >
                  <div className="flex items-center gap-3 mb-4">
                    <div className="p-2 bg-[var(--color-primary)]/10 rounded-lg">
                      {getTypeIcon(model.type)}
                    </div>
                    <div className="flex-1">
                      <h3 className="text-lg font-semibold text-[var(--color-text-primary)] group-hover:text-[var(--color-primary)] transition-colors">
                        {model.name}
                      </h3>
                      <span className={`px-2 py-1 rounded-full text-xs font-medium ${getStatusBadge(model.status)}`}>
                        {model.status}
                      </span>
                    </div>
                    <Star className="w-5 h-5 text-yellow-500 fill-current" />
                  </div>
                  <p className="text-[var(--color-text-secondary)] mb-4 line-clamp-2">
                    {model.shortDescription}
                  </p>
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-[var(--color-text-tertiary)]">{model.category}</span>
                    <ChevronRight className="w-4 h-4 text-[var(--color-text-tertiary)] group-hover:text-[var(--color-primary)] transition-colors" />
                  </div>
                </div>
              ))}
            </div>
          </div>
        </section>
      )}

      {/* Filters */}
      <section className="bg-[var(--color-bg-surface)] border-b border-[var(--color-border-primary)] sticky top-0 z-10">
        <div className="max-w-6xl mx-auto px-6 py-4">
          <div className="flex flex-col lg:flex-row gap-4">
            <div className="flex items-center gap-2">
              <Filter className="w-4 h-4 text-[var(--color-text-tertiary)]" />
              <span className="text-sm text-[var(--color-text-secondary)] mr-2">Filters:</span>
            </div>

            {/* Category Filter */}
            <div className="flex flex-wrap gap-2">
              {modelCategories.map((category) => (
                <button
                  key={category}
                  onClick={() => setSelectedCategory(category)}
                  className={`px-3 py-1.5 rounded-full text-sm font-medium transition-all ${
                    selectedCategory === category
                      ? 'bg-[var(--color-primary)] text-white'
                      : 'bg-[var(--color-bg-secondary)] text-[var(--color-text-secondary)] hover:bg-[var(--color-bg-tertiary)]'
                  }`}
                >
                  {category}
                </button>
              ))}
            </div>

            {/* Type Filter */}
            <div className="flex flex-wrap gap-2">
              {modelTypes.map((type) => (
                <button
                  key={type}
                  onClick={() => setSelectedType(type)}
                  className={`px-3 py-1.5 rounded-full text-sm font-medium transition-all ${
                    selectedType === type
                      ? 'bg-[var(--color-primary)] text-white'
                      : 'bg-[var(--color-bg-secondary)] text-[var(--color-text-secondary)] hover:bg-[var(--color-bg-tertiary)]'
                  }`}
                >
                  {type === 'All Types' ? type : type.replace('-', ' ').replace(/\b\w/g, l => l.toUpperCase())}
                </button>
              ))}
            </div>

            {/* Status Filter */}
            <div className="flex flex-wrap gap-2">
              {modelStatuses.map((status) => (
                <button
                  key={status}
                  onClick={() => setSelectedStatus(status)}
                  className={`px-3 py-1.5 rounded-full text-sm font-medium transition-all ${
                    selectedStatus === status
                      ? 'bg-[var(--color-primary)] text-white'
                      : 'bg-[var(--color-bg-secondary)] text-[var(--color-text-secondary)] hover:bg-[var(--color-bg-tertiary)]'
                  }`}
                >
                  {status === 'All Status' ? status : status.charAt(0).toUpperCase() + status.slice(1)}
                </button>
              ))}
            </div>
          </div>

          {/* Results count */}
          <div className="mt-4 text-sm text-[var(--color-text-secondary)]">
            <span className="font-medium text-[var(--color-text-primary)]">{filteredModels.length}</span> models found
          </div>
        </div>
      </section>

      {/* Models Grid */}
      <section className="py-12">
        <div className="max-w-6xl mx-auto px-6">
          <div className="grid md:grid-cols-2 xl:grid-cols-3 gap-6">
            {filteredModels.map((model) => {
              const isExpanded = expandedCards.has(model.id);
              return (
                <div
                  key={model.id}
                  className="group bg-[var(--color-bg-surface)] border border-[var(--color-border-primary)] rounded-2xl overflow-hidden hover:shadow-elevation-3 hover:border-[var(--color-border-secondary)] transition-all duration-300"
                >
                  {/* Card Header */}
                  <div className="p-6 pb-4">
                    <div className="flex items-start gap-3 mb-4">
                      <div className="flex-shrink-0 p-2 bg-[var(--color-primary)]/10 rounded-lg">
                        {getTypeIcon(model.type)}
                      </div>
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-2 mb-1">
                          <h3 className="text-lg font-semibold text-[var(--color-text-primary)] truncate">
                            {model.name}
                          </h3>
                          {model.featured && <Star className="w-4 h-4 text-yellow-500 fill-current flex-shrink-0" />}
                        </div>
                        <div className="flex items-center gap-2 mb-2">
                          <span className={`px-2 py-1 rounded-full text-xs font-medium ${getStatusBadge(model.status)}`}>
                            {model.status}
                          </span>
                          <span className="text-xs text-[var(--color-text-tertiary)]">{model.type.replace('-', ' ')}</span>
                        </div>
                      </div>
                    </div>

                    <p className="text-[var(--color-text-secondary)] text-sm mb-4 line-clamp-2">
                      {model.shortDescription}
                    </p>

                    {/* Key Metrics */}
                    <div className="grid grid-cols-2 gap-3 mb-4">
                      {model.metrics.accuracy && (
                        <div className="text-center p-2 bg-[var(--color-bg-secondary)] rounded-lg">
                          <div className="text-lg font-bold text-[var(--color-primary)]">{model.metrics.accuracy}</div>
                          <div className="text-xs text-[var(--color-text-tertiary)]">Accuracy</div>
                        </div>
                      )}
                      {model.metrics.f1Score && (
                        <div className="text-center p-2 bg-[var(--color-bg-secondary)] rounded-lg">
                          <div className="text-lg font-bold text-[var(--color-primary)]">{model.metrics.f1Score.split(',')[0]}</div>
                          <div className="text-xs text-[var(--color-text-tertiary)]">F1 Score</div>
                        </div>
                      )}
                      {(!model.metrics.accuracy && !model.metrics.f1Score) && (
                        <>
                          <div className="text-center p-2 bg-[var(--color-bg-secondary)] rounded-lg">
                            <div className="text-lg font-bold text-[var(--color-primary)]">{model.metrics.modelSize}</div>
                            <div className="text-xs text-[var(--color-text-tertiary)]">Size</div>
                          </div>
                          <div className="text-center p-2 bg-[var(--color-bg-secondary)] rounded-lg">
                            <div className="text-lg font-bold text-[var(--color-primary)]">{model.status}</div>
                            <div className="text-xs text-[var(--color-text-tertiary)]">Status</div>
                          </div>
                        </>
                      )}
                    </div>
                  </div>

                  {/* Expandable Content */}
                  {isExpanded && (
                    <div className="px-6 pb-4">
                      <div className="border-t border-[var(--color-border-primary)] pt-4">
                        <h4 className="text-sm font-medium text-[var(--color-text-primary)] mb-2">Key Capabilities</h4>
                        <div className="space-y-1 mb-4">
                          {model.capabilities[0]?.items.slice(0, 3).map((item, index) => (
                            <div key={index} className="flex items-start gap-2">
                              <CheckCircle className="w-3 h-3 text-green-500 mt-0.5 flex-shrink-0" />
                              <span className="text-xs text-[var(--color-text-secondary)]">{item}</span>
                            </div>
                          ))}
                        </div>

                        <h4 className="text-sm font-medium text-[var(--color-text-primary)] mb-2">Primary Use Cases</h4>
                        <div className="space-y-1">
                          {model.useCases.slice(0, 3).map((useCase, index) => (
                            <div key={index} className="flex items-start gap-2">
                              <Target className="w-3 h-3 text-[var(--color-primary)] mt-0.5 flex-shrink-0" />
                              <span className="text-xs text-[var(--color-text-secondary)]">{useCase}</span>
                            </div>
                          ))}
                        </div>
                      </div>
                    </div>
                  )}

                  {/* Card Footer */}
                  <div className="px-6 pb-6">
                    <div className="flex items-center justify-between">
                      <button
                        onClick={() => toggleCardExpansion(model.id)}
                        className="text-sm text-[var(--color-primary)] hover:text-[var(--color-primary)]/80 font-medium flex items-center gap-1"
                      >
                        {isExpanded ? 'Show Less' : 'Show More'}
                        <ChevronDown className={`w-4 h-4 transition-transform ${isExpanded ? 'rotate-180' : ''}`} />
                      </button>
                      <button
                        onClick={() => setSelectedModel(model)}
                        className="px-4 py-2 bg-[var(--color-primary)] text-white rounded-lg hover:bg-[var(--color-primary)]/90 transition-colors text-sm font-medium"
                      >
                        View Details
                      </button>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>

          {/* No Results */}
          {filteredModels.length === 0 && (
            <div className="text-center py-16">
              <Database className="w-16 h-16 text-[var(--color-text-tertiary)] mx-auto mb-4" />
              <h3 className="text-xl font-semibold text-[var(--color-text-primary)] mb-2">No models found</h3>
              <p className="text-[var(--color-text-secondary)] mb-6">
                Try adjusting your search query or filters
              </p>
              <button
                onClick={() => {
                  setSearchQuery('');
                  setSelectedCategory('All Models');
                  setSelectedType('All Types');
                  setSelectedStatus('All Status');
                }}
                className="px-6 py-3 bg-[var(--color-primary)] text-white rounded-lg hover:bg-[var(--color-primary)]/90 transition-colors"
              >
                Reset Filters
              </button>
            </div>
          )}
        </div>
      </section>

      {/* Integration Section */}
      <section className="py-16 bg-[var(--color-bg-surface)]">
        <div className="max-w-6xl mx-auto px-6">
          <div className="text-center mb-12">
            <h2 className="text-3xl font-bold text-[var(--color-text-primary)] mb-4">Ready to Deploy</h2>
            <p className="text-[var(--color-text-secondary)] max-w-2xl mx-auto">
              All models are production-ready with comprehensive documentation, API endpoints, and deployment guides.
            </p>
          </div>
          <div className="grid md:grid-cols-3 gap-8">
            <div className="text-center">
              <div className="p-3 bg-[var(--color-primary)]/10 rounded-xl w-fit mx-auto mb-4">
                <Github className="w-6 h-6 text-[var(--color-primary)]" />
              </div>
              <h3 className="text-lg font-semibold text-[var(--color-text-primary)] mb-2">Open Source</h3>
              <p className="text-[var(--color-text-secondary)] text-sm">
                Complete source code and training scripts available on GitHub
              </p>
            </div>
            <div className="text-center">
              <div className="p-3 bg-[var(--color-primary)]/10 rounded-xl w-fit mx-auto mb-4">
                <Zap className="w-6 h-6 text-[var(--color-primary)]" />
              </div>
              <h3 className="text-lg font-semibold text-[var(--color-text-primary)] mb-2">API Ready</h3>
              <p className="text-[var(--color-text-secondary)] text-sm">
                RESTful APIs and containerized deployment for easy integration
              </p>
            </div>
            <div className="text-center">
              <div className="p-3 bg-[var(--color-primary)]/10 rounded-xl w-fit mx-auto mb-4">
                <Shield className="w-6 h-6 text-[var(--color-primary)]" />
              </div>
              <h3 className="text-lg font-semibold text-[var(--color-text-primary)] mb-2">Enterprise Grade</h3>
              <p className="text-[var(--color-text-secondary)] text-sm">
                Built for production with monitoring, logging, and security features
              </p>
            </div>
          </div>
        </div>
      </section>
    </div>
  );
};

export default ModelsPageNew;