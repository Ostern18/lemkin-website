import React, { useState, useMemo } from 'react';
import { Search, Filter, Clock, Tag, ArrowRight, BookOpen } from 'lucide-react';
import { articles, Article } from './articlesData';
import { ArticleReader } from './ArticleReader';

export const ArticlesPage: React.FC = () => {
  const [selectedArticle, setSelectedArticle] = useState<Article | null>(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedCategory, setSelectedCategory] = useState<string>('all');
  const [selectedTags, setSelectedTags] = useState<string[]>([]);

  // Get unique tags from all articles
  const allTags = useMemo(() => {
    const tags = new Set<string>();
    articles.forEach(article => {
      article.tags.forEach(tag => tags.add(tag));
    });
    return Array.from(tags).sort();
  }, []);

  // Filter articles based on search and filters
  const filteredArticles = useMemo(() => {
    return articles.filter(article => {
      // Search filter
      if (searchQuery) {
        const query = searchQuery.toLowerCase();
        const matchesSearch =
          article.title.toLowerCase().includes(query) ||
          article.excerpt.toLowerCase().includes(query) ||
          article.tags.some(tag => tag.toLowerCase().includes(query));
        if (!matchesSearch) return false;
      }

      // Category filter
      if (selectedCategory !== 'all' && article.category !== selectedCategory) {
        return false;
      }

      // Tags filter
      if (selectedTags.length > 0) {
        const hasMatchingTag = selectedTags.some(tag => article.tags.includes(tag));
        if (!hasMatchingTag) return false;
      }

      return true;
    });
  }, [searchQuery, selectedCategory, selectedTags]);

  const toggleTag = (tag: string) => {
    setSelectedTags(prev =>
      prev.includes(tag)
        ? prev.filter(t => t !== tag)
        : [...prev, tag]
    );
  };

  const getCategoryColor = (category: Article['category']) => {
    switch (category) {
      case 'technical':
        return 'bg-[var(--color-primary)]/10 text-[var(--color-primary)] border border-[var(--color-primary)]/20';
      case 'operational':
        return 'bg-[var(--color-primary)]/10 text-[var(--color-primary)] border border-[var(--color-primary)]/20';
      case 'legal':
        return 'bg-[var(--color-primary)]/10 text-[var(--color-primary)] border border-[var(--color-primary)]/20';
      case 'analytical':
        return 'bg-[var(--color-primary)]/10 text-[var(--color-primary)] border border-[var(--color-primary)]/20';
      default:
        return 'bg-[var(--color-bg-secondary)] text-[var(--color-text-secondary)] border border-[var(--color-border-primary)]';
    }
  };

  // If an article is selected, show the reader
  if (selectedArticle) {
    return <ArticleReader article={selectedArticle} onBack={() => setSelectedArticle(null)} />;
  }

  return (
    <div className="min-h-screen bg-[var(--color-bg-primary)]">
      {/* Hero Section */}
      <section className="bg-gradient-to-b from-[var(--color-bg-surface)] to-[var(--color-bg-primary)] border-b border-[var(--color-border-primary)]">
        <div className="max-w-6xl mx-auto px-6 py-20">
          <div className="text-center max-w-4xl mx-auto">
            <h1 className="text-6xl font-bold text-[var(--color-text-primary)] mb-6 leading-[1.1] tracking-tight">Research & Insights</h1>
            <p className="text-xl text-[var(--color-text-secondary)] leading-relaxed max-w-3xl mx-auto mb-8">
              Exploring the intersection of AI, human rights, and international justice through
              technical deep-dives, operational guides, and legal analyses.
            </p>

            {/* Search Bar */}
            <div className="max-w-2xl mx-auto">
              <div className="relative">
                <Search className="absolute left-4 top-1/2 transform -translate-y-1/2 text-[var(--color-text-tertiary)] w-5 h-5" />
                <input
                  type="text"
                  placeholder="Search articles by title, content, or tags..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="w-full pl-12 pr-6 py-4 border border-[var(--color-border-primary)] rounded-2xl bg-[var(--color-bg-elevated)] text-[var(--color-text-primary)] placeholder-[var(--color-text-tertiary)] focus:outline-none focus:ring-2 focus:ring-[var(--color-primary)] focus:border-transparent transition-all text-lg shadow-sm"
                />
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Filters Section */}
      <section className="bg-[var(--color-bg-surface)] border-b border-[var(--color-border-primary)] sticky top-0 z-10">
        <div className="max-w-6xl mx-auto px-6 py-4">
          <div className="flex flex-col lg:flex-row gap-4">
            {/* Category Filter */}
            <div className="flex items-center gap-2">
              <Filter className="w-4 h-4 text-[var(--color-text-tertiary)]" />
              <span className="text-sm text-[var(--color-text-secondary)] mr-2">Category:</span>
              <div className="flex flex-wrap gap-2">
                {['all', 'technical', 'operational', 'legal', 'analytical'].map((cat) => (
                  <button
                    key={cat}
                    onClick={() => setSelectedCategory(cat)}
                    className={`px-4 py-2.5 rounded-xl font-medium text-sm transition-all duration-200 ${
                      selectedCategory === cat
                        ? 'bg-[var(--color-primary)] text-white shadow-sm scale-105'
                        : 'bg-[var(--color-bg-surface)] text-[var(--color-text-secondary)] hover:bg-[var(--color-bg-secondary)] border border-[var(--color-border-primary)]'
                    }`}
                  >
                    {cat === 'all' ? 'All' : cat.charAt(0).toUpperCase() + cat.slice(1)}
                  </button>
                ))}
              </div>
            </div>

            {/* Tags Filter (Show selected) */}
            {selectedTags.length > 0 && (
              <div className="flex items-center gap-2 flex-wrap">
                <span className="text-sm text-[var(--color-text-secondary)]">Tags:</span>
                {selectedTags.map(tag => (
                  <button
                    key={tag}
                    onClick={() => toggleTag(tag)}
                    className="px-3 py-1 bg-[var(--color-primary)]/10 text-[var(--color-primary)] rounded-full text-sm hover:bg-[var(--color-primary)]/20 transition-colors flex items-center gap-1 border border-[var(--color-primary)]/20"
                  >
                    {tag}
                    <span className="ml-1">×</span>
                  </button>
                ))}
                <button
                  onClick={() => setSelectedTags([])}
                  className="text-sm text-[var(--color-text-tertiary)] hover:text-[var(--color-text-secondary)]"
                >
                  Clear all
                </button>
              </div>
            )}
          </div>

          {/* Results count */}
          <div className="mt-4 text-sm text-[var(--color-text-secondary)]">
            <span className="font-medium text-[var(--color-text-primary)]">{filteredArticles.length}</span> of {articles.length} articles
          </div>
        </div>
      </section>

      {/* Articles Grid */}
      <section className="py-12">
        <div className="max-w-6xl mx-auto px-6">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            {filteredArticles.map((article) => (
              <article
                key={article.id}
                className="group cursor-pointer bg-[var(--color-bg-surface)] border border-[var(--color-border-primary)] rounded-2xl p-8 hover:shadow-elevation-3 hover:border-[var(--color-border-secondary)] hover:scale-[1.02] transition-all duration-300"
                onClick={() => setSelectedArticle(article)}
              >
                {/* Category Badge */}
                <div className="flex items-center justify-between mb-4">
                  <span className={`px-3 py-1 rounded-full text-xs font-medium ${getCategoryColor(article.category)}`}>
                    {article.category.charAt(0).toUpperCase() + article.category.slice(1)}
                  </span>
                  <div className="flex items-center gap-1 text-[var(--color-text-tertiary)] text-sm">
                    <Clock className="w-3 h-3" />
                    <span>{article.readTime}</span>
                  </div>
                </div>

                {/* Title */}
                <h3 className="text-2xl font-bold text-[var(--color-text-primary)] mb-4 group-hover:text-[var(--color-primary)] transition-colors leading-tight">
                  {article.title}
                </h3>

                {/* Excerpt */}
                <p className="text-[var(--color-text-secondary)] mb-6 leading-relaxed line-clamp-3">
                  {article.excerpt}
                </p>

                {/* Tags */}
                <div className="flex flex-wrap gap-2 mb-4">
                  {article.tags.slice(0, 3).map((tag) => (
                    <button
                      key={tag}
                      onClick={(e) => {
                        e.stopPropagation();
                        toggleTag(tag);
                      }}
                      className="px-2 py-1 bg-[var(--color-bg-secondary)] text-[var(--color-text-secondary)] rounded text-xs hover:bg-[var(--color-bg-tertiary)] transition-colors"
                    >
                      {tag}
                    </button>
                  ))}
                  {article.tags.length > 3 && (
                    <span className="px-2 py-1 text-[var(--color-text-tertiary)] text-xs">
                      +{article.tags.length - 3} more
                    </span>
                  )}
                </div>

                {/* Footer */}
                <div className="flex items-center justify-between pt-4 border-t border-[var(--color-border-primary)]">
                  <div className="flex items-center gap-3">
                    <div className="w-8 h-8 rounded-full bg-[var(--color-primary)]/10 flex items-center justify-center">
                      <span className="text-xs font-medium text-[var(--color-primary)]">
                        {article.author.split(' ').map(n => n[0]).join('')}
                      </span>
                    </div>
                    <div className="text-sm">
                      <div className="font-medium text-[var(--color-text-primary)]">{article.author}</div>
                      <div className="text-[var(--color-text-tertiary)]">{new Date(article.date).toLocaleDateString()}</div>
                    </div>
                  </div>
                  <div className="flex items-center gap-1 text-[var(--color-primary)] group-hover:gap-2 transition-all">
                    <span className="text-sm font-medium">Read</span>
                    <ArrowRight className="w-4 h-4" />
                  </div>
                </div>
              </article>
            ))}
          </div>

          {/* No results message */}
          {filteredArticles.length === 0 && (
            <div className="text-center py-16">
              <div className="max-w-md mx-auto">
                <BookOpen className="w-12 h-12 text-[var(--color-text-tertiary)] mx-auto mb-4" />
                <h3 className="text-lg font-semibold text-[var(--color-text-primary)] mb-2">No articles found</h3>
                <p className="text-[var(--color-text-secondary)] mb-6">
                  Try adjusting your search terms or category filters.
                </p>
                <button
                  onClick={() => {
                    setSearchQuery('');
                    setSelectedCategory('all');
                    setSelectedTags([]);
                  }}
                  className="inline-flex items-center gap-2 px-4 py-2 bg-[var(--color-primary)] text-white rounded-lg hover:bg-[var(--color-primary)]/90 transition-colors"
                >
                  Reset filters
                </button>
              </div>
            </div>
          )}
        </div>
      </section>

      {/* Popular Tags Section */}
      <section className="py-12 bg-[var(--color-bg-surface)]">
        <div className="max-w-6xl mx-auto px-6">
          <h2 className="text-2xl font-semibold mb-6 text-[var(--color-text-primary)]">
            Explore Topics
          </h2>
          <div className="flex flex-wrap gap-3">
            {allTags.slice(0, 20).map(tag => (
              <button
                key={tag}
                onClick={() => toggleTag(tag)}
                className={`px-4 py-2 rounded-full text-sm transition-all ${
                  selectedTags.includes(tag)
                    ? 'bg-[var(--color-primary)] text-white hover:bg-[var(--color-primary)]/90'
                    : 'bg-[var(--color-bg-secondary)] text-[var(--color-text-secondary)] hover:bg-[var(--color-bg-tertiary)] border border-[var(--color-border-primary)]'
                }`}
              >
                <Tag className="w-3 h-3 inline mr-1" />
                {tag}
              </button>
            ))}
          </div>
        </div>
      </section>
    </div>
  );
};