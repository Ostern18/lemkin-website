import React, { useState, useEffect, useMemo } from 'react';
import { ArrowLeft, Clock, Calendar, ArrowRight } from 'lucide-react';
import { Article, articles } from './articlesData';

interface ArticleReaderProps {
  article: Article;
  onBack: () => void;
}

export const ArticleReader: React.FC<ArticleReaderProps> = ({ article, onBack }) => {
  const [content, setContent] = useState<string>('');
  const [loading, setLoading] = useState(true);

  // Get related articles based on shared tags
  const relatedArticles = useMemo(() => {
    const currentTags = new Set(article.tags);
    return articles
      .filter(a => a.id !== article.id)
      .map(a => ({
        ...a,
        sharedTags: a.tags.filter(tag => currentTags.has(tag)).length
      }))
      .filter(a => a.sharedTags > 0)
      .sort((a, b) => b.sharedTags - a.sharedTags)
      .slice(0, 3);
  }, [article]);

  useEffect(() => {
    const loadArticle = async () => {
      try {
        setLoading(true);
        const response = await fetch(article.filePath);

        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }

        const text = await response.text();
        setContent(text);
      } catch (error) {
        console.error('Error loading article:', error);
        setContent('# Error Loading Article\n\nUnable to load the article content. Please try again later.');
      } finally {
        setLoading(false);
      }
    };

    loadArticle();
  }, [article]);

  // Convert markdown to HTML-like JSX
  const renderMarkdown = (markdown: string) => {
    const lines = markdown.split('\n');
    const elements: JSX.Element[] = [];
    let currentParagraph: string[] = [];
    let inCodeBlock = false;
    let codeContent: string[] = [];

    const flushParagraph = () => {
      if (currentParagraph.length > 0) {
        const text = currentParagraph.join(' ').trim();
        if (text) {
          elements.push(
            <p key={elements.length} className="mb-6 text-[var(--color-text-secondary)] leading-relaxed text-lg">
              {text}
            </p>
          );
        }
        currentParagraph = [];
      }
    };

    lines.forEach((line, index) => {
      // Code blocks
      if (line.startsWith('```')) {
        if (inCodeBlock) {
          elements.push(
            <pre key={elements.length} className="bg-[var(--surface)] border border-[var(--line)] p-5 rounded-xl overflow-x-auto mb-6 text-sm">
              <code className="text-[var(--ink)] font-mono">
                {codeContent.join('\n')}
              </code>
            </pre>
          );
          codeContent = [];
          inCodeBlock = false;
        } else {
          flushParagraph();
          inCodeBlock = true;
        }
        return;
      }

      if (inCodeBlock) {
        codeContent.push(line);
        return;
      }

      // Headers
      if (line.startsWith('# ')) {
        flushParagraph();
        elements.push(
          <h1 key={elements.length} className="text-3xl font-bold mb-6 mt-12 text-[var(--ink)] tracking-tight">
            {line.substring(2)}
          </h1>
        );
      } else if (line.startsWith('## ')) {
        flushParagraph();
        elements.push(
          <h2 key={elements.length} className="text-2xl font-semibold mb-4 mt-10 text-[var(--ink)] tracking-tight">
            {line.substring(3)}
          </h2>
        );
      } else if (line.startsWith('### ')) {
        flushParagraph();
        elements.push(
          <h3 key={elements.length} className="text-xl font-semibold mb-3 mt-8 text-[var(--ink)]">
            {line.substring(4)}
          </h3>
        );
      } else if (line.startsWith('#### ')) {
        flushParagraph();
        elements.push(
          <h4 key={elements.length} className="text-lg font-semibold mb-2 mt-6 text-[var(--muted)]">
            {line.substring(5)}
          </h4>
        );
      }
      // Bullet points
      else if (line.startsWith('- ') || line.startsWith('* ')) {
        flushParagraph();
        elements.push(
          <li key={elements.length} className="ml-6 mb-2 text-[var(--ink)] list-disc marker:text-[var(--subtle)]">
            {line.substring(2)}
          </li>
        );
      }
      // Numbered lists
      else if (/^\d+\.\s/.test(line)) {
        flushParagraph();
        const content = line.replace(/^\d+\.\s/, '');
        elements.push(
          <li key={elements.length} className="ml-6 mb-2 text-[var(--ink)] list-decimal marker:text-[var(--subtle)]">
            {content}
          </li>
        );
      }
      // Blockquotes
      else if (line.startsWith('> ')) {
        flushParagraph();
        elements.push(
          <blockquote key={elements.length} className="border-l-4 border-[var(--accent)]/30 pl-6 py-3 mb-6 text-[var(--muted)] italic bg-[var(--surface)] rounded-r-lg">
            {line.substring(2)}
          </blockquote>
        );
      }
      // Horizontal rule
      else if (line === '---' || line === '***') {
        flushParagraph();
        elements.push(
          <hr key={elements.length} className="my-8 border-[var(--color-border-primary)]" />
        );
      }
      // Empty line - flush current paragraph
      else if (line.trim() === '') {
        flushParagraph();
      }
      // Regular paragraph text
      else {
        currentParagraph.push(line);
      }
    });

    // Flush any remaining paragraph
    flushParagraph();

    return elements;
  };


  if (loading) {
    return (
      <div className="min-h-screen bg-[var(--color-bg-primary)] py-12">
        <div className="max-w-4xl mx-auto px-6">
          <div className="animate-pulse">
            <div className="h-4 bg-[var(--color-bg-secondary)] rounded w-24 mb-8"></div>
            <div className="h-12 bg-[var(--color-bg-secondary)] rounded mb-6"></div>
            <div className="flex gap-6 mb-8">
              <div className="h-4 bg-[var(--color-bg-secondary)] rounded w-32"></div>
              <div className="h-4 bg-[var(--color-bg-secondary)] rounded w-24"></div>
              <div className="h-4 bg-[var(--color-bg-secondary)] rounded w-28"></div>
            </div>
            <div className="space-y-4">
              <div className="h-4 bg-[var(--color-bg-secondary)] rounded"></div>
              <div className="h-4 bg-[var(--color-bg-secondary)] rounded"></div>
              <div className="h-4 bg-[var(--color-bg-secondary)] rounded w-5/6"></div>
            </div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-[var(--bg)]">
      <article className="py-12">
        <div className="max-w-4xl mx-auto px-6">
          {/* Back button */}
          <button
            onClick={onBack}
            className="inline-flex items-center gap-2 text-[var(--subtle)] hover:text-[var(--ink)] transition-all duration-200 hover:gap-3 mb-8 text-sm"
          >
            <ArrowLeft className="w-4 h-4" />
            <span className="font-medium">Back to Articles</span>
          </button>

          {/* Article header */}
          <header className="mb-12">
            {/* Category badge */}
            <div className="mb-4">
              <span className="inline-flex items-center px-3 py-1.5 rounded-full text-xs font-semibold bg-[var(--accent)]/8 text-[var(--accent)] border border-[var(--accent)]/15 uppercase tracking-wide">
                {article.category}
              </span>
            </div>

            <h1 className="text-3xl lg:text-5xl font-bold text-[var(--ink)] mb-6 leading-[1.15] tracking-tight">
              {article.title}
            </h1>

            <div className="flex flex-wrap items-center gap-6 text-sm text-[var(--color-text-secondary)]">
              <div className="flex items-center gap-2">
                {article.authors.map((author, idx) => (
                  <div key={idx} className="px-3 py-1.5 rounded-full bg-[var(--surface)] border border-[var(--line)] text-xs font-medium text-[var(--muted)]">
                    {author}
                  </div>
                ))}
              </div>
              <div className="flex items-center gap-1 text-[var(--color-text-tertiary)]">
                <Calendar className="w-4 h-4" />
                <span>{new Date(article.date).toLocaleDateString('en-US', {
                  year: 'numeric',
                  month: 'long',
                  day: 'numeric'
                })}</span>
              </div>
              <div className="flex items-center gap-1 text-[var(--color-text-tertiary)]">
                <Clock className="w-4 h-4" />
                <span>{article.readTime}</span>
              </div>
            </div>

            <div className="mt-8 p-6 bg-[var(--surface)] border border-[var(--line)] rounded-xl">
              <p className="text-lg text-[var(--muted)] leading-relaxed font-light">
                {article.excerpt}
              </p>
            </div>
          </header>

          {/* Article content */}
          <div className="prose prose-lg max-w-none text-[var(--ink)] leading-[1.75] space-y-6"
               style={{
                 fontSize: '1.0625rem',
                 lineHeight: '1.75',
                 fontFamily: 'Inter, system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif'
               }}>
            {renderMarkdown(content)}
          </div>

          {/* Tags */}
          <div className="mt-12 pt-8 border-t border-[var(--line)]">
            <div className="flex flex-wrap gap-2">
              {article.tags.map((tag: string) => (
                <span key={tag} className="px-3 py-1.5 bg-[var(--surface)] text-[var(--muted)] rounded-lg text-sm font-medium hover:bg-[var(--elevated)] transition-colors border border-[var(--line)]">
                  #{tag}
                </span>
              ))}
            </div>
          </div>

          {/* Related Articles */}
          {relatedArticles.length > 0 && (
            <section className="mt-16 pt-8 border-t border-[var(--line)]">
              <h2 className="text-2xl font-bold text-[var(--ink)] mb-8">Related Articles</h2>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                {relatedArticles.map((relatedArticle: any) => (
                  <article
                    key={relatedArticle.id}
                    className="group cursor-pointer bg-[var(--surface)] border border-[var(--line)] rounded-xl p-6 hover:shadow-md hover:bg-[var(--elevated)] hover:-translate-y-0.5 transition-all duration-200"
                    onClick={() => {
                      window.scrollTo(0, 0);
                      onBack();
                      // Small delay to ensure state updates
                      setTimeout(() => {
                        const event = new CustomEvent('selectArticle', { detail: relatedArticle });
                        window.dispatchEvent(event);
                      }, 100);
                    }}
                  >
                    <div className="flex items-center justify-between mb-3">
                      <span className="text-xs font-semibold uppercase tracking-wide text-[var(--accent)]">
                        {relatedArticle.category}
                      </span>
                      <span className="text-xs text-[var(--subtle)]">{relatedArticle.readTime}</span>
                    </div>
                    <h3 className="text-lg font-semibold text-[var(--ink)] mb-2 group-hover:text-[var(--accent)] transition-colors line-clamp-2">
                      {relatedArticle.title}
                    </h3>
                    <p className="text-sm text-[var(--muted)] line-clamp-3 mb-4">
                      {relatedArticle.excerpt}
                    </p>
                    <div className="flex items-center justify-between">
                      <span className="text-xs text-[var(--subtle)]">
                        {new Date(relatedArticle.date).toLocaleDateString()}
                      </span>
                      <span className="inline-flex items-center gap-1 text-xs font-medium text-[var(--accent)] group-hover:gap-2 transition-all">
                        Read <ArrowRight className="w-3 h-3" />
                      </span>
                    </div>
                  </article>
                ))}
              </div>
            </section>
          )}

          {/* Footer */}
          <footer className="mt-16 pt-8 border-t border-[var(--line)]">
            <div className="flex items-center justify-between">
              <button
                onClick={onBack}
                className="inline-flex items-center gap-2 text-[var(--accent)] hover:text-[var(--accent)]/80 font-medium transition-colors"
              >
                <ArrowLeft className="w-4 h-4" />
                <span>Back to all articles</span>
              </button>

              <div className="text-sm text-[var(--subtle)]">
                Last updated {new Date(article.date).toLocaleDateString('en-US', {
                  year: 'numeric',
                  month: 'long',
                  day: 'numeric'
                })}
              </div>
            </div>
          </footer>
        </div>
      </article>
    </div>
  );
};