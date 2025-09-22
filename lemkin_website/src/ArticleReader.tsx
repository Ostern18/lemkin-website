import React, { useState, useEffect } from 'react';
import { ArrowLeft, Clock, Calendar } from 'lucide-react';
import { Article } from './articlesData';

interface ArticleReaderProps {
  article: Article;
  onBack: () => void;
}

export const ArticleReader: React.FC<ArticleReaderProps> = ({ article, onBack }) => {
  const [content, setContent] = useState<string>('');
  const [loading, setLoading] = useState(true);

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
            <pre key={elements.length} className="bg-[var(--color-bg-secondary)] border border-[var(--color-border-primary)] p-4 rounded-lg overflow-x-auto mb-6">
              <code className="text-sm text-[var(--color-text-primary)]">
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
          <h1 key={elements.length} className="text-4xl font-bold mb-8 mt-8 text-[var(--color-text-primary)]">
            {line.substring(2)}
          </h1>
        );
      } else if (line.startsWith('## ')) {
        flushParagraph();
        elements.push(
          <h2 key={elements.length} className="text-3xl font-semibold mb-6 mt-10 text-[var(--color-text-primary)]">
            {line.substring(3)}
          </h2>
        );
      } else if (line.startsWith('### ')) {
        flushParagraph();
        elements.push(
          <h3 key={elements.length} className="text-2xl font-semibold mb-4 mt-8 text-[var(--color-text-primary)]">
            {line.substring(4)}
          </h3>
        );
      } else if (line.startsWith('#### ')) {
        flushParagraph();
        elements.push(
          <h4 key={elements.length} className="text-xl font-semibold mb-3 mt-6 text-[var(--color-text-secondary)]">
            {line.substring(5)}
          </h4>
        );
      }
      // Bullet points
      else if (line.startsWith('- ') || line.startsWith('* ')) {
        flushParagraph();
        elements.push(
          <li key={elements.length} className="ml-6 mb-2 text-[var(--color-text-secondary)] list-disc">
            {line.substring(2)}
          </li>
        );
      }
      // Numbered lists
      else if (/^\d+\.\s/.test(line)) {
        flushParagraph();
        const content = line.replace(/^\d+\.\s/, '');
        elements.push(
          <li key={elements.length} className="ml-6 mb-2 text-[var(--color-text-secondary)] list-decimal">
            {content}
          </li>
        );
      }
      // Blockquotes
      else if (line.startsWith('> ')) {
        flushParagraph();
        elements.push(
          <blockquote key={elements.length} className="border-l-4 border-[var(--color-primary)] pl-4 py-2 mb-6 italic text-[var(--color-text-secondary)]">
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
    <div className="min-h-screen bg-[var(--color-bg-primary)]">
      <article className="py-12">
        <div className="max-w-4xl mx-auto px-6">
          {/* Back button */}
          <button
            onClick={onBack}
            className="inline-flex items-center gap-2 text-[var(--color-text-secondary)] hover:text-[var(--color-text-primary)] transition-all duration-200 hover:gap-3 mb-6"
          >
            <ArrowLeft className="w-4 h-4" />
            <span className="text-sm font-medium">All Articles</span>
          </button>

          {/* Article header */}
          <header className="mb-12">
            {/* Category badge */}
            <div className="mb-4">
              <span className="inline-flex items-center px-3 py-1 rounded-full text-xs font-medium bg-[var(--color-primary)]/10 text-[var(--color-primary)] border border-[var(--color-primary)]/20">
                {article.category}
              </span>
            </div>

            <h1 className="text-5xl font-bold text-[var(--color-text-primary)] mb-8 leading-[1.1] tracking-tight">
              {article.title}
            </h1>

            <div className="flex flex-wrap items-center gap-6 text-sm text-[var(--color-text-secondary)]">
              <div className="flex items-center gap-2">
                <div className="w-8 h-8 rounded-full bg-[var(--color-primary)]/10 flex items-center justify-center">
                  <span className="text-xs font-medium text-[var(--color-primary)]">
                    {article.author.split(' ').map((n: string) => n[0]).join('')}
                  </span>
                </div>
                <span className="font-medium">{article.author}</span>
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

            <div className="mt-6 p-4 bg-[var(--color-bg-surface)] border border-[var(--color-border-primary)] rounded-lg">
              <p className="text-[var(--color-text-secondary)] italic">
                {article.excerpt}
              </p>
            </div>
          </header>

          {/* Article content */}
          <div className="text-[var(--color-text-primary)] leading-relaxed space-y-8"
               style={{
                 fontSize: '1.125rem',
                 lineHeight: '1.8',
                 fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif'
               }}>
            {renderMarkdown(content)}
          </div>

          {/* Footer */}
          <footer className="mt-16 pt-8 border-t border-[var(--color-border-primary)]">
            <div className="flex flex-wrap gap-2 mb-8">
              {article.tags.map((tag: string) => (
                <span key={tag} className="px-3 py-1.5 bg-[var(--color-bg-secondary)] text-[var(--color-text-secondary)] rounded-lg text-sm font-medium hover:bg-[var(--color-bg-tertiary)] transition-colors">
                  {tag}
                </span>
              ))}
            </div>

            <div className="flex items-center justify-between pt-6">
              <button
                onClick={onBack}
                className="inline-flex items-center gap-2 text-[var(--color-primary)] hover:text-[var(--color-primary)]/80 font-medium transition-colors"
              >
                <ArrowLeft className="w-4 h-4" />
                <span>Back to all articles</span>
              </button>

              <div className="text-sm text-[var(--color-text-tertiary)]">
                Published {new Date(article.date).toLocaleDateString('en-US', {
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