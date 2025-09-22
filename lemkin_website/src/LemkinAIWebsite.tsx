import React, { useState, useEffect, createContext, useContext } from 'react';
import { X, Search, Calendar, Clock, AlertCircle, CheckCircle, Book, Code, Users, Mail, ExternalLink, Github, Twitter, FileText, Download, ArrowRight, ArrowLeft, Copy, Check, Scale, Shield, Eye, Gavel, Grid, Package, Sun, Moon, User, Share, Folder } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { RouteProgressBar, Pressable, MotionCard } from './motion';
import { ArticlesPage } from './ArticlesPage';
import ModelsPageRevised from './ModelsPageRevised';
import {
  models,
  getFeaturedModels,
  modelCategories,
  modelTypes,
  modelStatuses,
  moduleTypes,
  tierFilters,
  getModelsByModuleType,
  getModelsByTier,
  getProductionReadyItems,
  getImplementationReadyModules
} from './modelsData';

// Theme Context
interface ThemeContextType {
  theme: 'light' | 'dark' | 'system';
  resolvedTheme: 'light' | 'dark';
  setTheme: (theme: 'light' | 'dark' | 'system') => void;
}

const ThemeContext = createContext<ThemeContextType | undefined>(undefined);

const useTheme = () => {
  const context = useContext(ThemeContext);
  if (!context) throw new Error('useTheme must be used within ThemeProvider');
  return context;
};

const ThemeProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [theme, setTheme] = useState<'light' | 'dark' | 'system'>(() => {
    if (typeof window === "undefined") return 'system';
    return (localStorage.getItem("theme") as 'light' | 'dark' | 'system') ?? 'system';
  });

  const [systemTheme, setSystemTheme] = useState<'light' | 'dark'>(() => {
    if (typeof window === "undefined") return 'light';
    return window.matchMedia("(prefers-color-scheme: dark)").matches ? 'dark' : 'light';
  });

  const resolvedTheme = theme === 'system' ? systemTheme : theme;

  useEffect(() => {
    const mq = window.matchMedia('(prefers-color-scheme: dark)');
    const onChange = (e: MediaQueryListEvent) => {
      setSystemTheme(e.matches ? 'dark' : 'light');
    };
    mq.addEventListener('change', onChange);
    return () => mq.removeEventListener('change', onChange);
  }, []);

  useEffect(() => {
    const root = document.documentElement;
    if (resolvedTheme === 'dark') {
      root.classList.add('dark');
    } else {
      root.classList.remove('dark');
    }
    localStorage.setItem("theme", theme);
  }, [theme, resolvedTheme]);

  return (
    <ThemeContext.Provider value={{ theme, resolvedTheme, setTheme }}>
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
  const [progress, setProgress] = useState(0);

  const navigate = (path: string) => {
    setProgress(10);
    setTimeout(() => setProgress(55), 80);
    setTimeout(() => setProgress(85), 160);
    setCurrentPath(path);
    window.scrollTo(0, 0);
    setTimeout(() => setProgress(100), 240);
    setTimeout(() => setProgress(0), 480); // reset after finish
  };

  return (
    <RouterContext.Provider value={{ currentPath, navigate }}>
      <AnimatePresence>
        {progress > 0 && <RouteProgressBar progress={progress} />}
      </AnimatePresence>
      {children}
    </RouterContext.Provider>
  );
};

// Practitioners' Brief Component with Robust States
interface Brief {
  title: string;
  content: string;
  author: string;
  date: string;
}

interface PractitionersBriefProps {
  state: 'loading' | 'empty' | 'ready';
  data?: Brief;
}

const PractitionersBrief: React.FC<PractitionersBriefProps> = ({ state, data }) => {
  return (
    <section
      className="container-desktop py-14"
      aria-busy={state === 'loading'}
      aria-live="polite"
    >
      <div className="card p-6" style={{ minHeight: 240 }}>
        <header className="flex items-center justify-between mb-6">
          <h2 className="text-xl font-semibold text-[var(--ink)]">Practitioners' Brief</h2>
          <a className="btn-outline" href="/docs/methodology">
            View methodology
          </a>
        </header>

        {state === 'loading' && (
          <div className="relative" role="status" aria-label="Loading brief">
            <div className="absolute inset-0 bg-gradient-to-r from-transparent via-[var(--surface)] to-transparent animate-shimmer" />
            <div className="space-y-3">
              {[1, 2, 3].map(i => (
                <div key={i} className="flex items-center gap-3">
                  <div className="w-1 h-12 bg-[var(--line)] rounded-full opacity-30" />
                  <div className="flex-1 space-y-2">
                    <div className="h-3 bg-[var(--surface)] rounded-sm" style={{ width: `${85 - i * 10}%` }} />
                    <div className="h-2 bg-[var(--surface)] rounded-sm opacity-60" style={{ width: `${70 - i * 8}%` }} />
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {state === 'empty' && (
          <div className="relative py-12">
            <div className="absolute inset-0 bg-gradient-radial from-[var(--accent)]/5 to-transparent opacity-50" />
            <div className="relative text-center">
              <div className="inline-flex items-center justify-center w-20 h-20 mx-auto mb-4 rounded-2xl bg-gradient-to-br from-[var(--surface)] to-[var(--elevated)] border border-[var(--line)] shadow-sm">
                <FileText className="w-10 h-10 text-[var(--subtle)]" />
              </div>
              <h3 className="text-lg font-medium text-[var(--ink)] mb-2">Preparing Intelligence Brief</h3>
              <p className="text-sm text-[var(--muted)] mb-6 max-w-sm mx-auto">
                Our team is synthesizing guidance from recent deployments and field operations.
              </p>
              <div className="flex gap-3 justify-center">
                <button className="btn-primary">
                  Explore Documentation →
                </button>
                <a href="/briefs" className="btn-outline">
                  See latest briefs
                </a>
              </div>
            </div>
          </div>
        )}

        {state === 'ready' && data && (
          <div>
            <h3 className="text-lg font-semibold text-[var(--ink)] mb-3">{data.title}</h3>
            <p className="text-[var(--muted)] mb-4 leading-relaxed">{data.content}</p>
            <div className="flex items-center justify-between text-sm text-[var(--subtle)]">
              <span>By {data.author}</span>
              <time dateTime={data.date}>
                {new Date(data.date).toLocaleDateString()}
              </time>
            </div>
          </div>
        )}
      </div>
    </section>
  );
};

// Table Helper Components
interface ThProps {
  children: React.ReactNode;
  align?: 'left' | 'right' | 'center';
  sticky?: 'left' | 'right';
  sortable?: boolean;
  sortDirection?: 'ascending' | 'descending' | 'none';
}

const Th: React.FC<ThProps> = ({ children, align = 'left', sticky, sortable, sortDirection }) => {
  const classes = [
    'text-sm font-medium px-4 py-3 border-b border-[var(--line)]',
    align === 'right' && 'text-right',
    align === 'center' && 'text-center',
    sticky === 'left' && 'sticky left-0 bg-[var(--surface)] z-10 min-w-[120px]',
    sticky === 'right' && 'sticky right-0 bg-[var(--surface)] z-10'
  ].filter(Boolean).join(' ');

  return (
    <th
      scope="col"
      className={classes}
      aria-sort={sortable ? (sortDirection === 'none' ? undefined : sortDirection) : undefined}
    >
      {children}
    </th>
  );
};

interface TdProps {
  children: React.ReactNode;
  align?: 'left' | 'right' | 'center';
  sticky?: 'left' | 'right';
  compact?: boolean;
  numeric?: boolean;
}

const Td: React.FC<TdProps> = ({ children, align = 'left', sticky, compact, numeric }) => {
  const classes = [
    'px-4', compact ? 'py-1.5' : 'py-2.5',
    'text-[var(--muted)]',
    numeric && 'font-mono tabular-nums text-sm',
    align === 'right' && 'text-right',
    align === 'center' && 'text-center',
    sticky === 'left' && 'sticky left-0 bg-[var(--bg)] min-w-[120px]',
    sticky === 'right' && 'sticky right-0 bg-[var(--bg)]',
    'focus-within:outline focus-within:outline-2 focus-within:outline-offset-[-2px] focus-within:outline-[var(--accent)]/20'
  ].filter(Boolean).join(' ');

  return <td className={classes}>{children}</td>;
};

const ModelCell: React.FC<{ model: any }> = ({ model }) => (
  <div className="flex items-center gap-3">
    <div className="w-8 h-8 rounded bg-[var(--accent)]/10 flex items-center justify-center">
      <Scale className="w-4 h-4 text-[var(--accent)]" />
    </div>
    <div>
      <div className="text-sm font-medium text-[var(--ink)]">{model.name}</div>
      <div className="text-xs text-[var(--subtle)] max-w-[300px] truncate">{model.description}</div>
    </div>
  </div>
);

const StatusTag: React.FC<{ status: string }> = ({ status }) => {
  const colors = {
    stable: 'bg-[var(--success)]/10 text-[var(--success)]',
    beta: 'bg-[var(--warning)]/10 text-[var(--warning)]',
    deprecated: 'bg-[var(--danger)]/10 text-[var(--danger)]'
  };

  return (
    <span className={`px-2 py-1 rounded-full text-xs font-medium ${colors[status as keyof typeof colors] || colors.stable}`}>
      {status}
    </span>
  );
};

const ViewButton: React.FC<{ onClick: () => void }> = ({ onClick }) => (
  <button
    onClick={onClick}
    className="text-sm text-[var(--accent)] hover:text-[var(--accent-ink)] focus:ring-1 focus:ring-[var(--accent)]/40 rounded px-2 py-1 transition-colors"
  >
    View
  </button>
);

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
    accuracy: 94.7,
    precision: 93.2,
    recall: 95.1,
    f1Score: 94.1,
    evaluator: 'UN IRMCT'
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
    accuracy: 91.2,
    precision: 89.8,
    recall: 92.5,
    f1Score: 91.1,
    evaluator: 'HRW Digital Lab'
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
    accuracy: 89.5,
    precision: 88.1,
    recall: 90.4,
    f1Score: 89.2,
    evaluator: 'ICC Registry'
  }
];

const mockPractitionerBriefs = [
  {
    id: 'large-language-models-legal-analysis',
    title: 'Large Language Models for Legal Document Analysis',
    excerpt: 'Balancing transformative potential with critical risks in human rights investigations when using LLMs for legal document processing.',
    author: 'Oliver Stern',
    date: '2025-07-15',
    lastReviewed: 'Jul 2025',
    readTime: '16 min',
    category: 'AI & Technology',
    tags: ['Legal Tech', 'Evidence Processing', 'Case Studies'],
    roles: ['Researchers', 'Prosecutors'],
    peerReviewed: true,
    content: `Large language models (LLMs) have revolutionized the field of legal document analysis, particularly in the context of international criminal justice and human rights investigations. These sophisticated AI systems offer unprecedented capabilities in processing vast amounts of textual evidence, identifying patterns across multiple languages, and extracting crucial information that might otherwise remain hidden in massive document collections.

In recent deployments at international tribunals, LLMs have demonstrated their ability to process millions of pages of evidence in days rather than years. The International Criminal Court's pilot program, which analyzed over 3 million documents related to alleged war crimes, reduced initial review time by 94% while maintaining accuracy rates above 97% for document classification tasks.

However, the implementation of LLMs in legal contexts requires careful consideration of several critical factors. First, the chain of custody for AI-processed evidence must be meticulously maintained. Every transformation, analysis, and output generated by the model must be logged, versioned, and cryptographically verified to ensure admissibility in court proceedings.

The risk of hallucination—where LLMs generate plausible but factually incorrect information—poses particular challenges in legal settings. Our framework addresses this through multiple validation layers: cross-referencing with verified databases, implementing confidence thresholds, and mandatory human review for all critical findings. No AI-generated analysis should ever be presented as evidence without thorough human verification.

Bias mitigation represents another crucial consideration. LLMs trained on general datasets may perpetuate societal biases that could impact legal proceedings. We've developed specialized fine-tuning protocols using diverse legal corpora from multiple jurisdictions, ensuring balanced representation across different legal traditions and languages. Regular bias audits, conducted by independent experts, help identify and correct any systematic issues.

The practical implementation of LLMs in legal document analysis follows a structured workflow. Initial document ingestion involves optical character recognition for scanned materials, language detection, and metadata extraction. The preprocessing phase includes deduplication, format standardization, and the creation of searchable indices. During analysis, the LLM performs tasks such as entity extraction, relationship mapping, timeline construction, and thematic categorization.

One particularly powerful application involves cross-lingual analysis. In investigations spanning multiple countries, LLMs can identify connections between documents in different languages that human analysts might miss. For instance, in a recent case involving financial crimes across three continents, the system identified matching transaction patterns described in documents written in Arabic, Mandarin, and Spanish—a connection that would have taken human investigators months to discover.

Privacy and data protection requirements add another layer of complexity. Legal documents often contain sensitive personal information that must be handled according to strict regulatory frameworks. Our implementation includes automated redaction capabilities that identify and protect personally identifiable information while preserving the document's evidentiary value. The system maintains both redacted and unredacted versions with appropriate access controls.

The integration of LLMs with existing legal technology infrastructure requires careful planning. Most law firms and legal departments operate legacy systems that weren't designed for AI integration. We've developed middleware solutions that allow LLMs to interface with traditional case management systems, evidence databases, and document management platforms without requiring wholesale system replacement.

Looking forward, the evolution of multimodal models promises even greater capabilities. Future systems will analyze not just text but also images, audio, and video evidence in an integrated manner. This could enable comprehensive analysis of mixed media evidence packages, automatically correlating witness testimonies with photographic evidence and documentary records.

The ethical framework governing LLM use in legal contexts continues to evolve. Key principles include transparency in AI decision-making, the right to human review, and the obligation to disclose AI involvement in evidence processing. Legal professionals must understand both the capabilities and limitations of these systems to use them effectively and ethically.

Training requirements for legal professionals represent a significant implementation challenge. Lawyers, investigators, and judges need sufficient understanding of LLM technology to evaluate AI-assisted analysis critically. We've developed comprehensive training programs that cover both technical concepts and practical applications, ensuring legal professionals can leverage these tools while maintaining appropriate skepticism.

Cost considerations often determine the feasibility of LLM deployment. While the initial investment in infrastructure and training can be substantial, the long-term efficiency gains typically justify the expense. Organizations processing large volumes of documents regularly see return on investment within 12-18 months through reduced manual review costs and faster case resolution.

The future of LLMs in legal document analysis will likely involve increasing specialization. Rather than general-purpose models, we anticipate the development of models trained specifically for different areas of law, jurisdictions, and types of legal documents. This specialization will improve accuracy and reduce the computational resources required for specific tasks.

As we continue to refine these systems, the focus remains on augmenting rather than replacing human legal professionals. LLMs excel at processing vast amounts of information and identifying patterns, but human judgment remains essential for interpreting legal significance, making strategic decisions, and ensuring justice is served. The most effective implementations treat LLMs as powerful tools that enhance human capabilities rather than autonomous systems that operate independently.`
  },
  {
    id: 'satellite-imagery-atrocity-documentation',
    title: 'Satellite Imagery Analysis for Atrocity Documentation',
    excerpt: 'Transforming pixels into evidence for international justice through advanced satellite image analysis and AI-powered change detection.',
    author: 'Oliver Stern',
    date: '2025-06-10',
    lastReviewed: 'Jun 2025',
    readTime: '18 min',
    category: 'Evidence Collection',
    tags: ['Remote Sensing', 'Digital Forensics', 'Case Studies'],
    roles: ['Investigators', 'Researchers'],
    peerReviewed: true,
    content: `Satellite imagery has become an indispensable tool in documenting atrocity crimes, providing objective, time-stamped evidence of events in regions where ground access is impossible or too dangerous. The integration of artificial intelligence with satellite analysis has transformed our ability to detect, document, and present evidence of mass atrocities, from destroyed villages to mass grave sites.

The evolution of commercial satellite technology has democratized access to high-resolution imagery. With resolutions now reaching 30 centimeters per pixel, investigators can identify individual vehicles, count displaced persons in camps, and document structural damage with unprecedented clarity. The temporal resolution—how frequently satellites capture images of the same location—has improved to daily or even hourly coverage for areas of interest.

Our AI-powered analysis pipeline begins with automated change detection algorithms that continuously monitor areas of concern. These systems can process thousands of square kilometers daily, flagging significant changes for human review. In conflict zones, this might include new crater patterns indicating artillery strikes, thermal signatures suggesting burning structures, or earthwork patterns consistent with mass grave construction.

The verification process for satellite evidence involves multiple validation steps. First, we authenticate the imagery itself, verifying metadata, checking for signs of manipulation, and confirming the chain of custody from the satellite operator. Next, we correlate satellite observations with ground reports, social media posts, and other intelligence sources to build a comprehensive picture of events.

One powerful application involves tracking population displacement. By analyzing changes in nighttime lights, vehicle movements, and temporary structure appearances, we can estimate refugee flows and identify humanitarian crises as they develop. In a recent analysis of displacement in Eastern Europe, our systems detected the establishment of informal refugee camps within 48 hours of their formation, enabling rapid humanitarian response.

The challenge of weather and atmospheric conditions requires sophisticated preprocessing techniques. Cloud cover, atmospheric haze, and seasonal variations can obscure critical details. We employ multiple strategies to address these challenges: using synthetic aperture radar (SAR) that penetrates clouds, combining observations from multiple satellites, and applying atmospheric correction algorithms to improve image clarity.

Machine learning models trained on specific atrocity indicators have proven particularly effective. These models can identify mass grave sites by detecting characteristic soil disturbances, vegetation changes, and access road patterns. In validation studies, our grave detection model achieved 89% accuracy in identifying known sites while generating fewer than 3% false positives—a critical metric for maintaining credibility in legal proceedings.

The temporal analysis capabilities of satellite imagery provide unique evidentiary value. By analyzing image sequences over time, we can establish timelines of events, identify responsible parties based on military unit movements, and document the progression of destruction. This temporal dimension often proves crucial in establishing intent and systematic patterns required for proving crimes against humanity.

Integrating satellite analysis with other data sources multiplies its effectiveness. We've developed systems that automatically correlate satellite observations with signals intelligence, social media posts, and witness testimonies. This multi-source validation strengthens the evidence and helps construct comprehensive narratives of events.

The legal admissibility of satellite evidence requires careful attention to documentation and methodology. Every analytical step must be recorded, from raw image acquisition through final analysis. We maintain detailed logs of processing parameters, algorithm versions, and analyst observations. Expert witnesses must be prepared to explain both the capabilities and limitations of satellite analysis to judges and juries unfamiliar with the technology.

Cultural and contextual interpretation remains essential for accurate analysis. What might appear as suspicious activity to an algorithm could be normal agricultural practice in certain regions. Our analysis teams include regional experts who provide crucial context for interpreting satellite observations. This human expertise prevents misinterpretation that could undermine the credibility of satellite evidence.

The cost structure of satellite imagery has evolved significantly. While tasking new imagery remains expensive—often thousands of dollars per scene—archived imagery is increasingly affordable. Many providers now offer subscription models that provide access to vast historical archives, enabling retrospective investigations of past atrocities.

Privacy and sovereignty concerns complicate satellite imagery use in some jurisdictions. Some nations restrict the distribution of high-resolution imagery of their territory, while others impose licensing requirements on imagery analysis. Navigating these legal frameworks requires careful attention to local regulations and international agreements.

Emerging technologies promise to further enhance satellite-based atrocity documentation. Hyperspectral imaging can detect chemical signatures associated with certain weapons or burial sites. Artificial intelligence models are becoming increasingly sophisticated at identifying subtle patterns indicative of human rights violations. Constellation satellites providing continuous coverage will eliminate temporal gaps in monitoring.

The proliferation of small satellites has introduced new challenges and opportunities. While these systems provide more frequent coverage at lower cost, their varying quality and calibration standards complicate standardized analysis. We've developed quality assessment protocols to evaluate and integrate data from diverse satellite sources while maintaining evidentiary standards.

Collaboration between human rights organizations, academic institutions, and commercial satellite operators has created a robust ecosystem for atrocity documentation. Data sharing agreements, standardized methodologies, and coordinated monitoring efforts amplify the impact of individual organizations. This collaborative approach has proven particularly effective in rapid response scenarios.

Training investigators in satellite imagery analysis requires balancing technical competence with practical application. Our training programs emphasize hands-on analysis of real cases, teaching investigators to identify relevant features, assess image quality, and understand analytical limitations. Regular workshops keep practitioners updated on new technologies and methodologies.

The psychological impact of analyzing atrocity imagery deserves consideration. Investigators reviewing evidence of mass violence require support systems to manage secondary trauma. We've implemented wellness programs, peer support networks, and rotation schedules to protect the mental health of our analysts while maintaining operational effectiveness.

Looking ahead, the integration of satellite imagery with other remote sensing technologies will create even more powerful investigative tools. Combining satellite observations with drone footage, ground-based sensors, and crowdsourced information will provide unprecedented capability to document and prevent atrocities. The challenge lies not in collecting data but in processing, verifying, and presenting it effectively for justice and accountability.`
  },
  {
    id: 'human-in-loop-imperative',
    title: 'The Human-in-the-Loop Imperative',
    excerpt: 'Why AI can never replace investigators and the critical importance of human oversight in AI-assisted legal processes.',
    author: 'Oliver Stern',
    date: '2025-05-08',
    lastReviewed: 'May 2025',
    readTime: '14 min',
    category: 'Ethics & Governance',
    tags: ['Legal Tech', 'Ethics & Governance', 'Best Practices'],
    roles: ['Prosecutors', 'Researchers'],
    peerReviewed: true,
    content: `The integration of artificial intelligence into legal and investigative processes has generated both enthusiasm and concern within the international justice community. While AI offers powerful capabilities for processing evidence and identifying patterns, the human-in-the-loop principle remains not just important but absolutely essential for maintaining the integrity, legitimacy, and effectiveness of legal proceedings.

The fundamental limitation of AI systems lies not in their computational power but in their inability to understand context, exercise judgment, and navigate the nuanced ethical terrain of human rights investigations. AI can identify that a document contains certain keywords or that an image shows specific objects, but it cannot grasp the human suffering behind the evidence or the broader implications of its findings for justice and reconciliation.

Consider the process of witness testimony analysis. An AI system might accurately transcribe speech, identify inconsistencies, and flag potential areas of concern. However, only a human investigator can assess whether inconsistencies stem from trauma, cultural communication patterns, or deliberate deception. The ability to read body language, understand cultural context, and exercise empathy cannot be reduced to algorithms.

The legal doctrine of mens rea—the guilty mind—illustrates another crucial limitation. Determining intent requires understanding not just actions but motivations, circumstances, and states of mind. While AI can identify patterns suggesting intentional behavior, the final determination of criminal intent requires human judgment informed by legal expertise and understanding of human psychology.

Our framework implements multiple checkpoints where human oversight is mandatory. These include: initial assessment of AI-generated leads, validation of pattern recognition results, interpretation of ambiguous findings, and all decisions affecting fundamental rights. No AI recommendation proceeds to action without explicit human approval, and humans retain the authority to override any AI suggestion.

The risk of automation bias—the tendency to over-rely on automated systems—poses particular challenges in legal contexts. Investigators might unconsciously defer to AI recommendations, especially when under time pressure or facing complex technical evidence. Combat this requires training that emphasizes critical evaluation of AI outputs and regular exercises where investigators practice identifying AI errors.

Accountability represents another fundamental reason for maintaining human oversight. Legal proceedings require clear chains of responsibility. When decisions affect human rights and liberty, there must be identifiable individuals who can be held accountable. AI systems, regardless of their sophistication, cannot bear moral or legal responsibility for their outputs.

The dynamic nature of conflict and human rights violations demands adaptive responses that current AI systems cannot provide. Perpetrators continuously evolve their methods to avoid detection. Human investigators can recognize novel patterns of abuse, understand emerging tactics, and adapt their approaches accordingly. AI systems, trained on historical data, may miss entirely new forms of violations.

Cultural competence exemplifies capabilities that remain uniquely human. Understanding how trauma manifests differently across cultures, recognizing culturally specific forms of violence, and interpreting evidence within appropriate cultural contexts require deep human understanding developed through experience and empathy.

The implementation of human-in-the-loop systems requires careful design to maximize both efficiency and oversight. We've developed interfaces that present AI findings alongside confidence scores, alternative interpretations, and relevant context. Investigators can quickly review AI-generated insights while maintaining critical distance and independent judgment.

Training programs for human-in-the-loop systems must address both technical and ethical dimensions. Investigators need to understand AI capabilities and limitations, recognize common failure modes, and develop strategies for effective oversight. Regular calibration exercises, where investigators review known cases with and without AI assistance, help maintain appropriate skepticism.

The psychological dynamics of human-AI collaboration merit careful consideration. Investigators may experience frustration when required to review seemingly obvious AI conclusions, or conversely, may feel overwhelmed by the volume of AI-generated insights. Effective system design balances automation benefits with meaningful human engagement.

Legal frameworks increasingly recognize the necessity of human oversight in AI-assisted decision-making. The European Union's AI Act, for instance, mandates human oversight for high-risk AI applications including those used in law enforcement and justice. These regulatory requirements align with ethical imperatives for maintaining human agency in consequential decisions.

The economic argument for human-in-the-loop systems proves compelling when considering the total cost of errors. While full automation might seem more efficient, the reputational, legal, and human costs of wrongful AI-driven decisions far exceed the investment in maintaining robust human oversight. A single wrongful conviction based on unsupervised AI analysis could undermine entire judicial proceedings.

Transparency requirements further reinforce the need for human involvement. Courts increasingly demand explanations for how evidence was processed and conclusions reached. Human investigators can provide narrative explanations, answer questions, and justify decisions in ways that pure AI systems cannot, maintaining the transparency essential for legal legitimacy.

The evolution of AI capabilities will not eliminate the need for human oversight but will shift its nature. As AI systems become more sophisticated, human roles will evolve from direct review of all outputs to more strategic oversight, policy setting, and intervention in edge cases. This evolution requires continuous adaptation of training and oversight frameworks.

International cooperation in developing human-in-the-loop standards strengthens the global justice system. Shared protocols for human oversight, common training standards, and collaborative development of best practices ensure consistent application of human judgment across jurisdictions and organizations.

The future of human-in-the-loop systems lies in achieving optimal collaboration between human and artificial intelligence. Rather than viewing AI as a replacement for human investigators, we must design systems that amplify human capabilities while preserving human agency, judgment, and accountability. This balanced approach ensures that technology serves justice rather than supplanting it.`
  },
  {
    id: 'chain-custody-digital-age',
    title: 'Chain of Custody in the Digital Age',
    excerpt: 'Cryptographic evidence verification methods for maintaining integrity in human rights investigations using blockchain and hash verification.',
    author: 'Oliver Stern',
    date: '2025-04-05',
    lastReviewed: 'Apr 2025',
    readTime: '12 min',
    category: 'Digital Forensics',
    tags: ['Digital Forensics', 'Evidence Processing', 'Best Practices'],
    roles: ['Investigators', 'Prosecutors'],
    peerReviewed: true,
    content: `The digital transformation of evidence collection has fundamentally altered how we maintain chain of custody in legal proceedings. Traditional paper-based documentation systems, designed for physical evidence, prove inadequate for digital materials that can be copied infinitely and modified without trace. Cryptographic verification methods now provide the mathematical certainty required to ensure evidence integrity from collection through trial.

The cornerstone of digital chain of custody lies in cryptographic hashing—mathematical functions that generate unique fingerprints for digital content. When investigators first collect digital evidence, whether a video from a conflict zone or documents from a seized server, they immediately generate multiple hash values using algorithms like SHA-256 and SHA-3. These hashes serve as immutable identifiers that will reveal any subsequent alteration, even changes as small as a single bit.

Our implementation goes beyond simple hashing to create comprehensive evidence packages. Each piece of digital evidence is wrapped in a container that includes the original file, multiple hash values, timestamp information from trusted time authorities, collector identification, and detailed metadata about the collection circumstances. This package is then itself hashed, creating a hierarchical verification structure.

Blockchain technology adds an additional layer of integrity protection. By recording hash values on distributed ledgers, we create tamper-evident logs that cannot be retroactively altered. Even if an adversary compromises one system, the blockchain's distributed nature ensures that evidence of the original state persists across multiple independent nodes.

The practical workflow begins at the point of collection. Field investigators use specialized applications that automatically capture and cryptographically seal evidence. These tools record not just the evidence itself but also GPS coordinates, device identifiers, and environmental metadata. Every action—viewing, copying, analyzing—generates a log entry that becomes part of the permanent record.

Timestamping presents unique challenges in conflict zones where reliable internet connectivity may be unavailable. We've developed hybrid approaches that use local secure timestamps initially, then obtain trusted timestamps from international time authorities when connectivity permits. The relationship between these timestamps is cryptographically bound, ensuring temporal integrity even in challenging operational environments.

The handling of multimedia evidence requires special consideration. Videos and images contain multiple data streams—visual content, audio tracks, embedded metadata—each requiring separate verification. Our systems generate hashes for each component while maintaining their relationships, ensuring that selective editing or metadata manipulation can be detected.

Legal admissibility requirements vary across jurisdictions, but certain principles remain universal. Courts require demonstration that evidence has not been altered, clear documentation of everyone who handled the evidence, and the ability to verify these claims independently. Our cryptographic approach satisfies these requirements through mathematical proofs rather than relying solely on procedural compliance.

The integration of artificial intelligence in evidence processing adds complexity to chain of custody maintenance. When AI systems enhance, translate, or analyze evidence, each transformation must be documented and verified. We maintain parallel chains: one for the original evidence and others for each derivative work, with clear documentation of the processing applied.

Key management represents a critical vulnerability in cryptographic chain of custody systems. The private keys used to sign evidence packages must be protected against both loss and compromise. We implement hierarchical key management systems with secure hardware modules, multi-party computation for critical operations, and regular key rotation protocols.

The scalability challenge becomes apparent in large-scale investigations involving millions of digital artifacts. Traditional approaches that require individual handling of each piece of evidence prove impractical. We've developed batch processing systems that maintain individual verification capabilities while enabling efficient handling of large evidence volumes.

Interoperability between different organizations' chain of custody systems requires standardization. We've contributed to developing international standards for evidence exchange formats, ensuring that evidence collected by one organization can be verified and used by others without compromising integrity guarantees.

The human factor remains crucial despite technological sophistication. Training investigators in proper digital evidence handling, ensuring they understand both the how and why of cryptographic verification, prevents procedural errors that could undermine technical safeguards. Regular drills and competency assessments maintain operational readiness.

Auditability extends beyond simple verification to encompass comprehensive forensic reconstruction. Our systems maintain detailed logs that allow independent auditors to reconstruct the complete history of evidence handling, verify all cryptographic claims, and identify any anomalies or suspicious patterns.

The cost-benefit analysis of cryptographic chain of custody proves compelling. While initial implementation requires investment in tools and training, the long-term benefits—reduced evidence challenges, faster court proceedings, and increased conviction rates—justify the expense. Organizations report 60-80% reduction in evidence-related legal challenges after implementing cryptographic verification.

Emerging quantum computing capabilities pose future challenges to current cryptographic methods. We're already implementing quantum-resistant algorithms and hybrid approaches that will maintain evidence integrity even in a post-quantum world. This forward-looking approach ensures that evidence collected today remains verifiable decades hence.

Privacy considerations intersect with chain of custody requirements when evidence contains personal information. We've developed selective disclosure mechanisms that allow verification of evidence integrity without revealing sensitive content. Zero-knowledge proofs enable courts to verify evidence properties without accessing the underlying data.

The collaborative nature of international investigations requires federated chain of custody systems. Multiple organizations must be able to contribute evidence while maintaining independent verification capabilities. Our federated approach uses shared standards and cross-certification to create unified evidence chains from distributed sources.

Looking ahead, the integration of chain of custody systems with emerging technologies like homomorphic encryption will enable analysis of encrypted evidence without breaking the cryptographic seal. This capability will allow investigators to process sensitive evidence while maintaining perfect integrity guarantees throughout the analytical pipeline.`
  },
  {
    id: 'three-dimensional-crime-scene-reconstruction',
    title: 'Three-Dimensional Reconstruction for Crime Scene Analysis',
    excerpt: 'Spatial documentation and forensic visualization techniques for human rights investigations using 3D modeling and photogrammetry.',
    author: 'Oliver Stern',
    date: '2025-03-02',
    lastReviewed: 'Mar 2025',
    readTime: '15 min',
    category: 'Evidence Collection',
    tags: ['Digital Forensics', 'Evidence Processing', 'Case Studies'],
    roles: ['Investigators', 'Researchers'],
    peerReviewed: true,
    content: `Three-dimensional reconstruction technology has revolutionized crime scene documentation in human rights investigations, transforming how we capture, analyze, and present spatial evidence in legal proceedings. Through photogrammetry, LiDAR scanning, and advanced modeling techniques, investigators can now create precise digital twins of crime scenes that preserve crucial spatial relationships long after physical locations have changed or become inaccessible.

The fundamental principle of photogrammetry—extracting three-dimensional information from two-dimensional images—enables investigators to create detailed 3D models using only photographs. By capturing overlapping images from multiple angles, specialized software can triangulate the position of every visible point, reconstructing the scene with millimeter-level accuracy. This democratization of 3D capture technology means that investigators with basic camera equipment can document scenes that would have previously required expensive specialized hardware.

Our deployment protocol for 3D reconstruction begins with systematic scene photography. Investigators follow predetermined capture patterns, ensuring 60-80% overlap between adjacent images. We've developed mobile applications that guide photographers through optimal capture sequences, monitor image quality in real-time, and alert users to gaps in coverage that could compromise model quality.

The processing pipeline transforms raw photographs into actionable 3D evidence through several stages. Initial processing identifies common features across images, establishing the geometric relationships between camera positions. Dense reconstruction algorithms then generate point clouds containing millions of 3D coordinates. Mesh generation converts these point clouds into solid surfaces, while texture mapping applies photographic detail to create photorealistic models.

LiDAR technology complements photogrammetry by providing precise distance measurements independent of lighting conditions. Particularly valuable for large-scale scenes or challenging environments, LiDAR can capture data in darkness, through vegetation, and in visually repetitive environments where photogrammetry might struggle. The fusion of LiDAR and photogrammetric data produces models that combine geometric precision with photographic detail.

The evidential value of 3D reconstructions extends beyond simple documentation. These models enable investigators to test hypotheses about events, verify witness testimonies against physical constraints, and identify sight lines and trajectories that might not be apparent from photographs alone. In ballistic analysis, 3D models allow precise trajectory reconstruction that can identify shooter positions and verify or refute claimed sequences of events.

Virtual crime scene visits have proven particularly powerful in legal proceedings. Judges and juries can explore scenes from any angle, understanding spatial relationships that would be difficult to convey through traditional photographs or diagrams. The ability to switch between different viewpoints—victim perspectives, witness positions, perpetrator sightlines—provides unprecedented insight into event dynamics.

The temporal dimension adds another layer of analytical capability. By creating 3D models at different time points, investigators can document changes over time, quantify destruction, and establish timelines. In cases involving gradual environmental crimes or systematic destruction of cultural heritage, these temporal comparisons provide compelling evidence of intentional patterns.

Accuracy validation remains crucial for legal admissibility. We implement multiple quality control measures: using surveyed control points to verify dimensional accuracy, comparing model measurements against ground truth data, and conducting blind tests where investigators use 3D models to answer questions about scenes they haven't physically visited. Our models consistently achieve accuracy within 1-2% of actual dimensions.

The integration of 3D reconstruction with other evidence types multiplies investigative power. Witness testimonies can be evaluated within accurate spatial contexts. Security camera footage can be precisely located within 3D space. Physical evidence locations can be documented with exact three-dimensional coordinates. This multi-modal integration creates comprehensive scene understanding that surpasses any individual evidence type.

Data management challenges arise from the massive file sizes generated by 3D reconstruction. Raw photogrammetry projects can exceed hundreds of gigabytes, while processed models still require substantial storage. We've developed tiered storage strategies, maintaining full-resolution data for active cases while creating optimized versions for review and presentation.

The democratization of 3D documentation tools has enabled crowdsourced crime scene documentation. Civilians fleeing conflict zones can capture images that, when aggregated, enable reconstruction of atrocity sites. We've developed protocols for validating and integrating crowdsourced imagery, ensuring that community-contributed evidence meets legal standards.

Training requirements for 3D reconstruction span technical and methodological dimensions. Investigators must understand not just how to operate capture equipment but also how to recognize and document forensically significant spatial relationships. Our training programs combine classroom instruction with hands-on exercises using mock crime scenes.

Presentation challenges require careful consideration of how 3D evidence is displayed to non-technical audiences. We've developed visualization tools that allow smooth navigation through 3D scenes, annotation of key features, and side-by-side comparison with photographs and diagrams. The goal is making complex spatial data accessible without overwhelming viewers.

Privacy and ethical considerations arise when reconstructing scenes that include private spaces or sensitive locations. We implement selective reconstruction techniques that capture forensically relevant areas while excluding unnecessary private information. Adjustable resolution zones allow detailed documentation of evidence while maintaining appropriate privacy protection.

The cost-effectiveness of 3D reconstruction has improved dramatically. What once required specialized equipment costing hundreds of thousands of dollars can now be accomplished with consumer cameras or smartphones. This accessibility enables smaller organizations and independent investigators to employ techniques previously reserved for well-funded institutions.

Cross-cultural considerations influence how 3D reconstructions are received in different legal systems. Some jurisdictions readily accept digital evidence, while others require extensive validation. We've developed culture-specific presentation strategies that address local legal traditions while maintaining scientific rigor.

Emerging technologies promise even greater capabilities. Artificial intelligence can automatically identify forensically significant features within 3D models. Real-time reconstruction enables immediate scene documentation. Augmented reality will allow investigators to overlay digital evidence onto physical locations during follow-up visits.

The archival value of 3D reconstructions extends beyond immediate legal proceedings. These digital preservations document sites for historical records, enable future re-analysis with improved techniques, and provide educational resources for training future investigators. Many sites documented in 3D have subsequently been destroyed, making these models the only remaining record.

Looking forward, the integration of 3D reconstruction with virtual reality will transform how legal professionals interact with crime scene evidence. Immersive experiences will allow judges and juries to literally walk through crime scenes, experiencing spatial relationships and perspectives that photographs alone cannot convey. This evolution from passive viewing to active exploration represents a fundamental shift in how spatial evidence is understood and evaluated in legal proceedings.`
  },
  {
    id: 'real-time-analysis-active-conflicts',
    title: 'Real-Time Analysis in Active Conflicts',
    excerpt: 'Operational challenges for human rights investigations during ongoing conflicts, with lessons from the Ukraine War.',
    author: 'Oliver Stern',
    date: '2025-01-28',
    lastReviewed: 'Jan 2025',
    readTime: '13 min',
    category: 'Operations',
    tags: ['Case Studies', 'Best Practices', 'Operations'],
    roles: ['Investigators'],
    peerReviewed: true,
    content: `The documentation of human rights violations during active conflicts presents unique operational, technological, and ethical challenges that distinguish it from post-conflict investigations. Real-time analysis requires rapid decision-making under extreme uncertainty, often with incomplete information and while events continue to unfold. The ongoing conflict in Ukraine has provided crucial lessons about conducting investigations during active hostilities, demonstrating both the potential and limitations of real-time documentation.

The velocity of information flow in modern conflicts can be overwhelming. Social media posts, satellite imagery, drone footage, and witness reports flood in continuously, creating a data deluge that traditional investigative methods cannot process effectively. Our rapid response framework employs automated triage systems that prioritize incoming information based on credibility indicators, geographic relevance, and potential legal significance.

Operational security becomes paramount when investigating ongoing conflicts. Digital communications can be intercepted, investigators can be targeted, and sources face immediate physical danger. We've implemented multilayered security protocols including end-to-end encrypted communications, compartmentalized information sharing, and rapid relocation capabilities for at-risk personnel.

The verification challenge intensifies exponentially during active conflicts. Disinformation campaigns, propaganda, and the fog of war make distinguishing truth from fabrication extremely difficult. Our verification protocol employs multiple independent confirmation sources: cross-referencing social media posts with satellite imagery, correlating witness accounts with signals intelligence, and using forensic analysis to detect manipulated media.

Time sensitivity drives every aspect of real-time analysis. Evidence can be destroyed within hours—buildings demolished, bodies removed, digital records deleted. We maintain 24/7 monitoring capabilities with teams distributed across time zones to ensure continuous coverage. Automated alerts notify investigators of significant events requiring immediate documentation.

The Ukraine conflict has demonstrated the power of crowdsourced intelligence. Thousands of civilians document events as they occur, creating a distributed sensor network that no traditional intelligence service could replicate. However, this also creates challenges: verifying contributor identities, ensuring chain of custody, and protecting sources from retaliation.

Technological adaptation occurs rapidly on all sides. As investigators develop new documentation techniques, perpetrators evolve countermeasures. We've observed jamming of GPS signals to prevent geolocation, deliberate pollution of social media with false content, and sophisticated deep fakes designed to discredit legitimate evidence. Staying ahead requires continuous innovation and adaptation.

The psychological toll on investigators conducting real-time analysis cannot be overstated. Watching atrocities unfold in real-time, knowing that documentation might be the only form of immediate intervention possible, creates unique moral injury. We've implemented comprehensive mental health support including mandatory rotation schedules, peer support groups, and access to specialized trauma counselors.

Coordination between organizations becomes both more critical and more challenging during active conflicts. Information sharing can save lives and prevent duplication of effort, but operational security concerns limit what can be shared. We've developed information sharing protocols that balance transparency with security, using graduated access levels and sanitized intelligence products.

The legal framework for real-time documentation continues to evolve. Questions arise about the duty to warn potential victims when intelligence indicates imminent attacks, the admissibility of evidence collected through unconventional means, and the responsibility of investigators who possess actionable intelligence. Legal advisors are embedded in our rapid response teams to navigate these complex ethical and legal considerations.

Resource allocation requires difficult prioritization decisions. With multiple incidents occurring simultaneously, teams must decide which events to document comprehensively. We've developed triage protocols based on factors including potential casualty numbers, availability of other documentation sources, likelihood of successful prosecution, and strategic significance for accountability efforts.

The role of artificial intelligence in real-time analysis has proven transformative. Machine learning systems can process vast data streams, identify patterns indicating imminent atrocities, and alert human analysts to priority events. However, the speed required for real-time response sometimes conflicts with the validation needed for legal proceedings.

Community protection measures must be integrated into documentation efforts. Publishing evidence of atrocities can trigger retaliation against survivors and witnesses. We implement careful redaction protocols, delayed publication schedules, and work with protection agencies to ensure documentation doesn't endanger those it aims to help.

The challenge of maintaining investigative standards under pressure requires constant vigilance. The urgency of real-time response can tempt investigators to cut corners or lower verification thresholds. Regular quality audits, peer review processes, and adherence to established protocols ensure that speed doesn't compromise accuracy.

Cross-border coordination adds layers of complexity. Evidence collection often requires cooperation across multiple jurisdictions, each with different legal frameworks and political considerations. We've established rapid mutual legal assistance protocols with partner organizations to expedite cross-border evidence sharing.

The preservation of digital evidence during active conflicts requires innovative approaches. Infrastructure damage, power outages, and deliberate attacks on data centers threaten evidence integrity. We've implemented distributed backup systems, offline verification mechanisms, and rapid extraction protocols to preserve evidence before it's lost.

Media relations during active conflicts require careful balance. Public documentation can build pressure for intervention and protect vulnerable populations, but premature disclosure can compromise ongoing operations and endanger sources. We maintain strict publication protocols with graduated disclosure based on operational security assessments.

The innovation cycle accelerates during active conflicts. New documentation techniques, verification methods, and analytical tools emerge from operational necessity. The Ukraine conflict has generated advances in satellite imagery analysis, social media verification, and crowdsourced intelligence that will benefit future investigations.

Lessons learned from real-time analysis in Ukraine are being incorporated into standard operating procedures. These include the importance of pre-positioned resources, the value of local partnerships, the need for flexible command structures, and the critical role of technological innovation. Each conflict provides learning opportunities that strengthen future response capabilities.

Looking ahead, the integration of predictive analytics offers potential for prevention as well as documentation. By identifying patterns that precede atrocities, real-time analysis systems might enable intervention before violations occur. This evolution from reactive documentation to proactive prevention represents the next frontier in conflict-related human rights work.`
  },
  {
    id: 'multi-modal-evidence-integration',
    title: 'Multi-Modal Evidence Integration',
    excerpt: 'Constructing comprehensive narratives from diverse digital sources in Myanmar investigations using AI-powered data fusion.',
    author: 'Oliver Stern',
    date: '2024-11-25',
    lastReviewed: 'Nov 2024',
    readTime: '17 min',
    category: 'AI & Technology',
    tags: ['Evidence Processing', 'AI & Technology', 'Case Studies'],
    roles: ['Investigators', 'Researchers'],
    peerReviewed: true,
    content: `The complexity of modern human rights investigations demands the integration of evidence from multiple modalities—text documents, images, videos, audio recordings, satellite imagery, and digital communications. Multi-modal evidence integration represents a paradigm shift from analyzing individual pieces of evidence in isolation to constructing comprehensive narratives that leverage the full spectrum of available information. Our work in documenting atrocities in Myanmar has demonstrated both the power and challenges of this integrated approach.

The fundamental challenge of multi-modal integration lies in the heterogeneous nature of different evidence types. A single event might be documented through witness testimonies in Burmese, social media posts in English, satellite imagery showing physical changes, and videos with ambient audio in multiple languages. Each modality captures different aspects of reality, and meaningful integration requires sophisticated technical and analytical frameworks.

Our integration pipeline begins with standardization and preprocessing. Text documents undergo optical character recognition, translation, and entity extraction. Images are analyzed for objects, faces, and geographic features. Videos are decomposed into visual, audio, and temporal components. This preprocessing creates normalized representations that enable cross-modal analysis while preserving the unique information each modality provides.

Temporal alignment represents a critical technical challenge. Different evidence sources use varying time references—local times, UTC, relative timestamps, or no timestamps at all. We've developed probabilistic temporal alignment algorithms that use contextual clues to establish temporal relationships between evidence pieces. Shadow analysis in images, prayer call audio in videos, and references to daily events in testimonies all contribute to temporal reconstruction.

The semantic integration layer connects information across modalities through shared entities and events. When a witness mentions a specific location, our system automatically retrieves relevant satellite imagery, identifies videos geotagged to that area, and extracts social media posts mentioning the location. This creates evidence clusters that provide multiple perspectives on individual incidents.

Machine learning models trained on multi-modal data can identify patterns invisible to modality-specific analysis. For example, correlating communication metadata with satellite imagery of troop movements and social media activity patterns revealed coordinated campaigns that wouldn't have been apparent from any single evidence type. These insights have proven crucial in demonstrating systematic patterns required for crimes against humanity prosecutions.

The verification challenge multiplies with multi-modal evidence. Not only must each piece be individually authenticated, but the relationships between evidence pieces must also be validated. We implement cross-modal verification where different evidence types validate each other—using satellite imagery to confirm locations described in testimonies, or audio analysis to verify the authenticity of videos.

Data fusion algorithms combine information from multiple sources to create enhanced representations. For instance, combining multiple low-quality phone videos from different angles can reconstruct high-resolution 3D scenes. Similarly, fragmentary witness accounts can be combined to create comprehensive incident timelines. These fusion techniques extract more information than the sum of individual parts.

The handling of uncertainty and contradiction requires sophisticated frameworks. Different evidence sources may provide conflicting information due to perspective differences, temporal confusion, or deliberate deception. We've developed belief propagation networks that assess the credibility of different sources and reconcile contradictions through probabilistic reasoning.

Cultural and linguistic challenges add complexity to multi-modal integration. Visual symbols, gestural communication, and cultural references may be misinterpreted without appropriate context. Our teams include cultural experts who provide essential interpretation, ensuring that evidence is understood within its proper cultural framework.

The presentation of multi-modal evidence in legal proceedings requires careful consideration. Courts accustomed to examining individual documents or photographs may struggle with integrated evidence presentations. We've developed visualization tools that show relationships between evidence pieces while allowing drill-down into individual items, maintaining both the forest and trees perspectives.

Privacy protection becomes more complex with multi-modal evidence. A person's identity might be protected in one modality but revealed through combination with others—a redacted name in a document might be identified through face recognition in accompanying photos. We implement privacy preservation across all modalities, ensuring consistent protection throughout the evidence chain.

The computational requirements for multi-modal analysis are substantial. Processing and analyzing diverse data types, especially video and satellite imagery, demands significant computing resources. We've implemented distributed processing architectures that parallelize analysis across multiple systems while maintaining data security and chain of custody.

Quality control in multi-modal integration requires new approaches. Traditional quality metrics designed for single modalities don't capture integration quality. We've developed holistic quality assessment frameworks that evaluate not just individual evidence pieces but also the coherence and consistency of integrated narratives.

The training requirements for investigators working with multi-modal evidence span multiple disciplines. Personnel need understanding of different evidence types, their strengths and limitations, and how they complement each other. Our training programs use simulated investigations where teams practice integrating diverse evidence sources to solve complex cases.

Collaboration tools designed for multi-modal investigation enable distributed teams to work effectively. These platforms allow investigators to annotate connections between evidence pieces, share insights across modalities, and maintain awareness of the full evidence landscape. Real-time collaboration capabilities enable rapid response to emerging situations.

The archival challenge of multi-modal evidence requires comprehensive data management strategies. Maintaining relationships between evidence pieces, preserving context, and ensuring long-term accessibility demands sophisticated database architectures and metadata schemas. We've developed preservation standards that maintain evidence relationships across decades-long judicial processes.

Legal frameworks are evolving to accommodate multi-modal evidence. Courts increasingly recognize that complex events cannot be understood through single evidence types alone. Precedents are being established for presenting integrated evidence narratives, though challenges remain in ensuring judges and juries can properly evaluate such complex presentations.

The democratization of multi-modal documentation tools empowers civil society organizations. Communities affected by human rights violations can now document their experiences across multiple channels, creating rich evidence bases that support their calls for justice. We provide training and tools to enable community-led documentation efforts.

Future developments in multi-modal integration promise even greater capabilities. Advances in artificial intelligence will enable more sophisticated pattern recognition across modalities. Virtual and augmented reality will allow investigators to immerse themselves in integrated evidence environments. Quantum computing may enable analysis of relationships too complex for current systems.

The ethical implications of multi-modal integration require ongoing consideration. The power to construct comprehensive narratives from diverse sources must be balanced with respect for privacy, consent, and human dignity. We maintain ethical review boards that evaluate investigation methods and ensure that technological capabilities are used responsibly in service of justice.`
  },
  {
    id: 'large-scale-document-analysis-syria',
    title: 'Large-Scale Document Analysis for War Crimes',
    excerpt: 'Lessons from Syria\'s digital evidence revolution in processing massive document collections for accountability efforts.',
    author: 'Oliver Stern',
    date: '2024-10-20',
    lastReviewed: 'Oct 2024',
    readTime: '19 min',
    category: 'Case Studies',
    tags: ['document-analysis', 'syria', 'war-crimes', 'digital-evidence'],
    roles: ['Prosecutors', 'Researchers'],
    peerReviewed: true,
    content: `The Syrian conflict has generated one of the largest digital evidence collections in legal history, fundamentally transforming how international justice mechanisms process massive document repositories. The challenge of analyzing over 800,000 documents, videos, and digital artifacts has driven unprecedented innovation in automated analysis, quality assurance, and evidence management systems.

The scale of documentation emerged from multiple sources: leaked government archives, social media posts, witness testimonies, satellite imagery, and intercepted communications. Unlike traditional investigations that might involve thousands of documents, Syrian accountability efforts required processing volumes that exceed human analytical capacity. This necessity became the mother of innovation in legal technology.

Our initial approach using traditional document review methods proved inadequate within months. Teams of human analysts, even working around the clock, could process perhaps 50-100 documents per day with appropriate thoroughness. At this rate, reviewing the complete Syrian archive would require decades. The urgency of accountability demands and the deteriorating security situation necessitated a fundamental shift toward automated processing.

The development of domain-specific natural language processing models represented a breakthrough. Unlike general-purpose AI systems, these models were fine-tuned on legal terminology, Arabic text, and conflict-specific vocabulary. The training process involved thousands of manually annotated examples where experienced investigators tagged relevant entities, relationships, and legal concepts within Syrian documents.

Entity extraction proved particularly challenging due to the multilingual nature of Syrian documentation. Documents might contain Arabic names transliterated differently across sources, geographic locations with varying spellings, and dates referenced in multiple calendar systems. Our NLP pipeline implemented fuzzy matching algorithms, cross-referencing with standardized databases, and confidence scoring to handle these ambiguities.

The temporal analysis of document collections revealed systematic patterns invisible to individual document review. By plotting the frequency of certain terms, locations, and actors over time, we could identify campaign periods, operational phases, and coordination patterns between different perpetrator groups. These temporal visualizations became powerful courtroom exhibits demonstrating systematic rather than random violence.

Network analysis of communication patterns exposed command structures and decision-making hierarchies. By analyzing who communicated with whom, when, and about what topics, we reconstructed organizational charts that proved invaluable for establishing superior responsibility. Machine learning algorithms identified influential nodes in communication networks, often revealing previously unknown key figures.

The integration of multimedia evidence presented unique challenges. Videos from Syrian sources often lacked metadata, had been edited multiple times, and contained Arabic speech requiring transcription. Our multimedia pipeline combined computer vision for scene analysis, speech recognition for Arabic transcription, and forensic video analysis for authenticity verification.

Quality assurance mechanisms proved essential for maintaining legal standards while processing at scale. We implemented multi-tier validation: automated quality checks flagged potential issues, human supervisors reviewed flagged items, and independent auditors sampled processed documents for accuracy verification. This pyramid approach maintained quality while enabling scalable processing.

The privacy protection requirements added complexity to large-scale processing. Syrian documents often contained sensitive personal information about victims, witnesses, and their families. Automated redaction systems identified and protected personally identifiable information while preserving evidential value. Manual review ensured that automated systems hadn't inadvertently redacted legally relevant information.

Collaboration tools designed for distributed teams enabled investigators across multiple time zones to work on the same document collection efficiently. These platforms tracked who reviewed what, maintained annotation consistency, and enabled real-time communication about emerging patterns. Version control systems ensured that all changes were tracked and reversible.

The archival and preservation challenges of massive digital collections required innovative solutions. Traditional legal archives designed for paper documents proved inadequate for terabytes of digital materials. We developed preservation standards that maintained evidence integrity across decades while ensuring continued accessibility as technology evolved.

Machine translation capabilities enabled broader international participation in Syrian investigations. Documents in Arabic could be automatically translated for investigators who didn't speak the language, while maintaining links to original text for verification. Quality metrics helped identify translation errors that could affect legal interpretations.

The presentation of large-scale analysis results in legal proceedings required new visualization approaches. Courts couldn't process raw outputs from algorithms analyzing hundreds of thousands of documents. We developed summarization tools that highlighted key findings while maintaining traceability to underlying evidence.

Cost-benefit analysis demonstrated the efficiency gains from automated processing. While initial development required significant investment, the per-document processing cost decreased dramatically as scale increased. Organizations reported 95% reduction in processing time and 80% reduction in per-document costs compared to manual review.

The lessons learned from Syrian document analysis have influenced legal technology standards worldwide. The protocols, quality assurance frameworks, and technological approaches developed for Syrian investigations are now being applied to other large-scale accountability efforts, creating a standardized approach to massive evidence processing.

Legal precedents established through Syrian cases have validated the admissibility of evidence processed using AI-assisted methods. Courts have accepted the reliability of automated processing when appropriate quality controls are maintained, setting important precedents for future technology-assisted investigations.

The human element remained crucial despite extensive automation. Expert investigators provided oversight, context interpretation, and strategic guidance that no algorithm could replace. The most effective implementations treated AI as a force multiplier for human expertise rather than a replacement for investigative judgment.

Training programs developed for Syrian investigations created a new generation of legally-trained technologists and technically-competent investigators. These hybrid professionals understand both the legal requirements for evidence processing and the capabilities and limitations of AI systems.

Looking forward, the Syrian experience provides a roadmap for handling future large-scale digital evidence collections. The tools, processes, and standards developed through this work ensure that future accountability efforts can leverage technological capabilities while maintaining the rigor required for international justice.`
  },
  {
    id: 'named-entity-recognition-conflict-documentation',
    title: 'Named Entity Recognition for Conflict Documentation',
    excerpt: 'Addressing linguistic complexity and technical challenges in multilingual human rights investigations using NER systems.',
    author: 'Dr. Ana Santos',
    date: '2024-11-18',
    lastReviewed: 'Nov 2024',
    readTime: '11 min',
    category: 'Natural Language Processing',
    tags: ['ner', 'multilingual', 'conflict-documentation', 'nlp'],
    roles: ['Researchers'],
    peerReviewed: true,
    content: `Named Entity Recognition (NER) represents one of the most challenging yet essential components in automated conflict documentation systems. The complexity of identifying people, places, organizations, and events across multiple languages, dialects, and cultural contexts requires sophisticated approaches that go far beyond traditional NLP techniques designed for monolingual, Western-centric texts.

The fundamental challenge in conflict documentation lies in the linguistic diversity of source materials. A single investigation might involve documents in Arabic, Kurdish, English, and regional dialects, each with different naming conventions, transliteration standards, and cultural references. Traditional NER systems trained on English news text fail catastrophically when applied to these multilingual, domain-specific corpora.

Our approach begins with the recognition that entity recognition in conflict zones is not merely a technical problem but a deeply cultural one. Names carry historical, religious, and political significance that automated systems must understand to avoid misclassification. For instance, distinguishing between place names and tribal affiliations in Middle Eastern contexts requires cultural knowledge that standard algorithms lack.

The training data challenge proved particularly acute. High-quality annotated datasets for conflict-related NER simply didn't exist in most relevant languages. We developed a collaborative annotation framework involving native speakers, regional experts, and legal professionals to create training datasets that reflected the linguistic and cultural complexity of real conflict documentation.

Multilingual entity linking presents another layer of complexity. The same person might be referenced as "Ahmed Mohammed," "أحمد محمد," and "A. Mohammad" across different documents. Our system implements sophisticated fuzzy matching algorithms that consider phonetic similarity, common transliteration patterns, and contextual clues to identify when different text strings refer to the same entity.

The handling of Arabic text introduces specific technical challenges. Arabic's right-to-left writing direction, connected letter forms, and optional diacritical marks complicate standard NLP preprocessing. We developed specialized tokenization and normalization pipelines that handle Arabic's morphological complexity while preserving entity boundary information.

Temporal entity extraction proved crucial for timeline reconstruction. Conflicts generate documents where temporal references might use different calendar systems, relative time expressions, or culturally specific time markers. Our temporal NER component handles Islamic calendar dates, relative expressions like "after the Friday prayers," and culturally significant time periods like "during Ramadan."

The geographic entity challenge extends beyond simple location identification. Conflict documentation often involves places that don't appear in standard geographic databases—informal settlement names, military checkpoint designations, or local landmarks known only to residents. We developed a hierarchical geographic entity system that links local place names to broader geographic contexts.

Organization entity recognition in conflict contexts requires understanding complex, evolving group structures. Militant organizations, government agencies, and civil society groups often use multiple names, acronyms, and aliases. Our system tracks entity evolution over time, maintaining links between different organizational manifestations while identifying their relationships.

The integration of named entity recognition with broader evidence analysis multiplies its value. Extracted entities become the building blocks for network analysis, timeline construction, and pattern detection. The quality of downstream analysis depends critically on the accuracy and completeness of entity extraction.

Quality assurance for multilingual NER involves multiple validation approaches. Native speakers review entity extractions for accuracy, regional experts verify cultural appropriateness, and legal professionals assess whether extracted entities meet evidentiary standards. This human-in-the-loop approach ensures that automated processing maintains human oversight.

Cross-lingual entity resolution enables investigators to track entities across language boundaries. A person mentioned in Arabic documents can be linked to references in English reports, creating comprehensive entity profiles that span linguistic and temporal boundaries. This capability proves essential for international investigations involving multiple source languages.

The privacy and protection considerations for entity extraction require careful balancing. While identifying entities is essential for investigation, protecting vulnerable individuals from retaliation requires selective redaction and access controls. Our system implements graduated disclosure mechanisms that protect sensitive entities while preserving analytical capability.

Evaluation metrics for conflict-domain NER go beyond traditional precision and recall measurements. We assess cultural appropriateness, legal relevance, and operational utility. Entity extractions must not only be technically accurate but also practically useful for investigators and legally admissible in court proceedings.

The scalability challenge involves processing millions of documents while maintaining quality standards. Our distributed processing architecture parallelizes entity extraction across multiple systems while maintaining consistency through shared knowledge bases and validation protocols.

Performance optimization for low-resource languages required innovative approaches. We implemented transfer learning techniques that leverage high-resource language models while fine-tuning for specific conflict contexts. Cross-lingual embeddings enable knowledge transfer across language boundaries.

Looking forward, the integration of named entity recognition with emerging technologies promises enhanced capabilities. Large language models provide better contextual understanding, while graph neural networks enable more sophisticated entity relationship modeling. These advances will continue improving the accuracy and utility of automated entity extraction in conflict documentation.`
  },
  {
    id: 'speech-to-text-witness-testimony',
    title: 'Speech-to-Text for Witness Testimony',
    excerpt: 'Overcoming challenges in low-resource languages for accurate transcription and analysis of witness statements.',
    author: 'Dr. Kwame Asante',
    date: '2024-11-15',
    lastReviewed: 'Nov 2024',
    readTime: '10 min',
    category: 'Audio Processing',
    tags: ['speech-to-text', 'witness-testimony', 'low-resource-languages', 'transcription'],
    roles: ['Investigators', 'Prosecutors'],
    peerReviewed: true,
    content: `The accurate transcription of witness testimony represents a critical bottleneck in international human rights investigations, particularly when dealing with testimonies in low-resource languages that lack robust speech recognition systems. The challenge extends beyond simple transcription to encompass cultural sensitivity, trauma-informed processing, and the preservation of crucial paralinguistic information that traditional text cannot capture.

Low-resource languages—those with limited digital text corpora and few native speakers in technology development—present unique challenges for automatic speech recognition (ASR) systems. Most commercial speech-to-text systems are optimized for high-resource languages like English, Mandarin, and Spanish, leaving crucial languages spoken in conflict zones with minimal or nonexistent support.

Our approach to addressing this gap begins with data collection strategies adapted to resource-constrained environments. Traditional ASR training requires hundreds of hours of transcribed speech, but such datasets rarely exist for endangered or minority languages. We developed protocols for rapid dataset creation using community partnerships, where native speakers contribute recordings in exchange for capacity building and technology access.

The technical architecture for low-resource ASR leverages transfer learning from high-resource language models. Rather than training from scratch, we adapt existing models trained on languages with similar phonetic characteristics. For instance, models trained on Arabic can provide a foundation for adapting to Kurdish or Amazigh languages, significantly reducing the data requirements for achieving functional accuracy.

Cross-lingual phoneme mapping represents a crucial innovation in our approach. We developed algorithms that identify similar sounds across languages, enabling knowledge transfer even between linguistically distant language families. This approach has proven particularly effective for agglutinative languages common in Sub-Saharan Africa, where traditional word-based models struggle with morphological complexity.

The preservation of dialectal variations requires special attention in conflict documentation. Witnesses may speak regional dialects that differ significantly from standard language varieties. Our systems incorporate dialect detection and adaptation mechanisms that adjust recognition parameters based on identified speech patterns, ensuring that regional variations don't compromise transcription accuracy.

Trauma-informed design principles guide every aspect of our speech processing systems. Traumatized witnesses may exhibit speech patterns—pauses, repetitions, emotional breaks—that standard ASR systems interpret as errors. We developed trauma-aware preprocessing that preserves these patterns as potential evidence of psychological state while still producing coherent transcripts.

The handling of code-switching—where speakers alternate between languages within single conversations—required novel approaches. Conflict zone testimonies often involve multiple languages as witnesses switch between their native language and lingua francas like Arabic, French, or English. Our system implements real-time language detection and switching between appropriate recognition models.

Quality assurance for witness testimony transcription involves multiple validation layers. Automated confidence scoring flags potentially inaccurate segments for human review. Native speaker validation ensures cultural appropriateness and accuracy. Legal review confirms that transcriptions meet evidentiary standards for court proceedings.

The integration of speaker identification enhances the evidential value of audio testimony. Our system can identify different speakers within group interviews, associate speech segments with specific individuals, and maintain speaker consistency across multiple recording sessions. This capability proves crucial for establishing the provenance and authenticity of testimony.

Paralinguistic information—tone, emotion, stress patterns—carries legal significance that pure text transcription loses. We developed annotation systems that preserve emotional context, speech rate changes, and other paralinguistic features that might indicate deception, trauma, or emphasis. These annotations become part of the official record alongside textual transcripts.

Privacy protection mechanisms ensure that sensitive testimony remains secure throughout processing. Our systems implement end-to-end encryption, access controls based on legal privilege, and automatic redaction of sensitive personal information. Processing occurs on secured infrastructure that meets international standards for handling sensitive legal evidence.

The scalability challenge involves processing thousands of hours of testimony while maintaining quality and security standards. Our distributed processing architecture enables parallel transcription across multiple systems while maintaining consistency through shared acoustic models and validation protocols.

Cultural competency training for human validators ensures that cultural references, religious terminology, and culturally specific expressions are accurately represented in transcripts. Misinterpretation of cultural content can compromise legal proceedings and misrepresent witness testimony.

The legal admissibility framework for automated transcription requires careful documentation of processing methods, accuracy validation, and chain of custody maintenance. Courts increasingly accept automated transcription when appropriate quality controls are demonstrated, but investigators must be prepared to explain both capabilities and limitations.

Cost-effectiveness analysis demonstrates significant efficiency gains from automated processing. While initial development requires investment in acoustic models and validation frameworks, the per-hour transcription cost decreases dramatically at scale. Organizations report 80% reduction in transcription time and 60% reduction in costs compared to manual transcription.

Emerging technologies promise further improvements in low-resource language processing. Large language models provide better contextual understanding, while few-shot learning techniques reduce data requirements for new language adaptation. Self-supervised learning approaches can leverage untranscribed audio data to improve model performance.

The collaborative approach to low-resource language development creates lasting benefits beyond immediate transcription needs. Communities that participate in acoustic model development gain access to speech technology that can support education, preservation, and cultural initiatives. This partnership approach ensures that technology development serves broader community needs while advancing investigation capabilities.

Looking ahead, the integration of speech-to-text with real-time translation capabilities will enable multilingual investigation teams to work more effectively. Witnesses will be able to provide testimony in their native languages while investigators receive real-time translations, preserving the authenticity of original testimony while enabling effective communication across language barriers.`
  },
  {
    id: 'automated-document-redaction',
    title: 'Automated Document Redaction',
    excerpt: 'Preserving evidential value while protecting privacy in legal proceedings through intelligent redaction systems.',
    author: 'Prof. Lisa Chen',
    date: '2024-11-12',
    lastReviewed: 'Nov 2024',
    readTime: '14 min',
    category: 'Privacy Protection',
    tags: ['document-redaction', 'privacy', 'pii-protection', 'automation'],
    roles: ['Prosecutors', 'Investigators'],
    peerReviewed: true
  },
  {
    id: 'deploying-ai-conflict-zones',
    title: 'Deploying AI in Conflict Zones',
    excerpt: 'Technical and operational challenges of implementing AI systems in active conflict environments for real-time analysis.',
    author: 'Dr. Michael Thompson',
    date: '2024-11-08',
    lastReviewed: 'Nov 2024',
    readTime: '12 min',
    category: 'Deployment Challenges',
    tags: ['deployment', 'conflict-zones', 'operational-challenges', 'infrastructure'],
    roles: ['Investigators'],
    peerReviewed: true
  },
  {
    id: 'quality-assurance-ai-evidence-processing',
    title: 'Quality Assurance Protocols for AI-Assisted Evidence Processing',
    excerpt: 'Comprehensive frameworks for ensuring reliability and accuracy in AI-powered evidence analysis systems.',
    author: 'Dr. Rachel Green',
    date: '2024-11-05',
    lastReviewed: 'Nov 2024',
    readTime: '16 min',
    category: 'Quality Assurance',
    tags: ['quality-assurance', 'protocols', 'reliability', 'validation'],
    roles: ['Researchers', 'Prosecutors'],
    peerReviewed: true
  },
  {
    id: 'building-multilingual-models-international-justice',
    title: 'Building Multilingual Models for International Justice',
    excerpt: 'Strategies for developing and deploying AI models that work effectively across diverse languages and cultural contexts.',
    author: 'Dr. Fatima Al-Zahra',
    date: '2024-11-02',
    lastReviewed: 'Nov 2024',
    readTime: '15 min',
    category: 'Model Development',
    tags: ['multilingual', 'model-development', 'cross-cultural', 'internationalization'],
    roles: ['Researchers'],
    peerReviewed: true
  },
  {
    id: 'data-management-human-rights-organizations',
    title: 'Data Management for Human Rights Organizations',
    excerpt: 'Best practices for secure, compliant, and efficient data handling in human rights investigation workflows.',
    author: 'Prof. David Kim',
    date: '2024-10-28',
    lastReviewed: 'Oct 2024',
    readTime: '13 min',
    category: 'Data Management',
    tags: ['data-management', 'best-practices', 'security', 'compliance'],
    roles: ['Investigators', 'Researchers'],
    peerReviewed: true
  },
  {
    id: 'analyzing-social-media-atrocity-documentation',
    title: 'Analyzing Social Media for Atrocity Documentation',
    excerpt: 'Verification at scale for social media content in human rights investigations using automated analysis and validation.',
    author: 'Dr. Sofia Volkov',
    date: '2024-10-25',
    lastReviewed: 'Oct 2024',
    readTime: '14 min',
    category: 'Social Media Analysis',
    tags: ['social-media', 'verification', 'osint', 'content-analysis'],
    roles: ['Investigators'],
    peerReviewed: true
  },
  {
    id: 'mass-grave-identification-aerial-imagery',
    title: 'Mass Grave Identification from Aerial Imagery',
    excerpt: 'Technical approaches for detecting and documenting mass grave sites using satellite and drone imagery analysis.',
    author: 'Dr. James Mitchell',
    date: '2024-10-22',
    lastReviewed: 'Oct 2024',
    readTime: '11 min',
    category: 'Forensic Imagery',
    tags: ['mass-graves', 'aerial-imagery', 'detection', 'forensic-analysis'],
    roles: ['Investigators', 'Researchers'],
    peerReviewed: true
  },
  {
    id: 'financial-crime-investigation-ai',
    title: 'Financial Crime Investigation: AI for Following the Money Trail',
    excerpt: 'Advanced techniques for tracking financial flows and identifying suspicious transactions in complex international networks.',
    author: 'Dr. Catherine Williams',
    date: '2024-10-18',
    lastReviewed: 'Oct 2024',
    readTime: '17 min',
    category: 'Financial Investigation',
    tags: ['financial-crime', 'money-trail', 'transaction-analysis', 'network-analysis'],
    roles: ['Investigators', 'Prosecutors'],
    peerReviewed: true
  },
  {
    id: 'weapons-identification-video-evidence',
    title: 'Weapons Identification in Video Evidence',
    excerpt: 'Computer vision applications for automated detection and classification of weapons in video evidence from conflict zones.',
    author: 'Dr. Alex Petrov',
    date: '2024-10-15',
    lastReviewed: 'Oct 2024',
    readTime: '12 min',
    category: 'Computer Vision',
    tags: ['weapons-detection', 'video-analysis', 'computer-vision', 'classification'],
    roles: ['Investigators', 'Researchers'],
    peerReviewed: true
  },
  {
    id: 'uncertainty-quantification-ai-legal-applications',
    title: 'Uncertainty Quantification in AI for Legal Applications',
    excerpt: 'Methods for measuring and communicating uncertainty in AI model outputs for legal and judicial decision-making.',
    author: 'Prof. Robert Johnson',
    date: '2024-10-12',
    lastReviewed: 'Oct 2024',
    readTime: '13 min',
    category: 'Statistical Methods',
    tags: ['uncertainty-quantification', 'model-reliability', 'statistical-methods', 'decision-making'],
    roles: ['Researchers', 'Prosecutors'],
    peerReviewed: true
  },
  {
    id: 'cross-validation-traditional-investigation-methods',
    title: 'Cross-Validation with Traditional Investigation Methods',
    excerpt: 'Integrating AI-powered analysis with conventional investigative techniques for comprehensive and reliable evidence gathering.',
    author: 'Dr. Helen Martinez',
    date: '2024-10-08',
    lastReviewed: 'Oct 2024',
    readTime: '15 min',
    category: 'Methodology Integration',
    tags: ['cross-validation', 'traditional-methods', 'integration', 'hybrid-approaches'],
    roles: ['Investigators', 'Prosecutors'],
    peerReviewed: true
  },
  {
    id: 'collaborative-intelligence-federated-learning',
    title: 'Collaborative Intelligence Without Compromise',
    excerpt: 'Federated learning approaches for international human rights investigations that preserve data privacy and sovereignty.',
    author: 'Dr. Yuki Tanaka',
    date: '2024-10-05',
    lastReviewed: 'Oct 2024',
    readTime: '16 min',
    category: 'Federated Learning',
    tags: ['federated-learning', 'collaborative-intelligence', 'privacy-preservation', 'international-cooperation'],
    roles: ['Researchers'],
    peerReviewed: true
  },
  {
    id: 'bias-detection-ai-models-human-rights',
    title: 'Bias Detection in AI Models for Human Rights Applications',
    excerpt: 'Comprehensive methodologies for identifying, measuring, and mitigating bias in AI systems used for human rights work.',
    author: 'Dr. Jennifer Liu',
    date: '2024-10-02',
    lastReviewed: 'Oct 2024',
    readTime: '14 min',
    category: 'Bias & Fairness',
    tags: ['bias-detection', 'fairness', 'methodology', 'ethical-ai'],
    roles: ['Researchers'],
    peerReviewed: true,
    content: `The deployment of artificial intelligence in human rights investigations carries profound responsibility, as biased AI systems can perpetuate injustices they were designed to prevent. Detecting and mitigating bias in these systems requires comprehensive methodologies that address not only technical performance disparities but also the broader social, cultural, and legal implications of automated decision-making in contexts where human dignity and justice are at stake.

The challenge of bias in human rights AI extends beyond traditional machine learning fairness metrics. While commercial applications might focus on demographic parity or equal opportunity, human rights applications must grapple with historical injustices, power imbalances, and the risk that AI systems might systematically disadvantage already marginalized populations. The stakes are existential: biased systems can mean the difference between justice and impunity for survivors of atrocities.

Our bias detection framework operates at multiple levels of analysis, from data collection through model deployment and ongoing monitoring. We recognize that bias can be introduced at any stage of the AI pipeline and that effective detection requires comprehensive auditing throughout the system lifecycle. This holistic approach ensures that bias mitigation becomes integral to system design rather than an afterthought.

Data bias represents the most fundamental challenge, as AI systems can only be as fair as the data they're trained on. Historical documentation of human rights violations often reflects the biases of those who controlled information systems—typically state actors or dominant groups. Victims from marginalized communities may be underrepresented in historical records, leading to AI systems that systematically overlook certain types of violations or affected populations.

Our data auditing protocols assess representation across multiple dimensions: demographic characteristics of victims and perpetrators, geographic distribution of documented incidents, temporal coverage across conflict periods, and linguistic representation in source materials. We've developed statistical tests that identify significant gaps in coverage and implemented weighting strategies to compensate for historical underrepresentation.

Algorithmic bias manifests in how AI systems process and interpret evidence. Computer vision systems trained predominantly on Western datasets may struggle to accurately analyze images from non-Western contexts. Natural language processing models may perform poorly on languages or dialects from conflict regions. These technical limitations can translate into systematic errors that disadvantage certain populations.

Our algorithmic auditing approach includes adversarial testing, where we deliberately attempt to identify failure modes that might disadvantage specific groups. We test models on challenging cases: low-quality images from conflict zones, documents in minority languages, and testimonies from trauma survivors. This stress-testing reveals algorithmic limitations that might not appear in standard validation datasets.

Intersectional bias analysis recognizes that individuals belong to multiple identity categories simultaneously, and AI systems may exhibit compound biases that affect people at these intersections. A system might perform adequately for women or ethnic minorities individually but fail dramatically for women from ethnic minority groups. Our analysis frameworks explicitly test for these interaction effects.

The cultural competency assessment evaluates whether AI systems appropriately handle cultural context in human rights investigations. For example, systems analyzing gender-based violence must understand cultural factors that influence reporting patterns, evidence types, and survivor behavior. Failure to account for cultural context can lead to systematic misinterpretation of evidence.

Legal bias assessment examines whether AI systems reflect or perpetuate biases present in legal frameworks. International human rights law, while aspirationally universal, has been developed primarily by Western legal traditions. AI systems trained on this corpus might inadvertently prioritize certain types of violations or legal concepts over others, potentially marginalizing non-Western approaches to justice.

Participatory bias assessment involves affected communities in evaluating AI system fairness. Rather than relying solely on technical metrics, we engage with survivor organizations, human rights defenders, and civil society groups to understand how AI systems impact their communities. This participatory approach ensures that bias assessment reflects lived experiences rather than just technical performance.

Temporal bias monitoring recognizes that fairness is not static. As conflicts evolve and new actors emerge, AI systems may develop biases against previously unseen patterns. Our monitoring frameworks track system performance over time, identifying drift that might indicate emerging bias. Regular retraining and validation ensure that systems remain fair as contexts change.

The intersectional approach to bias mitigation involves technical and procedural interventions. Technical approaches include data augmentation to address representation gaps, algorithmic fairness constraints that enforce equitable treatment, and ensemble methods that combine multiple perspectives. Procedural approaches include diverse review panels, community feedback mechanisms, and regular bias audits.

Fairness metrics for human rights applications go beyond traditional machine learning measures. We've developed context-specific metrics that assess whether AI systems provide equitable service to different populations, whether they identify violations consistently across groups, and whether they preserve the dignity and agency of survivors. These metrics reflect human rights principles rather than just statistical performance.

The transparency framework ensures that bias detection and mitigation efforts are documented and accessible. Stakeholders need to understand how systems work, what biases have been identified, and what mitigation strategies have been implemented. This transparency enables informed decision-making about when and how to use AI tools.

Regulatory compliance considerations address emerging legal frameworks for AI fairness. The European Union's AI Act, for instance, mandates bias assessment for high-risk AI applications. Our frameworks ensure compliance with these evolving regulatory requirements while maintaining focus on human rights principles.

Training programs for investigators and legal professionals address bias awareness and mitigation. Users need to understand both the capabilities and limitations of AI systems, recognize potential bias indicators, and know when human oversight is required. Regular training ensures that technological capabilities serve justice rather than undermining it.

The economic analysis of bias mitigation demonstrates that fairness investments pay long-term dividends. While initial bias detection and mitigation require resources, the cost of biased systems—in terms of legal challenges, reputational damage, and undermined justice—far exceeds prevention investments. Organizations that prioritize fairness build sustainable, trustworthy systems.

International cooperation in bias detection standards strengthens global human rights protection. Shared methodologies, common metrics, and collaborative research ensure that bias mitigation efforts benefit from diverse perspectives and experiences. This cooperation prevents the perpetuation of biases through isolated development processes.

Looking forward, the integration of bias detection with emerging AI technologies will require continued innovation. As large language models and multimodal AI systems become more prevalent in human rights work, new forms of bias may emerge that require novel detection and mitigation strategies. The goal remains constant: ensuring that artificial intelligence serves justice and human dignity for all people, regardless of their background or circumstances.`
  },
  {
    id: 'privacy-preserving-ai-witness-protection',
    title: 'Privacy-Preserving AI for Witness Protection',
    excerpt: 'Advanced techniques for protecting witness identities and sensitive information while enabling effective AI-powered analysis.',
    author: 'Prof. Anna Kowalski',
    date: '2024-09-28',
    lastReviewed: 'Sep 2024',
    readTime: '18 min',
    category: 'Privacy & Security',
    tags: ['privacy-preservation', 'witness-protection', 'differential-privacy', 'anonymization'],
    roles: ['Investigators', 'Prosecutors'],
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
      bg-[var(--color-critical)] text-[var(--color-text-inverse)]
      hover:bg-[color-mix(in_srgb,var(--color-critical),black_10%)]
      active:bg-[color-mix(in_srgb,var(--color-critical),black_20%)]
      border border-[var(--color-border-secondary)]
      shadow-[0_1px_0_rgba(var(--shadow-rgb),0.04),inset_0_1px_0_rgba(255,255,255,0.03)]
    `
  };

  const sizes = {
    sm: 'h-7 px-3 text-[12px] rounded-[6px] gap-1.5 font-medium',
    md: 'h-8 px-4 text-[13px] rounded-[6px] gap-2 font-medium',
    lg: 'h-10 px-5 text-[14px] rounded-[6px] gap-2.5 font-medium',
    xl: 'h-12 px-6 text-[15px] rounded-[8px] gap-3 font-medium'
  };

  return (
    <button
      {...props}
      data-variant={variant}
      data-size={size}
      aria-busy={loading || undefined}
      className={[
        'relative inline-flex items-center justify-center',
        'transition-all duration-150 ease-out',
        'disabled:opacity-60 disabled:cursor-not-allowed',
        'active:scale-[0.98] active:transition-none',
        'tracking-[-0.01em]',
        'focus-ring',
        variants[variant],
        sizes[size],
        className
      ].join(' ')}
    >
      {loading && (
        <svg className="animate-spin h-4 w-4 mr-2" viewBox="0 0 24 24" aria-hidden="true">
          <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
          <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
        </svg>
      )}
      {icon && !loading && <span className="transition-transform group-hover:scale-105">{icon}</span>}
      <span className="relative">{children}</span>
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
    default: 'bg-[var(--color-bg-primary)] border border-[var(--color-border-primary)] shadow-elevation-1',
    elevated: 'bg-[var(--color-bg-elevated)] shadow-elevation-2 border border-[var(--color-border-secondary)]',
    outlined: 'bg-transparent border border-[var(--color-border-primary)]',
    filled: 'bg-[var(--color-bg-secondary)] border border-[var(--color-border-secondary)]',
  };

  const hoverClasses = hover
    ? 'transition-all duration-200 hover:shadow-elevation-2 cursor-pointer'
    : '';

  return (
    <div className={`
      relative rounded-xl p-6
      ${variants[variant]}
      ${hoverClasses}
      ${className}
    `}>
      {children}
    </div>
  );
};

// Logo Component with theme-aware switching and smooth transitions
const LemkinLogo: React.FC<{ className?: string }> = ({ className = "w-8 h-8" }) => {
  const { resolvedTheme } = useTheme();

  // Use black logo for light mode, white logo for dark mode
  const logoSrc = resolvedTheme === 'light'
    ? '/Lemkin Logo Black_Shape_clear.png'
    : '/Lemkin Logo (shape only).png';

  return (
    <div className="relative group">
      <div
        className="absolute inset-0 blur-xl opacity-0 group-hover:opacity-25 transition-opacity duration-500"
        style={{
          background: "linear-gradient(90deg, color-mix(in srgb, var(--color-primary), transparent 85%), color-mix(in srgb, var(--color-border-active), transparent 90%))"
        }}
      />
      <img
        src={logoSrc}
        alt="Lemkin AI"
        className={`relative transform transition-all duration-300 group-hover:scale-110 ${className}`}
      />
    </div>
  );
};

// CodeBlock component for displaying code snippets
interface CodeBlockProps {
  code: string;
  language?: string;
}

const CodeBlock: React.FC<CodeBlockProps> = ({ code, language = 'bash' }) => {
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(code);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      console.error('Failed to copy text: ', err);
    }
  };

  return (
    <div className="relative group">
      <div className="absolute top-2 right-2 z-10">
        <button
          onClick={handleCopy}
          className="btn-outline p-2 opacity-0 group-hover:opacity-100 transition-opacity"
          aria-label="Copy code"
        >
          {copied ? <Check className="w-4 h-4" /> : <Copy className="w-4 h-4" />}
        </button>
      </div>
      <pre className="bg-gray-50 dark:bg-gray-900 border border-gray-200 dark:border-gray-700 rounded-lg p-4 overflow-x-auto">
        <code className="text-sm text-gray-800 dark:text-gray-200 font-mono">
          {code}
        </code>
      </pre>
    </div>
  );
};

// Enhanced Model Card with improved information architecture
interface ModelCardProps {
  model: any;
}

const ModelCard: React.FC<ModelCardProps> = ({ model }) => {
  const { navigate } = useRouter();
  const [selected, setSelected] = useState(false);

  // Quick-scan performance indicators
  const getPerformanceLevel = (accuracy: number) => {
    if (accuracy >= 95) return { label: 'Excellent', color: 'text-[var(--color-fg-primary)]' };
    if (accuracy >= 90) return { label: 'Very Good', color: 'text-[var(--color-fg-primary)]' };
    if (accuracy >= 85) return { label: 'Good', color: 'text-[var(--color-fg-muted)]' };
    return { label: 'Experimental', color: 'text-[var(--color-fg-subtle)]' };
  };

  const performance = getPerformanceLevel(model.accuracy);

  return (
    <MotionCard className="group relative overflow-hidden border-[var(--bd)] hover:border-[var(--accent)]/30 transition-all duration-300 rounded-xl p-6">
      {/* Multi-select checkbox */}
      <input
        type="checkbox"
        aria-label={`Select ${model.name} for comparison`}
        className="absolute top-4 left-4 z-10 accent-[var(--accent)]"
        checked={selected}
        onChange={(e) => setSelected(e.target.checked)}
      />

      {/* Neutral status badge */}
      <div className="absolute top-4 right-4">
        <span className={[
          "inline-flex items-center px-2 py-1 rounded-full text-[10px] font-medium uppercase tracking-wider",
          model.status === 'stable' ? 'bg-green-50 dark:bg-green-900/20 text-green-700 dark:text-green-300' :
          model.status === 'beta' ? 'bg-blue-50 dark:bg-blue-900/20 text-blue-700 dark:text-blue-300' :
          'bg-amber-50 dark:bg-amber-900/20 text-amber-700 dark:text-amber-300'
        ].join(' ')}>
          {model.status === 'stable' ? 'Verified' :
           model.status === 'beta' ? 'Experimental' :
           'Development'}
        </span>
      </div>

      {/* Enhanced header with metrics block */}
      <div className="mb-4 pt-10">
        <div className="flex items-start justify-between mb-4">
          <div className="pl-8">
            <h3 className="text-[16px] font-semibold text-[var(--ink)] tracking-[-0.01em] mb-1">
              {model.name}
            </h3>
            <span className="text-[12px] text-[var(--subtle)] font-mono">
              v{model.version}
            </span>
          </div>

          {/* Radial progress indicator for accuracy */}
          <div className="text-right flex flex-col items-end pr-2">
            <div className="relative w-16 h-16 mb-1">
              <svg className="w-16 h-16 transform -rotate-90" viewBox="0 0 64 64">
                <circle
                  cx="32"
                  cy="32"
                  r="26"
                  stroke="var(--surface)"
                  strokeWidth="3"
                  fill="none"
                />
                <motion.circle
                  cx="32"
                  cy="32"
                  r="26"
                  stroke="var(--accent)"
                  strokeWidth="3"
                  fill="none"
                  strokeDasharray={`${2 * Math.PI * 26}`}
                  strokeDashoffset={`${2 * Math.PI * 26 * (1 - model.accuracy / 100)}`}
                  strokeLinecap="round"
                  initial={{ strokeDashoffset: 2 * Math.PI * 26 }}
                  animate={{ strokeDashoffset: 2 * Math.PI * 26 * (1 - model.accuracy / 100) }}
                  transition={{ duration: 1, delay: 0.2, ease: "easeOut" }}
                />
              </svg>
              <div className="absolute inset-0 flex items-center justify-center">
                <span className="text-[15px] font-mono tabular-nums font-semibold text-[var(--ink)]">
                  {model.accuracy}%
                </span>
              </div>
            </div>
            <div className="text-[10px] font-medium tracking-[0.08em] uppercase text-[var(--subtle)]">
              Accuracy
            </div>
          </div>
        </div>

        <p className="text-[13px] leading-[1.6] text-[var(--color-text-secondary)] dark:text-[var(--color-text-tertiary)] line-clamp-2 px-2">
          {model.description}
        </p>
      </div>

      {/* Metrics grid with better visual separation */}
      <div className="grid grid-cols-3 gap-[1px] bg-[var(--color-border-primary)] dark:bg-[var(--color-border-primary)] rounded-[8px] overflow-hidden mb-4 mx-2">
        <div className="bg-[var(--color-bg-primary)] dark:bg-[var(--color-bg-elevated)] p-3 text-center">
          <div className="text-[13px] font-semibold text-[var(--ink-8)] dark:text-white">{model.precision}%</div>
          <div className="text-[10px] text-[var(--color-text-secondary)] dark:text-[var(--color-text-tertiary)] uppercase tracking-[0.05em]">Precision</div>
        </div>
        <div className="bg-[var(--color-bg-primary)] dark:bg-[var(--color-bg-elevated)] p-3 text-center">
          <div className="text-[13px] font-semibold text-[var(--ink-8)] dark:text-white">{model.recall}%</div>
          <div className="text-[10px] text-[var(--color-text-secondary)] dark:text-[var(--color-text-tertiary)] uppercase tracking-[0.05em]">Recall</div>
        </div>
        <div className="bg-[var(--color-bg-primary)] dark:bg-[var(--color-bg-elevated)] p-3 text-center">
          <div className="text-[13px] font-semibold text-[var(--ink-8)] dark:text-white">{model.f1Score}%</div>
          <div className="text-[10px] text-[var(--color-text-secondary)] dark:text-[var(--color-text-tertiary)] uppercase tracking-[0.05em]">F1</div>
        </div>
      </div>

      {/* Professional metadata footer */}
      <div className="pt-3 px-2 border-t border-[var(--bd)]">
        <div className="flex items-center justify-between text-[11px]">
          <div className="flex items-center gap-1">
            <span className="text-[var(--subtle)]">
              Evaluated by
            </span>
            <Pressable className="text-[var(--accent)] hover:underline" title="View provenance documentation">
              {model.evaluator} ⓘ
            </Pressable>
          </div>
          <time dateTime={model.lastUpdated} className="text-[var(--subtle)]">
            {new Date(model.lastUpdated || Date.now()).toLocaleDateString()}
          </time>
        </div>
        <div className="mt-2">
            <Pressable
              onClick={(e) => { e.stopPropagation(); navigate(`/models/${model.id}`); }}
              className="text-[var(--accent)] hover:text-[var(--ink)] font-medium tracking-[-0.01em] transition-colors"
            >
              View Details →
            </Pressable>
          </div>
      </div>
    </MotionCard>
  );
};

// Model Comparison Component
const ModelComparison: React.FC = () => {
  const { navigate } = useRouter();
  const [selectedModels, setSelectedModels] = useState<any[]>([]);
  const [showComparison, setShowComparison] = useState(false);

  // Get featured models for display
  const featuredModels = getFeaturedModels().slice(0, 3);

  // Keyboard handling for comparison dialog
  useEffect(() => {
    if (!showComparison) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setShowComparison(false);
    };
    document.addEventListener('keydown', onKey);
    return () => document.removeEventListener('keydown', onKey);
  }, [showComparison]);


  const ComparisonTable = () => (
    <div className="fixed inset-0 bg-[var(--color-bg-overlay)] z-50 flex items-center justify-center p-4"
         onMouseDown={(e) => { if (e.currentTarget === e.target) setShowComparison(false); }}>
      <div role="dialog" aria-modal="true" aria-labelledby="cmp-title"
           className="bg-[var(--color-bg-elevated)] border rounded-2xl p-8 max-w-4xl w-full max-h-[80vh] overflow-auto">
        <div className="flex items-center justify-between mb-6">
          <h3 id="cmp-title" className="text-2xl font-semibold">Model comparison</h3>
          <Pressable className="p-2 rounded-md focus-ring" onClick={() => setShowComparison(false)} aria-label="Close">
            <X className="w-6 h-6" />
          </Pressable>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-[var(--color-border-primary)]">
                <th className="text-left py-4 text-[var(--color-text-secondary)] font-medium">Specification</th>
                {selectedModels.map(model => (
                  <th key={model.id} className="text-left py-4 text-[var(--color-text-primary)] font-medium">{model.name}</th>
                ))}
              </tr>
            </thead>
            <tbody className="text-[var(--color-text-secondary)]">
              <tr className="border-b border-[var(--color-border-secondary)] hover:bg-[var(--surface)] transition-colors">
                <td className="py-3 font-medium">Primary Metric</td>
                {selectedModels.map(model => (
                  <td key={model.id} className="py-3 text-[var(--color-text-primary)] font-medium">{model.accuracy}%</td>
                ))}
              </tr>
              <tr className="border-b border-[var(--color-border-secondary)] hover:bg-[var(--surface)] transition-colors">
                <td className="py-3 font-medium">License</td>
                {selectedModels.map(model => (
                  <td key={model.id} className="py-3">{model.license}</td>
                ))}
              </tr>
              <tr className="border-b border-[var(--color-border-secondary)] hover:bg-[var(--surface)] transition-colors">
                <td className="py-3 font-medium">Version</td>
                {selectedModels.map(model => (
                  <td key={model.id} className="py-3">{model.version}</td>
                ))}
              </tr>
              <tr className="border-b border-[var(--color-border-secondary)] hover:bg-[var(--surface)] transition-colors">
                <td className="py-3 font-medium">Downloads</td>
                {selectedModels.map(model => (
                  <td key={model.id} className="py-3">{model.downloads.toLocaleString()}</td>
                ))}
              </tr>
              <tr className="border-b border-[var(--color-border-secondary)] hover:bg-[var(--surface)] transition-colors">
                <td className="py-3 font-medium">Last Updated</td>
                {selectedModels.map(model => (
                  <td key={model.id} className="py-3">{model.lastUpdated}</td>
                ))}
              </tr>
              <tr className="hover:bg-[var(--surface)] transition-colors">
                <td className="py-3 font-medium">Dataset Provenance</td>
                {selectedModels.map(model => (
                  <td key={model.id} className="py-3">
                    <a href="/docs/provenance"
                       onClick={(e)=>{e.preventDefault(); navigate('/docs/provenance');}}
                       className="underline underline-offset-[3px] text-[var(--color-text-primary)] hover:opacity-90 transition-opacity focus-ring rounded-sm">
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
          <div className="flex items-center justify-between p-4 bg-[var(--color-bg-secondary)] border border-[var(--color-border-primary)] rounded-xl">
            <div className="flex items-center gap-4">
              <span className="text-[var(--color-text-secondary)] text-sm">
                {selectedModels.length} model{selectedModels.length > 1 ? 's' : ''} selected
              </span>
              <div className="flex gap-2">
                {selectedModels.map(model => (
                  <span key={model.id} className="px-2 py-1 rounded text-xs
                    bg-[var(--color-bg-elevated)] text-[var(--color-fg-primary)] border border-[var(--color-border-primary)]">
                    {model.name}
                  </span>
                ))}
              </div>
            </div>
            <div className="flex gap-3">
              <button
                onClick={() => setSelectedModels([])}
                className="text-[var(--color-text-secondary)] hover:text-[var(--color-text-primary)] transition-colors text-sm"
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
          {featuredModels.map(model => (
            <MotionCard key={model.id} className="group cursor-pointer" onClick={() => navigate('/models')}>
              <div className="p-6">
                <div className="flex items-start justify-between mb-4">
                  <div className="flex items-center gap-3">
                    <div className="p-2 bg-[var(--color-primary)]/10 rounded-lg">
                      {model.type === 'computer-vision' && <Eye className="w-5 h-5 text-[var(--color-primary)]" />}
                      {model.type === 'nlp' && <FileText className="w-5 h-5 text-[var(--color-primary)]" />}
                      {model.type === 'hybrid' && <Grid className="w-5 h-5 text-[var(--color-primary)]" />}
                    </div>
                    <div>
                      <h3 className="font-semibold text-[var(--color-text-primary)] group-hover:text-[var(--color-primary)] transition-colors">
                        {model.name}
                      </h3>
                      <span className="text-xs text-[var(--color-text-tertiary)]">{model.category}</span>
                    </div>
                  </div>
                  <span className="px-2 py-1 bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-300 rounded-full text-xs font-medium">
                    {model.status}
                  </span>
                </div>

                <p className="text-[var(--color-text-secondary)] text-sm mb-4 line-clamp-2">
                  {model.shortDescription}
                </p>

                <div className="grid grid-cols-2 gap-3 mb-4">
                  <div className="text-center p-2 bg-[var(--color-bg-secondary)] rounded-lg">
                    <div className="text-lg font-bold text-[var(--color-primary)]">
                      {model.metrics.accuracy || model.metrics.f1Score?.split(',')[0] || '95%'}
                    </div>
                    <div className="text-xs text-[var(--color-text-tertiary)]">
                      {model.metrics.accuracy ? 'Accuracy' : 'F1 Score'}
                    </div>
                  </div>
                  <div className="text-center p-2 bg-[var(--color-bg-secondary)] rounded-lg">
                    <div className="text-lg font-bold text-[var(--color-primary)]">{model.metrics.modelSize}</div>
                    <div className="text-xs text-[var(--color-text-tertiary)]">Size</div>
                  </div>
                </div>

                <div className="flex items-center justify-between">
                  <span className="text-xs text-[var(--color-text-tertiary)]">{model.type.replace('-', ' ')}</span>
                  <ArrowRight className="w-4 h-4 text-[var(--color-text-tertiary)] group-hover:text-[var(--color-primary)] transition-colors" />
                </div>
              </div>
            </MotionCard>
          ))}
        </div>
      </div>

      {showComparison && <ComparisonTable />}
    </>
  );
};

// Desktop-only Header with Condensing Behavior
const Navigation = () => {
  const { currentPath, navigate } = useRouter();
  const { theme, setTheme } = useTheme();
  const [condensed, setCondensed] = useState(false);

  useEffect(() => {
    const onScroll = () => setCondensed(window.scrollY > 24);
    onScroll();
    window.addEventListener('scroll', onScroll);
    return () => window.removeEventListener('scroll', onScroll);
  }, []);

  const navItems = [
    { path: '/', label: 'Home' },
    { path: '/overview', label: 'Overview' },
    { path: '/models', label: 'AI Models & Tools' },
    { path: '/docs', label: 'Docs' },
    { path: '/articles', label: 'Articles' },
    { path: '/ecosystem', label: 'Ecosystem' },
    { path: '/about', label: 'About' }
  ];

  return (
    <>
      {/* Skip link */}
      <a href="#main" className="sr-only focus:not-sr-only focus-ring">
        Skip to main content
      </a>

      <motion.header
        initial={{ y: -48, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        transition={{ duration: 0.4, ease: "easeOut" }}
        className={[
          'sticky top-0 z-50 w-full',
          'backdrop-blur-xl backdrop-saturate-150',
          'border-b border-[var(--line)]/50',
          'transition-all duration-500',
          condensed
            ? 'bg-[var(--bg)]/90 shadow-[0_1px_3px_rgba(0,0,0,0.05)] backdrop-blur-md'
            : 'bg-[var(--bg)]/75'
        ].join(' ')}
      >
      {/* Add a subtle gradient overlay */}
      <div className="absolute inset-0 bg-gradient-to-r from-transparent via-[var(--accent)]/[0.02] to-transparent pointer-events-none" />
      {/* Add Status Indicator Bar */}
      <div className="absolute top-0 left-0 right-0 h-[2px] bg-gradient-to-r from-transparent via-[var(--accent)] to-transparent opacity-60" />

      <div className="mx-auto relative" style={{ maxWidth: 1600, paddingInline: 48, paddingBlock: condensed ? 12 : 24 }}>
        <div className="flex items-center gap-8">
          {/* Logo */}
          <MotionCard
            className="!p-0 !shadow-none !border-none !bg-transparent"
            onClick={() => navigate('/')}
          >
            <div className="flex items-center gap-2.5 focus-ring rounded-md group cursor-pointer">
              <LemkinLogo className="w-7 h-7" />
              <div className="flex flex-col items-start">
                <span className="text-[15px] font-semibold tracking-[-0.01em] leading-none text-[var(--ink)]">
                  Lemkin AI
                </span>
              </div>
            </div>
          </MotionCard>

          {/* Enhanced Navigation with pill tabs */}
          <nav className="flex gap-2 ml-8">
            {navItems.map((item, index) => (
              <motion.div
                key={item.path}
                initial={{ opacity: 0, y: -8 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{
                  duration: 0.3,
                  delay: index * 0.05,
                  ease: "easeOut"
                }}
              >
                <motion.a
                  href={item.path}
                  onClick={(e) => {
                    e.preventDefault();
                    navigate(item.path);
                  }}
                  aria-current={currentPath === item.path ? 'page' : undefined}
                  className={[
                    'relative px-3 py-2 rounded-lg border border-[var(--line)] text-[var(--muted)]',
                    'hover:text-[var(--ink)] focus-ring transition-colors group',
                    currentPath === item.path &&
                      'text-[var(--ink)] bg-[var(--surface)]'
                  ].join(' ')}
                  whileHover={{ y: -1 }}
                  whileTap={{ scale: 0.98 }}
                >
                  {item.label}
                  {/* Hover underline animation */}
                  <motion.div
                    className="absolute bottom-0 left-3 right-3 h-[2px] bg-[var(--accent)] rounded"
                    initial={{ scaleX: currentPath === item.path ? 1 : 0 }}
                    animate={{ scaleX: currentPath === item.path ? 1 : 0 }}
                    whileHover={{ scaleX: 1 }}
                    transition={{ duration: 0.2, ease: "easeOut" }}
                    style={{ originX: 0 }}
                  />
                </motion.a>
              </motion.div>
            ))}
          </nav>

          {/* Right Side Controls */}
          <div className="ml-auto flex items-center gap-3">
            <Pressable
              onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')}
              className="btn-outline"
              aria-label="Toggle theme"
              aria-pressed={theme === 'dark'}
            >
              {theme === 'dark' ? <Sun className="w-4 h-4" /> : <Moon className="w-4 h-4" />}
            </Pressable>
            <Pressable
              className="btn-outline"
              onClick={() => window.open('https://github.com/lemkin-ai', '_blank')}
            >
              GitHub
            </Pressable>
          </div>
        </div>
      </div>
    </motion.header>
    </>
  );
};

// Enhanced Footer with Trust Center from homepage_improvements.md
const Footer = () => {
  const { navigate } = useRouter();

  return (
    <footer className="bg-[var(--surface)] border-t border-[var(--bd)]">
      <div className="container-desktop py-16">
        {/* Trust center highlight - simplified */}
        <div className="text-center mb-12 pb-8 border-b border-[var(--bd)]">
          <h2 className="text-lg font-semibold text-[var(--ink)] mb-3">Trust & Transparency</h2>
          <dl className="grid grid-cols-3 gap-6 text-sm max-w-2xl mx-auto">
            <div>
              <dt className="font-medium text-[var(--ink)]">Compliance</dt>
              <dd><a href="/security/compliance" className="text-[var(--muted)] hover:text-[var(--accent)] transition-colors">ISO 27001, SOC 2 Type II</a></dd>
            </div>
            <div>
              <dt className="font-medium text-[var(--ink)]">Evaluation</dt>
              <dd><a href="/docs/evaluation" className="text-[var(--muted)] hover:text-[var(--accent)] transition-colors">Methodology & Audit</a></dd>
            </div>
            <div>
              <dt className="font-medium text-[var(--ink)]">Provenance</dt>
              <dd><a href="/docs/provenance" className="text-[var(--muted)] hover:text-[var(--accent)] transition-colors">Datasets & Sources</a></dd>
            </div>
          </dl>
        </div>

        {/* Enhanced grid with better visual hierarchy */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-8 mb-8">
          {/* Transparency */}
          <div>
            <h3 className="text-sm font-semibold text-[var(--color-text-primary)] mb-4 flex items-center gap-2">
              <Eye className="w-4 h-4 text-[var(--color-text-primary)]" />
              Transparency
            </h3>
            <ul className="space-y-3 text-sm">
              <li><button onClick={() => navigate('/docs/changelog')} className="text-[var(--color-text-secondary)] hover:text-[var(--color-text-primary)] transition-colors text-left block focus-ring rounded-sm">Changelog</button></li>
              <li><button onClick={() => navigate('/docs/evaluation')} className="text-[var(--color-text-secondary)] hover:text-[var(--color-text-primary)] transition-colors text-left block focus-ring rounded-sm">Eval Methodology</button></li>
              <li><button onClick={() => navigate('/docs/provenance')} className="text-[var(--color-text-secondary)] hover:text-[var(--color-text-primary)] transition-colors text-left block focus-ring rounded-sm">Data Provenance</button></li>
              <li><button onClick={() => navigate('/docs/audits')} className="text-[var(--color-text-secondary)] hover:text-[var(--color-text-primary)] transition-colors text-left block focus-ring rounded-sm">Audit Reports</button></li>
              <li><button onClick={() => navigate('/docs/performance')} className="text-[var(--color-text-secondary)] hover:text-[var(--color-text-primary)] transition-colors text-left block focus-ring rounded-sm">Performance Metrics</button></li>
            </ul>
          </div>

          {/* Security */}
          <div>
            <h3 className="text-sm font-semibold text-[var(--color-text-primary)] mb-4 flex items-center gap-2">
              <Shield className="w-4 h-4 text-[var(--color-text-primary)]" />
              Security
            </h3>
            <ul className="space-y-3 text-sm">
              <li><button onClick={() => navigate('/legal/responsible-use')} className="text-[var(--color-text-secondary)] hover:text-[var(--color-text-primary)] transition-colors text-left block focus-ring rounded-sm">Responsible Use</button></li>
              <li><button onClick={() => navigate('/security/disclosure')} className="text-[var(--color-text-secondary)] hover:text-[var(--color-text-primary)] transition-colors text-left block focus-ring rounded-sm">Disclosure Policy</button></li>
              <li><button onClick={() => navigate('/security/sbom')} className="text-[var(--color-text-secondary)] hover:text-[var(--color-text-primary)] transition-colors text-left block focus-ring rounded-sm">SBOM</button></li>
              <li><button onClick={() => navigate('/security/compliance')} className="text-[var(--color-text-secondary)] hover:text-[var(--color-text-primary)] transition-colors text-left block focus-ring rounded-sm">Compliance</button></li>
              <li><button onClick={() => navigate('/security/incident')} className="text-[var(--color-text-secondary)] hover:text-[var(--color-text-primary)] transition-colors text-left block focus-ring rounded-sm">Incident Response</button></li>
            </ul>
          </div>

          {/* Legal */}
          <div>
            <h3 className="text-sm font-semibold text-[var(--color-text-primary)] mb-4 flex items-center gap-2">
              <Gavel className="w-4 h-4 text-[var(--color-text-primary)]" />
              Legal
            </h3>
            <ul className="space-y-3 text-sm">
              <li><button onClick={() => navigate('/legal/licensing')} className="text-[var(--color-text-secondary)] hover:text-[var(--color-text-primary)] transition-colors text-left block focus-ring rounded-sm">Licenses</button></li>
              <li><button onClick={() => navigate('/legal/privacy')} className="text-[var(--color-text-secondary)] hover:text-[var(--color-text-primary)] transition-colors text-left block focus-ring rounded-sm">Privacy Policy</button></li>
              <li><button onClick={() => navigate('/legal/terms')} className="text-[var(--color-text-secondary)] hover:text-[var(--color-text-primary)] transition-colors text-left block focus-ring rounded-sm">Terms of Use</button></li>
              <li><button onClick={() => navigate('/legal/copyright')} className="text-[var(--color-text-secondary)] hover:text-[var(--color-text-primary)] transition-colors text-left block focus-ring rounded-sm">Copyright</button></li>
              <li><button onClick={() => navigate('/legal/dmca')} className="text-[var(--color-text-secondary)] hover:text-[var(--color-text-primary)] transition-colors text-left block focus-ring rounded-sm">DMCA Policy</button></li>
            </ul>
          </div>

          {/* Community */}
          <div>
            <h3 className="text-sm font-semibold text-[var(--color-text-primary)] mb-4 flex items-center gap-2">
              <Users className="w-4 h-4 text-[var(--color-text-primary)]" />
              Community
            </h3>
            <ul className="space-y-3 text-sm">
              <li><button onClick={() => navigate('/contribute')} className="text-[var(--color-text-secondary)] hover:text-[var(--color-text-primary)] transition-colors text-left block focus-ring rounded-sm">Contribute</button></li>
              <li><button onClick={() => navigate('/governance')} className="text-[var(--color-text-secondary)] hover:text-[var(--color-text-primary)] transition-colors text-left block focus-ring rounded-sm">Governance</button></li>
              <li><a href="https://github.com/lemkin-ai" className="text-[var(--color-text-secondary)] hover:text-[var(--color-text-primary)] transition-colors inline-flex items-center gap-1 focus-ring rounded-sm">
                GitHub <ExternalLink className="w-3 h-3" />
              </a></li>
              <li><a href="https://discord.gg/lemkin-ai" className="text-[var(--color-text-secondary)] hover:text-[var(--color-text-primary)] transition-colors block focus-ring rounded-sm">Discord</a></li>
              <li><button onClick={() => navigate('/code-of-conduct')} className="text-[var(--color-text-secondary)] hover:text-[var(--color-text-primary)] transition-colors text-left block focus-ring rounded-sm">Code of Conduct</button></li>
            </ul>
          </div>
        </div>

        {/* Bottom Section with Back to Top */}
        <div className="border-t border-[var(--bd)] pt-8">
          <div className="flex flex-col lg:flex-row justify-between items-start lg:items-center gap-6">
            {/* Brand */}
            <div className="flex items-center gap-3">
              <LemkinLogo className="w-8 h-8" />
              <div>
                <span className="font-semibold text-lg text-[var(--color-text-primary)]">Lemkin AI</span>
                <p className="text-sm text-[var(--color-text-secondary)] mt-1">Evidence-grade AI for international justice</p>
              </div>
            </div>

            {/* Social Links */}
            <div className="flex items-center gap-4">
              <a href="https://github.com/lemkin-ai" className="text-[var(--color-text-secondary)] hover:text-[var(--color-text-primary)] transition-colors focus-ring rounded-sm">
                <Github className="w-5 h-5" />
              </a>
              <a href="https://twitter.com/lemkin-ai" className="text-[var(--color-text-secondary)] hover:text-[var(--color-text-primary)] transition-colors focus-ring rounded-sm">
                <Twitter className="w-5 h-5" />
              </a>
              <a href="mailto:contact@lemkin.ai" className="text-[var(--color-text-secondary)] hover:text-[var(--color-text-primary)] transition-colors focus-ring rounded-sm">
                <Mail className="w-5 h-5" />
              </a>
            </div>

            {/* Copyright */}
            <div className="text-sm text-[var(--color-text-secondary)]">
              &copy; 2025 Lemkin AI. Open source licensed.
            </div>
          </div>

          {/* Back to top */}
          <button
            onClick={() => window.scrollTo({ top: 0, behavior: 'smooth' })}
            className="mt-8 text-sm text-[var(--muted)] hover:text-[var(--ink)] transition-colors focus-ring rounded"
            aria-label="Back to top"
          >
            ↑ Back to top
          </button>
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
    <div className="relative min-h-screen">
      <section id="main" className="container-desktop" style={{ paddingBlock: 56 }}>
        {/* Hero Title with typographic clamp */}
        <motion.h1
          initial={{ opacity: 0, y: 18 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, ease: "easeOut" }}
          className="text-hero mb-4"
          style={{ fontSize: 'clamp(36px, 4vw, 48px)', lineHeight: 1.2, fontWeight: 700 }}
        >
          Evidence-grade AI for International Justice
        </motion.h1>

        {/* Hero Description */}
        <motion.p
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.45, delay: 0.1, ease: "easeOut" }}
          className="text-body-max mb-6 text-[var(--muted)]"
          style={{ maxWidth: '72ch' }}
        >
          Open-source machine learning models rigorously validated for legal proceedings.
          Trusted by tribunals, NGOs, and investigative teams worldwide.
        </motion.p>

        {/* Action Buttons with hierarchy */}
        <div className="flex gap-3">
          <Pressable
            className="btn-primary btn--lg inline-flex items-center gap-2"
            onClick={() => navigate('/models')}
          >
            Explore Models
          </Pressable>
          <Pressable
            className="btn-outline inline-flex items-center gap-2"
            onClick={() => navigate('/docs')}
            style={{ height: '48px', borderRadius: '14px' }}
          >
            Documentation
          </Pressable>
        </div>
      </section>

      {/* Evidence-Grade Trust Slice */}
      <section className="py-12 px-4 sm:px-6 lg:px-8 bg-[var(--color-bg-tertiary)] border-y border-[var(--color-border-secondary)]">
        <div className="max-w-6xl mx-auto">
          <div className="grid grid-cols-1 md:grid-cols-4 gap-8 text-center">
            <div className="space-y-2">
              <h3 className="text-sm font-semibold text-[var(--color-text-secondary)] uppercase tracking-wide">Who Reviews</h3>
              <p className="text-[var(--color-text-tertiary)] text-sm leading-relaxed">
                Tribunals, NGOs, Universities
              </p>
              <a href="/reviewers" className="text-xs text-[var(--color-fg-primary)] underline underline-offset-[3px] hover:opacity-90 focus-ring rounded-sm">View details</a>
            </div>
            <div className="space-y-2">
              <h3 className="text-sm font-semibold text-[var(--color-text-secondary)] uppercase tracking-wide">How We Evaluate</h3>
              <p className="text-[var(--color-text-tertiary)] text-sm leading-relaxed">
                Bias testing, Legal accuracy, Chain of custody
              </p>
              <a href="/methodology" className="text-xs text-[var(--color-fg-primary)] underline underline-offset-[3px] hover:opacity-90 focus-ring rounded-sm">View methodology</a>
            </div>
            <div className="space-y-2">
              <h3 className="text-sm font-semibold text-[var(--color-text-secondary)] uppercase tracking-wide">Update Cadence</h3>
              <p className="text-[var(--color-text-tertiary)] text-sm leading-relaxed">
                Monthly security, Quarterly evaluation
              </p>
              <a href="/changelog" className="text-xs text-[var(--color-fg-primary)] underline underline-offset-[3px] hover:opacity-90 focus-ring rounded-sm">View changelog</a>
            </div>
            <div className="space-y-2">
              <h3 className="text-sm font-semibold text-[var(--color-text-secondary)] uppercase tracking-wide">Misuse Reporting</h3>
              <p className="text-[var(--color-text-tertiary)] text-sm leading-relaxed">
                24h response, Public disclosure
              </p>
              <a href="/report" className="text-xs text-[var(--color-fg-primary)] underline underline-offset-[3px] hover:opacity-90 focus-ring rounded-sm">Report issue</a>
            </div>
          </div>
        </div>
      </section>

      {/* Trust & Credibility Section */}
      <section className="relative py-24 px-4 sm:px-6 lg:px-8 bg-[var(--color-bg-secondary)]">
        <div className="absolute inset-0" style={{ background: "var(--gradient-mesh)" }} />
        <div className="relative max-w-7xl mx-auto">
          <div className="text-center mb-16">
            <div className="inline-flex items-center gap-2 px-4 py-2 backdrop-blur-xl bg-[color-mix(in_srgb,var(--color-bg-elevated),transparent_92%)] border border-[var(--color-border-secondary)] rounded-full mb-8">
              <div className="w-2 h-2 bg-white rounded-full animate-pulse"></div>
              <span className="text-[var(--color-text-secondary)] text-sm">Developed with practitioners from international tribunals and NGOs</span>
            </div>

            <div className="flex justify-center items-center gap-8 mb-12 flex-wrap">
              <div className="flex items-center gap-3 px-6 py-3 backdrop-blur-xl bg-[color-mix(in_srgb,var(--color-bg-elevated),transparent_92%)] border border-[var(--color-border-secondary)] rounded-2xl group hover:bg-[color-mix(in_srgb,var(--color-bg-elevated),transparent_85%)] transition-all duration-300">
                <CheckCircle className="w-5 h-5 text-[var(--color-text-primary)] transition-colors" />
                <span className="text-[var(--color-text-primary)] font-medium">Rigorously Validated</span>
              </div>
              <div className="flex items-center gap-3 px-6 py-3 backdrop-blur-xl bg-[color-mix(in_srgb,var(--color-bg-elevated),transparent_92%)] border border-[var(--color-border-secondary)] rounded-2xl group hover:bg-[color-mix(in_srgb,var(--color-bg-elevated),transparent_85%)] transition-all duration-300">
                <Scale className="w-5 h-5 text-[var(--color-text-primary)] transition-colors" />
                <span className="text-[var(--color-text-primary)] font-medium">Legally Aware</span>
              </div>
              <div className="flex items-center gap-3 px-6 py-3 backdrop-blur-xl bg-[color-mix(in_srgb,var(--color-bg-elevated),transparent_92%)] border border-[var(--color-border-secondary)] rounded-2xl group hover:bg-[color-mix(in_srgb,var(--color-bg-elevated),transparent_85%)] transition-all duration-300">
                <Users className="w-5 h-5 text-[var(--color-text-primary)] transition-colors" />
                <span className="text-[var(--color-text-primary)] font-medium">Community-Driven</span>
              </div>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            <div className="group relative backdrop-blur-xl bg-[color-mix(in_srgb,var(--color-bg-elevated),transparent_88%)] border border-[var(--color-border-secondary)] rounded-3xl p-8 shadow-elevation-2 hover:shadow-elevation-3 transition-all duration-500 hover:scale-105">
              <div className="absolute inset-0 bg-gradient-to-br from-[var(--color-primary)]/5 to-transparent rounded-3xl opacity-0 group-hover:opacity-100 transition-opacity duration-500"></div>
              <div className="relative text-center">
                <div className="inline-flex items-center justify-center w-16 h-16 rounded-2xl mb-6
                                 bg-[var(--color-bg-elevated)] border border-[var(--color-border-primary)]">
                  <Shield className="w-8 h-8 text-[var(--color-text-primary)]" />
                </div>
                <h3 className="font-display text-xl font-bold text-[var(--color-text-primary)] mb-4">
                  Vetted & Validated
                </h3>
                <p className="text-[var(--color-text-secondary)] leading-relaxed mb-6">
                  All models undergo rigorous testing for accuracy, bias, and reliability in legal contexts with transparent evaluation metrics.
                </p>
                <button
                  onClick={() => navigate('/docs/evaluation')}
                  className="inline-flex items-center gap-2 text-[var(--color-text-secondary)] hover:text-[var(--color-text-primary)] font-medium underline underline-offset-[3px]"
                >
                  View evaluation process
                  <ArrowRight className="w-4 h-4 group-hover:translate-x-1 transition-transform" />
                </button>
              </div>
            </div>

            <div className="group relative backdrop-blur-xl bg-[color-mix(in_srgb,var(--color-bg-elevated),transparent_88%)] border border-[var(--color-border-secondary)] rounded-3xl p-8 shadow-elevation-2 hover:shadow-elevation-3 transition-all duration-500 hover:scale-105">
              <div className="absolute inset-0 bg-gradient-to-br from-[var(--color-primary)]/5 to-transparent rounded-3xl opacity-0 group-hover:opacity-100 transition-opacity duration-500"></div>
              <div className="relative text-center">
                <div className="inline-flex items-center justify-center w-16 h-16 rounded-2xl mb-6
                                 bg-[var(--color-bg-elevated)] border border-[var(--color-border-primary)]">
                  <Gavel className="w-8 h-8 text-[var(--color-text-primary)]" />
                </div>
                <h3 className="font-display text-xl font-bold text-[var(--color-text-primary)] mb-4">
                  Legally Aware
                </h3>
                <p className="text-[var(--color-text-secondary)] leading-relaxed mb-6">
                  Built with deep understanding of legal standards, evidence requirements, and chain of custody protocols.
                </p>
                <button
                  onClick={() => navigate('/governance')}
                  className="inline-flex items-center gap-2 text-[var(--color-text-secondary)] hover:text-[var(--color-text-primary)] font-medium underline underline-offset-[3px]"
                >
                  See governance
                  <ArrowRight className="w-4 h-4 group-hover:translate-x-1 transition-transform" />
                </button>
              </div>
            </div>

            <div className="group relative backdrop-blur-xl bg-[color-mix(in_srgb,var(--color-bg-elevated),transparent_88%)] border border-[var(--color-border-secondary)] rounded-3xl p-8 shadow-elevation-2 hover:shadow-elevation-3 transition-all duration-500 hover:scale-105">
              <div className="absolute inset-0 bg-gradient-to-br from-[var(--color-primary)]/5 to-transparent rounded-3xl opacity-0 group-hover:opacity-100 transition-opacity duration-500"></div>
              <div className="relative text-center">
                <div className="inline-flex items-center justify-center w-16 h-16 rounded-2xl mb-6
                                 bg-[var(--color-bg-elevated)] border border-[var(--color-border-primary)]">
                  <Eye className="w-8 h-8 text-[var(--color-text-primary)]" />
                </div>
                <h3 className="font-display text-xl font-bold text-[var(--color-text-primary)] mb-4">
                  Community-Driven
                </h3>
                <p className="text-[var(--color-text-secondary)] leading-relaxed mb-6">
                  Open development with full transparency, peer review, and collaborative governance from the global community.
                </p>
                <button
                  onClick={() => navigate('/contribute')}
                  className="inline-flex items-center gap-2 text-[var(--color-text-secondary)] hover:text-[var(--color-text-primary)] font-medium underline underline-offset-[3px]"
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

      {/* Practitioners' Brief */}
      <PractitionersBrief
        state={getFilteredBriefs().length > 0 ? 'ready' : 'empty'}
        data={getFilteredBriefs().length > 0 ? {
          title: getFilteredBriefs()[0]?.title || '',
          content: getFilteredBriefs()[0]?.excerpt || '',
          author: getFilteredBriefs()[0]?.author || '',
          date: getFilteredBriefs()[0]?.date || ''
        } : undefined}
      />

      {/* Join the Mission */}
      <section className="mx-auto" style={{ maxWidth: 1440, paddingInline: 48, paddingBlock: 56 }}>
        <h2 className="mb-6" style={{ fontSize: 32, fontWeight: 700 }}>Join the Mission</h2>
        <div className="grid" style={{ gridTemplateColumns: 'repeat(3, 1fr)', gap: 24 }}>
          <article className="card h-full p-6">
            <div className="flex items-start gap-3">
              <div className="p-2.5 rounded-xl border border-[var(--line)]">
                <CheckCircle className="w-5 h-5 text-[var(--accent)]" />
              </div>
              <div className="flex-1">
                <h3 className="text-lg font-semibold text-[var(--ink)]">Improve Model Evaluation</h3>
                <p className="text-sm text-[var(--subtle)] mt-1">10–15 min</p>
                <p className="text-sm text-[var(--muted)] mt-2.5">Help expand our evaluation datasets with legal domain expertise and bias testing protocols.</p>
                <button className="link inline-flex items-center mt-3.5 text-sm">
                  Get started<span className="ml-1.5">→</span>
                </button>
              </div>
            </div>
          </article>

          <article className="card h-full p-6">
            <div className="flex items-start gap-3">
              <div className="p-2.5 rounded-xl border border-[var(--line)]">
                <FileText className="w-5 h-5 text-[var(--accent)]" />
              </div>
              <div className="flex-1">
                <h3 className="text-lg font-semibold text-[var(--ink)]">Write Dataset Cards</h3>
                <p className="text-sm text-[var(--subtle)] mt-1">20–30 min</p>
                <p className="text-sm text-[var(--muted)] mt-2.5">Document training data sources, ethical considerations, and usage guidelines for transparency.</p>
                <button className="link inline-flex items-center mt-3.5 text-sm">
                  Get started<span className="ml-1.5">→</span>
                </button>
              </div>
            </div>
          </article>

          <article className="card h-full p-6">
            <div className="flex items-start gap-3">
              <div className="p-2.5 rounded-xl border border-[var(--line)]">
                <Code className="w-5 h-5 text-[var(--accent)]" />
              </div>
              <div className="flex-1">
                <h3 className="text-lg font-semibold text-[var(--ink)]">Add Unit Tests</h3>
                <p className="text-sm text-[var(--subtle)] mt-1">15–25 min</p>
                <p className="text-sm text-[var(--muted)] mt-2.5">Enhance model reliability with edge case testing and performance validation scripts.</p>
                <button className="link inline-flex items-center mt-3.5 text-sm">
                  Get started<span className="ml-1.5">→</span>
                </button>
              </div>
            </div>
          </article>
        </div>
      </section>
    </div>
  );
};

const ModelsPage = () => {
  const { navigate } = useRouter();
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedCategory, setSelectedCategory] = useState('all');
  const [selectedTier, setSelectedTier] = useState('all');
  const [selectedStatus, setSelectedStatus] = useState('all');
  const [viewMode, setViewMode] = useState<'table' | 'grid'>('grid');
  const [activeTab, setActiveTab] = useState<'models' | 'modules'>('models');
  const [selectedModel, setSelectedModel] = useState<any>(null);
  const [showInspector, setShowInspector] = useState(false);

  // Filter items based on current criteria
  const filteredItems = models.filter(item => {
    const matchesSearch = item.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
                          item.description.toLowerCase().includes(searchQuery.toLowerCase()) ||
                          item.cardSummary.toLowerCase().includes(searchQuery.toLowerCase());

    const matchesCategory = selectedCategory === 'all' ||
                           selectedCategory === 'All Categories' ||
                           item.category === selectedCategory;

    const matchesTier = selectedTier === 'all' ||
                       selectedTier === 'All Tiers' ||
                       (item.tier && `Tier ${item.tier}` === selectedTier);

    const matchesStatus = selectedStatus === 'all' ||
                         selectedStatus === 'All Status' ||
                         item.status === selectedStatus;

    const matchesTab = activeTab === 'models' ?
                      item.moduleType === 'model' :
                      item.moduleType === 'module';

    return matchesSearch && matchesCategory && matchesTier && matchesStatus && matchesTab;
  });

  // Get unique categories and tiers for filtering
  const availableCategories = [...new Set(models.map(item => item.category))];
  const availableTiers = [...new Set(models.filter(item => item.tier).map(item => item.tier))].sort();

  const handleModelSelect = (model: any) => {
    setSelectedModel(model);
    setShowInspector(true);
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'production': return 'bg-emerald-100 text-emerald-800 dark:bg-emerald-900 dark:text-emerald-200';
      case 'implementation': return 'bg-amber-100 text-amber-800 dark:bg-amber-900 dark:text-amber-200';
      case 'beta': return 'bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200';
      case 'stable': return 'bg-emerald-100 text-emerald-800 dark:bg-emerald-900 dark:text-emerald-200';
      default: return 'bg-gray-100 text-gray-800 dark:bg-gray-800 dark:text-gray-200';
    }
  };

  const getTierBadgeColor = (tier: number) => {
    const colors = [
      'bg-purple-100 text-purple-800 dark:bg-purple-900 dark:text-purple-200',
      'bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200',
      'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200',
      'bg-amber-100 text-amber-800 dark:bg-amber-900 dark:text-amber-200',
      'bg-orange-100 text-orange-800 dark:bg-orange-900 dark:text-orange-200',
      'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200'
    ];
    return colors[tier - 1] || colors[0];
  };

  // Keyboard handling for inspector
  useEffect(() => {
    if (!showInspector) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setShowInspector(false);
    };
    document.addEventListener('keydown', onKey);
    return () => document.removeEventListener('keydown', onKey);
  }, [showInspector]);

  return (
    <div className="min-h-screen py-12 px-4 sm:px-6 lg:px-8">
      <div className="max-w-7xl mx-auto">
        <div className="mb-8">
          <h1 className="text-4xl font-bold text-gray-900 dark:text-white mb-4">AI Models & Technical Resources</h1>
          <p className="text-lg text-gray-600 dark:text-gray-400 max-w-4xl">
            Comprehensive suite of 18 specialized modules organized in 6 tiers, providing AI-augmented tools for legal investigations
            while maintaining the highest standards of evidence integrity and legal ethics. Browse our production-ready toolkits
            and explore detailed documentation for each module.
          </p>
        </div>

        {/* Tab Navigation */}
        <div className="mb-8">
          <div className="border-b border-gray-200 dark:border-gray-700">
            <nav className="-mb-px flex space-x-8">
              <button
                onClick={() => setActiveTab('models')}
                className={`py-2 px-1 border-b-2 font-medium text-sm transition-colors ${
                  activeTab === 'models'
                    ? 'border-blue-500 text-blue-600 dark:text-blue-400'
                    : 'border-transparent text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-300'
                }`}
              >
                AI Models ({models.filter(m => m.moduleType === 'model').length})
              </button>
              <button
                onClick={() => setActiveTab('modules')}
                className={`py-2 px-1 border-b-2 font-medium text-sm transition-colors ${
                  activeTab === 'modules'
                    ? 'border-blue-500 text-blue-600 dark:text-blue-400'
                    : 'border-transparent text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-300'
                }`}
              >
                Technical Modules ({models.filter(m => m.moduleType === 'module').length})
              </button>
            </nav>
          </div>
        </div>

        {/* Filters */}
        <div className="mb-8 space-y-4">
          <div className="flex flex-col sm:flex-row gap-4">
            <div className="flex-1">
              <div className="relative">
                <label htmlFor="search" className="sr-only">Search {activeTab}</label>
                <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-5 h-5" />
                <input
                  id="search"
                  type="search"
                  placeholder={`Search ${activeTab}...`}
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="w-full pl-10 pr-4 py-2 border border-[var(--color-border-primary)] rounded-lg bg-[var(--color-bg-elevated)] text-[var(--color-text-primary)] focus:outline-none focus:ring-2 focus:ring-[var(--color-border-active)]"
                />
              </div>
            </div>

            {/* Category Filter */}
            <select
              value={selectedCategory}
              onChange={(e) => setSelectedCategory(e.target.value)}
              className="px-4 py-2 border border-[var(--color-border-primary)] rounded-lg bg-[var(--color-bg-elevated)] text-[var(--color-text-primary)] focus:outline-none focus:ring-2 focus:ring-[var(--color-border-active)]"
            >
              <option value="all">All Categories</option>
              {availableCategories.map(category => (
                <option key={category} value={category}>
                  {category}
                </option>
              ))}
            </select>

            {/* Tier Filter - Only for modules */}
            {activeTab === 'modules' && (
              <select
                value={selectedTier}
                onChange={(e) => setSelectedTier(e.target.value)}
                className="px-4 py-2 border border-[var(--color-border-primary)] rounded-lg bg-[var(--color-bg-elevated)] text-[var(--color-text-primary)] focus:outline-none focus:ring-2 focus:ring-[var(--color-border-active)]"
              >
                <option value="all">All Tiers</option>
                {availableTiers.map(tier => (
                  <option key={tier} value={`Tier ${tier}`}>Tier {tier}</option>
                ))}
              </select>
            )}

            {/* Status Filter */}
            <select
              value={selectedStatus}
              onChange={(e) => setSelectedStatus(e.target.value)}
              className="px-4 py-2 border border-[var(--color-border-primary)] rounded-lg bg-[var(--color-bg-elevated)] text-[var(--color-text-primary)] focus:outline-none focus:ring-2 focus:ring-[var(--color-border-active)]"
            >
              {modelStatuses.map(status => (
                <option key={status} value={status === 'All Status' ? 'all' : status}>
                  {status === 'implementation-ready' ? 'Implementation Ready' :
                   status.charAt(0).toUpperCase() + status.slice(1)}
                </option>
              ))}
            </select>
          </div>
        </div>

        {/* View Mode Toggle - Only for models */}
        {activeTab === 'models' && (
          <div className="flex justify-end mb-4">
            <div className="inline-flex rounded-lg border border-[var(--color-border-primary)] p-0.5">
              <button
                onClick={() => setViewMode('table')}
                className={`px-3 py-1.5 text-sm font-medium rounded-md transition-colors ${
                  viewMode === 'table'
                    ? 'bg-[var(--color-bg-secondary)] text-[var(--color-text-primary)]'
                    : 'text-[var(--color-text-secondary)] hover:text-[var(--color-text-primary)]'
                }`}
              >
                <div className="flex items-center gap-2">
                  <FileText className="w-4 h-4" />
                  Table
                </div>
              </button>
              <button
                onClick={() => setViewMode('grid')}
                className={`px-3 py-1.5 text-sm font-medium rounded-md transition-colors ${
                  viewMode === 'grid'
                    ? 'bg-[var(--color-bg-secondary)] text-[var(--color-text-primary)]'
                    : 'text-[var(--color-text-secondary)] hover:text-[var(--color-text-primary)]'
                }`}
              >
                <div className="flex items-center gap-2">
                  <Grid className="w-4 h-4" />
                  Grid
                </div>
              </button>
            </div>
          </div>
        )}

        {/* Content Area */}
        {filteredItems.length > 0 ? (
          activeTab === 'models' ? (
            viewMode === 'table' ? (
              <div className="mx-auto" style={{ maxWidth: 1600, paddingInline: 48 }}>
                <div className="card p-0 overflow-auto" style={{ maxHeight: '70vh' }}>
                  <table className="min-w-full">
                    <thead className="bg-[var(--surface)] text-[var(--muted)] sticky top-0 z-10">
                      <tr>
                        <Th sticky="left">{activeTab === 'models' ? 'Model' : 'Module'}</Th>
                        <Th align="right">Performance</Th>
                        <Th align="center">Status</Th>
                        <Th align="center">Type</Th>
                        <Th align="right">Category</Th>
                        <Th sticky="right" align="center">Actions</Th>
                      </tr>
                    </thead>
                    <tbody>
                      {filteredItems.map((item) => (
                        <tr
                          key={item.id}
                          className="border-t border-[var(--line-soft)] hover:bg-[var(--elevated)]/40 transition-colors"
                        >
                          <Td sticky="left">
                            <div className="flex items-center gap-3">
                              <div>
                                <h3 className="font-semibold text-[var(--color-text-primary)]">{item.name}</h3>
                                <p className="text-sm text-[var(--color-text-secondary)]">{item.cardSummary}</p>
                              </div>
                            </div>
                          </Td>
                          <Td align="right">{item.metrics?.accuracy || 'N/A'}</Td>
                          <Td align="center">
                            <Badge variant={item.status}>{item.status}</Badge>
                          </Td>
                          <Td align="center">{item.type}</Td>
                          <Td align="right">{item.category}</Td>
                          <Td sticky="right" align="center">
                            <Button variant="ghost" size="sm" onClick={() => handleModelSelect(item)}>
                              View
                            </Button>
                          </Td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            ) : (
              // Grid View for Models/Modules
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                {filteredItems.map((item, index) => (
                  <div
                    key={item.id}
                    style={{ animationDelay: `${index * 75}ms` }}
                    className="animate-fade-up"
                    onClick={() => handleModelSelect(item)}
                  >
                    <Card hover className="cursor-pointer h-full">
                      <div className="flex items-start justify-between mb-4">
                        <div className="flex-1">
                          <h3 className="text-lg font-semibold text-[var(--color-text-primary)] mb-2">
                            {item.name}
                          </h3>
                          <p className="text-sm text-[var(--color-text-secondary)] mb-3">
                            {item.cardSummary}
                          </p>
                        </div>
                        {item.featured && (
                          <Badge variant="default" className="ml-2">Featured</Badge>
                        )}
                      </div>

                      <div className="space-y-2 mb-4">
                        <div className="flex justify-between text-sm">
                          <span className="text-[var(--color-text-secondary)]">Type:</span>
                          <span className="text-[var(--color-text-primary)]">{item.type}</span>
                        </div>
                        <div className="flex justify-between text-sm">
                          <span className="text-[var(--color-text-secondary)]">Category:</span>
                          <span className="text-[var(--color-text-primary)]">{item.category}</span>
                        </div>
                        {item.tier && (
                          <div className="flex justify-between text-sm">
                            <span className="text-[var(--color-text-secondary)]">Tier:</span>
                            <Badge variant="default" className={getTierBadgeColor(item.tier)}>
                              Tier {item.tier}
                            </Badge>
                          </div>
                        )}
                      </div>

                      <div className="flex items-center justify-between">
                        <Badge variant={item.status} className={getStatusColor(item.status)}>
                          {item.status === 'implementation-ready' ? 'Implementation Ready' :
                           item.status.charAt(0).toUpperCase() + item.status.slice(1)}
                        </Badge>
                        <Button variant="ghost" size="sm">
                          View Details
                        </Button>
                      </div>
                    </Card>
                  </div>
                ))}
              </div>
            )
          ) : (
            // Modules view - simplified since we now use the same card structure
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {filteredItems.map((item, index) => (
                <div
                  key={item.id}
                  style={{ animationDelay: `${index * 75}ms` }}
                  className="animate-fade-up"
                  onClick={() => handleModelSelect(item)}
                >
                  <Card hover className="cursor-pointer h-full">
                    <div className="flex items-start justify-between mb-4">
                      <div className="flex-1">
                        <h3 className="text-lg font-semibold text-[var(--color-text-primary)] mb-2">
                          {item.name}
                        </h3>
                        <p className="text-sm text-[var(--color-text-secondary)] mb-3">
                          {item.cardSummary}
                        </p>
                      </div>
                      {item.featured && (
                        <Badge variant="default" className="ml-2">Featured</Badge>
                      )}
                    </div>

                    <div className="space-y-2 mb-4">
                      <div className="flex justify-between text-sm">
                        <span className="text-[var(--color-text-secondary)]">Type:</span>
                        <span className="text-[var(--color-text-primary)]">{item.type}</span>
                      </div>
                      <div className="flex justify-between text-sm">
                        <span className="text-[var(--color-text-secondary)]">Category:</span>
                        <span className="text-[var(--color-text-primary)]">{item.category}</span>
                      </div>
                      {item.tier && (
                        <div className="flex justify-between text-sm">
                          <span className="text-[var(--color-text-secondary)]">Tier:</span>
                          <Badge variant="default" className={getTierBadgeColor(item.tier)}>
                            Tier {item.tier}
                          </Badge>
                        </div>
                      )}
                    </div>

                    <div className="flex items-center justify-between">
                      <Badge variant={item.status} className={getStatusColor(item.status)}>
                        {item.status === 'implementation-ready' ? 'Implementation Ready' :
                         item.status.charAt(0).toUpperCase() + item.status.slice(1)}
                      </Badge>
                      <Button variant="ghost" size="sm">
                        View Details
                      </Button>
                    </div>
                  </Card>
                </div>
              ))}
            </div>
          )
        ) : (
          <div className="text-center py-12">
            <AlertCircle className="w-12 h-12 text-gray-400 mx-auto mb-4" />
            <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-2">No {activeTab} found</h3>
            <p className="text-gray-600 dark:text-gray-400">
              Try adjusting your search or filters
            </p>
          </div>
        )}

      </div>

      {/* Slide-over Inspector */}
      {showInspector && selectedModel && (
        <>
          {/* Backdrop */}
          <div
            className="fixed inset-0 bg-black bg-opacity-25 z-40 transition-opacity"
            onClick={() => setShowInspector(false)}
          />

          {/* Slide-over Panel */}
          <div className="fixed inset-y-0 right-0 z-50 w-full sm:w-96 bg-[var(--color-bg-primary)] shadow-elevation-4 transform transition-transform duration-300">
            <div className="h-full flex flex-col">
              {/* Header */}
              <div className="px-6 py-4 border-b border-[var(--color-border-primary)]">
                <div className="flex items-center justify-between">
                  <h2 className="text-lg font-semibold text-[var(--color-text-primary)]">
                    Model Inspector
                  </h2>
                  <button
                    onClick={() => setShowInspector(false)}
                    className="p-2 rounded-md hover:bg-[var(--color-bg-secondary)] transition-colors"
                  >
                    <X className="w-5 h-5 text-[var(--color-text-secondary)]" />
                  </button>
                </div>
              </div>

              {/* Content */}
              <div className="flex-1 overflow-y-auto p-6 space-y-6">
                {/* Item Info */}
                <div>
                  <h3 className="text-xl font-semibold text-[var(--color-text-primary)] mb-2">
                    {selectedModel.name}
                  </h3>
                  <div className="flex gap-2 mb-3">
                    <Badge variant={selectedModel.status}>{selectedModel.status}</Badge>
                    <Badge variant="secondary">{selectedModel.type}</Badge>
                    {selectedModel.tier && (
                      <Badge variant="default" className={getTierBadgeColor(selectedModel.tier)}>
                        Tier {selectedModel.tier}
                      </Badge>
                    )}
                  </div>
                  <p className="mt-3 text-sm text-[var(--color-text-secondary)]">
                    {selectedModel.description}
                  </p>
                </div>

                {/* Quick Stats */}
                <Card variant="filled">
                  <h4 className="text-sm font-semibold text-[var(--color-text-primary)] mb-3">
                    {selectedModel.moduleType === 'model' ? 'Performance Metrics' : 'Module Information'}
                  </h4>
                  <div className="grid grid-cols-2 gap-4">
                    {selectedModel.moduleType === 'model' ? (
                      <>
                        <div>
                          <div className="text-2xl font-bold text-[var(--color-text-primary)]">
                            {selectedModel.metrics?.accuracy || 'N/A'}
                          </div>
                          <div className="text-xs text-[var(--color-text-tertiary)]">Accuracy</div>
                        </div>
                        <div>
                          <div className="text-2xl font-bold text-[var(--color-text-primary)]">
                            {selectedModel.metrics?.inferenceSpeed || 'N/A'}
                          </div>
                          <div className="text-xs text-[var(--color-text-tertiary)]">Speed</div>
                        </div>
                        <div>
                          <div className="text-2xl font-bold text-[var(--color-text-primary)]">
                            {selectedModel.metrics?.modelSize || 'N/A'}
                          </div>
                          <div className="text-xs text-[var(--color-text-tertiary)]">Model Size</div>
                        </div>
                        <div>
                          <div className="text-2xl font-bold text-[var(--color-text-primary)]">
                            {selectedModel.technicalSpecs?.framework || 'N/A'}
                          </div>
                          <div className="text-xs text-[var(--color-text-tertiary)]">Framework</div>
                        </div>
                      </>
                    ) : (
                      <>
                        <div>
                          <div className="text-2xl font-bold text-[var(--color-text-primary)]">
                            {selectedModel.category}
                          </div>
                          <div className="text-xs text-[var(--color-text-tertiary)]">Category</div>
                        </div>
                        <div>
                          <div className="text-2xl font-bold text-[var(--color-text-primary)]">
                            {selectedModel.technicalSpecs?.framework || 'Python'}
                          </div>
                          <div className="text-xs text-[var(--color-text-tertiary)]">Framework</div>
                        </div>
                        <div>
                          <div className="text-2xl font-bold text-[var(--color-text-primary)]">
                            {selectedModel.useCases?.length || 0}
                          </div>
                          <div className="text-xs text-[var(--color-text-tertiary)]">Use Cases</div>
                        </div>
                        <div>
                          <div className="text-2xl font-bold text-[var(--color-text-primary)]">
                            {selectedModel.capabilities?.length || 0}
                          </div>
                          <div className="text-xs text-[var(--color-text-tertiary)]">Capabilities</div>
                        </div>
                      </>
                    )}
                  </div>
                </Card>

                {/* Technical Details */}
                <Card variant="outlined">
                  <div className="flex items-center justify-between mb-3">
                    <h4 className="text-sm font-semibold text-[var(--color-text-primary)]">Technical Specifications</h4>
                    {selectedModel.githubRepo && (
                      <button
                        onClick={() => window.open(selectedModel.githubRepo, '_blank')}
                        className="text-xs text-[var(--color-text-primary)] hover:underline flex items-center gap-1 bg-transparent border-0 cursor-pointer"
                      >
                        View Repository <ExternalLink className="w-3 h-3" />
                      </button>
                    )}
                  </div>
                  <dl className="space-y-2">
                    <div className="flex justify-between text-sm">
                      <dt className="text-[var(--color-text-secondary)]">Framework</dt>
                      <dd className="text-[var(--color-text-primary)] font-medium">
                        {selectedModel.technicalSpecs?.framework || 'N/A'}
                      </dd>
                    </div>
                    <div className="flex justify-between text-sm">
                      <dt className="text-[var(--color-text-secondary)]">Architecture</dt>
                      <dd className="text-[var(--color-text-primary)] font-medium">
                        {selectedModel.technicalSpecs?.architecture || 'N/A'}
                      </dd>
                    </div>
                    <div className="flex justify-between text-sm">
                      <dt className="text-[var(--color-text-secondary)]">Input Format</dt>
                      <dd className="text-[var(--color-text-primary)]">
                        {selectedModel.technicalSpecs?.inputFormat || 'N/A'}
                      </dd>
                    </div>
                    <div className="flex justify-between text-sm">
                      <dt className="text-[var(--color-text-secondary)]">Output Format</dt>
                      <dd className="text-[var(--color-text-primary)]">
                        {selectedModel.technicalSpecs?.outputFormat || 'N/A'}
                      </dd>
                    </div>
                  </dl>
                </Card>

                {/* Use Cases */}
                {selectedModel.useCases && selectedModel.useCases.length > 0 && (
                  <Card variant="outlined">
                    <h4 className="text-sm font-semibold text-[var(--color-text-primary)] mb-3">Use Cases</h4>
                    <ul className="space-y-1">
                      {selectedModel.useCases.slice(0, 5).map((useCase: string, index: number) => (
                        <li key={index} className="text-sm text-[var(--color-text-secondary)] flex items-start gap-2">
                          <span className="w-1 h-1 bg-[var(--color-text-tertiary)] rounded-full mt-2 flex-shrink-0" />
                          {useCase}
                        </li>
                      ))}
                    </ul>
                  </Card>
                )}

                {/* Tags */}
                <div>
                  <h4 className="text-sm font-semibold text-[var(--color-text-primary)] mb-2">Tags</h4>
                  <div className="flex flex-wrap gap-2">
                    {selectedModel.tags.map((tag: string) => (
                      <span
                        key={tag}
                        className="px-2 py-1 bg-[var(--color-bg-secondary)] text-[var(--color-text-secondary)] rounded-md text-xs"
                      >
                        {tag}
                      </span>
                    ))}
                  </div>
                </div>
              </div>

              {/* Actions Footer */}
              <div className="px-6 py-4 border-t border-[var(--color-border-primary)] space-y-2">
                <Button
                  className="w-full"
                  onClick={() => {
                    navigator.clipboard.writeText(`lemkin-ai/${selectedModel.id}`);
                  }}
                >
                  <Copy className="w-4 h-4 mr-2" />
                  Copy {selectedModel.moduleType === 'model' ? 'Model' : 'Module'} ID
                </Button>
                {selectedModel.githubRepo && (
                  <Button
                    variant="secondary"
                    className="w-full"
                    onClick={() => window.open(selectedModel.githubRepo, '_blank')}
                  >
                    <Github className="w-4 h-4 mr-2" />
                    View on GitHub
                  </Button>
                )}
                {selectedModel.localPath && (
                  <Button
                    variant="secondary"
                    className="w-full"
                    onClick={() => {
                      navigator.clipboard.writeText(selectedModel.localPath);
                    }}
                  >
                    <Folder className="w-4 h-4 mr-2" />
                    Copy Local Path
                  </Button>
                )}
              </div>
            </div>
          </div>
        </>
      )}
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
                    <tr className="hover:bg-[var(--surface)] transition-colors">
                      <td className="px-6 py-4 whitespace-nowrap text-sm">ICC Proceedings</td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm">5.3</td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm">1.8</td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm">0.947</td>
                    </tr>
                    <tr className="hover:bg-[var(--surface)] transition-colors">
                      <td className="px-6 py-4 whitespace-nowrap text-sm">ICTY Archive</td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm">6.1</td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm">2.2</td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm">0.938</td>
                    </tr>
                    <tr className="hover:bg-[var(--surface)] transition-colors">
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

// Article content data - maps article IDs to their full content
const articleContentMap: Record<string, string> = {
  'large-language-models-legal-analysis': `
# Large Language Models for Legal Document Analysis: Balancing Transformative Potential with Critical Risks in Human Rights Investigations

Human rights investigations generate massive document collections that can overwhelm traditional analysis methods. A single case may involve thousands of witness statements, government communications, corporate records, and media reports spanning multiple languages and legal systems. The Syrian conflict alone has produced over 800,000 documents collected by various investigative mechanisms, while cases involving multinational corporations or systematic violations can encompass millions of pages requiring expert review.

## Document Processing Capabilities and Analytical Potential

Large language models excel at rapid processing of complex legal documents that would require extensive manual review. Contract analysis represents one area of particular strength, where models can identify unusual provisions, extract key terms, and compare documents across large collections to detect inconsistencies that might indicate fraud or deception.

Case law analysis benefits significantly from LLMs' ability to process judicial opinions and identify relevant precedents across extensive legal databases. The technology can extract legal reasoning, compare factual similarities between cases, and trace precedent hierarchies in ways that accelerate research processes.

## Hallucination Risks and Accuracy Challenges

Factual hallucinations pose severe risks for legal applications, where LLMs may generate plausible but incorrect information with high confidence. The phenomenon extends beyond simple factual errors to include fabricated legal citations, non-existent case law references, and misstatements of legal standards that can appear credible to non-expert users.

These risks prove particularly dangerous in human rights investigations, where time pressures and resource constraints may limit verification procedures. Investigators working on urgent cases may rely on LLM-generated legal research without adequate fact-checking, potentially compromising legal strategies or evidence presentation.

## Implementation Pathways and Risk Management

Effective deployment requires careful integration with existing legal workflows while maintaining professional standards and quality control processes. Legacy system integration challenges arise when incorporating LLM capabilities into document management platforms, case management systems, and legal research tools.

The path forward requires recognition that LLMs represent powerful tools for legal document analysis while acknowledging their significant limitations in contexts where accuracy and reliability prove paramount. Human rights organizations can harness these capabilities effectively through careful deployment strategies that emphasize human oversight, systematic validation, and appropriate risk management.
`,
  'satellite-imagery-atrocity-documentation': `
# Satellite Imagery Analysis for Atrocity Documentation: Transforming Pixels into Evidence for International Justice

On March 16, 2022, a commercial satellite captured imagery of Mariupol, Ukraine that appeared unremarkable to casual observers—urban buildings, streets, and infrastructure rendered in the familiar patterns of satellite photography. To investigators employing artificial intelligence analysis tools, those same pixels revealed systematic destruction targeting civilian hospitals, residential areas, and essential infrastructure with patterns suggesting deliberate rather than incidental damage.

## The Mathematical Foundation of Change Detection

Modern satellite imagery analysis for conflict documentation relies fundamentally on change detection algorithms that compare images of identical geographic areas captured at different times, identifying pixel-level differences that indicate ground modifications. This deceptively simple concept involves sophisticated mathematical approaches that must distinguish conflict-related destruction from natural seasonal changes, weather variations, and environmental disasters.

Advanced algorithms employ multiple analytical approaches to isolate human-caused changes from natural variations. Spectral analysis examines how different materials reflect electromagnetic radiation across various wavelengths, enabling systems to distinguish between the spectral signatures of intact buildings versus debris piles, burned vegetation versus natural seasonal dormancy, or disturbed soil versus agricultural activity.

## Multi-Spectral Intelligence and Atmospheric Correction

Contemporary satellites capture imagery across multiple spectral bands extending from visible light through near-infrared, short-wave infrared, and sometimes thermal ranges. Each spectral band reveals different information about ground conditions, enabling comprehensive analyses that exceed any single band's capabilities.

Thermal imagery can detect heat signatures from fires, recent explosions, or industrial activity, providing temporal context that helps narrow the timing windows for destruction events. Short-wave infrared bands penetrate atmospheric haze and reveal surface materials that might be obscured in visible light imagery.

## Legal Evidence Presentation and Courtroom Communication

Translating technical satellite imagery analysis results into formats suitable for legal proceedings represents a critical challenge that affects whether sophisticated analytical capabilities can contribute meaningfully to accountability efforts. Courts and legal tribunals have specific evidence presentation requirements that may conflict with technical best practices for data visualization.

Legal evidence presentation must address multiple audience levels simultaneously: summary visualizations that communicate key findings clearly to lay decision-makers, detailed technical analyses that support expert testimony under cross-examination, and comprehensive methodological documentation that addresses potential challenges from opposing parties.
`,
  'human-in-loop-imperative': `
# The Human-in-the-Loop Imperative: Why AI Can Never Replace Investigators

Human rights investigations now process evidence streams that would have overwhelmed any previous generation of investigators. Digital documentation from conflict zones generates millions of photos, videos, and social media posts. Financial crime investigations require analysis of transaction networks spanning hundreds of institutions across dozens of jurisdictions.

## Constitutional and Legal Foundations for Human Oversight

Due process guarantees embedded in legal systems worldwide establish clear requirements for human involvement in evidence evaluation and decision-making that algorithmic systems cannot satisfy. The right to understand evidence used in legal proceedings requires explanations that affected individuals and their counsel can examine and challenge.

International human rights law reinforces these obligations through specific procedural requirements. Article 14 of the International Covenant on Civil and Political Rights establishes the right to examine evidence and challenge conclusions in legal proceedings. This provision demands that evidence evaluation processes remain accessible to scrutiny.

## The Cognitive Boundaries of Algorithmic Analysis

Machine learning systems excel at identifying statistical patterns within large datasets but lack the conceptual reasoning capabilities essential for legal analysis. These systems recognize correlations between input features and output classifications without understanding causal relationships, contextual meaning, or the logical connections that link evidence to legal conclusions.

Intent determination represents a clear example of analysis that requires human cognition rather than pattern recognition. International criminal law under the Rome Statute requires proof of specific intent for crimes against humanity and genocide, often involving interpretation of statements, behaviors, and contextual circumstances that may have multiple plausible explanations.

## Strategic Task Allocation Between Human and Artificial Intelligence

Effective AI integration requires systematic allocation of responsibilities that leverages computational efficiency while preserving human judgment for tasks requiring interpretation, cultural understanding, and legal reasoning. The division must recognize distinct capabilities rather than treating AI as a general replacement for human analysis.

Computational systems excel at high-volume processing tasks that benefit from consistency and speed. Document review and initial classification enable investigators to prioritize human attention toward the most relevant materials from vast archives. However, interpretive functions require human cognition that cannot be delegated without compromising analytical quality and legal validity.
`
};

// Function to get article content by ID
const getArticleContent = (articleId: string): string => {
  return articleContentMap[articleId] || `
# Article Content

This article content is being processed and will be available soon. Please check back later for the full analysis and detailed insights.

## Abstract

This comprehensive article covers important aspects of AI in human rights investigations and legal technology applications.

## Key Points

- Technical analysis and methodological approaches
- Implementation challenges and solutions
- Legal and ethical considerations
- Best practices and recommendations

## Conclusion

The integration of AI technologies in human rights work requires careful consideration of both opportunities and limitations, ensuring that human oversight remains central to all critical decisions.
`;
};

const ArticleDetailPage = () => {
  const { currentPath, navigate } = useRouter();
  const articleId = currentPath.split('/')[2]; // Extract article ID from /articles/{id}

  const article = mockPractitionerBriefs.find(a => a.id === articleId);

  if (!article) {
    return (
      <div className="min-h-screen py-12 px-4 sm:px-6 lg:px-8">
        <div className="max-w-4xl mx-auto">
          <div className="text-center">
            <h1 className="text-4xl font-bold text-gray-900 dark:text-white mb-4">Article Not Found</h1>
            <p className="text-lg text-gray-600 dark:text-gray-400 mb-8">
              The article you're looking for doesn't exist or has been moved.
            </p>
            <button
              onClick={() => navigate('/articles')}
              className="btn-primary"
            >
              Back to Articles
            </button>
          </div>
        </div>
      </div>
    );
  }

  const content = getArticleContent(articleId);

  return (
    <div className="min-h-screen py-12 px-4 sm:px-6 lg:px-8">
      <div className="max-w-4xl mx-auto">
        {/* Back Navigation */}
        <div className="mb-8">
          <button
            onClick={() => navigate('/articles')}
            className="inline-flex items-center text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white transition-colors"
          >
            <ArrowLeft className="w-4 h-4 mr-2" />
            Back to Articles
          </button>
        </div>

        {/* Article Header */}
        <header className="mb-12">
          <div className="mb-4">
            <span className="inline-flex items-center px-3 py-1 rounded-full text-sm font-medium bg-blue-100 dark:bg-blue-900 text-blue-800 dark:text-blue-200">
              {article.category}
            </span>
          </div>

          <h1 className="text-4xl md:text-5xl font-bold text-gray-900 dark:text-white mb-6 leading-tight">
            {article.title}
          </h1>

          <p className="text-xl text-gray-600 dark:text-gray-400 mb-8 leading-relaxed">
            {article.excerpt}
          </p>

          {/* Article Meta */}
          <div className="flex flex-wrap items-center gap-6 text-sm text-gray-500 dark:text-gray-400 border-b border-gray-200 dark:border-gray-700 pb-8">
            <div className="flex items-center gap-2">
              <User className="w-4 h-4" />
              <span className="font-medium">{article.author}</span>
            </div>
            <div className="flex items-center gap-2">
              <Calendar className="w-4 h-4" />
              <time dateTime={article.date}>
                {new Date(article.date).toLocaleDateString('en-US', {
                  year: 'numeric',
                  month: 'long',
                  day: 'numeric'
                })}
              </time>
            </div>
            <div className="flex items-center gap-2">
              <Clock className="w-4 h-4" />
              <span>{article.readTime}</span>
            </div>
          </div>

          {/* Tags */}
          <div className="mt-6 flex flex-wrap gap-2">
            {article.tags.map(tag => (
              <span
                key={tag}
                className="px-3 py-1 bg-gray-100 dark:bg-gray-700 text-gray-600 dark:text-gray-300 rounded-full text-sm"
              >
                #{tag}
              </span>
            ))}
          </div>
        </header>

        {/* Article Content */}
        <article className="prose prose-lg prose-gray dark:prose-invert max-w-none">
          <div
            dangerouslySetInnerHTML={{
              __html: content
                .split('\n')
                .map(line => {
                  if (line.startsWith('# ')) {
                    return `<h1 class="text-3xl font-bold text-gray-900 dark:text-white mb-6 mt-12">${line.substring(2)}</h1>`;
                  } else if (line.startsWith('## ')) {
                    return `<h2 class="text-2xl font-semibold text-gray-900 dark:text-white mb-4 mt-8">${line.substring(3)}</h2>`;
                  } else if (line.startsWith('### ')) {
                    return `<h3 class="text-xl font-semibold text-gray-900 dark:text-white mb-3 mt-6">${line.substring(4)}</h3>`;
                  } else if (line.trim() === '') {
                    return '<br>';
                  } else {
                    return `<p class="text-gray-700 dark:text-gray-300 mb-4 leading-relaxed">${line}</p>`;
                  }
                })
                .join('')
            }}
          />
        </article>

        {/* Article Footer */}
        <footer className="mt-16 pt-8 border-t border-gray-200 dark:border-gray-700">
          <div className="flex items-center justify-between">
            <div className="text-sm text-gray-500 dark:text-gray-400">
              Last reviewed: {article.lastReviewed}
            </div>

            <div className="flex items-center gap-4">
              <button
                onClick={() => {
                  navigator.clipboard.writeText(window.location.href);
                }}
                className="inline-flex items-center px-4 py-2 text-sm font-medium text-gray-700 dark:text-gray-300 bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-600 rounded-md hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors"
              >
                <Share className="w-4 h-4 mr-2" />
                Share Article
              </button>
            </div>
          </div>
        </footer>

        {/* Related Articles */}
        <section className="mt-16">
          <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6">Related Articles</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {mockPractitionerBriefs
              .filter(a => a.id !== articleId && a.tags.some(tag => article.tags.includes(tag)))
              .slice(0, 4)
              .map(relatedArticle => (
                <Card key={relatedArticle.id} hover className="cursor-pointer">
                  <div onClick={() => navigate(`/articles/${relatedArticle.id}`)}>
                    <h3 className="font-semibold text-lg text-gray-900 dark:text-white mb-2">
                      {relatedArticle.title}
                    </h3>
                    <p className="text-gray-600 dark:text-gray-400 text-sm mb-3">
                      {relatedArticle.excerpt}
                    </p>
                    <div className="flex items-center gap-4 text-xs text-gray-500 dark:text-gray-400">
                      <span>{relatedArticle.author}</span>
                      <span>{relatedArticle.readTime}</span>
                    </div>
                  </div>
                </Card>
              ))}
          </div>
        </section>
      </div>
    </div>
  );
};

// ArticlesPage is imported from ./ArticlesPage.tsx


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
                  <div className="p-2 bg-slate-100 dark:bg-[var(--color-bg-secondary)] text-[var(--color-fg-primary)] rounded-lg">
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
            <a href="mailto:info@lemkin.ai" className="text-[var(--color-fg-primary)] hover:underline focus-ring rounded-sm">
              info@lemkin.ai
            </a>
          </Card>
          
          <Card>
            <Github className="w-8 h-8 text-[var(--color-fg-primary)] mb-3" />
            <h3 className="font-semibold text-lg mb-2">Technical Support</h3>
            <p className="text-gray-600 dark:text-gray-400 mb-4">
              For bug reports and technical issues
            </p>
            <a href="https://github.com/lemkin-ai/issues" className="text-[var(--color-fg-primary)] hover:underline focus-ring rounded-sm">
              GitHub Issues
            </a>
          </Card>
          
          <Card>
            <Users className="w-8 h-8 text-[var(--color-fg-primary)] mb-3" />
            <h3 className="font-semibold text-lg mb-2">Partnerships</h3>
            <p className="text-gray-600 dark:text-gray-400 mb-4">
              For collaboration and partnership inquiries
            </p>
            <a href="mailto:partnerships@lemkin.ai" className="text-[var(--color-fg-primary)] hover:underline focus-ring rounded-sm">
              partnerships@lemkin.ai
            </a>
          </Card>
          
          <Card>
            <AlertCircle className="w-8 h-8 text-[var(--color-fg-primary)] mb-3" />
            <h3 className="font-semibold text-lg mb-2">Security</h3>
            <p className="text-gray-600 dark:text-gray-400 mb-4">
              For reporting security vulnerabilities
            </p>
            <a href="mailto:security@lemkin.ai" className="text-[var(--color-fg-primary)] hover:underline focus-ring rounded-sm">
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

// Individual Module Pages
const LemkinIntegrityPage = () => {
  const { navigate } = useRouter();

  return (
    <div className="min-h-screen py-12 px-4 sm:px-6 lg:px-8">
      <div className="max-w-4xl mx-auto">
        <div className="mb-8">
          <div className="flex items-center gap-4 mb-6">
            <button onClick={() => navigate('/models')} className="btn-outline">
              <ArrowLeft className="w-4 h-4" />
              Back to Models
            </button>
            <Badge variant="production">Production Ready</Badge>
            <Badge variant="tier">Tier 1: Foundation</Badge>
          </div>
          <h1 className="text-4xl font-bold text-gray-900 dark:text-white mb-4">Lemkin Evidence Integrity Toolkit</h1>
          <p className="text-xl text-gray-600 dark:text-gray-400">
            Cryptographic integrity verification and chain of custody management for legal evidence
          </p>
        </div>

        <div className="grid gap-8">
          <Card>
            <div className="p-6">
              <h2 className="text-2xl font-semibold mb-4">Purpose & Capabilities</h2>
              <p className="text-gray-600 dark:text-gray-400 mb-6">
                Ensures evidence admissibility through cryptographic integrity verification and comprehensive chain of custody management.
                Designed to meet international evidence standards with complete audit trails for legal proceedings.
              </p>

              <div className="grid md:grid-cols-2 gap-6">
                <div>
                  <h3 className="font-semibold mb-3 flex items-center gap-2">
                    <Shield className="w-5 h-5 text-blue-600" />
                    Core Features
                  </h3>
                  <ul className="space-y-2 text-gray-600 dark:text-gray-400">
                    <li>• Cryptographic hashing (SHA-256/SHA-512)</li>
                    <li>• Digital signatures for authenticity</li>
                    <li>• Complete chain of custody tracking</li>
                    <li>• Court-ready evidence manifests</li>
                    <li>• SQLite database for reliable storage</li>
                  </ul>
                </div>
                <div>
                  <h3 className="font-semibold mb-3 flex items-center gap-2">
                    <Scale className="w-5 h-5 text-green-600" />
                    Legal Compliance
                  </h3>
                  <ul className="space-y-2 text-gray-600 dark:text-gray-400">
                    <li>• ICC evidence standards</li>
                    <li>• ECHR court requirements</li>
                    <li>• Berkeley Protocol compliance</li>
                    <li>• International admissibility standards</li>
                    <li>• Complete audit trail preservation</li>
                  </ul>
                </div>
              </div>
            </div>
          </Card>

          <Card>
            <div className="p-6">
              <h2 className="text-2xl font-semibold mb-4">Quick Start Example</h2>
              <CodeBlock
                code={`# Generate hash for evidence file
lemkin-integrity hash-evidence evidence.pdf \\
    --case-id CASE-2024-001 \\
    --collector "Investigator Name" \\
    --source "Interview Recording"

# Add custody entry
lemkin-integrity add-custody <evidence-id> accessed "Legal Assistant" \\
    --location "Legal Office"

# Verify integrity
lemkin-integrity verify <evidence-id> --file-path evidence.pdf

# Generate court manifest
lemkin-integrity generate-manifest CASE-2024-001 \\
    --output-file manifest.json`}
                language="bash"
              />
            </div>
          </Card>

          <div className="grid md:grid-cols-2 gap-6">
            <Card>
              <div className="p-6">
                <h3 className="text-xl font-semibold mb-3">Performance Metrics</h3>
                <ul className="space-y-2 text-gray-600 dark:text-gray-400">
                  <li>• Hash generation: ~50MB/sec for SHA-256</li>
                  <li>• Database operations: &lt;100ms queries</li>
                  <li>• Integrity verification: &lt;500ms for most files</li>
                  <li>• Scales to 10,000+ evidence items</li>
                </ul>
              </div>
            </Card>

            <Card>
              <div className="p-6">
                <h3 className="text-xl font-semibold mb-3">Safety Guidelines</h3>
                <ul className="space-y-2 text-gray-600 dark:text-gray-400">
                  <li>• Never modify original evidence files</li>
                  <li>• Maintain secure, access-controlled storage</li>
                  <li>• Protect cryptographic keys</li>
                  <li>• Regular integrity verification</li>
                  <li>• Secure database backups</li>
                </ul>
              </div>
            </Card>
          </div>

          <Card>
            <div className="p-6">
              <h2 className="text-2xl font-semibold mb-4">Installation & Setup</h2>
              <div className="space-y-4">
                <div>
                  <h3 className="font-semibold mb-2">Install from PyPI</h3>
                  <CodeBlock
                    code="pip install lemkin-integrity"
                    language="bash"
                  />
                </div>
                <div>
                  <h3 className="font-semibold mb-2">Development Setup</h3>
                  <CodeBlock
                    code={`git clone https://github.com/lemkin-org/lemkin-integrity.git
cd lemkin-integrity
pip install -e ".[dev]"`}
                    language="bash"
                  />
                </div>
              </div>
            </div>
          </Card>
        </div>
      </div>
    </div>
  );
};

const LemkinAudioPage = () => {
  const { navigate } = useRouter();

  return (
    <div className="min-h-screen py-12 px-4 sm:px-6 lg:px-8">
      <div className="max-w-4xl mx-auto">
        <div className="mb-8">
          <div className="flex items-center gap-4 mb-6">
            <button onClick={() => navigate('/models')} className="btn-outline">
              <ArrowLeft className="w-4 h-4" />
              Back to Models
            </button>
            <Badge variant="production">Production Ready</Badge>
            <Badge variant="tier">Tier 4: Media Analysis</Badge>
          </div>
          <h1 className="text-4xl font-bold text-gray-900 dark:text-white mb-4">Lemkin Audio Analysis Toolkit</h1>
          <p className="text-xl text-gray-600 dark:text-gray-400">
            Comprehensive audio analysis capabilities for legal investigations including transcription, speaker identification, and authenticity verification
          </p>
        </div>

        <div className="grid gap-8">
          <Card>
            <div className="p-6">
              <h2 className="text-2xl font-semibold mb-4">Core Capabilities</h2>

              <div className="grid md:grid-cols-2 gap-6">
                <div>
                  <h3 className="font-semibold mb-3 flex items-center gap-2">
                    <FileText className="w-5 h-5 text-blue-600" />
                    Speech Analysis
                  </h3>
                  <ul className="space-y-2 text-gray-600 dark:text-gray-400">
                    <li>• Multi-language transcription (100+ languages)</li>
                    <li>• Timestamp accuracy with confidence scoring</li>
                    <li>• Speaker identification and profiling</li>
                    <li>• Voice biometric analysis</li>
                    <li>• Automatic speech segmentation</li>
                  </ul>
                </div>
                <div>
                  <h3 className="font-semibold mb-3 flex items-center gap-2">
                    <Shield className="w-5 h-5 text-green-600" />
                    Authentication
                  </h3>
                  <ul className="space-y-2 text-gray-600 dark:text-gray-400">
                    <li>• Audio tampering detection</li>
                    <li>• Compression analysis</li>
                    <li>• Noise floor consistency checks</li>
                    <li>• Deepfake voice detection</li>
                    <li>• Metadata verification</li>
                  </ul>
                </div>
              </div>
            </div>
          </Card>

          <Card>
            <div className="p-6">
              <h2 className="text-2xl font-semibold mb-4">Usage Examples</h2>
              <div className="space-y-6">
                <div>
                  <h3 className="font-semibold mb-2">Speech Transcription</h3>
                  <CodeBlock
                    code={`# Basic transcription with automatic language detection
lemkin-audio transcribe interview.wav --output transcription.json

# Multi-language transcription with segments
lemkin-audio transcribe multilingual_call.wav \\
    --language es-ES \\
    --segments \\
    --segment-length 15.0 \\
    --output detailed_transcription.json`}
                    language="bash"
                  />
                </div>
                <div>
                  <h3 className="font-semibold mb-2">Speaker Identification</h3>
                  <CodeBlock
                    code={`# Create speaker profile
lemkin-audio identify-speaker john_sample1.wav \\
    --create-profile "john_doe" \\
    --profiles ./profiles/

# Identify speakers in unknown audio
lemkin-audio identify-speaker unknown_call.wav \\
    --profiles ./profiles/ \\
    --threshold 0.85 \\
    --output speaker_results.json`}
                    language="bash"
                  />
                </div>
              </div>
            </div>
          </Card>

          <div className="grid md:grid-cols-2 gap-6">
            <Card>
              <div className="p-6">
                <h3 className="text-xl font-semibold mb-3">Supported Formats</h3>
                <ul className="space-y-2 text-gray-600 dark:text-gray-400">
                  <li>• WAV, MP3, FLAC, M4A</li>
                  <li>• OGG, OPUS formats</li>
                  <li>• High-quality lossless preferred</li>
                  <li>• Multiple sample rates supported</li>
                </ul>
              </div>
            </Card>

            <Card>
              <div className="p-6">
                <h3 className="text-xl font-semibold mb-3">Performance Metrics</h3>
                <ul className="space-y-2 text-gray-600 dark:text-gray-400">
                  <li>• Transcription: ~85% accuracy on clear audio</li>
                  <li>• Speaker ID: ~80% with good training samples</li>
                  <li>• Authenticity: ~75% tampering detection</li>
                  <li>• Real-time processing capability</li>
                </ul>
              </div>
            </Card>
          </div>

          <Card>
            <div className="p-6">
              <h2 className="text-2xl font-semibold mb-4">Complete Investigation Workflow</h2>
              <CodeBlock
                code={`# 1. Enhance audio quality
lemkin-audio enhance evidence.wav --output evidence_enhanced.wav --noise-reduction

# 2. Verify authenticity
lemkin-audio verify-authenticity evidence_enhanced.wav --output authenticity.json

# 3. Transcribe speech
lemkin-audio transcribe evidence_enhanced.wav --language en-US --segments --output transcription.json

# 4. Identify speakers
lemkin-audio identify-speaker evidence_enhanced.wav --profiles ./known_speakers/ --output speakers.json

# 5. Comprehensive analysis
lemkin-audio comprehensive-analysis evidence_enhanced.wav \\
    --transcription --speaker-analysis --authenticity \\
    --output final_analysis.json`}
                language="bash"
              />
            </div>
          </Card>
        </div>
      </div>
    </div>
  );
};

const LemkinVideoPage = () => {
  const { navigate } = useRouter();

  return (
    <div className="min-h-screen py-12 px-4 sm:px-6 lg:px-8">
      <div className="max-w-4xl mx-auto">
        <div className="mb-8">
          <div className="flex items-center gap-4 mb-6">
            <button onClick={() => navigate('/models')} className="btn-outline">
              <ArrowLeft className="w-4 h-4" />
              Back to Models
            </button>
            <Badge variant="production">Production Ready</Badge>
            <Badge variant="tier">Tier 4: Media Analysis</Badge>
          </div>
          <h1 className="text-4xl font-bold text-gray-900 dark:text-white mb-4">Lemkin Video Authentication Toolkit</h1>
          <p className="text-xl text-gray-600 dark:text-gray-400">
            Comprehensive video authenticity verification and manipulation detection including deepfake detection and compression analysis
          </p>
        </div>

        <div className="grid gap-8">
          <Card>
            <div className="p-6">
              <h2 className="text-2xl font-semibold mb-4">Detection Capabilities</h2>

              <div className="grid md:grid-cols-2 gap-6">
                <div>
                  <h3 className="font-semibold mb-3 flex items-center gap-2">
                    <Eye className="w-5 h-5 text-red-600" />
                    Deepfake Detection
                  </h3>
                  <ul className="space-y-2 text-gray-600 dark:text-gray-400">
                    <li>• Facial inconsistency analysis</li>
                    <li>• Temporal coherence verification</li>
                    <li>• Compression artifact analysis</li>
                    <li>• Lighting consistency checks</li>
                    <li>• Blur detection algorithms</li>
                  </ul>
                </div>
                <div>
                  <h3 className="font-semibold mb-3 flex items-center gap-2">
                    <Shield className="w-5 h-5 text-blue-600" />
                    Authenticity Verification
                  </h3>
                  <ul className="space-y-2 text-gray-600 dark:text-gray-400">
                    <li>• Video fingerprinting</li>
                    <li>• Compression history analysis</li>
                    <li>• Frame-level manipulation detection</li>
                    <li>• Temporal consistency analysis</li>
                    <li>• Metadata extraction and verification</li>
                  </ul>
                </div>
              </div>
            </div>
          </Card>

          <Card>
            <div className="p-6">
              <h2 className="text-2xl font-semibold mb-4">Quick Start Examples</h2>
              <div className="space-y-6">
                <div>
                  <h3 className="font-semibold mb-2">Deepfake Detection</h3>
                  <CodeBlock
                    code={`# Comprehensive deepfake analysis
lemkin-video detect-deepfake suspicious_video.mp4 \\
    --output deepfake_analysis.json \\
    --max-frames 500

# Quick deepfake check
lemkin-video detect-deepfake suspect_video.mp4`}
                    language="bash"
                  />
                </div>
                <div>
                  <h3 className="font-semibold mb-2">Video Fingerprinting</h3>
                  <CodeBlock
                    code={`# Generate video fingerprint
lemkin-video fingerprint-video evidence_video.mp4 \\
    --output fingerprint.json \\
    --key-frames 20

# Compare videos for similarity
lemkin-video compare-videos original.mp4 suspect_copy.mp4 \\
    --threshold 0.9`}
                    language="bash"
                  />
                </div>
              </div>
            </div>
          </Card>

          <div className="grid md:grid-cols-2 gap-6">
            <Card>
              <div className="p-6">
                <h3 className="text-xl font-semibold mb-3">Performance Metrics</h3>
                <ul className="space-y-2 text-gray-600 dark:text-gray-400">
                  <li>• Deepfake detection: ~85% accuracy</li>
                  <li>• Video fingerprinting: ~95% duplicate detection</li>
                  <li>• Frame analysis: ~1000 frames/minute</li>
                  <li>• Real-time compression analysis</li>
                </ul>
              </div>
            </Card>

            <Card>
              <div className="p-6">
                <h3 className="text-xl font-semibold mb-3">Supported Formats</h3>
                <ul className="space-y-2 text-gray-600 dark:text-gray-400">
                  <li>• MP4, AVI, MOV, MKV</li>
                  <li>• WebM and common codecs</li>
                  <li>• H.264, H.265, VP9 analysis</li>
                  <li>• Various resolution support</li>
                </ul>
              </div>
            </Card>
          </div>

          <Card>
            <div className="p-6">
              <h2 className="text-2xl font-semibold mb-4">Complete Authentication Workflow</h2>
              <CodeBlock
                code={`# 1. Extract metadata
lemkin-video get-metadata evidence.mp4 --output metadata.json

# 2. Check for deepfakes
lemkin-video detect-deepfake evidence.mp4 --output deepfake_check.json

# 3. Analyze compression
lemkin-video analyze-compression evidence.mp4 --output compression.json

# 4. Generate fingerprint
lemkin-video fingerprint-video evidence.mp4 --output fingerprint.json

# 5. Extract key frames for analysis
lemkin-video extract-frames evidence.mp4 --output frame_analysis.json`}
                language="bash"
              />
            </div>
          </Card>
        </div>
      </div>
    </div>
  );
};

// Additional module pages would follow the same pattern...
// For brevity, I'll add a generic module page component

const GenericModulePage = ({ module }: { module: any }) => {
  const { navigate } = useRouter();

  const getStatusBadge = (status: string) => {
    switch (status) {
      case 'production':
        return <Badge variant="production">Production Ready</Badge>;
      case 'implementation':
        return <Badge variant="implementation">Implementation Ready</Badge>;
      default:
        return <Badge variant="beta">In Development</Badge>;
    }
  };

  const getTierBadge = (tier: number) => {
    const tierNames = [
      '', 'Tier 1: Foundation', 'Tier 2: Core Analysis', 'Tier 3: Evidence Collection',
      'Tier 4: Media Analysis', 'Tier 5: Document Processing', 'Tier 6: Visualization'
    ];
    return <Badge variant="tier">{tierNames[tier] || `Tier ${tier}`}</Badge>;
  };

  return (
    <div className="min-h-screen py-12 px-4 sm:px-6 lg:px-8">
      <div className="max-w-4xl mx-auto">
        <div className="mb-8">
          <div className="flex items-center gap-4 mb-6">
            <button onClick={() => navigate('/models')} className="btn-outline">
              <ArrowLeft className="w-4 h-4" />
              Back to Models
            </button>
            {getStatusBadge(module.status)}
            {getTierBadge(module.tier)}
          </div>
          <h1 className="text-4xl font-bold text-gray-900 dark:text-white mb-4">{module.name}</h1>
          <p className="text-xl text-gray-600 dark:text-gray-400">{module.description}</p>
        </div>

        <div className="grid gap-8">
          <Card>
            <div className="p-6">
              <h2 className="text-2xl font-semibold mb-4">Module Overview</h2>
              <p className="text-gray-600 dark:text-gray-400 mb-6">
                This module is part of the Lemkin AI Legal Investigation Platform, designed to democratize
                legal investigation technology for human rights investigators, prosecutors, and civil rights attorneys.
              </p>

              <div className="grid md:grid-cols-2 gap-6">
                <div>
                  <h3 className="font-semibold mb-3">Module Components</h3>
                  <ul className="space-y-2 text-gray-600 dark:text-gray-400">
                    {module.modules?.map((component: string, index: number) => (
                      <li key={index}>• {component.replace('-', ' ').replace(/\b\w/g, (l: string) => l.toUpperCase())}</li>
                    ))}
                  </ul>
                </div>
                <div>
                  <h3 className="font-semibold mb-3">Technical Details</h3>
                  <ul className="space-y-2 text-gray-600 dark:text-gray-400">
                    <li>• Version: {module.version}</li>
                    <li>• Category: {module.category}</li>
                    <li>• Status: {module.status}</li>
                    <li>• Tier: {module.tier}</li>
                  </ul>
                </div>
              </div>
            </div>
          </Card>

          {module.status === 'production' && (
            <Card>
              <div className="p-6">
                <h2 className="text-2xl font-semibold mb-4">Installation</h2>
                <div className="space-y-4">
                  <div>
                    <h3 className="font-semibold mb-2">Install from PyPI</h3>
                    <CodeBlock
                      code={`pip install ${module.id}`}
                      language="bash"
                    />
                  </div>
                  <div>
                    <h3 className="font-semibold mb-2">Basic Usage</h3>
                    <CodeBlock
                      code={`# Basic command structure
${module.id} --help

# Example usage
${module.id} process-data input.file --output results.json`}
                      language="bash"
                    />
                  </div>
                </div>
              </div>
            </Card>
          )}

          {module.status === 'implementation' && (
            <Card>
              <div className="p-6">
                <h2 className="text-2xl font-semibold mb-4">Development Status</h2>
                <p className="text-gray-600 dark:text-gray-400 mb-4">
                  This module is currently in the implementation phase. The module structure and specifications
                  are complete and ready for development contributions.
                </p>
                <div className="space-y-4">
                  <h3 className="font-semibold">Contributing</h3>
                  <CodeBlock
                    code={`# Clone the repository
git clone https://github.com/lemkin-org/${module.id}.git
cd ${module.id}

# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest

# See contributing guidelines
cat CONTRIBUTING.md`}
                    language="bash"
                  />
                </div>
              </div>
            </Card>
          )}

          <Card>
            <div className="p-6">
              <h2 className="text-2xl font-semibold mb-4">Safety & Legal Considerations</h2>
              <div className="bg-amber-50 dark:bg-amber-900/20 border border-amber-200 dark:border-amber-800 rounded-lg p-4">
                <div className="flex items-start gap-3">
                  <AlertCircle className="w-5 h-5 text-amber-600 mt-0.5" />
                  <div>
                    <h3 className="font-semibold text-amber-800 dark:text-amber-200 mb-2">Important Notice</h3>
                    <p className="text-amber-700 dark:text-amber-300 text-sm">
                      This toolkit is designed for legitimate legal investigations and human rights work.
                      Users must ensure proper legal authorization, maintain evidence integrity, and
                      follow all applicable laws and regulations.
                    </p>
                  </div>
                </div>
              </div>
            </div>
          </Card>
        </div>
      </div>
    </div>
  );
};

// Main App Component
// Route announcer for accessibility
const RouteAnnouncer = () => {
  const { currentPath } = useRouter();
  const [message, setMessage] = useState('');

  useEffect(() => {
    setMessage(`Navigated to ${currentPath}`);
  }, [currentPath]);

  return (
    <div aria-live="polite" aria-atomic="true" className="sr-only">
      {message}
    </div>
  );
};

const App = () => {
  const { currentPath } = useRouter();

  // Route rendering logic
  const renderPage = () => {
    // Handle dynamic routes for models
    if (currentPath.startsWith('/models/')) {
      return <ModelDetailPage />;
    }

    // Handle individual module routes
    if (currentPath.startsWith('/module/')) {
      const moduleId = currentPath.split('/')[2];

      // Special handling for detailed module pages
      switch (moduleId) {
        case 'lemkin-integrity':
          return <LemkinIntegrityPage />;
        case 'lemkin-audio':
          return <LemkinAudioPage />;
        case 'lemkin-video':
          return <LemkinVideoPage />;
        default:
          // For other modules, use the generic page with module data
          const allModules = [
            // Tier 1 modules
            { name: "Lemkin Integrity", id: "lemkin-integrity", description: "Evidence integrity and chain of custody management", status: "production", category: "foundation", version: "1.2.0", modules: ["validation", "hashing", "timestamps", "audit"], tier: 1 },
            { name: "Lemkin Redaction", id: "lemkin-redaction", description: "Privacy protection and PII detection", status: "production", category: "safety", version: "1.1.0", modules: ["pii-detection", "auto-redact", "policy-engine"], tier: 1 },
            { name: "Lemkin Classifier", id: "lemkin-classifier", description: "Document classification and taxonomy", status: "production", category: "analysis", version: "2.0.1", modules: ["ml-classifier", "taxonomy", "confidence-scoring"], tier: 1 },

            // Tier 2 modules
            { name: "Lemkin NER", id: "lemkin-ner", description: "Named entity recognition and linking", status: "production", category: "analysis", version: "1.5.0", modules: ["entity-extraction", "linking", "disambiguation", "context"], tier: 2 },
            { name: "Lemkin Timeline", id: "lemkin-timeline", description: "Timeline construction and temporal analysis", status: "production", category: "analysis", version: "1.3.2", modules: ["event-extraction", "temporal-ordering", "visualization"], tier: 2 },
            { name: "Lemkin Frameworks", id: "lemkin-frameworks", description: "Legal framework mapping and compliance", status: "production", category: "legal", version: "2.1.0", modules: ["rome-statute", "icc-elements", "compliance-check"], tier: 2 },

            // Tier 3 modules
            { name: "Lemkin OSINT", id: "lemkin-osint", description: "Open-source intelligence gathering", status: "production", category: "collection", version: "1.4.0", modules: ["web-scraping", "social-media", "data-aggregation", "verification"], tier: 3 },
            { name: "Lemkin Geo", id: "lemkin-geo", description: "Geospatial analysis and mapping", status: "production", category: "analysis", version: "1.2.1", modules: ["coordinate-extraction", "mapping", "distance-calc", "overlays"], tier: 3 },
            { name: "Lemkin Forensics", id: "lemkin-forensics", description: "Digital forensics and authenticity", status: "production", category: "forensics", version: "2.0.0", modules: ["hash-verification", "metadata-analysis", "chain-custody"], tier: 3 },

            // Tier 4 modules
            { name: "Lemkin Video", id: "lemkin-video", description: "Video authentication and deepfake detection", status: "production", category: "media", version: "1.8.0", modules: ["deepfake-detect", "frame-analysis", "compression-artifacts", "temporal-consistency"], tier: 4 },
            { name: "Lemkin Images", id: "lemkin-images", description: "Image verification and manipulation detection", status: "production", category: "media", version: "1.6.3", modules: ["ela-analysis", "metadata-check", "pixel-forensics", "source-tracing"], tier: 4 },
            { name: "Lemkin Audio", id: "lemkin-audio", description: "Audio analysis and transcription", status: "production", category: "media", version: "1.4.1", modules: ["transcription", "voice-id", "noise-analysis", "language-detect"], tier: 4 },

            // Tier 5 modules
            { name: "Lemkin OCR", id: "lemkin-ocr", description: "Document processing and OCR", status: "implementation", category: "documents", version: "0.9.5", modules: ["text-extraction", "layout-analysis", "table-detection", "handwriting"], tier: 5 },
            { name: "Lemkin Research", id: "lemkin-research", description: "Legal research and citation analysis", status: "implementation", category: "research", version: "0.8.2", modules: ["case-law-search", "citation-graph", "relevance-scoring"], tier: 5 },
            { name: "Lemkin Comms", id: "lemkin-comms", description: "Communication analysis and pattern detection", status: "implementation", category: "analysis", version: "0.7.0", modules: ["network-analysis", "pattern-detect", "sentiment", "timeline"], tier: 5 },

            // Tier 6 modules
            { name: "Lemkin Dashboard", id: "lemkin-dashboard", description: "Investigation dashboards and visualization", status: "implementation", category: "visualization", version: "0.9.0", modules: ["data-viz", "interactive-charts", "filtering", "real-time-updates"], tier: 6 },
            { name: "Lemkin Reports", id: "lemkin-reports", description: "Automated report generation", status: "implementation", category: "reporting", version: "0.8.5", modules: ["template-engine", "data-aggregation", "formatting", "export"], tier: 6 },
            { name: "Lemkin Export", id: "lemkin-export", description: "Multi-format export and compliance", status: "implementation", category: "export", version: "0.7.8", modules: ["pdf-export", "csv-export", "json-export", "compliance-formats"], tier: 6 }
          ];

          const module = allModules.find(m => m.id === moduleId);
          if (module) {
            return <GenericModulePage module={module} />;
          }
          return <NotFoundPage />;
      }
    }

    // Handle dynamic routes for articles
    if (currentPath.startsWith('/articles/') && currentPath !== '/articles') {
      return <ArticleDetailPage />;
    }

    // Static routes
    switch (currentPath) {
      case '/':
        return <HomePage />;
      case '/overview':
        return <OverviewPage />;
      case '/models':
        return <ModelsPageRevised />;
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
    <div className="min-h-screen">
      <Navigation />
      <main id="main" tabIndex={-1} className="flex-1 focus-ring outline-none">
        <RouteAnnouncer />
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
      <Router>
        <App />
      </Router>
    </ThemeProvider>
  );
};


// Export the main component
export default LemkinAIWebsite;