export interface Article {
  id: string;
  title: string;
  category: 'technical' | 'operational' | 'legal' | 'analytical';
  authors: string[];
  date: string;
  readTime: string;
  tags: string[];
  filePath: string;
  excerpt: string;
}

export const articles: Article[] = [
  {
    id: 'real-time-analysis',
    title: 'Real-Time Analysis in Active Conflicts',
    category: 'operational',
    authors: ['Oliver Stern', 'Lemkin AI Research Team'],
    date: '2025-03-12',
    readTime: '18 min',
    tags: ['Ukraine', 'Real-Time Processing', 'Evidence Collection', 'Verification'],
    filePath: '/articles/Real-Time-Analysis-in-Active-Conflicts_-Operational-Challenges-for-Human-Rights-Investigations-During-the-Ukraine-War.md',
    excerpt: 'The conflict in Ukraine has generated digital evidence at unprecedented scale. This article examines operational challenges for human rights investigations during active warfare.'
  },
  {
    id: 'social-media-analysis',
    title: 'Analyzing Social Media for Atrocity Documentation',
    category: 'analytical',
    authors: ['Oliver Stern', 'Lemkin AI Research Team'],
    date: '2024-10-03',
    readTime: '15 min',
    tags: ['Social Media', 'Verification', 'Documentation', 'Digital Evidence'],
    filePath: '/articles/Analyzing-Social-Media-for-Atrocity-Documentation_-Verification-at-Scale.md',
    excerpt: 'Verification at scale presents unique challenges when analyzing social media for atrocity documentation. Learn about methodologies and tools for processing massive volumes of user-generated content.'
  },
  {
    id: 'llm-legal-analysis',
    title: 'Large Language Models for Legal Document Analysis',
    category: 'technical',
    authors: ['Oliver Stern', 'Lemkin AI Research Team'],
    date: '2025-07-15',
    readTime: '20 min',
    tags: ['LLMs', 'Legal Tech', 'AI Ethics', 'Document Analysis'],
    filePath: '/articles/Large-Language-Models-for-Legal-Document-Analysis_-Balancing-Transformative-Potential-with-Critical-Risks-in-Human-Rights-Investigations.md',
    excerpt: 'Exploring the transformative potential and critical risks of using Large Language Models in human rights investigations and legal document analysis.'
  },
  {
    id: 'syria-document-analysis',
    title: 'Large-Scale Document Analysis for War Crimes',
    category: 'operational',
    authors: ['Oliver Stern', 'Lemkin AI Research Team'],
    date: '2025-01-09',
    readTime: '22 min',
    tags: ['Syria', 'Document Analysis', 'War Crimes', 'Digital Evidence'],
    filePath: '/articles/Large-Scale-Document-Analysis-for-War-Crimes_-Lessons-from-Syria_s-Digital-Evidence-Revolution.md',
    excerpt: "Lessons from Syria's digital evidence revolution and its implications for future war crimes investigations."
  },
  {
    id: 'quality-assurance',
    title: 'Quality Assurance Protocols for AI-Assisted Evidence',
    category: 'technical',
    authors: ['Oliver Stern', 'Lemkin AI Research Team'],
    date: '2024-11-23',
    readTime: '16 min',
    tags: ['Quality Assurance', 'AI Validation', 'Evidence Processing', 'Standards'],
    filePath: '/articles/Quality-Assurance-Protocols-for-AI-Assisted-Evidence-Processing.md',
    excerpt: 'Establishing rigorous quality assurance protocols for AI-assisted evidence processing in human rights investigations.'
  },
  {
    id: 'cross-validation',
    title: 'Cross-Validation with Traditional Investigation Methods',
    category: 'operational',
    authors: ['Oliver Stern', 'Lemkin AI Research Team'],
    date: '2025-05-20',
    readTime: '14 min',
    tags: ['Validation', 'Traditional Methods', 'Hybrid Approaches', 'Investigation'],
    filePath: '/articles/Cross-Validation-with-Traditional-Investigation-Methods.md',
    excerpt: 'Bridging AI-powered analysis with traditional investigation methods through systematic cross-validation approaches.'
  },
  {
    id: 'automated-redaction',
    title: 'Automated Document Redaction',
    category: 'legal',
    authors: ['Oliver Stern', 'Lemkin AI Research Team'],
    date: '2025-08-03',
    readTime: '17 min',
    tags: ['Privacy', 'Redaction', 'Legal Compliance', 'Automation'],
    filePath: '/articles/Automated-Document-Redaction_-Preserving-Evidential-Value-While-Protecting-Privacy-in-Legal-Proceedings.md',
    excerpt: 'Preserving evidential value while protecting privacy through automated document redaction systems.'
  },
  {
    id: 'witness-protection',
    title: 'Privacy-Preserving AI for Witness Protection',
    category: 'legal',
    authors: ['Oliver Stern', 'Lemkin AI Research Team'],
    date: '2025-02-14',
    readTime: '19 min',
    tags: ['Privacy', 'Witness Protection', 'Security', 'AI Ethics'],
    filePath: '/articles/Privacy-Preserving-AI-for-Witness-Protection.md',
    excerpt: 'Developing privacy-preserving AI systems that protect witness identities while maintaining evidential integrity.'
  },
  {
    id: 'multilingual-models',
    title: 'Building Multilingual Models for International Justice',
    category: 'technical',
    authors: ['Oliver Stern', 'Lemkin AI Research Team'],
    date: '2024-12-07',
    readTime: '21 min',
    tags: ['Multilingual', 'NLP', 'International Justice', 'Language Processing'],
    filePath: '/articles/Building-Multilingual-Models-for-International-Justice.md',
    excerpt: 'Addressing the linguistic diversity of international justice through specialized multilingual AI models.'
  },
  {
    id: 'human-in-loop',
    title: 'The Human-in-the-Loop Imperative',
    category: 'operational',
    authors: ['Oliver Stern', 'Lemkin AI Research Team'],
    date: '2025-06-11',
    readTime: '16 min',
    tags: ['Human Oversight', 'AI Ethics', 'Investigation Methods', 'Best Practices'],
    filePath: '/articles/The-Human-in-the-Loop-Imperative_-Why-AI-Can-Never-Replace-Investigators.md',
    excerpt: 'Why AI can never replace human investigators and the critical importance of human oversight in automated systems.'
  },
  {
    id: 'conflict-zone-deployment',
    title: 'Deploying AI in Conflict Zones',
    category: 'operational',
    authors: ['Oliver Stern', 'Lemkin AI Research Team'],
    date: '2024-10-29',
    readTime: '18 min',
    tags: ['Conflict Zones', 'Deployment', 'Security', 'Infrastructure'],
    filePath: '/articles/Deploying-AI-in-Conflict-Zones_-Technical-and-Operational-Challenges.md',
    excerpt: 'Technical and operational challenges of deploying AI systems in active conflict zones.'
  },
  {
    id: 'uncertainty-quantification',
    title: 'Uncertainty Quantification in AI for Legal Applications',
    category: 'legal',
    authors: ['Oliver Stern', 'Lemkin AI Research Team'],
    date: '2025-09-18',
    readTime: '20 min',
    tags: ['Uncertainty', 'Legal Standards', 'AI Reliability', 'Evidence'],
    filePath: '/articles/Uncertainty-Quantification-in-AI-for-Legal-Applications.md',
    excerpt: 'Methods for quantifying and communicating AI uncertainty in legal contexts.'
  },
  {
    id: 'weapons-identification',
    title: 'Weapons Identification in Video Evidence',
    category: 'technical',
    authors: ['Oliver Stern', 'Lemkin AI Research Team'],
    date: '2025-04-02',
    readTime: '17 min',
    tags: ['Computer Vision', 'Weapons ID', 'Video Analysis', 'Evidence'],
    filePath: '/articles/Weapons-Identification-in-Video-Evidence_-Computer-Vision-Applications.md',
    excerpt: 'Computer vision applications for identifying weapons in video evidence from conflict zones.'
  },
  {
    id: 'financial-crime',
    title: 'Financial Crime Investigation',
    category: 'analytical',
    authors: ['Oliver Stern', 'Lemkin AI Research Team'],
    date: '2024-11-11',
    readTime: '19 min',
    tags: ['Financial Crime', 'Money Trails', 'Investigation', 'Analytics'],
    filePath: '/articles/Financial-Crime-Investigation_-AI-for-Following-the-Money-Trail.md',
    excerpt: 'AI techniques for following money trails in financial crime and corruption investigations.'
  },
  {
    id: 'speech-to-text',
    title: 'Speech-to-Text for Witness Testimony',
    category: 'technical',
    authors: ['Oliver Stern', 'Lemkin AI Research Team'],
    date: '2025-07-27',
    readTime: '16 min',
    tags: ['Speech Recognition', 'Low-Resource Languages', 'Testimony', 'NLP'],
    filePath: '/articles/Speech-to-Text-for-Witness-Testimony_-Challenges-in-Low-Resource-Languages.md',
    excerpt: 'Challenges and solutions for speech-to-text systems in low-resource languages for witness testimony.'
  },
  {
    id: 'satellite-imagery',
    title: 'Satellite Imagery Analysis for Atrocity Documentation',
    category: 'analytical',
    authors: ['Oliver Stern', 'Lemkin AI Research Team'],
    date: '2025-01-26',
    readTime: '21 min',
    tags: ['Satellite Imagery', 'Remote Sensing', 'Evidence', 'Analysis'],
    filePath: '/articles/Satellite-Imagery-Analysis-for-Atrocity-Documentation_-Transforming-Pixels-into-Evidence-for-International-Justice.md',
    excerpt: 'Transforming satellite imagery into admissible evidence for international justice proceedings.'
  },
  {
    id: 'named-entity-recognition',
    title: 'Named Entity Recognition for Conflict Documentation',
    category: 'technical',
    authors: ['Oliver Stern', 'Lemkin AI Research Team'],
    date: '2024-12-19',
    readTime: '18 min',
    tags: ['NER', 'NLP', 'Documentation', 'Linguistic Analysis'],
    filePath: '/articles/Named-Entity-Recognition-for-Conflict-Documentation_-Linguistic-Complexity-and-Technical-Challenges-in-Human-Rights-Investigations.md',
    excerpt: 'Addressing linguistic complexity in named entity recognition systems for conflict documentation.'
  },
  {
    id: 'chain-of-custody',
    title: 'Chain of Custody in the Digital Age',
    category: 'legal',
    authors: ['Oliver Stern', 'Lemkin AI Research Team'],
    date: '2025-08-22',
    readTime: '17 min',
    tags: ['Chain of Custody', 'Cryptography', 'Digital Evidence', 'Legal Standards'],
    filePath: '/articles/Chain-of-Custody-in-the-Digital-Age_-Cryptographic-Evidence-Verification-for-Human-Rights-Investigations.md',
    excerpt: 'Cryptographic methods for maintaining chain of custody in digital evidence collection.'
  },
  {
    id: 'federated-learning',
    title: 'Collaborative Intelligence Without Compromise',
    category: 'technical',
    authors: ['Oliver Stern', 'Lemkin AI Research Team'],
    date: '2025-03-30',
    readTime: '20 min',
    tags: ['Federated Learning', 'Privacy', 'Collaboration', 'Machine Learning'],
    filePath: '/articles/Collaborative-Intelligence-Without-Compromise_-Federated-Learning-for-International-Human-Rights-Investigations.md',
    excerpt: 'Federated learning approaches for international collaboration without compromising data sovereignty.'
  },
  {
    id: 'bias-detection',
    title: 'Bias Detection in AI Models',
    category: 'technical',
    authors: ['Oliver Stern', 'Lemkin AI Research Team'],
    date: '2024-10-07',
    readTime: '19 min',
    tags: ['Bias Detection', 'AI Ethics', 'Fairness', 'Model Evaluation'],
    filePath: '/articles/Bias-Detection-in-AI-Models-for-Human-Rights-Applications.md',
    excerpt: 'Methods for detecting and mitigating bias in AI models used for human rights applications.'
  },
  {
    id: 'multi-modal-evidence',
    title: 'Multi-Modal Evidence Integration',
    category: 'analytical',
    authors: ['Oliver Stern', 'Lemkin AI Research Team'],
    date: '2025-09-05',
    readTime: '22 min',
    tags: ['Multi-Modal', 'Evidence Integration', 'Myanmar', 'Analysis'],
    filePath: '/articles/Multi-Modal-Evidence-Integration_-Constructing-Comprehensive-Narratives-from-Diverse-Digital-Sources-in-Myanmar-Investigations.md',
    excerpt: 'Constructing comprehensive narratives from diverse digital sources in Myanmar investigations.'
  },
  {
    id: 'data-management',
    title: 'Data Management for Human Rights Organizations',
    category: 'operational',
    authors: ['Oliver Stern', 'Lemkin AI Research Team'],
    date: '2025-02-03',
    readTime: '15 min',
    tags: ['Data Management', 'Best Practices', 'Infrastructure', 'Organizations'],
    filePath: '/articles/Data-Management-for-Human-Rights-Organizations_-Best-Practices.md',
    excerpt: 'Best practices for data management infrastructure in human rights organizations.'
  },
  {
    id: '3d-reconstruction',
    title: 'Three-Dimensional Reconstruction for Crime Scene Analysis',
    category: 'technical',
    authors: ['Oliver Stern', 'Lemkin AI Research Team'],
    date: '2025-06-25',
    readTime: '20 min',
    tags: ['3D Reconstruction', 'Crime Scenes', 'Visualization', 'Forensics'],
    filePath: '/articles/Three-Dimensional-Reconstruction-for-Crime-Scene-Analysis_-Spatial-Documentation-and-Forensic-Visualization-in-Human-Rights-Investigations.md',
    excerpt: 'Spatial documentation and forensic visualization techniques for human rights investigations.'
  },
  {
    id: 'mass-grave-identification',
    title: 'Mass Grave Identification from Aerial Imagery',
    category: 'analytical',
    authors: ['Oliver Stern', 'Lemkin AI Research Team'],
    date: '2025-04-18',
    readTime: '18 min',
    tags: ['Remote Sensing', 'Mass Graves', 'Investigation', 'Imagery Analysis'],
    filePath: '/articles/Mass-Grave-Identification-from-Aerial-Imagery_-Technical-Approaches.md',
    excerpt: 'Technical approaches for identifying mass graves using aerial and satellite imagery.'
  }
];