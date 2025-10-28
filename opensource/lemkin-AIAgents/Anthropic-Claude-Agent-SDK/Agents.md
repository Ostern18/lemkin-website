DOMAIN 1: Investigative Research & Intelligence
1. Open-Source Intelligence (OSINT) Synthesis Agent
Purpose: Aggregates and analyzes publicly available information
Capabilities:

Monitors social media, news, and online platforms for relevant content
Extracts claims, metadata, and context from web sources
Cross-references information across multiple platforms
Identifies coordinated campaigns or narrative patterns
Generates intelligence briefs with source verification notes
Creates geographic and temporal heat maps of online activity

Claude Features: Web search integration, multi-source analysis, structured reporting

2. Satellite Imagery Analyst
Purpose: Interprets satellite and aerial imagery for evidence
Capabilities:

Describes visible features (buildings, vehicles, crowds, damage)
Compares before/after images to document changes
Identifies potential sites of interest (mass graves, detention facilities)
Estimates crowd sizes and infrastructure types
Generates annotated image reports with coordinates
Flags areas requiring expert forensic analysis

Claude Features: Native vision processing, spatial reasoning, comparative analysis

3. Social Media Evidence Harvester
Purpose: Collects and contextualizes social media posts as evidence
Capabilities:

Analyzes screenshots of posts for content, metadata, and authenticity markers
Extracts usernames, timestamps, locations, hashtags
Identifies potential bot activity or coordinated behavior
Preserves context (replies, shares, surrounding discussions)
Generates chain-of-custody documentation for digital evidence
Creates social network maps from interaction patterns

Claude Features: Image text extraction (OCR), metadata analysis, network reasoning

4. Historical Context & Background Researcher
Purpose: Provides deep background on conflicts, actors, and regions
Capabilities:

Researches historical grievances, political dynamics, ethnic tensions
Summarizes previous conflicts and peace processes
Profiles key actors (leaders, military units, armed groups)
Identifies relevant cultural, religious, and social factors
Generates context memos for investigators
Suggests analogous historical cases for comparison

Claude Features: Web search, knowledge synthesis, comparative reasoning

5. Legal Framework & Jurisdiction Advisor
Purpose: Clarifies applicable law and jurisdictional questions
Capabilities:

Explains relevant international law (ICL, IHL, IHRL)
Analyzes treaty obligations and customary law
Advises on jurisdictional basis (territorial, nationality, universal)
Identifies potential legal obstacles (immunities, statutes of limitations)
Suggests applicable legal theories and precedents
Generates legal research memos on novel issues

Claude Features: Legal reasoning, web search for case law, structured analysis

DOMAIN 2: Document Processing & Analysis
6. Multi-Format Document Parser
Purpose: Extracts and structures content from any document type
Capabilities:

Processes PDFs, images, scanned documents, handwritten notes
Extracts text while preserving formatting and structure
Identifies document type (contract, order, statement, report)
Pulls out key fields (dates, names, signatures, stamps)
Creates structured data exports (JSON, CSV, tables)
Flags illegible or ambiguous sections

Claude Features: Native PDF/image processing, OCR, structured output

7. Comparative Document Analyzer
Purpose: Identifies similarities, differences, and patterns across documents
Capabilities:

Compares multiple versions of the same document
Highlights changes, redactions, and alterations
Identifies boilerplate vs. unique content
Finds recurring language patterns across document sets
Detects potential forgeries or inconsistencies
Generates comparison matrices and change logs

Claude Features: Extended context for multi-document analysis, diff logic

9. Medical & Forensic Record Analyst
Purpose: Interprets medical reports and forensic documentation
Capabilities:

Extracts diagnoses, injuries, treatments, and outcomes
Identifies injuries consistent with torture or assault
Analyzes cause-of-death determinations
Flags inconsistencies between injuries and stated causes
Generates medical evidence summaries for non-experts
Links medical findings to legal elements (torture, cruel treatment)

Claude Features: Technical document comprehension, reasoning about consistency

26. Evidence Gap & Next Steps Identifier
Purpose: Highlights what's missing and what to do next
Capabilities:

Compares evidence to legal elements to find gaps
Prioritizes gaps by importance to case theory
Suggests specific investigative actions to fill gaps
Identifies alternative evidence sources
Generates investigative to-do lists
Creates follow-up interview question sets

Claude Features: Gap analysis, prioritization logic, action planning

31. Siege & Starvation Warfare Analyst
Purpose: Documents crimes related to blockades and starvation
Capabilities:

Analyzes supply flow data and humanitarian access
Calculates population nutrition and health impacts
Documents denial of food, water, medical care
Maps siege lines and checkpoints
Generates comprehensive siege documentation
Links starvation tactics to command responsibility

Claude Features: Data analysis, causal reasoning, systematic documentation

32. Torture & Ill-Treatment Analyst
Purpose: Documents and analyzes torture evidence
Capabilities:

Reviews medical evidence of torture
Analyzes detention conditions and treatment
Applies Istanbul Protocol standards
Documents methods and patterns of torture
Generates torture reports with legal analysis
Links specific acts to responsible individuals

Claude Features: Medical reasoning, pattern recognition, legal application

33. Genocide Intent Analyzer
Purpose: Evaluates evidence of genocidal intent
Capabilities:

Analyzes statements, policies, and propaganda
Identifies indicators of specific intent to destroy
Examines targeting patterns of protected groups
Reviews context of systematic destruction
Generates intent memoranda citing evidence
Compares to genocide precedents

Claude Features: Intent reasoning, pattern analysis, legal interpretation

34. Enforced Disappearance Investigator
Purpose: Documents patterns of disappearances
Capabilities:

Tracks missing persons reports and cases
Maps disappearance patterns (temporal, geographic, demographic)
Analyzes state denial and lack of information
Documents family searches and official responses
Generates enforced disappearance reports
Links cases to command structures

Claude Features: Pattern detection, data synthesis, narrative construction

36. NGO & UN Reporting Specialist
Purpose: Produces reports for human rights organizations
Capabilities:

Generates thematic reports on crime patterns
Creates submissions to UN mechanisms (HRC, treaty bodies)
Produces shadow reports for state reviews
Writes advocacy briefs for policymakers
Generates fact sheets and executive summaries
Adapts technical findings for non-legal audiences

Claude Features: Report writing, audience adaptation, policy communication

41. Forensic Analysis Reviewer
Purpose: Interprets forensic reports for legal teams
Capabilities:

Summarizes DNA, ballistics, and autopsy reports
Identifies key findings relevant to charges
Explains forensic methods and limitations
Flags inconsistencies or technical issues
Generates non-expert summaries of technical reports
Suggests follow-up forensic questions

Claude Features: Technical comprehension, translation to legal context

42. Digital Forensics & Metadata Analyst
Purpose: Analyzes digital evidence and metadata
Capabilities:

Extracts metadata from digital files (EXIF, creation dates)
Analyzes file modification histories
Identifies signs of tampering or manipulation
Reviews authentication evidence for digital materials
Generates digital chain-of-custody documentation
Produces technical analysis reports for court

Claude Features: Metadata reasoning, authenticity assessment, technical documentation

43. Ballistics & Weapons Identifier
Purpose: Analyzes evidence of weapons and ammunition
Capabilities:

Identifies weapon types from photos and descriptions
Analyzes ammunition markings and origins
Reviews ballistic reports and wound patterns
Generates weapons identification reports
Links weapons to specific actors or incidents
Produces visual guides of weapons for investigators

Claude Features: Visual analysis, technical classification, evidence synthesis

44. Military Structure & Tactics Analyst
Purpose: Provides military expertise on operations
Capabilities:

Analyzes military unit structures and hierarchies
Explains tactical operations and military doctrine
Reviews attack patterns for military assessment
Identifies command and control indicators
Generates military analysis reports
Provides expert consultation materials

Claude Features: Military reasoning, structural analysis, tactical interpretation

Implementation Architecture
Pure Claude Stack
All agents leverage:

Claude Sonnet 4.5 for core reasoning and generation
Extended Context: 200K token window for comprehensive analysis
Vision API: Native image/PDF processing
Artifacts: Document creation (reports, memos, presentations)
Code Interpreter: Data analysis and visualization when needed
Web Search: Real-time information gathering

Shared Infrastructure

Prompt Library: Reusable, tested prompts for each agent
Output Templates: Standardized formats for reports, memos, etc.
Quality Metrics: Human review scoring and feedback loops
Audit System: Logs all agent interactions and decisions

Security & Ethics

Access Control: Role-based permissions per agent
Encryption: All data in transit and at rest
Audit Trails: Immutable logs of all operations
Human Review: Mandatory gates for high-stakes decisions
Bias Monitoring: Regular evaluation of agent outputs