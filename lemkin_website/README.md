# Lemkin AI

<div align="center">
  <img src="public/Lemkin Logo Black_Shape_clear.png" alt="Lemkin AI Logo" width="200" height="200">

  **Open-Source Legal Investigation Technology**

  *Democratizing access to powerful analysis tools for human rights investigators, civil rights attorneys, and organizations working for justice and accountability.*

  [![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
  [![CI](https://github.com/lemkin-org/lemkin_website/workflows/CI/badge.svg)](https://github.com/lemkin-org/lemkin_website/actions)
  [![Security](https://img.shields.io/badge/Security-Reviewed-green.svg)](SECURITY.md)
  [![Code of Conduct](https://img.shields.io/badge/Code%20of%20Conduct-Contributor%20Covenant-brightgreen.svg)](CODE_OF_CONDUCT.md)
</div>

## About Lemkin AI

Lemkin AI is a comprehensive ecosystem of open-source tools designed specifically for legal investigations, human rights documentation, and evidence analysis. Named after [Raphael Lemkin](https://en.wikipedia.org/wiki/Raphael_Lemkin), who coined the term "genocide" and advocated for international justice, our mission is to strengthen the capacity of those working for justice and accountability worldwide.

### Why Lemkin AI?

Legal investigations often require sophisticated technical capabilities that are either:
- **Expensive**: Commercial tools can cost thousands of dollars per license
- **Inaccessible**: Require specialized technical knowledge to operate
- **Closed**: Black-box algorithms that can't be audited or trusted in legal proceedings
- **Limited**: Don't meet the specific needs of human rights and legal investigations

Lemkin AI addresses these challenges by providing:
- **Open Source**: All tools are freely available and auditable
- **Legal-First Design**: Built specifically for legal evidence standards and workflows
- **User-Friendly**: Designed for legal professionals, not just technical experts
- **Comprehensive**: End-to-end analysis capabilities across multiple domains
- **Secure**: Enterprise-grade security with human rights protections

## 🚀 Quick Start

### Website Development

```bash
# Clone the repository
git clone https://github.com/lemkin-org/lemkin_website.git
cd lemkin_website

# Install dependencies
npm install

# Start development server
npm start

# Build for production
npm run build
```

### Using Docker

```bash
# Start the full development environment
docker-compose --profile dev up

# Production deployment
docker-compose up lemkin-website

# With documentation server
docker-compose --profile docs up
```

### Tool Installation

```bash
# Install a specific tool (example: audio analysis)
pip install lemkin-audio

# Install all tools
pip install lemkin-toolkit

# Development installation
cd resources/lemkin-audio
pip install -e ".[dev]"
```

## 🛠️ Available Tools

### Core Analysis Tools

| Tool | Purpose | Status | Installation |
|------|---------|--------|--------------|
| **[lemkin-audio](resources/lemkin-audio/)** | Audio transcription, speaker ID, authenticity verification | ✅ Stable | `pip install lemkin-audio` |
| **[lemkin-images](resources/lemkin-images/)** | Image analysis, manipulation detection, metadata extraction | ✅ Stable | `pip install lemkin-images` |
| **[lemkin-video](resources/lemkin-video/)** | Video analysis, deepfake detection, timeline extraction | ✅ Stable | `pip install lemkin-video` |
| **[lemkin-forensics](resources/lemkin-forensics/)** | Digital forensics, file recovery, system analysis | ✅ Stable | `pip install lemkin-forensics` |
| **[lemkin-ner](resources/lemkin-ner/)** | Named entity recognition, PII detection, entity linking | ✅ Stable | `pip install lemkin-ner` |

### Specialized Analysis Tools

| Tool | Purpose | Status | Installation |
|------|---------|--------|--------------|
| **[lemkin-osint](resources/lemkin-osint/)** | Open source intelligence gathering and analysis | ✅ Stable | `pip install lemkin-osint` |
| **[lemkin-geo](resources/lemkin-geo/)** | Geographic analysis, location verification, mapping | ✅ Stable | `pip install lemkin-geo` |
| **[lemkin-comms](resources/lemkin-comms/)** | Communication pattern analysis, network mapping | 🔧 Beta | `pip install lemkin-comms` |
| **[lemkin-timeline](resources/lemkin-timeline/)** | Timeline analysis and chronological reconstruction | 🔧 Beta | `pip install lemkin-timeline` |
| **[lemkin-classifier](resources/lemkin-classifier/)** | Document classification and content analysis | ✅ Stable | `pip install lemkin-classifier` |

### Utility and Framework Tools

| Tool | Purpose | Status | Installation |
|------|---------|--------|--------------|
| **[lemkin-integrity](resources/lemkin-integrity/)** | Evidence integrity, chain of custody, verification | ✅ Stable | `pip install lemkin-integrity` |
| **[lemkin-redaction](resources/lemkin-redaction/)** | Privacy protection, PII redaction, data anonymization | ✅ Stable | `pip install lemkin-redaction` |
| **[lemkin-ocr](resources/lemkin-ocr/)** | Optical character recognition, document digitization | 🔧 Beta | `pip install lemkin-ocr` |
| **[lemkin-export](resources/lemkin-export/)** | Data export, report generation, format conversion | 🔧 Beta | `pip install lemkin-export` |
| **[lemkin-frameworks](resources/lemkin-frameworks/)** | Shared frameworks and utilities for all tools | ✅ Stable | `pip install lemkin-frameworks` |

### Dashboard and Reporting

| Tool | Purpose | Status | Installation |
|------|---------|--------|--------------|
| **[lemkin-dashboard](resources/lemkin-dashboard/)** | Investigation dashboard, case management | 🔧 Beta | `pip install lemkin-dashboard` |
| **[lemkin-reports](resources/lemkin-reports/)** | Automated report generation, templates | 🔧 Beta | `pip install lemkin-reports` |
| **[lemkin-research](resources/lemkin-research/)** | Research tools, data analysis, statistical analysis | 🔧 Beta | `pip install lemkin-research` |

**Status Legend:**
- ✅ **Stable**: Production-ready with comprehensive testing
- 🔧 **Beta**: Functional but may have some limitations
- 🚧 **Alpha**: Early development, may be unstable

## 📋 Use Cases

### Human Rights Investigations
- **Evidence Analysis**: Verify authenticity of photos, videos, and audio recordings
- **Timeline Reconstruction**: Build chronological timelines from multiple evidence sources
- **Communication Analysis**: Map networks and analyze communication patterns
- **Location Verification**: Verify locations in media using geospatial analysis
- **Witness Protection**: Redact identifying information while preserving evidence value

### Legal Proceedings
- **Discovery Assistance**: Process large volumes of documents and media
- **Expert Testimony**: Generate transparent, auditable analysis for court presentation
- **Chain of Custody**: Maintain cryptographic evidence integrity throughout analysis
- **Compliance**: Meet legal standards for evidence handling and digital forensics
- **Cross-Examination Preparation**: Understand analysis methods and limitations

### Civil Rights Organizations
- **Pattern Recognition**: Identify systemic issues through data analysis
- **Public Interest Litigation**: Support cases with comprehensive evidence analysis
- **Advocacy**: Use data visualization to support policy arguments
- **Training**: Educational resources for staff and volunteers
- **Collaboration**: Share analysis tools and methodologies across organizations

## 🏗️ Architecture

### Website Architecture

```
lemkin_website/
├── src/
│   ├── LemkinAIWebsite.tsx    # Main React application (14 pages)
│   ├── index.css              # Tailwind CSS with custom design system
│   └── index.tsx              # Application entry point
├── public/                    # Static assets and logos
├── .github/workflows/         # CI/CD automation
├── Dockerfile                 # Production containerization
└── docker-compose.yml        # Development environment
```

**Key Features:**
- **Single Page Application**: React 18 with TypeScript and custom routing
- **Design System**: Comprehensive Tailwind CSS customization with neural themes
- **Dark Mode**: Full system with automatic detection and manual toggle
- **Responsive**: Mobile-first design with comprehensive breakpoint coverage
- **Accessible**: WCAG 2.1 AA compliance with screen reader support
- **Performance**: Optimized build with code splitting and asset optimization

### Tool Architecture

Each tool follows a consistent architecture pattern:

```
lemkin-{tool}/
├── src/lemkin_{tool}/         # Core implementation
│   ├── __init__.py           # Public API
│   ├── core/                 # Core analysis algorithms
│   ├── models/               # Data models and schemas
│   ├── utils/                # Utility functions
│   └── cli/                  # Command-line interface
├── tests/                    # Comprehensive test suite
├── docs/                     # Documentation and examples
├── Makefile                  # Development and build automation
└── pyproject.toml           # Python packaging configuration
```

### Security Architecture

- **Zero Trust**: All inputs validated, all outputs verified
- **Encryption**: Data encrypted at rest and in transit
- **Audit Logging**: Comprehensive audit trails for all operations
- **Access Control**: Role-based access with principle of least privilege
- **Isolation**: Tools can run in isolated environments for sensitive data
- **Chain of Custody**: Cryptographic integrity verification throughout processing

## 🧪 Development

### Prerequisites

**Website Development:**
- Node.js 18+
- npm or yarn
- Modern web browser

**Tool Development:**
- Python 3.10+
- Git
- Docker (optional but recommended)

### Getting Started

1. **Fork and Clone**
   ```bash
   git clone https://github.com/YOUR_USERNAME/lemkin_website.git
   cd lemkin_website
   ```

2. **Install Dependencies**
   ```bash
   # Website
   npm install

   # Tools (example: audio)
   cd resources/lemkin-audio
   pip install -e ".[dev]"
   ```

3. **Run Tests**
   ```bash
   # Website
   npm test

   # Tools
   make test
   ```

4. **Start Development**
   ```bash
   # Website
   npm start

   # Full environment
   docker-compose --profile dev up
   ```

### Code Quality

We maintain high code quality standards:

```bash
# Website
npm run lint          # ESLint checking
npm run format        # Prettier formatting
npm run type-check    # TypeScript validation

# Tools (example pattern)
make lint            # Python linting with ruff
make format          # Code formatting with black
make type-check      # Type checking with mypy
make test            # Comprehensive testing
make security-scan   # Security vulnerability scanning
```

### Contributing

We welcome contributions from developers, legal professionals, and domain experts! Please see:

- **[Contributing Guidelines](CONTRIBUTING.md)**: Detailed contribution process
- **[Code of Conduct](CODE_OF_CONDUCT.md)**: Community standards and expectations
- **[Security Policy](SECURITY.md)**: Security vulnerability reporting and best practices

## 📖 Documentation

### User Documentation
- **[Website](https://lemkin.ai)**: Main website with tool overview and examples
- **[User Guides](docs/user-guides/)**: Step-by-step guides for each tool
- **[API Documentation](docs/api/)**: Comprehensive API reference
- **[Tutorials](docs/tutorials/)**: Real-world usage scenarios and walkthroughs

### Technical Documentation
- **[Architecture Guide](docs/architecture/)**: System design and technical details
- **[Deployment Guide](docs/deployment/)**: Production deployment and scaling
- **[Security Documentation](docs/security/)**: Security procedures and threat models
- **[Developer Guide](docs/development/)**: Development environment and contribution guide

### Legal Documentation
- **[Evidence Standards](docs/legal/evidence-standards.md)**: Legal evidence handling requirements
- **[Compliance Guide](docs/legal/compliance.md)**: Regulatory compliance information
- **[Expert Testimony](docs/legal/expert-testimony.md)**: Guidance for expert witness testimony
- **[Chain of Custody](docs/legal/chain-of-custody.md)**: Evidence integrity procedures

## 🔒 Security

Security is paramount in legal investigation technology. Our security approach includes:

### Technical Security
- **Encryption**: AES-256 encryption for data at rest, TLS 1.3+ for data in transit
- **Access Control**: Multi-factor authentication and role-based access control
- **Audit Logging**: Comprehensive logging of all data access and modifications
- **Vulnerability Management**: Regular security scanning and dependency updates
- **Incident Response**: Documented procedures for security incident handling

### Legal Security
- **Chain of Custody**: Cryptographic integrity verification for all evidence
- **Privacy Protection**: Built-in PII detection and redaction capabilities
- **Compliance**: GDPR, CCPA, and other relevant regulation compliance
- **Professional Standards**: Adherence to legal professional responsibility rules
- **Expert Standards**: Tools designed to meet Daubert and other evidentiary standards

**Report Security Issues**: security@lemkin.ai

See [SECURITY.md](SECURITY.md) for detailed security procedures.

## 🌍 Community

### Join Our Community

- **Discord**: Real-time chat and community support
- **GitHub Discussions**: Long-form community conversations and Q&A
- **Monthly Community Calls**: Regular updates and collaboration opportunities
- **Newsletter**: Monthly updates on new features and community highlights

### Community Guidelines

Our community is built on:
- **Respect**: Treating all members with dignity and professionalism
- **Inclusion**: Welcoming people of all backgrounds and skill levels
- **Mission Focus**: Supporting human rights and justice through technology
- **Collaboration**: Working together across disciplines and organizations
- **Ethics**: Maintaining the highest ethical standards in all activities

### Getting Help

- **Documentation**: Comprehensive guides and API references
- **Community Discord**: Real-time help from community members
- **GitHub Issues**: Bug reports and feature requests
- **Expert Consultation**: Access to legal technology experts for complex issues

## 📜 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

### Why Apache 2.0?

We chose Apache 2.0 to:
- **Enable wide adoption** by legal organizations and NGOs
- **Allow commercial use** for legal technology companies serving justice
- **Require attribution** to ensure transparency in legal proceedings
- **Protect contributors** with patent grants and liability limitations
- **Maintain compatibility** with other open source legal technology projects

## 🏛️ Legal Notice

**Important**: These tools are designed for legitimate legal investigations and human rights documentation. Users are responsible for:

- **Legal Authorization**: Obtaining proper legal authority for investigations
- **Jurisdiction Compliance**: Following applicable laws and regulations
- **Professional Standards**: Meeting ethical obligations for legal professionals
- **Evidence Standards**: Ensuring analysis meets relevant evidentiary standards
- **Privacy Protection**: Protecting the privacy rights of individuals

The Lemkin AI project and contributors provide these tools "as is" without warranty. Users assume full responsibility for their use in legal proceedings.

## 🙏 Acknowledgments

### Inspiration
- **Raphael Lemkin**: International lawyer who coined "genocide" and advocated for international justice
- **Human Rights Community**: Investigators and attorneys working for justice worldwide
- **Open Source Community**: Contributors and maintainers making technology accessible

### Contributors
- **Core Development Team**: Full-time developers and maintainers
- **Legal Advisory Board**: Legal professionals providing domain expertise
- **Security Experts**: Security researchers ensuring tool safety
- **Community Contributors**: Everyone who contributes code, documentation, and expertise

### Supporters
- **Legal Organizations**: Groups providing real-world testing and feedback
- **Academic Institutions**: Research partnerships and educational collaboration
- **Technology Partners**: Companies providing infrastructure and services
- **Individual Donors**: Community members supporting the mission financially

## 📞 Contact

### General Inquiries
- **Website**: https://lemkin.ai
- **Email**: contact@lemkin.ai
- **Community**: [Discord Server](https://discord.gg/lemkin-ai)

### Specialized Contact
- **Security**: security@lemkin.ai
- **Legal Questions**: legal@lemkin.ai
- **Ethics**: ethics@lemkin.ai
- **Research Collaboration**: research@lemkin.ai
- **Media Inquiries**: media@lemkin.ai

### Social Media
- **Twitter**: [@lemkin_ai](https://twitter.com/lemkin_ai)
- **LinkedIn**: [Lemkin AI](https://linkedin.com/company/lemkin-ai)
- **GitHub**: [@lemkin-org](https://github.com/lemkin-org)

---

<div align="center">
  <strong>Democratizing justice through technology</strong>

  *By building these tools together, we strengthen the capacity of those working for human rights, justice, and accountability around the world.*

  [![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg)](CODE_OF_CONDUCT.md)
</div>