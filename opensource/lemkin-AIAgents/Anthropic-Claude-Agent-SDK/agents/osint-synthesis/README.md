# OSINT Synthesis Agent

Aggregates and analyzes publicly available information for legal and investigative purposes.

## Purpose

Monitors open sources (social media, news, websites, forums) to gather intelligence, verify claims, detect coordinated campaigns, and generate actionable intelligence briefs for investigations.

## Capabilities

- **Multi-Source Monitoring**: Track mentions across platforms, identify trends
- **Claim Verification**: Extract claims and cross-reference across sources
- **Credibility Assessment**: Evaluate source reliability and bias
- **Pattern Detection**: Identify coordinated campaigns and information operations
- **Geographic/Temporal Analysis**: Map where and when information spreads
- **Intelligence Briefs**: Generate structured reports with confidence scores

## Usage

### Basic Monitoring

```python
from agents.osint_synthesis.agent import OSINTSynthesisAgent

agent = OSINTSynthesisAgent()

# Monitor keywords
result = agent.monitor_keywords(
    keywords=["protest", "#demonstration", "police violence"],
    time_period_days=7,
    case_id="CASE-2024-001"
)

print(result['executive_summary'])
print(f"Findings: {len(result['key_findings'])}")
```

### Claim Verification

```python
# Verify a specific claim
result = agent.verify_claim(
    claim="Military used tear gas on peaceful protesters on January 15",
    context="Protest in capital city",
    case_id="CASE-2024-002"
)

if 'verification_summary' in result:
    print(f"Status: {result['verification_summary']['status']}")
    print(f"Confidence: {result['verification_summary']['confidence']}")
```

### Event Analysis

```python
# Analyze event coverage
result = agent.analyze_event(
    event_description="Detention facility raid",
    location="City Center",
    date="2024-01-20",
    case_id="CASE-2024-003"
)

# Check for coordinated narratives
if result.get('narrative_analysis'):
    for narrative in result['narrative_analysis']['dominant_narratives']:
        print(f"Narrative: {narrative['narrative']}")
        print(f"Assessment: {narrative['authenticity_assessment']}")
```

## Output Format

Returns structured JSON with:
- Key findings with credibility assessment
- Claims identified and verification status
- Source credibility analysis
- Narrative analysis and dominant themes
- Coordination detection
- Geographic/temporal heat maps
- Actor profiles
- Red flags (bot activity, disinformation)
- Follow-up recommendations

## Configuration

```python
from agents.osint_synthesis.config import OSINTConfig

config = OSINTConfig(
    perform_credibility_assessment=True,
    detect_coordination=True,
    flag_bot_activity=True,
    min_credibility_score=0.6
)

agent = OSINTSynthesisAgent(config=config)
```

## Use Cases

- **Protest Monitoring**: Real-time tracking of demonstrations and police response
- **Claim Verification**: Verify allegations using open-source evidence
- **Disinformation Detection**: Identify coordinated false narratives
- **Actor Tracking**: Monitor individuals or organizations online
- **Event Documentation**: Gather public evidence of specific incidents
