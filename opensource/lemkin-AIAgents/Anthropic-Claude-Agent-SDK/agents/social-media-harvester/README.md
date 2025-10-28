# Social Media Evidence Harvester Agent

## Overview

The Social Media Evidence Harvester Agent specializes in collecting, analyzing, and documenting social media posts as legal evidence. It processes screenshots of social media content to extract evidentiary information while maintaining strict chain of custody and ensuring admissibility standards for legal proceedings.

## Capabilities

### Core Functions

1. **Content Analysis & Extraction**
   - Analyzes screenshots from any social media platform
   - Extracts usernames, handles, and profile information
   - Identifies timestamps, dates, and timezone information
   - Extracts post text, captions, hashtags, and mentions
   - Documents media attachments and external links

2. **Metadata Documentation**
   - Captures engagement metrics (likes, shares, comments)
   - Documents privacy settings and audience scope
   - Records platform-specific features (verification badges)
   - Extracts technical metadata from screenshots
   - Preserves platform UI elements for authenticity

3. **Authenticity Assessment**
   - Evaluates indicators of genuine vs. manipulated content
   - Detects potential deepfakes or edited media
   - Assesses account authenticity markers
   - Identifies coordinated inauthentic behavior
   - Flags technical inconsistencies

4. **Legal Chain of Custody**
   - Generates unique evidence identifiers
   - Documents collection time and method
   - Creates admissibility-focused documentation
   - Tracks all handling and analysis steps
   - Prepares legal authenticity attestations

5. **Context Preservation**
   - Documents conversation threads and reply chains
   - Preserves parent posts and surrounding context
   - Tracks hashtag usage and trending contexts
   - Records temporal context and breaking news connections
   - Maps viral spread patterns

6. **Network Analysis**
   - Maps interactions between accounts
   - Identifies coordinated posting patterns
   - Detects bot networks and suspicious behavior
   - Analyzes influence patterns and reach
   - Documents follower networks when visible

## Supported Platforms

- Twitter/X (all post types including threads)
- Facebook (posts, comments, reactions)
- Instagram (posts, stories, reels)
- TikTok (videos, comments, duets)
- LinkedIn (professional posts, comments)
- YouTube (comments, community posts)
- Telegram (public channels, groups)
- Reddit (posts, comments, threads)
- WhatsApp (screenshots of public content)
- Discord (public server content)

## Usage

### Basic Analysis

```python
from agents.social_media_harvester import SocialMediaHarvesterAgent

# Initialize agent
harvester = SocialMediaHarvesterAgent()

# Analyze a screenshot
result = harvester.process({
    'screenshot_data': 'path/to/screenshot.png',  # or base64 data
    'screenshot_filename': 'evidence_001.png',
    'case_id': 'CASE-2024-001',
    'collector_info': 'Investigator Smith',
    'collection_context': 'Documentation of threats against witness'
})

print(result['output_data']['summary'])
```

### Batch Processing

```python
# Process multiple screenshots
screenshots = [
    {
        'screenshot_data': 'path/to/screenshot1.png',
        'screenshot_filename': 'threat_001.png',
        'case_id': 'CASE-2024-001'
    },
    {
        'screenshot_data': 'path/to/screenshot2.png',
        'screenshot_filename': 'threat_002.png',
        'case_id': 'CASE-2024-001'
    }
]

results = harvester.process_batch(screenshots)
```

### Legal Evidence Configuration

```python
from agents.social_media_harvester.config import LEGAL_EVIDENCE_CONFIG

# Use high-precision configuration for court evidence
harvester = SocialMediaHarvesterAgent(config=LEGAL_EVIDENCE_CONFIG)
```

### Network Analysis Focus

```python
from agents.social_media_harvester.config import NETWORK_ANALYSIS_CONFIG

# Focus on coordinated behavior detection
harvester = SocialMediaHarvesterAgent(config=NETWORK_ANALYSIS_CONFIG)

result = harvester.process({
    'screenshot_data': 'coordinated_posts.png',
    'analysis_focus': ['coordinated_behavior', 'bot_detection', 'amplification']
})
```

## Output Format

The agent produces comprehensive JSON documentation including:

### Evidence Metadata
- Unique evidence identifier
- Collection timestamp and method
- File hash for integrity verification
- Chain of custody information

### Content Analysis
- Platform identification and version
- Account details (handle, display name, verification)
- Post content (text, media, links, hashtags)
- Engagement metrics and reactions
- Temporal data (timestamps, timezone)

### Authenticity Assessment
- Account credibility indicators
- Content authenticity markers
- Bot detection analysis
- Coordination pattern identification
- Confidence scores and red flags

### Legal Considerations
- Admissibility factor assessment
- Authentication requirements
- Privacy consideration analysis
- Chain of custody recommendations

### Network Analysis
- Interaction pattern mapping
- Coordinated behavior detection
- Influence metrics and reach analysis
- Related account identification

## Configuration Options

### Default Configuration
- Balanced analysis depth and speed
- Standard authenticity assessment
- Basic network analysis
- Medium precision thresholds

### Legal Evidence Configuration
- Maximum precision analysis
- Strict authenticity requirements
- Comprehensive legal assessment
- High-quality evidence standards

### Network Analysis Configuration
- Deep social network mapping
- Enhanced coordination detection
- Extended interaction analysis
- Lower detection thresholds

### Bulk Processing Configuration
- Optimized for speed and volume
- Reduced analysis depth
- Essential authenticity checks only
- Streamlined output format

## Authentication Markers

### Authentic Indicators
- Consistent posting history and patterns
- Natural engagement ratios
- Verified accounts with established history
- Original content creation
- Geographic consistency in posts
- Reasonable follower/following ratios

### Suspicious Indicators
- New accounts with unusually high activity
- Identical or near-identical posts across accounts
- Unrealistic engagement patterns
- Coordinated timing across multiple accounts
- Recycled media from unrelated events
- Metadata inconsistencies

## Legal Considerations

### Chain of Custody
- Every screenshot is assigned a unique evidence ID
- SHA-256 hash calculated for integrity verification
- Complete audit trail of all analysis steps
- Documentation of collector and collection method

### Authentication Requirements
- Assessment of factors needed to authenticate evidence
- Documentation of platform-specific verification needs
- Evaluation of technical requirements for court admission
- Identification of supporting evidence needs

### Privacy Analysis
- Public vs. private post status assessment
- Audience scope and visibility documentation
- Privacy setting implications for legal use
- Platform-specific privacy considerations

## Example Use Cases

### Documenting Threats and Harassment
```python
result = harvester.process({
    'screenshot_data': 'threat_screenshot.png',
    'case_id': 'HARASSMENT-001',
    'analysis_focus': ['authenticity', 'legal_admissibility'],
    'collection_context': 'Threats received by victim on Twitter'
})
```

### Investigating Coordinated Campaigns
```python
harvester = SocialMediaHarvesterAgent(config=NETWORK_ANALYSIS_CONFIG)
results = harvester.process_batch([
    {'screenshot_data': f'coordinated_{i}.png'} for i in range(1, 11)
])
```

### Preserving Protest Documentation
```python
result = harvester.process({
    'screenshot_data': 'protest_livestream.png',
    'analysis_focus': ['temporal_accuracy', 'geolocation', 'authenticity'],
    'collection_context': 'Real-time documentation of police response'
})
```

### Atrocity Evidence Collection
```python
result = harvester.process({
    'screenshot_data': 'eyewitness_account.png',
    'case_id': 'ATROCITY-INVESTIGATION-001',
    'analysis_focus': ['authenticity', 'verification', 'context_preservation'],
    'collector_info': 'Human Rights Investigator'
})
```

## Error Handling

The agent handles various error conditions:
- Invalid or corrupted screenshot data
- Unsupported image formats
- Network analysis failures
- JSON parsing errors in responses
- Chain of custody verification failures

## Integration with Other Agents

The Social Media Evidence Harvester works well with:

- **OSINT Synthesis Agent**: Provides individual post analysis for broader intelligence gathering
- **Comparative Document Analyzer**: Compares versions of posts across time
- **Digital Forensics & Metadata Analyst**: Provides technical metadata analysis
- **Evidence Gap & Next Steps Identifier**: Identifies missing social media evidence

## Best Practices

### Evidence Collection
1. Capture full screenshots including platform UI elements
2. Document collection time and circumstances immediately
3. Preserve original files with integrity hashes
4. Collect related posts for context

### Authentication
1. Look for multiple authentication markers
2. Cross-reference with other evidence sources
3. Document any suspicious indicators
4. Preserve metadata for technical analysis

### Legal Preparation
1. Use legal evidence configuration for court cases
2. Document chain of custody meticulously
3. Prepare authentication foundations
4. Consider privacy and ethical implications

## Technical Requirements

- Python 3.8+
- Anthropic Claude API access with vision capabilities
- PIL/Pillow for image processing
- Network connectivity for real-time verification (optional)
- Sufficient storage for evidence preservation

## Security Considerations

- All evidence is hashed for integrity verification
- Audit logs are immutable and cryptographically secured
- Chain of custody is maintained throughout analysis
- Sensitive content is handled according to legal standards
- Privacy considerations are documented and flagged