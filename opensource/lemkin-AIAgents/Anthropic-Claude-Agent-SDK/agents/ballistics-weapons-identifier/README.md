# Ballistics & Weapons Identifier Agent

The Ballistics & Weapons Identifier Agent identifies weapons and ammunition from photographs and descriptions, analyzes ballistics reports, assesses wound patterns, traces weapon origins, and links weapons to incidents and perpetrators for legal investigations.

## Overview

This agent provides expert weapons identification and ballistics analysis to support investigations and prosecutions. It identifies weapons from visual evidence, analyzes ammunition markings, interprets ballistics reports, and assesses IHL compliance for weapons use.

## Core Capabilities

### Weapon Identification
- **Type identification** - Rifles, pistols, machine guns, grenade launchers, mortars
- **Manufacturer/model determination** - Specific make, model, and variant
- **Characteristic assessment** - Range, rate of fire, caliber, capabilities
- **Modification detection** - Custom modifications and alterations
- **Condition evaluation** - Functionality and damage assessment
- **Dating** - Age determination from design features

### Ammunition Analysis
- **Type/caliber identification** - Ammunition types and calibers
- **Marking analysis** - Headstamps, lot numbers, date codes
- **Origin tracing** - Manufacturer and country of origin
- **Weapon compatibility** - Matching ammunition to weapon types
- **Characteristic assessment** - Lethality, intended use, special properties

### Ballistics Review
- **Bullet-weapon matching** - Linking bullets to specific weapons
- **Trajectory analysis** - Firing position, angle, distance
- **Firing pin marks** - Ejector and firing pin pattern analysis
- **Gunshot residue** - GSR analysis interpretation
- **Impact patterns** - Bullet impact and fragmentation patterns

### Wound Pattern Analysis
- **Injury-weapon correlation** - Matching wounds to weapon types
- **Entrance/exit wounds** - Wound characteristic analysis
- **Fragmentation patterns** - Shrapnel and bullet fragmentation
- **Distance assessment** - Determining firing distance from wounds
- **Consistency evaluation** - Assessing wound-weapon consistency

### Weapon Tracing & Attribution
- **Manufacturer tracing** - Supply chain identification
- **Serial number analysis** - Weapon registration and tracking
- **Unit/group attribution** - Linking weapons to military units or armed groups
- **State/non-state origin** - Determining state or non-state actor origin
- **Proliferation tracking** - Weapon distribution patterns

### Legal & IHL Analysis
- **Prohibited weapons** - Identifying IHL-prohibited weapons (explosive bullets, chemical, etc.)
- **Indiscriminate weapons** - Assessing indiscriminate use
- **Incident linking** - Connecting weapons to specific attacks
- **Perpetrator attribution** - Supporting attribution to individuals/groups
- **Legal documentation** - Evidence generation for proceedings

## Configuration Options

**Default**: Comprehensive weapons and ballistics analysis

**Investigation** (`INVESTIGATION_CONFIG`):
- Focus on weapon identification and attribution
- Incident and actor linking
- Extended analysis (12,000 tokens)

**Technical Analysis** (`TECHNICAL_ANALYSIS_CONFIG`):
- Deep ammunition and markings analysis
- Modification identification
- Technical specifications
- Maximum precision (temperature 0.05)

## Usage Examples

### Basic Weapon Identification

```python
from agents.ballistics_weapons_identifier.agent import BallisticsWeaponsIdentifierAgent

agent = BallisticsWeaponsIdentifierAgent()

result = agent.identify_weapon(
    weapon_image_or_description="""
    Photo shows assault rifle with wooden stock, curved magazine,
    distinctive front sight, appears to be 7.62mm caliber
    """,
    case_id='CASE-2024-001'
)

print(f"Weapon identified: {result['weapon_identification'][0]['identification']['model']}")
print(f"Confidence: {result['weapon_identification'][0]['identification']['identification_confidence']}")
```

### Comprehensive Ballistics Analysis

```python
result = agent.process({
    'weapon_images': [
        {'description': 'AK-pattern rifle', 'photo_path': 'weapon1.jpg'},
        {'description': 'Ammunition found at scene', 'photo_path': 'ammo.jpg'}
    ],
    'ballistics_reports': [
        """
        Ballistics Report: Bullet recovered from victim matches
        rifling characteristics consistent with 7.62x39mm weapon...
        """
    ],
    'wound_patterns': [
        """
        Medical Examiner: Entry wound 8mm diameter, exit wound 15mm,
        consistent with intermediate rifle cartridge...
        """
    ],
    'case_id': 'CASE-2024-001',
    'analysis_focus': ['identification', 'attribution', 'legality']
})

# Extract findings
weapons = result['weapon_identification']
attribution = result['attribution_analysis']
legality = result['weapon_legality_assessment']
```

### Multi-Agent Workflow

```python
from shared import AuditLogger, EvidenceHandler
from agents.satellite_imagery_analyst.agent import SatelliteImageryAnalystAgent
from agents.ballistics_weapons_identifier.agent import BallisticsWeaponsIdentifierAgent

shared_infra = {
    'audit_logger': AuditLogger(),
    'evidence_handler': EvidenceHandler()
}

imagery = SatelliteImageryAnalystAgent(**shared_infra)
weapons = BallisticsWeaponsIdentifierAgent(**shared_infra)

# Analyze satellite imagery for weapon systems
with open('artillery_position.jpg', 'rb') as f:
    imagery_result = imagery.analyze_military_features(
        image_data=f.read(),
        case_id='CASE-001'
    )

# Identify weapons visible in imagery
weapons_result = weapons.process({
    'weapon_images': [{'description': imagery_result['features_identified']}],
    'case_id': 'CASE-001'
})
```

## Integration with Other Agents

### Satellite Imagery Analyst
Identifies weapons visible in satellite/aerial imagery.

### Forensic Analysis Reviewer
Reviews ballistics reports as part of comprehensive forensic review.

### Medical Forensic Analyst
Analyzes wounds for consistency with identified weapons.

### Military Structure Analyst
Links weapons to specific military units and command structures.

## Evidentiary Standards

- **Visual Analysis**: Careful examination of photographs and physical evidence
- **Expert Standards**: Applies ballistics and weapons identification standards
- **Confidence Scoring**: All identifications include confidence levels
- **Traceability**: Documents origins and supply chains
- **Legal Framework**: Assesses IHL compliance
- **Chain of Custody**: All evidence handling logged

## Technical Requirements

- Python 3.9+
- Anthropic API key (Claude Sonnet 4.5 with Vision API)
- Access to weapon photographs and descriptions
- Ballistics reports (if available)

## Limitations

- Cannot perform physical forensic analysis (reviews photos/descriptions)
- Identification confidence depends on image/description quality
- Some modifications may not be visible in photos
- Supply chain tracing requires additional intelligence
- Cannot replace expert ballistics testimony for court

## Use Cases

- War crimes investigation (weapon identification)
- Arms trafficking investigations
- IHL compliance assessment
- Military targeting analysis
- Expert witness preparation
- Weapon proliferation tracking
- Forensic ballistics review
- Incident reconstruction
