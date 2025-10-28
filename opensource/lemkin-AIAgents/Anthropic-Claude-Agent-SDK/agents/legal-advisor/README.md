# Legal Framework & Jurisdiction Advisor Agent

## Overview

The Legal Framework & Jurisdiction Advisor Agent provides expert analysis on applicable international and domestic law, jurisdictional questions, and legal frameworks for human rights investigations and legal proceedings. It helps legal teams understand which laws apply, which courts have jurisdiction, and what legal strategies are available.

## Capabilities

### Core Functions

1. **International Law Analysis**
   - Analyze applicable international criminal law provisions
   - Explain international humanitarian law requirements
   - Assess international human rights law obligations
   - Interpret treaty provisions and customary law
   - Analyze relationships between different legal frameworks

2. **Jurisdictional Analysis**
   - Assess territorial, nationality, and universal jurisdiction
   - Analyze court competence and admissibility requirements
   - Evaluate complementarity principles (ICC/domestic courts)
   - Assess jurisdictional conflicts and forum selection
   - Analyze immunities and jurisdictional bars

3. **Legal Element Mapping**
   - Break down crimes into required legal elements
   - Analyze evidence requirements for each element
   - Assess strength of legal case for specific charges
   - Identify alternative charges and legal theories
   - Map facts to legal requirements

4. **Procedural Analysis**
   - Explain applicable procedural rules and requirements
   - Analyze evidence admissibility standards
   - Assess statute of limitations and temporal jurisdiction
   - Evaluate procedural rights and protections
   - Analyze appeals and review mechanisms

5. **Comparative Legal Analysis**
   - Compare different legal frameworks and their advantages
   - Analyze precedents from international and domestic courts
   - Assess legal developments and emerging jurisprudence
   - Evaluate different prosecution strategies
   - Analyze plea bargaining and cooperation opportunities

6. **Legal Strategy Development**
   - Recommend optimal legal frameworks and forums
   - Suggest charge selection and legal theories
   - Analyze risks and benefits of different approaches
   - Provide alternative legal strategies
   - Assess political and practical considerations

## Legal Frameworks Covered

### International Criminal Law (ICL)
- **Rome Statute**: ICC crimes (genocide, crimes against humanity, war crimes, aggression)
- **Geneva Conventions**: War crimes and grave breaches
- **Customary International Law**: Universal criminal responsibility principles
- **Specialized Treaties**: Torture Convention, Genocide Convention, etc.

### International Humanitarian Law (IHL)
- **Geneva Conventions & Additional Protocols**: Armed conflict rules
- **Hague Conventions**: Laws and customs of war
- **Customary IHL**: Universally applicable conflict rules
- **Rome Statute War Crimes**: Detailed war crimes provisions

### International Human Rights Law (IHRL)
- **Universal Treaties**: ICCPR, ICESCR, CAT, CERD, CRC, CEDAW
- **Regional Systems**: European, Inter-American, African human rights
- **Customary IHRL**: Fundamental human rights principles
- **Specialized Mechanisms**: Treaty body procedures, special procedures

### Domestic Legal Systems
- **Constitutional Law**: Fundamental rights and state obligations
- **Criminal Law**: Domestic crimes corresponding to international offenses
- **Civil Law**: Remedies and reparations for victims
- **Administrative Law**: State responsibilities and procedures

## Usage

### Basic Legal Analysis

```python
from agents.legal_advisor import LegalAdvisorAgent

# Initialize agent
legal_advisor = LegalAdvisorAgent()

# Analyze legal framework
result = legal_advisor.process({
    'legal_question': 'What charges are available for systematic torture in detention centers?',
    'factual_scenario': 'Government forces systematically tortured detainees in multiple facilities over 2-year period',
    'potential_charges': ['torture', 'crimes_against_humanity', 'war_crimes'],
    'case_id': 'LEGAL-ANALYSIS-001'
})

print(result['output_data']['executive_summary'])
```

### ICC Jurisdiction Analysis

```python
# Analyze ICC jurisdiction specifically
result = legal_advisor.analyze_icc_jurisdiction(
    factual_scenario='Mass killings of civilians during armed conflict in State Party to Rome Statute',
    potential_charges=['genocide', 'crimes_against_humanity', 'war_crimes'],
    case_id='ICC-ANALYSIS-001'
)

# Check complementarity assessment
complementarity = result['output_data']['jurisdictional_analysis']['available_jurisdictions'][0]['admissibility_assessment']['complementarity']
print(f"Complementarity analysis: {complementarity}")
```

### Universal Jurisdiction Assessment

```python
# Assess universal jurisdiction opportunities
result = legal_advisor.assess_universal_jurisdiction(
    factual_scenario='Torture and enforced disappearances by government officials',
    potential_charges=['torture', 'enforced_disappearance'],
    target_states=['Germany', 'Spain', 'Belgium', 'Netherlands'],
    case_id='UJ-ASSESSMENT-001'
)

# Review jurisdictional opportunities
for jurisdiction in result['output_data']['jurisdictional_analysis']['available_jurisdictions']:
    print(f"Court: {jurisdiction['court']}")
    print(f"Basis: {jurisdiction['jurisdictional_basis']}")
    print(f"Requirements met: {jurisdiction['requirements_met']}")
```

### Legal Element Mapping

```python
# Map legal elements for specific charges
result = legal_advisor.map_legal_elements(
    charges=['crimes_against_humanity_murder', 'crimes_against_humanity_persecution'],
    factual_scenario='Systematic killing and persecution of ethnic minority population',
    case_id='ELEMENTS-ANALYSIS-001'
)

# Review element analysis
for element_analysis in result['output_data']['legal_element_analysis']:
    print(f"Charge: {element_analysis['charge']}")
    for element in element_analysis['required_elements']:
        print(f"  Element: {element['element']}")
        print(f"  Evidence strength: {element['evidence_strength']}")
```

### Immunity Analysis

```python
# Analyze immunity issues
result = legal_advisor.analyze_immunity_issues(
    potential_defendants=['President X', 'General Y', 'Minister Z'],
    their_positions=['Head of State', 'Military Commander', 'Interior Minister'],
    factual_scenario='Alleged ordering and execution of crimes against humanity',
    case_id='IMMUNITY-ANALYSIS-001'
)

# Check immunity assessments
for immunity in result['output_data']['jurisdictional_analysis']['immunity_issues']:
    print(f"Immunity type: {immunity['immunity_type']}")
    print(f"Applicable to: {immunity['applicable_to']}")
    print(f"Exceptions: {immunity['exceptions']}")
```

### Forum Comparison

```python
# Compare different legal forums
result = legal_advisor.compare_legal_forums(
    factual_scenario='War crimes and crimes against humanity during armed conflict',
    potential_forums=['ICC', 'domestic_courts_state_A', 'universal_jurisdiction_germany'],
    evaluation_criteria=['likelihood_of_success', 'victim_participation', 'reparations'],
    case_id='FORUM-COMPARISON-001'
)

# Review forum analysis
for forum in result['output_data']['jurisdictional_analysis']['available_jurisdictions']:
    print(f"Forum: {forum['court']}")
    print(f"Advantages: {forum['advantages']}")
    print(f"Disadvantages: {forum['disadvantages']}")
```

### Prosecution Strategy Development

```python
# Develop comprehensive prosecution strategy
result = legal_advisor.develop_prosecution_strategy(
    factual_scenario='Systematic attacks on civilian population',
    available_evidence=['witness_testimony', 'documentary_evidence', 'satellite_imagery'],
    constraints=['limited_state_cooperation', 'fugitive_suspects'],
    objectives=['accountability', 'victim_justice', 'deterrence'],
    case_id='STRATEGY-DEVELOPMENT-001'
)

# Review strategic recommendations
recommendations = result['output_data']['recommendations']
print(f"Primary strategy: {recommendations['primary_recommendation']['recommended_approach']}")
print(f"Rationale: {recommendations['primary_recommendation']['rationale']}")
```

## Configuration Options

### Default Configuration
- Comprehensive analysis of all legal frameworks
- Balanced consideration of international and domestic law
- Standard evidence and procedural analysis
- Political and practical considerations included

### Comprehensive Analysis Configuration
```python
from agents.legal_advisor.config import COMPREHENSIVE_ANALYSIS_CONFIG

advisor = LegalAdvisorAgent(config=COMPREHENSIVE_ANALYSIS_CONFIG)
```
- Maximum depth analysis with extensive precedent review
- Detailed procedural and evidentiary analysis
- Comprehensive risk assessment and strategy development

### ICC-Focused Configuration
```python
from agents.legal_advisor.config import ICC_FOCUSED_CONFIG

advisor = LegalAdvisorAgent(config=ICC_FOCUSED_CONFIG)
```
- Specialized analysis for ICC jurisdiction and procedures
- Detailed complementarity assessment
- Rome Statute-specific legal element mapping

### Domestic Prosecution Configuration
```python
from agents.legal_advisor.config import DOMESTIC_PROSECUTION_CONFIG

advisor = LegalAdvisorAgent(config=DOMESTIC_PROSECUTION_CONFIG)
```
- Focus on domestic courts and universal jurisdiction
- Constitutional and domestic law emphasis
- Practical implementation considerations

### Strategic Planning Configuration
```python
from agents.legal_advisor.config import STRATEGIC_PLANNING_CONFIG

advisor = LegalAdvisorAgent(config=STRATEGIC_PLANNING_CONFIG)
```
- Enhanced strategic analysis and planning
- Detailed political and practical considerations
- Implementation-focused recommendations

## Output Format

### Legal Analysis Metadata
- Unique analysis identifier
- Analysis date and scope
- Legal frameworks considered
- Confidence assessments

### Applicable Law Analysis
- International criminal law provisions
- International humanitarian law requirements
- International human rights law obligations
- Domestic legal framework assessment
- Treaty ratification status and obligations

### Jurisdictional Analysis
- Available forums and their requirements
- Jurisdictional basis assessment (territorial, nationality, universal)
- Admissibility analysis (complementarity, gravity, interests of justice)
- Immunity issues and exceptions
- Practical considerations for each forum

### Legal Element Mapping
- Required elements for each potential charge
- Evidence requirements and availability
- Evidentiary challenges and gaps
- Mental element (intent/knowledge) analysis
- Contextual elements assessment

### Procedural Considerations
- Evidence admissibility standards
- Temporal jurisdiction and limitation periods
- Victim participation rights and mechanisms
- Authentication and chain of custody requirements
- Appeal and review procedures

### Precedent Analysis
- Relevant case law from international and domestic courts
- Legal principles established in precedents
- Applicability to current case
- Distinguishing factors and limitations
- Precedential value assessment

### Strategic Recommendations
- Primary recommended legal approach
- Alternative strategies and their advantages
- Implementation steps and timelines
- Risk assessment and mitigation strategies
- Resource requirements and practical considerations

## Jurisdictional Frameworks

### International Criminal Court (ICC)
- **Jurisdiction**: Genocide, crimes against humanity, war crimes, aggression
- **Temporal**: Crimes after July 1, 2002
- **Personal**: Natural persons only, no organizational liability
- **Complementarity**: Only when domestic courts unwilling/unable
- **Admissibility**: Gravity, interests of justice, ne bis in idem

### Universal Jurisdiction
- **Crimes Covered**: Genocide, torture, crimes against humanity, war crimes
- **No Territorial Link Required**: Based on nature of crime
- **Aut Dedere Aut Judicare**: Obligation to prosecute or extradite
- **State Implementation**: Varies by domestic legislation
- **Practical Challenges**: Arrest, evidence, cooperation

### Domestic Courts
- **Territorial Jurisdiction**: Crimes committed on state territory
- **Nationality Jurisdiction**: Crimes by or against nationals
- **Protective Jurisdiction**: Crimes affecting state security
- **Constitutional Review**: Human rights violations
- **Civil Remedies**: Compensation and reparations

### Regional Human Rights Courts
- **European Court of Human Rights**: Council of Europe members
- **Inter-American Court of Human Rights**: OAS members
- **African Court on Human and Peoples' Rights**: AU members
- **Individual Petition Rights**: After domestic remedies exhausted
- **State Responsibility**: Focus on state obligations

## Legal Element Analysis

### Genocide
- **Mental Element**: Intent to destroy protected group
- **Physical Element**: Listed acts (killing, causing harm, etc.)
- **Protected Groups**: National, ethnic, racial, religious
- **Scale Requirement**: Not required under law
- **Special Intent**: Dolus specialis requirement

### Crimes Against Humanity
- **Contextual Element**: Widespread or systematic attack
- **Knowledge Element**: Knowledge of attack
- **Civilian Population**: Attack directed against civilians
- **Listed Acts**: Murder, persecution, torture, etc.
- **Policy Element**: State or organizational policy

### War Crimes
- **Armed Conflict**: International or non-international
- **Nexus Requirement**: Connection to armed conflict
- **Protected Persons**: Hors de combat, civilians, etc.
- **Grave Breaches**: Specific violations of Geneva Conventions
- **Command Responsibility**: Superior responsibility for subordinate crimes

### Torture
- **Elements**: Severe pain/suffering, purpose, official capacity
- **Mental Element**: Intent to inflict suffering
- **Purposes**: Information, punishment, intimidation, discrimination
- **Official Capacity**: State officials or with acquiescence
- **Prohibition**: Absolute, no derogation allowed

## Evidence and Procedure

### Evidence Standards
- **International Courts**: Flexible evidence rules
- **Domestic Courts**: Strict national evidence rules
- **Burden of Proof**: Beyond reasonable doubt (criminal)
- **Admissibility**: Relevance, probative value, prejudice
- **Authentication**: Chain of custody, expert testimony

### Victim Rights
- **Participation**: Right to participate in proceedings
- **Legal Representation**: Right to legal counsel
- **Protection**: Witness and victim protection measures
- **Reparations**: Individual and collective reparations
- **Truth**: Right to know the truth about violations

### Procedural Safeguards
- **Fair Trial Rights**: Due process guarantees
- **Presumption of Innocence**: Burden on prosecution
- **Right to Counsel**: Effective legal representation
- **Right to Interpreter**: Language assistance
- **Appeals**: Right to appeal convictions and sentences

## Integration with Other Agents

The Legal Advisor works effectively with:

- **Historical Researcher**: Provides legal context for historical analysis
- **Evidence Gap Identifier**: Identifies legal evidence requirements
- **Comparative Document Analyzer**: Analyzes legal documents and precedents
- **OSINT Synthesis Agent**: Provides legal framework for intelligence analysis
- **NGO & UN Reporting Specialist**: Supplies legal framework for advocacy

## Best Practices

### Legal Analysis Quality
1. **Precision**: Use exact legal terminology and citations
2. **Comprehensiveness**: Consider all relevant legal frameworks
3. **Accuracy**: Verify all legal authorities and precedents
4. **Clarity**: Explain complex legal concepts clearly

### Strategic Thinking
1. **Practical Focus**: Consider real-world implementation challenges
2. **Risk Assessment**: Identify and evaluate all significant risks
3. **Alternative Planning**: Develop multiple strategic options
4. **Timeline Awareness**: Consider timing and sequencing issues

### Ethical Considerations
1. **Victim Focus**: Prioritize victim interests and rights
2. **Justice Goals**: Balance accountability with practical outcomes
3. **Legal Ethics**: Maintain professional standards
4. **Transparency**: Clear about limitations and uncertainties

## Limitations and Considerations

### Legal Complexity
- International law is complex and sometimes uncertain
- Different legal systems may reach different conclusions
- Political factors often influence legal outcomes
- Resource constraints may limit legal options

### Jurisdictional Challenges
- States may not cooperate with international proceedings
- Immunity claims may bar prosecution of senior officials
- Evidentiary challenges in international prosecutions
- Long timelines for international justice processes

### Strategic Considerations
- Legal perfectionism vs. practical achievability
- Victim expectations vs. realistic outcomes
- Political timing and international relations
- Resource allocation and priority setting