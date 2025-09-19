"""
Human Rights Frameworks analysis module.

This module provides analysis of evidence against international and regional human
rights instruments including ICCPR, ECHR, ACHR, ACHPR, and UDHR.
"""

from datetime import datetime
from enum import Enum
from typing import List, Dict, Optional, Set, Any
from uuid import UUID

from pydantic import BaseModel
from loguru import logger

from .core import (
    Evidence, LegalElement, ElementSatisfaction, FrameworkAnalysis,
    GapAnalysis, LegalFramework, ElementStatus
)
from .element_analyzer import ElementAnalyzer


class HumanRightsInstrument(str, Enum):
    """Human rights instruments and treaties."""
    UDHR = "udhr"  # Universal Declaration of Human Rights
    ICCPR = "iccpr"  # International Covenant on Civil and Political Rights
    ECHR = "echr"  # European Convention on Human Rights
    ACHR = "achr"  # American Convention on Human Rights
    ACHPR = "achpr"  # African Charter on Human and Peoples' Rights


class HumanRightsViolationType(str, Enum):
    """Types of human rights violations."""
    RIGHT_TO_LIFE = "right_to_life"
    PROHIBITION_OF_TORTURE = "prohibition_of_torture"
    RIGHT_TO_LIBERTY = "right_to_liberty"
    RIGHT_TO_FAIR_TRIAL = "right_to_fair_trial"
    FREEDOM_OF_EXPRESSION = "freedom_of_expression"
    FREEDOM_OF_ASSEMBLY = "freedom_of_assembly"
    RIGHT_TO_PRIVACY = "right_to_privacy"
    NON_DISCRIMINATION = "non_discrimination"
    FREEDOM_OF_RELIGION = "freedom_of_religion"
    RIGHT_TO_EDUCATION = "right_to_education"


class DerogationStatus(str, Enum):
    """Status of derogation from human rights obligations."""
    NO_DEROGATION = "no_derogation"
    VALID_DEROGATION = "valid_derogation"
    INVALID_DEROGATION = "invalid_derogation"
    NON_DEROGABLE_RIGHT = "non_derogable_right"


class ICCPRViolation(BaseModel):
    """Violation of International Covenant on Civil and Political Rights."""
    article: str
    title: str
    description: str
    violation_type: HumanRightsViolationType
    derogation_status: DerogationStatus
    elements: List[str]
    state_obligations: List[str]


class RegionalHumanRightsViolation(BaseModel):
    """Violation of regional human rights instruments.""" 
    instrument: HumanRightsInstrument
    article: str
    title: str
    description: str
    violation_type: HumanRightsViolationType
    margin_of_appreciation: bool
    positive_obligations: List[str]
    remedies_available: List[str]


class HumanRightsAnalysis(BaseModel):
    """Analysis results for human rights violations."""
    instruments_analyzed: List[HumanRightsInstrument]
    universal_rights_findings: List[Dict]
    regional_rights_findings: List[Dict]
    derogation_analysis: Dict[str, str]
    state_responsibility_assessment: Dict[str, Any]
    individual_remedies: List[str]
    systemic_issues_identified: List[str]
    overall_human_rights_assessment: str


class HumanRightsAnalyzer:
    """
    Analyzer for human rights framework violations.
    
    This class implements analysis of evidence against international and regional
    human rights instruments.
    """
    
    def __init__(self, framework: LegalFramework):
        """Initialize the human rights analyzer for a specific framework."""
        self.framework = framework
        self.element_analyzer = ElementAnalyzer()
        self.legal_elements = self._load_human_rights_elements(framework)
        logger.info(f"HumanRightsAnalyzer initialized for {framework.value}")
    
    def _load_human_rights_elements(self, framework: LegalFramework) -> Dict[str, LegalElement]:
        """Load human rights elements for the specified framework."""
        elements = {}
        
        if framework == LegalFramework.ICCPR:
            elements.update(self._load_iccpr_elements())
        elif framework == LegalFramework.ECHR:
            elements.update(self._load_echr_elements())
        elif framework == LegalFramework.ACHR:
            elements.update(self._load_achr_elements())
        elif framework == LegalFramework.ACHPR:
            elements.update(self._load_achpr_elements())
        elif framework == LegalFramework.UDHR:
            elements.update(self._load_udhr_elements())
        else:
            raise ValueError(f"Unsupported human rights framework: {framework}")
        
        logger.info(f"Loaded {len(elements)} elements for {framework.value}")
        return elements
    
    def _load_iccpr_elements(self) -> Dict[str, LegalElement]:
        """Load ICCPR elements."""
        elements = {}
        
        # Article 6 - Right to life
        elements["iccpr_art_6"] = LegalElement(
            id="iccpr_art_6",
            framework=LegalFramework.ICCPR,
            article="Article 6",
            title="Right to life",
            description="Every human being has the inherent right to life",
            requirements=[
                "No one shall be arbitrarily deprived of life",
                "Death penalty may only be imposed for most serious crimes",
                "Death penalty cannot be imposed for crimes committed by persons below 18",
                "Pregnant women cannot be executed"
            ],
            keywords=["right to life", "arbitrary deprivation", "death penalty", "execution"],
            citation="International Covenant on Civil and Political Rights, Article 6"
        )
        
        # Article 7 - Prohibition of torture
        elements["iccpr_art_7"] = LegalElement(
            id="iccpr_art_7", 
            framework=LegalFramework.ICCPR,
            article="Article 7",
            title="Prohibition of torture",
            description="No one shall be subjected to torture or to cruel, inhuman or degrading treatment or punishment",
            requirements=[
                "No one shall be subjected to torture",
                "No one shall be subjected to cruel, inhuman or degrading treatment",
                "No medical or scientific experimentation without consent",
                "This is a non-derogable right"
            ],
            keywords=["torture", "cruel treatment", "inhuman treatment", "degrading", "non-derogable"],
            citation="International Covenant on Civil and Political Rights, Article 7"
        )
        
        # Article 9 - Right to liberty and security
        elements["iccpr_art_9"] = LegalElement(
            id="iccpr_art_9",
            framework=LegalFramework.ICCPR,
            article="Article 9",
            title="Right to liberty and security of person",
            description="Everyone has the right to liberty and security of person",
            requirements=[
                "No one shall be subjected to arbitrary arrest or detention",
                "No one shall be deprived of liberty except on grounds and procedures established by law",
                "Anyone arrested shall be informed of reasons for arrest",
                "Anyone arrested shall be brought promptly before a judge",
                "Anyone detained has right to take proceedings for review of lawfulness"
            ],
            keywords=["liberty", "security", "arbitrary arrest", "detention", "habeas corpus"],
            citation="International Covenant on Civil and Political Rights, Article 9"
        )
        
        # Article 14 - Right to fair trial
        elements["iccpr_art_14"] = LegalElement(
            id="iccpr_art_14",
            framework=LegalFramework.ICCPR,
            article="Article 14", 
            title="Right to fair trial",
            description="Everyone shall be entitled to a fair and public hearing by a competent, independent and impartial tribunal",
            requirements=[
                "All persons are equal before courts and tribunals",
                "Everyone entitled to fair and public hearing",
                "Tribunal must be competent, independent and impartial",
                "Everyone charged with criminal offence has right to be presumed innocent",
                "Minimum guarantees in criminal proceedings must be respected"
            ],
            keywords=["fair trial", "independent tribunal", "presumption of innocence", "public hearing"],
            citation="International Covenant on Civil and Political Rights, Article 14"
        )
        
        # Article 19 - Freedom of expression
        elements["iccpr_art_19"] = LegalElement(
            id="iccpr_art_19",
            framework=LegalFramework.ICCPR,
            article="Article 19",
            title="Freedom of opinion and expression",
            description="Everyone shall have the right to freedom of expression",
            requirements=[
                "Everyone shall have right to hold opinions without interference",
                "Everyone shall have right to freedom of expression",
                "Right includes freedom to seek, receive and impart information and ideas",
                "Restrictions must be provided by law and necessary for respect of others' rights or protection of national security, public order, public health or morals"
            ],
            keywords=["freedom of expression", "opinion", "information", "media", "restrictions"],
            citation="International Covenant on Civil and Political Rights, Article 19"
        )
        
        # Article 21 - Right of peaceful assembly
        elements["iccpr_art_21"] = LegalElement(
            id="iccpr_art_21",
            framework=LegalFramework.ICCPR,
            article="Article 21",
            title="Right of peaceful assembly",
            description="The right of peaceful assembly shall be recognized",
            requirements=[
                "The right of peaceful assembly shall be recognized",
                "Restrictions may only be imposed if necessary in democratic society",
                "Restrictions must be in interests of national security, public safety, public order, protection of public health or morals or protection of others' rights"
            ],
            keywords=["peaceful assembly", "protest", "demonstration", "restrictions"],
            citation="International Covenant on Civil and Political Rights, Article 21",
            precedents=["Kivenmaa v. Finland (HRC)", "Zvozskov v. Belarus (HRC)"]
        )
        
        # Article 22 - Freedom of association
        elements["iccpr_art_22"] = LegalElement(
            id="iccpr_art_22",
            framework=LegalFramework.ICCPR,
            article="Article 22",
            title="Freedom of association",
            description="Everyone shall have the right to freedom of association with others, including the right to form and join trade unions",
            requirements=[
                "Everyone shall have right to freedom of association with others",
                "Right includes forming and joining trade unions for protection of interests",
                "Restrictions may only be prescribed by law and necessary in democratic society",
                "Restrictions must be in interests of national security, public safety, public order, protection of public health or morals or protection of others' rights and freedoms"
            ],
            keywords=["freedom of association", "trade unions", "organizations", "restrictions"],
            citation="International Covenant on Civil and Political Rights, Article 22"
        )
        
        # Article 25 - Right to participate in public affairs
        elements["iccpr_art_25"] = LegalElement(
            id="iccpr_art_25",
            framework=LegalFramework.ICCPR,
            article="Article 25",
            title="Right to participate in public affairs and elections",
            description="Every citizen shall have the right and opportunity to take part in the conduct of public affairs, vote and be elected",
            requirements=[
                "Every citizen has right to take part in conduct of public affairs, directly or through freely chosen representatives",
                "Every citizen has right to vote and to be elected at genuine periodic elections",
                "Elections shall be by universal and equal suffrage and held by secret ballot",
                "Elections shall guarantee free expression of will of electors"
            ],
            keywords=["participation", "public affairs", "elections", "voting", "universal suffrage"],
            citation="International Covenant on Civil and Political Rights, Article 25"
        )
        
        # Article 26 - Equality before the law
        elements["iccpr_art_26"] = LegalElement(
            id="iccpr_art_26",
            framework=LegalFramework.ICCPR,
            article="Article 26",
            title="Equality before the law",
            description="All persons are equal before the law and are entitled without any discrimination to the equal protection of the law",
            requirements=[
                "All persons are equal before the law",
                "All persons are entitled to equal protection of law without discrimination",
                "Law shall prohibit any discrimination",
                "Law shall guarantee to all persons equal and effective protection against discrimination on any ground"
            ],
            keywords=["equality", "non-discrimination", "equal protection", "law"],
            citation="International Covenant on Civil and Political Rights, Article 26"
        )
        
        # Article 27 - Rights of minorities
        elements["iccpr_art_27"] = LegalElement(
            id="iccpr_art_27",
            framework=LegalFramework.ICCPR,
            article="Article 27",
            title="Rights of minorities",
            description="Persons belonging to ethnic, religious or linguistic minorities shall not be denied the right to enjoy their own culture, profess and practice their own religion, or use their own language",
            requirements=[
                "Persons belong to ethnic, religious or linguistic minorities",
                "They shall not be denied right to enjoy their own culture",
                "They shall not be denied right to profess and practice their own religion",
                "They shall not be denied right to use their own language",
                "These rights are to be enjoyed in community with other members of their group"
            ],
            keywords=["minorities", "culture", "religion", "language", "ethnic", "linguistic"],
            citation="International Covenant on Civil and Political Rights, Article 27"
        )
        
        return elements
    
    def _load_echr_elements(self) -> Dict[str, LegalElement]:
        """Load ECHR elements."""
        elements = {}
        
        # Article 2 - Right to life
        elements["echr_art_2"] = LegalElement(
            id="echr_art_2",
            framework=LegalFramework.ECHR,
            article="Article 2",
            title="Right to life",
            description="Everyone's right to life shall be protected by law",
            requirements=[
                "Everyone's right to life shall be protected by law",
                "No one shall be deprived of life intentionally",
                "Exceptions: death resulting from use of force which is absolutely necessary",
                "State has positive obligation to protect life"
            ],
            keywords=["right to life", "intentional deprivation", "use of force", "positive obligations"],
            citation="European Convention on Human Rights, Article 2"
        )
        
        # Article 3 - Prohibition of torture
        elements["echr_art_3"] = LegalElement(
            id="echr_art_3",
            framework=LegalFramework.ECHR,
            article="Article 3",
            title="Prohibition of torture",
            description="No one shall be subjected to torture or to inhuman or degrading treatment or punishment",
            requirements=[
                "No one shall be subjected to torture",
                "No one shall be subjected to inhuman treatment or punishment",
                "No one shall be subjected to degrading treatment or punishment",
                "This is an absolute right with no exceptions"
            ],
            keywords=["torture", "inhuman treatment", "degrading treatment", "absolute right"],
            citation="European Convention on Human Rights, Article 3"
        )
        
        # Article 5 - Right to liberty and security
        elements["echr_art_5"] = LegalElement(
            id="echr_art_5",
            framework=LegalFramework.ECHR,
            article="Article 5",
            title="Right to liberty and security",
            description="Everyone has the right to liberty and security of person",
            requirements=[
                "No one shall be deprived of liberty save in specific cases and in accordance with procedure prescribed by law",
                "Everyone arrested must be informed promptly of reasons",
                "Everyone arrested shall be brought promptly before a judge",
                "Everyone detained has right to take proceedings to review lawfulness",
                "Everyone victim of unlawful arrest has enforceable right to compensation"
            ],
            keywords=["liberty", "security", "arrest", "detention", "compensation"],
            citation="European Convention on Human Rights, Article 5"
        )
        
        # Article 8 - Right to respect for private and family life
        elements["echr_art_8"] = LegalElement(
            id="echr_art_8",
            framework=LegalFramework.ECHR,
            article="Article 8",
            title="Right to respect for private and family life",
            description="Everyone has the right to respect for his private and family life, his home and his correspondence",
            requirements=[
                "Everyone has right to respect for private and family life",
                "Everyone has right to respect for home and correspondence",
                "No interference by public authority except as in accordance with law",
                "Interference must be necessary in democratic society",
                "State may have positive obligations to ensure effective respect"
            ],
            keywords=["private life", "family life", "home", "correspondence", "interference"],
            citation="European Convention on Human Rights, Article 8"
        )
        
        # Article 10 - Freedom of expression
        elements["echr_art_10"] = LegalElement(
            id="echr_art_10",
            framework=LegalFramework.ECHR,
            article="Article 10",
            title="Freedom of expression",
            description="Everyone has the right to freedom of expression",
            requirements=[
                "Everyone has right to freedom of expression",
                "Right includes freedom to hold opinions and receive and impart information and ideas",
                "Exercise may be subject to formalities, conditions, restrictions or penalties as prescribed by law",
                "Restrictions must be necessary in democratic society"
            ],
            keywords=["freedom of expression", "opinions", "information", "restrictions", "democratic society"],
            citation="European Convention on Human Rights, Article 10",
            precedents=["Sunday Times v. United Kingdom (ECtHR)", "Handyside v. United Kingdom (ECtHR)"]
        )
        
        # Article 11 - Freedom of assembly and association
        elements["echr_art_11"] = LegalElement(
            id="echr_art_11",
            framework=LegalFramework.ECHR,
            article="Article 11",
            title="Freedom of assembly and association",
            description="Everyone has the right to freedom of peaceful assembly and to freedom of association with others",
            requirements=[
                "Everyone has right to freedom of peaceful assembly",
                "Everyone has right to freedom of association with others",
                "Right includes forming and joining trade unions for protection of interests",
                "Restrictions must be prescribed by law and necessary in democratic society",
                "Restrictions must be in interests of national security or public safety, prevention of disorder or crime, protection of health or morals or protection of others' rights and freedoms"
            ],
            keywords=["peaceful assembly", "freedom of association", "trade unions", "restrictions"],
            citation="European Convention on Human Rights, Article 11"
        )
        
        # Article 6 - Right to fair trial
        elements["echr_art_6"] = LegalElement(
            id="echr_art_6",
            framework=LegalFramework.ECHR,
            article="Article 6",
            title="Right to a fair trial",
            description="Everyone is entitled to a fair and public hearing within a reasonable time by an independent and impartial tribunal established by law",
            requirements=[
                "Everyone entitled to fair and public hearing within reasonable time",
                "Tribunal must be independent and impartial and established by law",
                "Judgment shall be pronounced publicly",
                "In criminal proceedings: presumption of innocence",
                "Minimum rights in criminal proceedings must be guaranteed"
            ],
            keywords=["fair trial", "independent tribunal", "reasonable time", "presumption of innocence"],
            citation="European Convention on Human Rights, Article 6",
            precedents=["Golder v. United Kingdom (ECtHR)", "Delcourt v. Belgium (ECtHR)"]
        )
        
        # Article 9 - Freedom of thought, conscience and religion
        elements["echr_art_9"] = LegalElement(
            id="echr_art_9",
            framework=LegalFramework.ECHR,
            article="Article 9",
            title="Freedom of thought, conscience and religion",
            description="Everyone has the right to freedom of thought, conscience and religion",
            requirements=[
                "Everyone has right to freedom of thought, conscience and religion",
                "Right includes freedom to change religion or belief",
                "Right includes freedom to manifest religion or belief in worship, teaching, practice and observance",
                "Freedom to manifest may be subject to limitations prescribed by law",
                "Limitations must be necessary in democratic society for protection of public safety, order, health or morals, or protection of others' rights and freedoms"
            ],
            keywords=["freedom of religion", "conscience", "belief", "worship", "manifestation"],
            citation="European Convention on Human Rights, Article 9"
        )
        
        # Article 14 - Prohibition of discrimination
        elements["echr_art_14"] = LegalElement(
            id="echr_art_14",
            framework=LegalFramework.ECHR,
            article="Article 14",
            title="Prohibition of discrimination",
            description="The enjoyment of the rights and freedoms set forth in this Convention shall be secured without discrimination",
            requirements=[
                "Enjoyment of Convention rights shall be secured without discrimination",
                "Discrimination prohibited on any ground such as sex, race, colour, language, religion, political or other opinion, national or social origin, association with national minority, property, birth or other status",
                "Different treatment requires objective and reasonable justification",
                "Different treatment must pursue legitimate aim and be proportionate"
            ],
            keywords=["discrimination", "equal treatment", "objective justification", "proportionality"],
            citation="European Convention on Human Rights, Article 14"
        )
        
        return elements
    
    def _load_achr_elements(self) -> Dict[str, LegalElement]:
        """Load American Convention on Human Rights elements."""
        elements = {}
        
        # Article 4 - Right to life
        elements["achr_art_4"] = LegalElement(
            id="achr_art_4",
            framework=LegalFramework.ACHR,
            article="Article 4",
            title="Right to life",
            description="Every person has the right to have his life respected",
            requirements=[
                "Every person has right to have life respected",
                "This right shall be protected by law from moment of conception",
                "No one shall be arbitrarily deprived of life",
                "Death penalty may not be imposed for political offenses"
            ],
            keywords=["right to life", "conception", "arbitrary deprivation", "political offenses"],
            citation="American Convention on Human Rights, Article 4"
        )
        
        # Article 5 - Right to humane treatment
        elements["achr_art_5"] = LegalElement(
            id="achr_art_5",
            framework=LegalFramework.ACHR,
            article="Article 5",
            title="Right to humane treatment",
            description="Every person has the right to have his physical, mental, and moral integrity respected",
            requirements=[
                "Every person has right to physical, mental, and moral integrity",
                "No one shall be subjected to torture or cruel, inhuman, or degrading punishment or treatment",
                "All persons deprived of liberty shall be treated with respect for inherent dignity",
                "Accused persons shall be segregated from convicted persons"
            ],
            keywords=["humane treatment", "integrity", "torture", "dignity", "liberty"],
            citation="American Convention on Human Rights, Article 5"
        )
        
        return elements
    
    def _load_achpr_elements(self) -> Dict[str, LegalElement]:
        """Load African Charter on Human and Peoples' Rights elements."""
        elements = {}
        
        # Article 4 - Right to life
        elements["achpr_art_4"] = LegalElement(
            id="achpr_art_4",
            framework=LegalFramework.ACHPR,
            article="Article 4",
            title="Right to life and integrity of person",
            description="Human beings are inviolable. Every human being shall be entitled to respect for his life and the integrity of his person",
            requirements=[
                "Human beings are inviolable",
                "Every human being entitled to respect for life and integrity",
                "No one may be arbitrarily deprived of this right"
            ],
            keywords=["inviolable", "life", "integrity", "arbitrary deprivation"],
            citation="African Charter on Human and Peoples' Rights, Article 4"
        )
        
        # Article 5 - Prohibition of torture
        elements["achpr_art_5"] = LegalElement(
            id="achpr_art_5",
            framework=LegalFramework.ACHPR,
            article="Article 5",
            title="Prohibition of torture and degrading treatment",
            description="Every individual shall have the right to the respect of the dignity inherent in a human being",
            requirements=[
                "Every individual has right to respect of inherent human dignity",
                "All forms of exploitation and degradation are prohibited",
                "Slavery, slave trade, torture, cruel inhuman or degrading punishment and treatment are prohibited"
            ],
            keywords=["dignity", "exploitation", "degradation", "slavery", "torture"],
            citation="African Charter on Human and Peoples' Rights, Article 5"
        )
        
        return elements
    
    def _load_udhr_elements(self) -> Dict[str, LegalElement]:
        """Load Universal Declaration of Human Rights elements."""
        elements = {}
        
        # Article 3 - Right to life
        elements["udhr_art_3"] = LegalElement(
            id="udhr_art_3",
            framework=LegalFramework.UDHR,
            article="Article 3",
            title="Right to life, liberty and security of person",
            description="Everyone has the right to life, liberty and security of person",
            requirements=[
                "Everyone has right to life",
                "Everyone has right to liberty",
                "Everyone has right to security of person"
            ],
            keywords=["life", "liberty", "security", "person"],
            citation="Universal Declaration of Human Rights, Article 3"
        )
        
        # Article 5 - Prohibition of torture
        elements["udhr_art_5"] = LegalElement(
            id="udhr_art_5",
            framework=LegalFramework.UDHR,
            article="Article 5",
            title="Prohibition of torture",
            description="No one shall be subjected to torture or to cruel, inhuman or degrading treatment or punishment",
            requirements=[
                "No one shall be subjected to torture",
                "No one shall be subjected to cruel treatment or punishment",
                "No one shall be subjected to inhuman treatment or punishment",
                "No one shall be subjected to degrading treatment or punishment"
            ],
            keywords=["torture", "cruel", "inhuman", "degrading"],
            citation="Universal Declaration of Human Rights, Article 5"
        )
        
        return elements
    
    def analyze(self, evidence: List[Evidence]) -> FrameworkAnalysis:
        """
        Analyze evidence against human rights framework elements.
        
        Args:
            evidence: List of evidence to analyze
            
        Returns:
            FrameworkAnalysis: Complete human rights analysis
        """
        logger.info(f"Analyzing {len(evidence)} pieces of evidence against {self.framework.value}")
        
        # Analyze each legal element
        element_satisfactions = []
        violations_identified = []
        
        for element_id, legal_element in self.legal_elements.items():
            satisfaction = self.element_analyzer.analyze_element_satisfaction(
                evidence, legal_element
            )
            element_satisfactions.append(satisfaction)
            
            # Check if this represents a potential violation
            if satisfaction.status in [ElementStatus.SATISFIED, ElementStatus.PARTIALLY_SATISFIED]:
                if satisfaction.confidence >= 0.6:
                    violations_identified.append(f"{legal_element.article}: {legal_element.title}")
        
        # Generate gap analysis
        gap_analysis = self._generate_gap_analysis(element_satisfactions)
        
        # Calculate overall confidence
        satisfied_elements = [s for s in element_satisfactions 
                            if s.status in [ElementStatus.SATISFIED, ElementStatus.PARTIALLY_SATISFIED]]
        if satisfied_elements:
            overall_confidence = sum(s.confidence for s in satisfied_elements) / len(satisfied_elements)
        else:
            overall_confidence = 0.0
        
        # Generate analysis summary
        summary = self._generate_summary(element_satisfactions, violations_identified, overall_confidence)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(element_satisfactions, gap_analysis)
        
        analysis = FrameworkAnalysis(
            framework=self.framework,
            evidence_count=len(evidence),
            elements_analyzed=[e.id for e in self.legal_elements.values()],
            element_satisfactions=element_satisfactions,
            overall_confidence=overall_confidence,
            gap_analysis=gap_analysis,
            summary=summary,
            violations_identified=violations_identified,
            recommendations=recommendations
        )
        
        logger.info(f"{self.framework.value} analysis completed. Found {len(violations_identified)} potential violations")
        return analysis
    
    def _generate_gap_analysis(self, satisfactions: List[ElementSatisfaction]) -> GapAnalysis:
        """Generate gap analysis for human rights elements."""
        missing_elements = []
        weak_elements = []
        evidence_needs = {}
        recommendations = []
        
        for satisfaction in satisfactions:
            if satisfaction.status == ElementStatus.NOT_SATISFIED:
                missing_elements.append(satisfaction.element_id)
            elif satisfaction.status == ElementStatus.INSUFFICIENT_EVIDENCE or satisfaction.confidence < 0.5:
                weak_elements.append(satisfaction.element_id)
            
            if satisfaction.gaps:
                evidence_needs[satisfaction.element_id] = satisfaction.gaps
        
        # Generate human rights specific recommendations
        if missing_elements:
            recommendations.append("Document state responsibility for human rights violations")
        if weak_elements:
            recommendations.append("Strengthen evidence of state knowledge and acquiescence")
        
        # Evidence-specific recommendations
        if any("victim" in gap.lower() for gaps in evidence_needs.values() for gap in gaps):
            recommendations.append("Obtain detailed victim testimony and impact statements")
        if any("state" in gap.lower() for gaps in evidence_needs.values() for gap in gaps):
            recommendations.append("Document state involvement or failure to prevent/investigate")
        if any("remedy" in gap.lower() for gaps in evidence_needs.values() for gap in gaps):
            recommendations.append("Assess availability and effectiveness of domestic remedies")
        
        # Calculate priority score
        total_elements = len(satisfactions)
        problematic_elements = len(missing_elements) + len(weak_elements)
        priority_score = problematic_elements / total_elements if total_elements > 0 else 0.0
        
        return GapAnalysis(
            framework=self.framework,
            missing_elements=missing_elements,
            weak_elements=weak_elements,
            evidence_needs=evidence_needs,
            recommendations=recommendations,
            priority_score=priority_score
        )
    
    def _generate_summary(self, satisfactions: List[ElementSatisfaction], 
                         violations: List[str], overall_confidence: float) -> str:
        """Generate analysis summary."""
        framework_name = self.framework.value.upper().replace("_", " ")
        total_elements = len(satisfactions)
        satisfied = len([s for s in satisfactions if s.status == ElementStatus.SATISFIED])
        partial = len([s for s in satisfactions if s.status == ElementStatus.PARTIALLY_SATISFIED])
        
        summary = f"{framework_name} analysis of {total_elements} human rights elements: "
        summary += f"{satisfied} fully satisfied, {partial} partially satisfied. "
        summary += f"Overall confidence: {overall_confidence:.1%}. "
        
        if violations:
            summary += f"Identified {len(violations)} potential human rights violations including "
            summary += ", ".join([v.split(":")[0] for v in violations[:3]])
            if len(violations) > 3:
                summary += f" and {len(violations) - 3} others"
            summary += "."
        else:
            summary += "No clear human rights violations identified based on available evidence."
        
        return summary
    
    def _generate_recommendations(self, satisfactions: List[ElementSatisfaction],
                                gap_analysis: GapAnalysis) -> List[str]:
        """Generate recommendations for human rights analysis."""
        recommendations = []
        
        # Framework-specific recommendations
        if self.framework == LegalFramework.ICCPR:
            recommendations.extend([
                "Submit individual communication to Human Rights Committee if domestic remedies exhausted",
                "Document state party obligations under ICCPR",
                "Assess potential derogation claims by state"
            ])
        elif self.framework == LegalFramework.ECHR:
            recommendations.extend([
                "Consider application to European Court of Human Rights",
                "Assess margin of appreciation doctrine applicability",
                "Evaluate positive obligations of state"
            ])
        elif self.framework in [LegalFramework.ACHR, LegalFramework.ACHPR]:
            recommendations.extend([
                "Consider petition to regional human rights commission/court",
                "Document exhaustion of domestic remedies",
                "Assess state compliance with regional obligations"
            ])
        
        # Evidence collection recommendations
        recommendations.extend(gap_analysis.recommendations)
        
        # General human rights recommendations
        recommendations.extend([
            "Document pattern of violations and state responsibility",
            "Assess individual and structural remedies needed",
            "Consider interim measures if continuing violations"
        ])
        
        return recommendations


def analyze_human_rights_violations(evidence: List[Evidence], 
                                   framework: LegalFramework) -> HumanRightsAnalysis:
    """
    Convenience function to analyze human rights violations.
    
    Args:
        evidence: List of evidence to analyze
        framework: Human rights framework to analyze against
        
    Returns:
        HumanRightsAnalysis: Detailed human rights analysis
    """
    analyzer = HumanRightsAnalyzer(framework)
    framework_analysis = analyzer.analyze(evidence)
    
    # Determine if this is universal or regional instrument
    universal_instruments = [LegalFramework.UDHR, LegalFramework.ICCPR]
    regional_instruments = [LegalFramework.ECHR, LegalFramework.ACHR, LegalFramework.ACHPR]
    
    universal_findings = []
    regional_findings = []
    
    for satisfaction in framework_analysis.element_satisfactions:
        if satisfaction.status in [ElementStatus.SATISFIED, ElementStatus.PARTIALLY_SATISFIED]:
            finding_data = {
                "element": satisfaction.element_id,
                "article": next((e.article for e in analyzer.legal_elements.values() 
                               if e.id == satisfaction.element_id), "Unknown"),
                "confidence": satisfaction.confidence,
                "reasoning": satisfaction.reasoning
            }
            
            if framework in universal_instruments:
                universal_findings.append(finding_data)
            else:
                regional_findings.append(finding_data)
    
    # Derogation analysis (simplified)
    derogation_analysis = {
        "emergency_declared": "requires_assessment",
        "derogation_measures_proportionate": "requires_assessment",
        "non_derogable_rights_affected": "requires_assessment"
    }
    
    # State responsibility assessment
    state_responsibility = {
        "attribution_to_state": "requires_detailed_analysis",
        "breach_of_obligation": bool(framework_analysis.violations_identified),
        "circumstances_precluding_wrongfulness": "requires_assessment",
        "reparations_owed": "requires_assessment"
    }
    
    # Individual remedies
    individual_remedies = [
        "Compensation for material and moral damages",
        "Restitution where possible",
        "Rehabilitation services",
        "Guarantees of non-repetition"
    ]
    
    # Systemic issues
    systemic_issues = []
    if len(framework_analysis.violations_identified) > 3:
        systemic_issues.append("Pattern of violations suggests systemic issues")
    if framework_analysis.overall_confidence > 0.7:
        systemic_issues.append("High confidence violations indicate institutional failures")
    
    return HumanRightsAnalysis(
        instruments_analyzed=[HumanRightsInstrument(framework.value)],
        universal_rights_findings=universal_findings,
        regional_rights_findings=regional_findings,
        derogation_analysis=derogation_analysis,
        state_responsibility_assessment=state_responsibility,
        individual_remedies=individual_remedies,
        systemic_issues_identified=systemic_issues,
        overall_human_rights_assessment=framework_analysis.summary
    )