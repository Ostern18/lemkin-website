"""
Geneva Conventions and Additional Protocols analysis module.

This module provides detailed analysis of evidence against International Humanitarian
Law (IHL) violations under the Geneva Conventions of 1949 and their Additional Protocols.
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


class GenevaConvention(str, Enum):
    """Geneva Conventions and Additional Protocols."""
    GC_I = "gc_i"  # Geneva Convention I - Wounded and Sick in Armed Forces
    GC_II = "gc_ii"  # Geneva Convention II - Wounded, Sick and Shipwrecked at Sea
    GC_III = "gc_iii"  # Geneva Convention III - Prisoners of War
    GC_IV = "gc_iv"  # Geneva Convention IV - Civilian Persons
    AP_I = "ap_i"  # Additional Protocol I - International Armed Conflicts
    AP_II = "ap_ii"  # Additional Protocol II - Non-International Armed Conflicts


class ProtectedPersonCategory(str, Enum):
    """Categories of persons protected under IHL."""
    WOUNDED_SICK = "wounded_sick"
    PRISONERS_OF_WAR = "prisoners_of_war"
    CIVILIANS = "civilians"
    MEDICAL_PERSONNEL = "medical_personnel"
    RELIGIOUS_PERSONNEL = "religious_personnel"
    HUMANITARIAN_PERSONNEL = "humanitarian_personnel"
    PEACEKEEPERS = "peacekeepers"


class IHLViolationType(str, Enum):
    """Types of IHL violations."""
    GRAVE_BREACH = "grave_breach"
    SERIOUS_VIOLATION = "serious_violation"
    OTHER_VIOLATION = "other_violation"


class IHLViolation(BaseModel):
    """International Humanitarian Law violation."""
    convention: GenevaConvention
    article: str
    title: str
    description: str
    violation_type: IHLViolationType
    protected_persons: List[ProtectedPersonCategory]
    elements: List[str]
    applicable_conflicts: List[str]  # ["international", "non_international"]


class ProtectedPersonViolation(BaseModel):
    """Violation against protected persons under IHL."""
    protected_category: ProtectedPersonCategory
    convention: GenevaConvention
    article: str
    violation_description: str
    elements_violated: List[str]
    severity: str  # "grave_breach", "serious_violation", "other"


class GenevaAnalysis(BaseModel):
    """Analysis results for Geneva Conventions violations."""
    conventions_analyzed: List[GenevaConvention]
    conflict_classification: Dict[str, str]
    grave_breaches_found: List[Dict]
    serious_violations_found: List[Dict]
    protected_persons_analysis: List[ProtectedPersonViolation]
    medical_facilities_analysis: Dict[str, Any]
    civilian_objects_analysis: Dict[str, Any]
    proportionality_analysis: Dict[str, Any]
    overall_ihl_assessment: str


class GenevaAnalyzer:
    """
    Analyzer for Geneva Conventions and IHL violations.
    
    This class implements detailed analysis of evidence against International
    Humanitarian Law provisions in the Geneva Conventions and Additional Protocols.
    """
    
    def __init__(self):
        """Initialize the Geneva Conventions analyzer."""
        self.element_analyzer = ElementAnalyzer()
        self.legal_elements = self._load_geneva_elements()
        logger.info("GenevaAnalyzer initialized with IHL elements")
    
    def _load_geneva_elements(self) -> Dict[str, LegalElement]:
        """Load all Geneva Conventions legal elements."""
        elements = {}
        
        # Load elements from each Convention and Protocol
        elements.update(self._load_gc_i_elements())
        elements.update(self._load_gc_ii_elements())
        elements.update(self._load_gc_iii_elements())
        elements.update(self._load_gc_iv_elements())
        elements.update(self._load_ap_i_elements())
        elements.update(self._load_ap_ii_elements())
        
        logger.info(f"Loaded {len(elements)} Geneva Conventions legal elements")
        return elements
    
    def _load_gc_i_elements(self) -> Dict[str, LegalElement]:
        """Load Geneva Convention I elements (Wounded and Sick in Armed Forces)."""
        elements = {}
        
        # Article 12 - Protection of wounded and sick
        elements["gc_i_art_12"] = LegalElement(
            id="gc_i_art_12",
            framework=LegalFramework.GENEVA_CONVENTIONS,
            article="GC I, Article 12",
            title="Protection and care of wounded and sick",
            description="Members of the armed forces who are wounded or sick shall be respected and protected in all circumstances",
            requirements=[
                "Persons are wounded or sick members of armed forces",
                "They shall be respected and protected in all circumstances",
                "They shall be treated humanely without adverse distinction",
                "Any attempt on their lives or violence to their persons is prohibited"
            ],
            keywords=["wounded", "sick", "armed forces", "protection", "respect", "humane treatment"],
            citation="Geneva Convention I, Article 12",
            precedents=["ICRC Commentary on Geneva Convention I"]
        )
        
        # Article 24 - Protection of medical personnel
        elements["gc_i_art_24"] = LegalElement(
            id="gc_i_art_24",
            framework=LegalFramework.GENEVA_CONVENTIONS,
            article="GC I, Article 24",
            title="Protection of medical personnel",
            description="Medical personnel exclusively engaged in the search for, or the collection, transport or treatment of the wounded or sick shall be respected and protected",
            requirements=[
                "Personnel are exclusively engaged in medical duties",
                "Medical duties include search, collection, transport or treatment of wounded or sick",
                "They shall be respected and protected in all circumstances",
                "In no circumstances shall they be attacked"
            ],
            keywords=["medical personnel", "protection", "respect", "medical duties", "wounded", "sick"],
            citation="Geneva Convention I, Article 24"
        )
        
        # Article 35 - Medical transports
        elements["gc_i_art_35"] = LegalElement(
            id="gc_i_art_35",
            framework=LegalFramework.GENEVA_CONVENTIONS,
            article="GC I, Article 35",
            title="Protection of medical transports",
            description="Transports of wounded and sick or of medical equipment shall be respected and protected",
            requirements=[
                "Transports carry wounded and sick or medical equipment",
                "They shall be respected and protected",
                "They may not be attacked",
                "They shall display the distinctive emblem"
            ],
            keywords=["medical transports", "wounded", "sick", "medical equipment", "protection", "distinctive emblem"],
            citation="Geneva Convention I, Article 35"
        )
        
        # Article 15 - Search for casualties  
        elements["gc_i_art_15"] = LegalElement(
            id="gc_i_art_15",
            framework=LegalFramework.GENEVA_CONVENTIONS,
            article="GC I, Article 15",
            title="Search for casualties and collection of the dead",
            description="Parties to conflict shall search for the wounded, sick and dead and ensure their protection",
            requirements=[
                "After each engagement, parties shall search for wounded, sick and dead",
                "Measures shall be taken to prevent bodies being despoiled",
                "Local agreements may be concluded for removal or exchange of wounded",
                "Dead shall be honorably interred and graves respected"
            ],
            keywords=["search", "casualties", "dead", "collection", "burial", "graves"],
            citation="Geneva Convention I, Article 15"
        )
        
        return elements
    
    def _load_gc_ii_elements(self) -> Dict[str, LegalElement]:
        """Load Geneva Convention II elements (Wounded, Sick and Shipwrecked at Sea).""" 
        elements = {}
        
        # Article 12 - Protection at sea
        elements["gc_ii_art_12"] = LegalElement(
            id="gc_ii_art_12",
            framework=LegalFramework.GENEVA_CONVENTIONS,
            article="GC II, Article 12",
            title="Protection of wounded, sick and shipwrecked at sea",
            description="Members of armed forces wounded, sick or shipwrecked at sea shall be respected and protected",
            requirements=[
                "Persons are wounded, sick or shipwrecked members of armed forces at sea",
                "They shall be respected and protected in all circumstances", 
                "They shall be treated humanely without adverse distinction",
                "Any attempt on their lives or violence is prohibited"
            ],
            keywords=["wounded", "sick", "shipwrecked", "sea", "naval", "protection"],
            citation="Geneva Convention II, Article 12"
        )
        
        return elements
    
    def _load_gc_iii_elements(self) -> Dict[str, LegalElement]:
        """Load Geneva Convention III elements (Prisoners of War)."""
        elements = {}
        
        # Article 13 - Humane treatment of POWs
        elements["gc_iii_art_13"] = LegalElement(
            id="gc_iii_art_13",
            framework=LegalFramework.GENEVA_CONVENTIONS,
            article="GC III, Article 13", 
            title="Humane treatment of prisoners of war",
            description="Prisoners of war must at all times be humanely treated",
            requirements=[
                "Persons are prisoners of war",
                "They must be treated humanely at all times",
                "No violence to life and person is permitted",
                "No cruel treatment and torture is permitted",
                "No outrages upon personal dignity are permitted"
            ],
            keywords=["prisoners of war", "POW", "humane treatment", "torture", "dignity"],
            citation="Geneva Convention III, Article 13"
        )
        
        # Article 17 - Interrogation of POWs
        elements["gc_iii_art_17"] = LegalElement(
            id="gc_iii_art_17",
            framework=LegalFramework.GENEVA_CONVENTIONS,
            article="GC III, Article 17",
            title="Interrogation of prisoners of war",
            description="POWs may only be required to give name, rank, date of birth and army number",
            requirements=[
                "Person is a prisoner of war",
                "When questioned, may only be required to give: surname, first names, rank, date of birth, army number",
                "No physical or mental torture or coercion to secure information",
                "POWs who refuse to answer may not be threatened, insulted, or exposed to unpleasant treatment"
            ],
            keywords=["interrogation", "questioning", "torture", "coercion", "information"],
            citation="Geneva Convention III, Article 17"
        )
        
        # Article 130 - Grave breaches
        elements["gc_iii_art_130"] = LegalElement(
            id="gc_iii_art_130",
            framework=LegalFramework.GENEVA_CONVENTIONS,
            article="GC III, Article 130",
            title="Grave breaches against prisoners of war",
            description="Grave breaches include wilful killing, torture, inhuman treatment of POWs",
            requirements=[
                "Acts committed against persons protected by this Convention",
                "Acts include: wilful killing, torture or inhuman treatment, wilfully causing great suffering or serious injury",
                "Acts are committed wilfully",
                "Victims are protected persons under the Convention"
            ],
            keywords=["grave breaches", "wilful killing", "torture", "inhuman treatment", "serious injury"],
            citation="Geneva Convention III, Article 130"
        )
        
        return elements
    
    def _load_gc_iv_elements(self) -> Dict[str, LegalElement]:
        """Load Geneva Convention IV elements (Protection of Civilian Persons)."""
        elements = {}
        
        # Article 27 - Treatment of protected persons
        elements["gc_iv_art_27"] = LegalElement(
            id="gc_iv_art_27",
            framework=LegalFramework.GENEVA_CONVENTIONS,
            article="GC IV, Article 27",
            title="Treatment of protected persons",
            description="Protected persons are entitled to respect for their persons, honour, family rights, religious convictions and practices",
            requirements=[
                "Persons are protected under this Convention",
                "They are entitled to respect for their persons, honour, family rights",
                "They shall be treated humanely at all times",
                "They shall be protected against violence, threats, insults and public curiosity",
                "Women shall be especially protected against rape, enforced prostitution and indecent assault"
            ],
            keywords=["protected persons", "civilians", "humane treatment", "dignity", "rape", "assault"],
            citation="Geneva Convention IV, Article 27"
        )
        
        # Article 33 - Individual responsibility and collective penalties
        elements["gc_iv_art_33"] = LegalElement(
            id="gc_iv_art_33",
            framework=LegalFramework.GENEVA_CONVENTIONS,
            article="GC IV, Article 33",
            title="Individual responsibility, collective penalties, pillage and reprisals",
            description="No protected person may be punished for an offence he or she has not personally committed",
            requirements=[
                "Persons are protected under this Convention",
                "No protected person may be punished for offences not personally committed",
                "Collective penalties are prohibited",
                "Pillage is prohibited", 
                "Reprisals against protected persons and their property are prohibited"
            ],
            keywords=["collective punishment", "individual responsibility", "pillage", "reprisals"],
            citation="Geneva Convention IV, Article 33"
        )
        
        # Article 49 - Deportations, transfers, evacuations
        elements["gc_iv_art_49"] = LegalElement(
            id="gc_iv_art_49",
            framework=LegalFramework.GENEVA_CONVENTIONS,
            article="GC IV, Article 49",
            title="Deportations, transfers, evacuations",
            description="Individual or mass forcible transfers and deportations of protected persons are prohibited",
            requirements=[
                "Protected persons are subject to transfer or deportation",
                "Individual or mass forcible transfers are prohibited",
                "Deportations from occupied territory to territory of Occupying Power or any other country are prohibited",
                "Evacuation may only be undertaken for security or imperative military reasons",
                "Population shall be transferred back as soon as hostilities have ceased"
            ],
            keywords=["deportation", "transfer", "evacuation", "occupied territory", "forcible"],
            citation="Geneva Convention IV, Article 49"
        )
        
        # Article 147 - Grave breaches
        elements["gc_iv_art_147"] = LegalElement(
            id="gc_iv_art_147",
            framework=LegalFramework.GENEVA_CONVENTIONS,
            article="GC IV, Article 147",
            title="Grave breaches against protected persons",
            description="Grave breaches include wilful killing, torture, inhuman treatment, unlawful deportation",
            requirements=[
                "Acts committed against persons protected by this Convention",
                "Acts include: wilful killing, torture or inhuman treatment, wilfully causing great suffering",
                "Also: unlawful deportation or transfer, unlawful confinement, taking of hostages",
                "Acts are committed wilfully",
                "Victims are protected persons under the Convention"
            ],
            keywords=["grave breaches", "wilful killing", "torture", "deportation", "hostages"],
            citation="Geneva Convention IV, Article 147",
            precedents=["Prosecutor v. Tadić (ICTY)", "Prosecutor v. Krstić (ICTY)"]
        )
        
        # Article 53 - Destruction of property
        elements["gc_iv_art_53"] = LegalElement(
            id="gc_iv_art_53",
            framework=LegalFramework.GENEVA_CONVENTIONS,
            article="GC IV, Article 53",
            title="Prohibition of destruction and appropriation of property",
            description="Any destruction by the Occupying Power of real or personal property belonging individually or collectively to private persons, or to the State, is prohibited",
            requirements=[
                "There is an occupying power in control of territory",
                "Property belongs to private persons or the State",
                "Destruction or appropriation is carried out by occupying power",
                "Destruction is not absolutely necessary for military operations"
            ],
            keywords=["destruction", "property", "occupying power", "private persons", "military necessity"],
            citation="Geneva Convention IV, Article 53"
        )
        
        # Article 55 - Food and medical supplies for population
        elements["gc_iv_art_55"] = LegalElement(
            id="gc_iv_art_55",
            framework=LegalFramework.GENEVA_CONVENTIONS,
            article="GC IV, Article 55",
            title="Food and medical supplies for population",
            description="To the fullest extent of the means available to it, the Occupying Power has the duty of ensuring the food and medical supplies of the population",
            requirements=[
                "Territory is under occupation",
                "Occupying power must ensure food and medical supplies of population",
                "This duty applies to the fullest extent of means available",
                "Occupying power should bring in necessary supplies if local resources are inadequate"
            ],
            keywords=["food", "medical supplies", "population", "occupying power", "duty"],
            citation="Geneva Convention IV, Article 55"
        )
        
        # Article 64 - Penal laws in occupied territory
        elements["gc_iv_art_64"] = LegalElement(
            id="gc_iv_art_64",
            framework=LegalFramework.GENEVA_CONVENTIONS,
            article="GC IV, Article 64",
            title="Penal laws in occupied territory",
            description="The penal laws of the occupied territory shall remain in force, with the exception that they may be repealed or suspended by the Occupying Power",
            requirements=[
                "Territory is under occupation",
                "Penal laws of occupied territory remain in force as a general rule",
                "Occupying Power may only modify laws where absolutely necessary",
                "Occupying Power may publish and enforce necessary regulations for security and administration"
            ],
            keywords=["penal laws", "occupied territory", "occupying power", "regulations"],
            citation="Geneva Convention IV, Article 64"
        )
        
        return elements
    
    def _load_ap_i_elements(self) -> Dict[str, LegalElement]:
        """Load Additional Protocol I elements (International Armed Conflicts)."""
        elements = {}
        
        # Article 48 - Basic rule of distinction
        elements["ap_i_art_48"] = LegalElement(
            id="ap_i_art_48",
            framework=LegalFramework.GENEVA_CONVENTIONS,
            article="AP I, Article 48",
            title="Basic rule of distinction",
            description="Parties to conflict shall distinguish between civilian population and combatants",
            requirements=[
                "Parties to the conflict shall at all times distinguish between civilian population and combatants",
                "Parties shall distinguish between civilian objects and military objectives",
                "Operations may only be directed against military objectives"
            ],
            keywords=["distinction", "civilians", "combatants", "military objectives"],
            citation="Additional Protocol I, Article 48"
        )
        
        # Article 51 - Protection of civilian population
        elements["ap_i_art_51"] = LegalElement(
            id="ap_i_art_51",
            framework=LegalFramework.GENEVA_CONVENTIONS,
            article="AP I, Article 51",
            title="Protection of the civilian population",
            description="The civilian population and individual civilians shall enjoy general protection against dangers arising from military operations",
            requirements=[
                "The civilian population shall not be the object of attack",
                "Acts or threats of violence primarily to spread terror are prohibited",
                "Indiscriminate attacks are prohibited",
                "Attacks expected to cause excessive civilian harm are prohibited"
            ],
            keywords=["civilian population", "indiscriminate attacks", "terror", "proportionality"],
            citation="Additional Protocol I, Article 51"
        )
        
        # Article 52 - Protection of civilian objects
        elements["ap_i_art_52"] = LegalElement(
            id="ap_i_art_52",
            framework=LegalFramework.GENEVA_CONVENTIONS,
            article="AP I, Article 52",
            title="General protection of civilian objects",
            description="Civilian objects shall not be the object of attack or of reprisals",
            requirements=[
                "Civilian objects are all objects which are not military objectives",
                "Civilian objects shall not be the object of attack",
                "Reprisals against civilian objects are prohibited",
                "Military objectives are limited to objects which make effective contribution to military action"
            ],
            keywords=["civilian objects", "military objectives", "reprisals", "attack"],
            citation="Additional Protocol I, Article 52"
        )
        
        return elements
    
    def _load_ap_ii_elements(self) -> Dict[str, LegalElement]:
        """Load Additional Protocol II elements (Non-International Armed Conflicts)."""
        elements = {}
        
        # Article 4 - Fundamental guarantees
        elements["ap_ii_art_4"] = LegalElement(
            id="ap_ii_art_4",
            framework=LegalFramework.GENEVA_CONVENTIONS,
            article="AP II, Article 4", 
            title="Fundamental guarantees",
            description="All persons who do not take direct part in hostilities enjoy fundamental guarantees",
            requirements=[
                "Persons do not or no longer take direct part in hostilities",
                "They are treated humanely without adverse distinction", 
                "Violence to life, health and physical or mental well-being is prohibited",
                "Collective punishments, hostage-taking, acts of terrorism are prohibited",
                "Outrages upon personal dignity are prohibited"
            ],
            keywords=["fundamental guarantees", "humane treatment", "hostages", "terrorism", "dignity"],
            citation="Additional Protocol II, Article 4"
        )
        
        # Article 13 - Protection of civilian population
        elements["ap_ii_art_13"] = LegalElement(
            id="ap_ii_art_13",
            framework=LegalFramework.GENEVA_CONVENTIONS,
            article="AP II, Article 13",
            title="Protection of the civilian population",
            description="The civilian population shall enjoy general protection against the dangers arising from military operations",
            requirements=[
                "The civilian population shall not be the object of attack",
                "Acts or threats of violence primarily to spread terror among civilian population are prohibited",
                "Civilians shall enjoy protection unless and for such time as they take direct part in hostilities"
            ],
            keywords=["civilian population", "terror", "direct participation", "hostilities"],
            citation="Additional Protocol II, Article 13"
        )
        
        return elements
    
    def analyze(self, evidence: List[Evidence]) -> FrameworkAnalysis:
        """
        Analyze evidence against Geneva Conventions elements.
        
        Args:
            evidence: List of evidence to analyze
            
        Returns:
            FrameworkAnalysis: Complete Geneva Conventions analysis
        """
        logger.info(f"Analyzing {len(evidence)} pieces of evidence against Geneva Conventions")
        
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
            framework=LegalFramework.GENEVA_CONVENTIONS,
            evidence_count=len(evidence),
            elements_analyzed=[e.id for e in self.legal_elements.values()],
            element_satisfactions=element_satisfactions,
            overall_confidence=overall_confidence,
            gap_analysis=gap_analysis,
            summary=summary,
            violations_identified=violations_identified,
            recommendations=recommendations
        )
        
        logger.info(f"Geneva Conventions analysis completed. Found {len(violations_identified)} potential violations")
        return analysis
    
    def _generate_gap_analysis(self, satisfactions: List[ElementSatisfaction]) -> GapAnalysis:
        """Generate gap analysis for Geneva Conventions elements."""
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
        
        # Generate specific IHL recommendations
        if missing_elements:
            recommendations.append("Document armed conflict classification (international vs non-international)")
        if weak_elements:
            recommendations.append("Gather additional evidence on protected person status")
        
        # Evidence-specific recommendations
        if any("medical" in gap.lower() for gaps in evidence_needs.values() for gap in gaps):
            recommendations.append("Document attacks on medical facilities and personnel")
        if any("civilian" in gap.lower() for gaps in evidence_needs.values() for gap in gaps):
            recommendations.append("Establish civilian status of victims and objects")
        if any("military" in gap.lower() for gaps in evidence_needs.values() for gap in gaps):
            recommendations.append("Analyze military necessity and proportionality")
        
        # Calculate priority score
        total_elements = len(satisfactions)
        problematic_elements = len(missing_elements) + len(weak_elements)
        priority_score = problematic_elements / total_elements if total_elements > 0 else 0.0
        
        return GapAnalysis(
            framework=LegalFramework.GENEVA_CONVENTIONS,
            missing_elements=missing_elements,
            weak_elements=weak_elements,
            evidence_needs=evidence_needs,
            recommendations=recommendations,
            priority_score=priority_score
        )
    
    def _generate_summary(self, satisfactions: List[ElementSatisfaction], 
                         violations: List[str], overall_confidence: float) -> str:
        """Generate analysis summary."""
        total_elements = len(satisfactions)
        satisfied = len([s for s in satisfactions if s.status == ElementStatus.SATISFIED])
        partial = len([s for s in satisfactions if s.status == ElementStatus.PARTIALLY_SATISFIED])
        
        summary = f"Geneva Conventions analysis of {total_elements} IHL elements: "
        summary += f"{satisfied} fully satisfied, {partial} partially satisfied. "
        summary += f"Overall confidence: {overall_confidence:.1%}. "
        
        if violations:
            summary += f"Identified {len(violations)} potential IHL violations including "
            
            # Categorize violations
            grave_breaches = [v for v in violations if "grave breach" in v.lower()]
            if grave_breaches:
                summary += f"{len(grave_breaches)} grave breaches, "
            
            summary += ", ".join([v.split(":")[0] for v in violations[:3]])
            if len(violations) > 3:
                summary += f" and {len(violations) - 3} others"
            summary += "."
        else:
            summary += "No clear IHL violations identified based on available evidence."
        
        return summary
    
    def _generate_recommendations(self, satisfactions: List[ElementSatisfaction],
                                gap_analysis: GapAnalysis) -> List[str]:
        """Generate recommendations for Geneva Conventions analysis."""
        recommendations = []
        
        # Check for grave breaches
        grave_breach_elements = [s for s in satisfactions 
                               if "grave" in s.element_id and s.status == ElementStatus.SATISFIED]
        if grave_breach_elements:
            recommendations.append("Consider universal jurisdiction prosecution for grave breaches")
        
        # Evidence collection recommendations
        recommendations.extend(gap_analysis.recommendations)
        
        # IHL-specific recommendations
        recommendations.extend([
            "Classify the nature of armed conflict (international/non-international)",
            "Document protected person status under relevant Convention",
            "Assess military necessity and proportionality of attacks",
            "Verify compliance with precautionary obligations",
            "Document respect for distinctive emblems (Red Cross, etc.)"
        ])
        
        return recommendations


def assess_geneva_violations(evidence: List[Evidence]) -> GenevaAnalysis:
    """
    Convenience function to assess Geneva Conventions violations.
    
    Args:
        evidence: List of evidence to analyze
        
    Returns:
        GenevaAnalysis: Detailed Geneva Conventions analysis
    """
    analyzer = GenevaAnalyzer()
    framework_analysis = analyzer.analyze(evidence)
    
    # Classify conflict type based on evidence
    conflict_classification = {
        "type": "to_be_determined",  # Would require specific analysis
        "international": "unclear",
        "non_international": "unclear",
        "threshold_met": "requires_assessment"
    }
    
    # Categorize violations by type
    grave_breaches = []
    serious_violations = []
    protected_persons_violations = []
    
    for satisfaction in framework_analysis.element_satisfactions:
        if satisfaction.status in [ElementStatus.SATISFIED, ElementStatus.PARTIALLY_SATISFIED]:
            violation_data = {
                "element": satisfaction.element_id,
                "article": next((e.article for e in analyzer.legal_elements.values() 
                               if e.id == satisfaction.element_id), "Unknown"),
                "confidence": satisfaction.confidence,
                "reasoning": satisfaction.reasoning
            }
            
            if "grave" in satisfaction.element_id or "130" in satisfaction.element_id or "147" in satisfaction.element_id:
                grave_breaches.append(violation_data)
            else:
                serious_violations.append(violation_data)
            
            # Check if involves protected persons
            protected_keywords = ["prisoner", "civilian", "wounded", "sick", "medical"]
            if any(keyword in satisfaction.element_id for keyword in protected_keywords):
                protected_persons_violations.append(ProtectedPersonViolation(
                    protected_category=ProtectedPersonCategory.CIVILIANS,  # Simplified
                    convention=GenevaConvention.GC_IV,  # Simplified
                    article=violation_data["article"],
                    violation_description=satisfaction.reasoning,
                    elements_violated=[satisfaction.element_id],
                    severity="grave_breach" if violation_data in grave_breaches else "serious_violation"
                ))
    
    # Medical facilities analysis
    medical_analysis = {
        "facilities_attacked": "requires_assessment",
        "distinctive_emblems_respected": "requires_assessment", 
        "medical_personnel_protected": "requires_assessment"
    }
    
    # Civilian objects analysis
    civilian_objects_analysis = {
        "civilian_objects_attacked": "requires_assessment",
        "military_objectives_distinguished": "requires_assessment",
        "precautionary_measures_taken": "requires_assessment"
    }
    
    # Proportionality analysis
    proportionality_analysis = {
        "excessive_civilian_harm": "requires_assessment",
        "military_advantage_anticipated": "requires_assessment",
        "feasible_precautions_taken": "requires_assessment"
    }
    
    return GenevaAnalysis(
        conventions_analyzed=[GenevaConvention.GC_I, GenevaConvention.GC_II, 
                             GenevaConvention.GC_III, GenevaConvention.GC_IV,
                             GenevaConvention.AP_I, GenevaConvention.AP_II],
        conflict_classification=conflict_classification,
        grave_breaches_found=grave_breaches,
        serious_violations_found=serious_violations,
        protected_persons_analysis=protected_persons_violations,
        medical_facilities_analysis=medical_analysis,
        civilian_objects_analysis=civilian_objects_analysis,
        proportionality_analysis=proportionality_analysis,
        overall_ihl_assessment=framework_analysis.summary
    )