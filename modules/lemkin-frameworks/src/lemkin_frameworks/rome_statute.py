"""
Rome Statute of the International Criminal Court analysis module.

This module provides detailed analysis of evidence against the Rome Statute elements
for war crimes, crimes against humanity, and genocide.
"""

from datetime import datetime
from enum import Enum
from typing import List, Dict, Optional, Set, Any
from uuid import UUID

from pydantic import BaseModel
from loguru import logger

from .core import (
    Evidence, LegalElement, ElementSatisfaction, FrameworkAnalysis, 
    GapAnalysis, LegalFramework, ElementStatus, ConfidenceLevel
)
from .element_analyzer import ElementAnalyzer, ConfidenceScore


class RomeStatuteCrime(str, Enum):
    """Types of crimes under the Rome Statute."""
    GENOCIDE = "genocide"
    CRIMES_AGAINST_HUMANITY = "crimes_against_humanity"  
    WAR_CRIMES = "war_crimes"
    CRIME_OF_AGGRESSION = "crime_of_aggression"


class WarCrimeCategory(str, Enum):
    """Categories of war crimes under Article 8."""
    GRAVE_BREACHES_GENEVA = "grave_breaches_geneva"
    SERIOUS_VIOLATIONS_LAWS_CUSTOMS_INTERNATIONAL = "serious_violations_international"
    ARMED_CONFLICT_NOT_INTERNATIONAL = "armed_conflict_not_international"
    SERIOUS_VIOLATIONS_LAWS_CUSTOMS_NON_INTERNATIONAL = "serious_violations_non_international"


class WarCrime(BaseModel):
    """Specific war crime under the Rome Statute."""
    article: str
    paragraph: str
    subparagraph: Optional[str] = None
    title: str
    description: str
    category: WarCrimeCategory
    elements: List[str]
    contextual_elements: List[str]
    mental_element: str


class CrimeAgainstHumanity(BaseModel):
    """Crime against humanity under Article 7."""
    article: str = "7"
    paragraph: str
    subparagraph: Optional[str] = None
    title: str
    description: str
    elements: List[str]
    contextual_elements: List[str]
    mental_element: str


class Genocide(BaseModel):
    """Genocide under Article 6."""
    article: str = "6"
    paragraph: str
    title: str
    description: str
    elements: List[str]
    protected_groups: List[str]
    mental_element: str


class RomeStatuteAnalysis(BaseModel):
    """Analysis results for Rome Statute violations."""
    crimes_analyzed: List[RomeStatuteCrime]
    genocide_findings: List[Dict]
    crimes_against_humanity_findings: List[Dict]
    war_crimes_findings: List[Dict]
    jurisdictional_elements: Dict[str, bool]
    admissibility_assessment: Dict[str, Any]
    complementarity_analysis: Dict[str, str]
    overall_assessment: str


class RomeStatuteAnalyzer:
    """
    Analyzer for Rome Statute violations.
    
    This class implements detailed analysis of evidence against specific elements
    of genocide, crimes against humanity, and war crimes under the Rome Statute.
    """
    
    def __init__(self):
        """Initialize the Rome Statute analyzer."""
        self.element_analyzer = ElementAnalyzer()
        self.legal_elements = self._load_rome_statute_elements()
        logger.info("RomeStatuteAnalyzer initialized with Rome Statute elements")
    
    def _load_rome_statute_elements(self) -> Dict[str, LegalElement]:
        """Load all Rome Statute legal elements."""
        elements = {}
        
        # Load genocide elements (Article 6)
        elements.update(self._load_genocide_elements())
        
        # Load crimes against humanity elements (Article 7)
        elements.update(self._load_crimes_against_humanity_elements())
        
        # Load war crimes elements (Article 8)
        elements.update(self._load_war_crimes_elements())
        
        # Load general elements
        elements.update(self._load_general_elements())
        
        logger.info(f"Loaded {len(elements)} Rome Statute legal elements")
        return elements
    
    def _load_genocide_elements(self) -> Dict[str, LegalElement]:
        """Load genocide elements from Article 6."""
        elements = {}
        
        # Article 6(a) - Killing members of the group
        elements["genocide_6a"] = LegalElement(
            id="genocide_6a",
            framework=LegalFramework.ROME_STATUTE,
            article="6(a)",
            title="Genocide by killing members of the group",
            description="Killing members of a national, ethnical, racial or religious group with intent to destroy the group",
            requirements=[
                "The perpetrator killed one or more persons",
                "Such person or persons belonged to a particular national, ethnical, racial or religious group", 
                "The perpetrator intended to destroy, in whole or in part, that national, ethnical, racial or religious group",
                "The conduct took place in the context of a manifest pattern of similar conduct directed against that group"
            ],
            keywords=["killing", "murder", "execution", "assassination", "death", "genocide"],
            citation="Rome Statute, Article 6(a)",
            precedents=["Prosecutor v. Akayesu (ICTR)", "Prosecutor v. Krstić (ICTY)"]
        )
        
        # Article 6(b) - Causing serious bodily or mental harm
        elements["genocide_6b"] = LegalElement(
            id="genocide_6b", 
            framework=LegalFramework.ROME_STATUTE,
            article="6(b)",
            title="Genocide by causing serious bodily or mental harm",
            description="Causing serious bodily or mental harm to members of the group",
            requirements=[
                "The perpetrator caused serious bodily or mental harm to one or more persons",
                "Such person or persons belonged to a particular national, ethnical, racial or religious group",
                "The perpetrator intended to destroy, in whole or in part, that national, ethnical, racial or religious group",
                "The conduct took place in the context of a manifest pattern of similar conduct directed against that group"
            ],
            keywords=["torture", "harm", "injury", "mental", "psychological", "trauma"],
            citation="Rome Statute, Article 6(b)"
        )
        
        # Article 6(c) - Deliberately inflicting conditions of life
        elements["genocide_6c"] = LegalElement(
            id="genocide_6c",
            framework=LegalFramework.ROME_STATUTE, 
            article="6(c)",
            title="Genocide by deliberately inflicting conditions of life calculated to bring about physical destruction",
            description="Deliberately inflicting on the group conditions of life calculated to bring about its physical destruction in whole or in part",
            requirements=[
                "The perpetrator inflicted certain conditions of life upon one or more persons",
                "Such person or persons belonged to a particular national, ethnical, racial or religious group",
                "The perpetrator intended to destroy, in whole or in part, that national, ethnical, racial or religious group",
                "The conditions of life were calculated to bring about the physical destruction of that group, in whole or in part"
            ],
            keywords=["starvation", "deprivation", "conditions", "destruction", "blockade"],
            citation="Rome Statute, Article 6(c)"
        )
        
        # Article 6(d) - Imposing measures intended to prevent births
        elements["genocide_6d"] = LegalElement(
            id="genocide_6d",
            framework=LegalFramework.ROME_STATUTE,
            article="6(d)", 
            title="Genocide by imposing measures intended to prevent births within the group",
            description="Imposing measures intended to prevent births within the group",
            requirements=[
                "The perpetrator imposed certain measures upon one or more persons",
                "Such person or persons belonged to a particular national, ethnical, racial or religious group",
                "The perpetrator intended to destroy, in whole or in part, that national, ethnical, racial or religious group", 
                "The measures imposed were intended to prevent births within that group"
            ],
            keywords=["sterilization", "contraception", "birth", "reproduction", "pregnancy"],
            citation="Rome Statute, Article 6(d)"
        )
        
        # Article 6(e) - Forcibly transferring children
        elements["genocide_6e"] = LegalElement(
            id="genocide_6e",
            framework=LegalFramework.ROME_STATUTE,
            article="6(e)",
            title="Genocide by forcibly transferring children of the group to another group",
            description="Forcibly transferring children of the group to another group",
            requirements=[
                "The perpetrator forcibly transferred one or more persons",
                "Such person or persons were under the age of 18 years",
                "The perpetrator intended to destroy, in whole or in part, that national, ethnical, racial or religious group",
                "The transfer was from that group to another group"
            ],
            keywords=["children", "transfer", "adoption", "separation", "displacement"],
            citation="Rome Statute, Article 6(e)"
        )
        
        return elements
    
    def _load_crimes_against_humanity_elements(self) -> Dict[str, LegalElement]:
        """Load crimes against humanity elements from Article 7.""" 
        elements = {}
        
        # Article 7(1)(a) - Murder
        elements["cah_7_1_a"] = LegalElement(
            id="cah_7_1_a",
            framework=LegalFramework.ROME_STATUTE,
            article="7(1)(a)",
            title="Murder as a crime against humanity", 
            description="Murder when committed as part of a widespread or systematic attack directed against any civilian population",
            requirements=[
                "The perpetrator killed one or more persons",
                "The conduct was committed as part of a widespread or systematic attack directed against a civilian population",
                "The perpetrator knew that the conduct was part of or intended the conduct to be part of a widespread or systematic attack directed against a civilian population"
            ],
            keywords=["murder", "killing", "execution", "assassination", "homicide"],
            citation="Rome Statute, Article 7(1)(a)",
            precedents=["Prosecutor v. Katanga (ICC)", "Prosecutor v. Bemba (ICC)"]
        )
        
        # Article 7(1)(b) - Extermination
        elements["cah_7_1_b"] = LegalElement(
            id="cah_7_1_b",
            framework=LegalFramework.ROME_STATUTE,
            article="7(1)(b)", 
            title="Extermination as a crime against humanity",
            description="Extermination when committed as part of a widespread or systematic attack",
            requirements=[
                "The perpetrator killed one or more persons, including by inflicting conditions of life calculated to bring about the destruction of part of a population",
                "The conduct constituted, or took place as part of, a mass killing of members of a civilian population",
                "The conduct was committed as part of a widespread or systematic attack directed against a civilian population",
                "The perpetrator knew that the conduct was part of a widespread or systematic attack directed against a civilian population"
            ],
            keywords=["extermination", "mass killing", "massacre", "elimination"],
            citation="Rome Statute, Article 7(1)(b)"
        )
        
        # Article 7(1)(d) - Deportation or forcible transfer
        elements["cah_7_1_d"] = LegalElement(
            id="cah_7_1_d",
            framework=LegalFramework.ROME_STATUTE,
            article="7(1)(d)",
            title="Deportation or forcible transfer of population as a crime against humanity",
            description="Deportation or forcible transfer of population when committed as part of widespread or systematic attack",
            requirements=[
                "The perpetrator deported or forcibly transferred, without grounds permitted under international law, one or more persons to another State or location",
                "Such person or persons were lawfully present in the area from which they were so deported or transferred",
                "The perpetrator was aware of the factual circumstances that established the lawfulness of such presence",
                "The conduct was committed as part of a widespread or systematic attack directed against a civilian population",
                "The perpetrator knew that the conduct was part of a widespread or systematic attack"
            ],
            keywords=["deportation", "transfer", "displacement", "expulsion", "removal"],
            citation="Rome Statute, Article 7(1)(d)"
        )
        
        # Article 7(1)(f) - Torture
        elements["cah_7_1_f"] = LegalElement(
            id="cah_7_1_f", 
            framework=LegalFramework.ROME_STATUTE,
            article="7(1)(f)",
            title="Torture as a crime against humanity",
            description="Torture when committed as part of a widespread or systematic attack",
            requirements=[
                "The perpetrator inflicted severe physical or mental pain or suffering upon one or more persons",
                "Such person or persons were in the custody or under the control of the perpetrator",
                "Such pain or suffering did not arise only from, and was not inherent in or incidental to, lawful sanctions",
                "The conduct was committed as part of a widespread or systematic attack directed against a civilian population",
                "The perpetrator knew that the conduct was part of a widespread or systematic attack"
            ],
            keywords=["torture", "pain", "suffering", "cruelty", "abuse"],
            citation="Rome Statute, Article 7(1)(f)"
        )
        
        # Article 7(1)(g) - Rape
        elements["cah_7_1_g"] = LegalElement(
            id="cah_7_1_g",
            framework=LegalFramework.ROME_STATUTE, 
            article="7(1)(g)",
            title="Rape as a crime against humanity",
            description="Rape when committed as part of a widespread or systematic attack",
            requirements=[
                "The perpetrator invaded the body of a person by conduct resulting in penetration, however slight, of any part of the body of the victim or of the perpetrator with a sexual organ, or of the anal or genital opening of the victim with any object or any other part of the body",
                "The invasion was committed by force, or by threat of force or coercion, or by taking advantage of a coercive environment, or the invasion was committed against a person incapable of giving genuine consent",
                "The conduct was committed as part of a widespread or systematic attack directed against a civilian population",
                "The perpetrator knew that the conduct was part of a widespread or systematic attack"
            ],
            keywords=["rape", "sexual violence", "assault", "penetration"],
            citation="Rome Statute, Article 7(1)(g)",
            precedents=["Prosecutor v. Bemba (ICC)", "Prosecutor v. Ntaganda (ICC)"]
        )
        
        # Article 7(1)(h) - Persecution
        elements["cah_7_1_h"] = LegalElement(
            id="cah_7_1_h",
            framework=LegalFramework.ROME_STATUTE,
            article="7(1)(h)",
            title="Persecution as a crime against humanity",
            description="Persecution against any identifiable group or collectivity on political, racial, national, ethnic, cultural, religious, gender grounds",
            requirements=[
                "The perpetrator severely deprived, contrary to international law, one or more persons of fundamental rights",
                "The perpetrator targeted such person or persons by reason of the identity of a group or collectivity or targeted the group or collectivity as such",
                "Such targeting was based on political, racial, national, ethnic, cultural, religious, gender grounds, or other grounds universally recognized as impermissible under international law",
                "The conduct was committed as part of a widespread or systematic attack directed against a civilian population",
                "The perpetrator knew that the conduct was part of a widespread or systematic attack"
            ],
            keywords=["persecution", "discrimination", "fundamental rights", "identity", "group"],
            citation="Rome Statute, Article 7(1)(h)",
            precedents=["Prosecutor v. Krstić (ICTY)", "Prosecutor v. Al Mahdi (ICC)"]
        )
        
        # Article 7(1)(i) - Enforced disappearance
        elements["cah_7_1_i"] = LegalElement(
            id="cah_7_1_i",
            framework=LegalFramework.ROME_STATUTE,
            article="7(1)(i)",
            title="Enforced disappearance of persons as a crime against humanity",
            description="Enforced disappearance when committed as part of widespread or systematic attack",
            requirements=[
                "The perpetrator arrested, detained, or abducted one or more persons, or refused to acknowledge the arrest, detention or abduction, or to give information on the fate or whereabouts of such person or persons",
                "Such arrest, detention or abduction was followed or accompanied by a refusal to acknowledge that deprivation of freedom or to give information on the fate or whereabouts of such person or persons",
                "Such arrest, detention or abduction was carried out by, or with the authorization, support or acquiescence of, a State or political organization",
                "Such refusal was preceded or accompanied by that deprivation of freedom",
                "The perpetrator intended to remove such person or persons from the protection of the law for a prolonged period of time",
                "The conduct was committed as part of a widespread or systematic attack directed against a civilian population",
                "The perpetrator knew that the conduct was part of a widespread or systematic attack"
            ],
            keywords=["enforced disappearance", "detention", "abduction", "whereabouts", "fate unknown"],
            citation="Rome Statute, Article 7(1)(i)"
        )
        
        # Article 7(1)(j) - Crime of apartheid
        elements["cah_7_1_j"] = LegalElement(
            id="cah_7_1_j",
            framework=LegalFramework.ROME_STATUTE,
            article="7(1)(j)",
            title="Crime of apartheid as a crime against humanity",
            description="Inhumane acts committed in the context of an institutionalized regime of systematic oppression and domination by one racial group over another",
            requirements=[
                "The perpetrator committed an inhumane act against one or more persons",
                "Such act was an act referred to in article 7, paragraph 1, of the Statute, or was of a similar character",
                "The perpetrator was aware of the factual circumstances that established the character of the act",
                "The conduct was committed in the context of an institutionalized regime of systematic oppression and domination by one racial group over any other racial group or groups",
                "The perpetrator intended to maintain such regime by the conduct",
                "The conduct was committed as part of a widespread or systematic attack directed against a civilian population",
                "The perpetrator knew that the conduct was part of a widespread or systematic attack"
            ],
            keywords=["apartheid", "racial group", "systematic oppression", "domination", "institutionalized regime"],
            citation="Rome Statute, Article 7(1)(j)"
        )
        
        # Article 7(1)(k) - Other inhumane acts
        elements["cah_7_1_k"] = LegalElement(
            id="cah_7_1_k",
            framework=LegalFramework.ROME_STATUTE,
            article="7(1)(k)",
            title="Other inhumane acts as a crime against humanity",
            description="Other inhumane acts of a similar character intentionally causing great suffering, or serious injury to body or to mental or physical health",
            requirements=[
                "The perpetrator inflicted great suffering, or serious injury to body or to mental or physical health, by means of an inhumane act",
                "Such act was of a character similar to any other act referred to in article 7, paragraph 1, of the Statute",
                "The perpetrator was aware of the factual circumstances that established the character of the act",
                "The conduct was committed as part of a widespread or systematic attack directed against a civilian population",
                "The perpetrator knew that the conduct was part of a widespread or systematic attack"
            ],
            keywords=["inhumane acts", "great suffering", "serious injury", "mental health", "physical health"],
            citation="Rome Statute, Article 7(1)(k)"
        )
        
        return elements
    
    def _load_war_crimes_elements(self) -> Dict[str, LegalElement]:
        """Load war crimes elements from Article 8."""
        elements = {}
        
        # Article 8(2)(a)(i) - Wilful killing (Geneva Convention grave breach)
        elements["wc_8_2_a_i"] = LegalElement(
            id="wc_8_2_a_i",
            framework=LegalFramework.ROME_STATUTE,
            article="8(2)(a)(i)",
            title="War crime of wilful killing",
            description="Wilful killing of persons protected under Geneva Conventions",
            requirements=[
                "The perpetrator killed one or more persons",
                "Such person or persons were protected under one or more of the Geneva Conventions of 1949",
                "The perpetrator was aware of the factual circumstances that established that protected status",
                "The conduct took place in the context of and was associated with an international armed conflict"
            ],
            keywords=["killing", "murder", "execution", "protected persons"],
            citation="Rome Statute, Article 8(2)(a)(i)"
        )
        
        # Article 8(2)(a)(ii) - Torture or inhuman treatment
        elements["wc_8_2_a_ii"] = LegalElement(
            id="wc_8_2_a_ii",
            framework=LegalFramework.ROME_STATUTE,
            article="8(2)(a)(ii)", 
            title="War crime of torture or inhuman treatment",
            description="Torture or inhuman treatment of protected persons",
            requirements=[
                "The perpetrator inflicted severe physical or mental pain or suffering upon one or more persons",
                "Such person or persons were protected under one or more of the Geneva Conventions of 1949",
                "The perpetrator was aware of the factual circumstances that established that protected status",
                "The conduct took place in the context of and was associated with an international armed conflict"
            ],
            keywords=["torture", "inhuman treatment", "cruel treatment", "suffering"],
            citation="Rome Statute, Article 8(2)(a)(ii)"
        )
        
        # Article 8(2)(b)(i) - Intentionally directing attacks against civilians
        elements["wc_8_2_b_i"] = LegalElement(
            id="wc_8_2_b_i",
            framework=LegalFramework.ROME_STATUTE,
            article="8(2)(b)(i)",
            title="War crime of intentionally directing attacks against the civilian population",
            description="Intentionally directing attacks against the civilian population as such or against individual civilians not taking direct part in hostilities",
            requirements=[
                "The perpetrator directed an attack",
                "The object of the attack was a civilian population as such or individual civilians not taking direct part in hostilities", 
                "The perpetrator intended the civilian population as such or individual civilians not taking direct part in hostilities to be the object of the attack",
                "The conduct took place in the context of and was associated with an international armed conflict"
            ],
            keywords=["attack", "civilians", "targeting", "hostilities"],
            citation="Rome Statute, Article 8(2)(b)(i)"
        )
        
        # Article 8(2)(b)(ii) - Attacking civilian objects
        elements["wc_8_2_b_ii"] = LegalElement(
            id="wc_8_2_b_ii",
            framework=LegalFramework.ROME_STATUTE,
            article="8(2)(b)(ii)",
            title="War crime of intentionally directing attacks against civilian objects",
            description="Intentionally directing attacks against civilian objects, that is, objects which are not military objectives",
            requirements=[
                "The perpetrator directed an attack",
                "The object of the attack was civilian objects, that is, objects which are not military objectives",
                "The perpetrator intended such civilian objects to be the object of the attack", 
                "The conduct took place in the context of and was associated with an international armed conflict"
            ],
            keywords=["attack", "civilian objects", "military objectives", "targeting"],
            citation="Rome Statute, Article 8(2)(b)(ii)"
        )
        
        # Article 8(2)(b)(ix) - Attacking protected buildings
        elements["wc_8_2_b_ix"] = LegalElement(
            id="wc_8_2_b_ix",
            framework=LegalFramework.ROME_STATUTE,
            article="8(2)(b)(ix)",
            title="War crime of intentionally directing attacks against buildings dedicated to religion, education, art, science or charitable purposes, historic monuments, hospitals and places where the sick and wounded are collected",
            description="Intentionally directing attacks against protected buildings and sites",
            requirements=[
                "The perpetrator directed an attack",
                "The object of the attack was one or more buildings dedicated to religion, education, art, science or charitable purposes, historic monuments, hospitals or places where the sick and wounded are collected, which were not military objectives",
                "The perpetrator intended such buildings to be the object of the attack",
                "The conduct took place in the context of and was associated with an international armed conflict"
            ],
            keywords=["protected buildings", "hospitals", "schools", "religious sites", "cultural objects"],
            citation="Rome Statute, Article 8(2)(b)(ix)",
            precedents=["Prosecutor v. Al Mahdi (ICC)", "Prosecutor v. Strugar (ICTY)"]
        )
        
        # Article 8(2)(b)(xxii) - Rape as a war crime
        elements["wc_8_2_b_xxii"] = LegalElement(
            id="wc_8_2_b_xxii",
            framework=LegalFramework.ROME_STATUTE,
            article="8(2)(b)(xxii)",
            title="War crime of rape",
            description="Committing rape in the context of international armed conflict",
            requirements=[
                "The perpetrator invaded the body of a person by conduct resulting in penetration, however slight, of any part of the body of the victim or of the perpetrator with a sexual organ, or of the anal or genital opening of the victim with any object or any other part of the body",
                "The invasion was committed by force, or by threat of force or coercion, or by taking advantage of a coercive environment, or the invasion was committed against a person incapable of giving genuine consent",
                "The conduct took place in the context of and was associated with an international armed conflict"
            ],
            keywords=["rape", "sexual violence", "armed conflict", "coercion"],
            citation="Rome Statute, Article 8(2)(b)(xxii)",
            precedents=["Prosecutor v. Ntaganda (ICC)", "Prosecutor v. Bemba (ICC)"]
        )
        
        # Article 8(2)(b)(xxv) - Starvation
        elements["wc_8_2_b_xxv"] = LegalElement(
            id="wc_8_2_b_xxv",
            framework=LegalFramework.ROME_STATUTE,
            article="8(2)(b)(xxv)",
            title="War crime of intentionally using starvation of civilians as a method of warfare",
            description="Intentionally using starvation of civilians as a method of warfare by depriving them of objects indispensable to their survival",
            requirements=[
                "The perpetrator deprived civilians of objects indispensable to their survival",
                "The perpetrator intended to starve civilians as a method of warfare",
                "The conduct took place in the context of and was associated with an international armed conflict"
            ],
            keywords=["starvation", "civilians", "objects indispensable to survival", "method of warfare"],
            citation="Rome Statute, Article 8(2)(b)(xxv)"
        )
        
        # Article 8(2)(b)(xxvi) - Recruiting child soldiers
        elements["wc_8_2_b_xxvi"] = LegalElement(
            id="wc_8_2_b_xxvi",
            framework=LegalFramework.ROME_STATUTE,
            article="8(2)(b)(xxvi)",
            title="War crime of conscripting or enlisting children under the age of fifteen years",
            description="Conscripting or enlisting children under the age of fifteen years into the national armed forces or using them to participate actively in hostilities",
            requirements=[
                "The perpetrator conscripted or enlisted one or more persons into the national armed forces or used one or more persons to participate actively in hostilities",
                "Such person or persons were under the age of 15 years",
                "The perpetrator knew or should have known that such person or persons were under the age of 15 years",
                "The conduct took place in the context of and was associated with an international armed conflict"
            ],
            keywords=["child soldiers", "conscription", "enlisting", "children", "under fifteen", "hostilities"],
            citation="Rome Statute, Article 8(2)(b)(xxvi)",
            precedents=["Prosecutor v. Lubanga (ICC)", "Prosecutor v. Ntaganda (ICC)"]
        )
        
        # Non-international armed conflict elements
        
        # Article 8(2)(c)(i) - Violence to life and person
        elements["wc_8_2_c_i"] = LegalElement(
            id="wc_8_2_c_i",
            framework=LegalFramework.ROME_STATUTE,
            article="8(2)(c)(i)",
            title="War crime of violence to life and person in non-international armed conflict",
            description="Violence to life and person, in particular murder of all kinds, mutilation, cruel treatment and torture",
            requirements=[
                "The perpetrator killed or injured one or more persons or subjected them to physical or mental torture, cruel or humane treatment",
                "Such person or persons were either hors de combat, or were civilians, medical personnel, or religious personnel taking no active part in the hostilities",
                "The perpetrator was aware of the factual circumstances that established this status",
                "The conduct took place in the context of and was associated with an armed conflict not of an international character"
            ],
            keywords=["violence", "murder", "mutilation", "cruel treatment", "torture", "non-international"],
            citation="Rome Statute, Article 8(2)(c)(i)"
        )
        
        # Article 8(2)(e)(i) - Directing attacks against civilian population (NIAC)
        elements["wc_8_2_e_i"] = LegalElement(
            id="wc_8_2_e_i",
            framework=LegalFramework.ROME_STATUTE,
            article="8(2)(e)(i)",
            title="War crime of intentionally directing attacks against the civilian population in non-international armed conflict",
            description="Intentionally directing attacks against the civilian population as such or against individual civilians not taking direct part in hostilities",
            requirements=[
                "The perpetrator directed an attack",
                "The object of the attack was a civilian population as such or individual civilians not taking direct part in hostilities",
                "The perpetrator intended the civilian population as such or individual civilians not taking direct part in hostilities to be the object of the attack",
                "The conduct took place in the context of and was associated with an armed conflict not of an international character"
            ],
            keywords=["attack", "civilians", "non-international conflict", "direct part in hostilities"],
            citation="Rome Statute, Article 8(2)(e)(i)"
        )
        
        # Article 8(2)(e)(vii) - Child soldiers in NIAC
        elements["wc_8_2_e_vii"] = LegalElement(
            id="wc_8_2_e_vii",
            framework=LegalFramework.ROME_STATUTE,
            article="8(2)(e)(vii)",
            title="War crime of conscripting or enlisting children under fifteen years in non-international armed conflict",
            description="Conscripting or enlisting children under the age of fifteen years into armed forces or groups or using them to participate actively in hostilities",
            requirements=[
                "The perpetrator conscripted or enlisted one or more persons into an armed force or group or used one or more persons to participate actively in hostilities",
                "Such person or persons were under the age of 15 years",
                "The perpetrator knew or should have known that such person or persons were under the age of 15 years",
                "The conduct took place in the context of and was associated with an armed conflict not of an international character"
            ],
            keywords=["child soldiers", "armed groups", "non-international conflict", "conscription", "under fifteen"],
            citation="Rome Statute, Article 8(2)(e)(vii)",
            precedents=["Prosecutor v. Lubanga (ICC)", "Prosecutor v. Katanga (ICC)"]
        )
        
        return elements
    
    def _load_general_elements(self) -> Dict[str, LegalElement]:
        """Load general elements for Rome Statute crimes."""
        elements = {}
        
        # Widespread or systematic attack (for crimes against humanity)
        elements["cah_widespread_systematic"] = LegalElement(
            id="cah_widespread_systematic",
            framework=LegalFramework.ROME_STATUTE,
            article="7(1)",
            title="Widespread or systematic attack against civilian population",
            description="Attack directed against any civilian population that is either widespread or systematic",
            requirements=[
                "There was an attack directed against a civilian population",
                "The attack was either widespread or systematic",
                "The attack was pursuant to or in furtherance of a State or organizational policy"
            ],
            keywords=["widespread", "systematic", "attack", "civilian population", "policy"],
            citation="Rome Statute, Article 7(1)"
        )
        
        # Armed conflict nexus (for war crimes)
        elements["wc_armed_conflict"] = LegalElement(
            id="wc_armed_conflict", 
            framework=LegalFramework.ROME_STATUTE,
            article="8",
            title="Armed conflict nexus",
            description="Conduct taking place in the context of and associated with armed conflict",
            requirements=[
                "There was an armed conflict (international or non-international)",
                "The conduct took place in the context of and was associated with the armed conflict"
            ],
            keywords=["armed conflict", "hostilities", "war", "international", "non-international"],
            citation="Rome Statute, Article 8"
        )
        
        return elements
    
    def analyze(self, evidence: List[Evidence]) -> FrameworkAnalysis:
        """
        Analyze evidence against Rome Statute elements.
        
        Args:
            evidence: List of evidence to analyze
            
        Returns:
            FrameworkAnalysis: Complete Rome Statute analysis
        """
        logger.info(f"Analyzing {len(evidence)} pieces of evidence against Rome Statute")
        
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
            framework=LegalFramework.ROME_STATUTE,
            evidence_count=len(evidence),
            elements_analyzed=[e.id for e in self.legal_elements.values()],
            element_satisfactions=element_satisfactions,
            overall_confidence=overall_confidence,
            gap_analysis=gap_analysis,
            summary=summary,
            violations_identified=violations_identified,
            recommendations=recommendations
        )
        
        logger.info(f"Rome Statute analysis completed. Found {len(violations_identified)} potential violations")
        return analysis
    
    def _generate_gap_analysis(self, satisfactions: List[ElementSatisfaction]) -> GapAnalysis:
        """Generate gap analysis for Rome Statute elements."""
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
        
        # Generate recommendations based on gaps
        if missing_elements:
            recommendations.append("Collect evidence for completely missing elements")
        if weak_elements:
            recommendations.append("Strengthen evidence for weakly supported elements")
        if any("witness" in gap.lower() for gaps in evidence_needs.values() for gap in gaps):
            recommendations.append("Obtain additional witness testimony")
        if any("document" in gap.lower() for gaps in evidence_needs.values() for gap in gaps):
            recommendations.append("Secure additional documentary evidence")
        
        # Calculate priority score
        total_elements = len(satisfactions)
        problematic_elements = len(missing_elements) + len(weak_elements)
        priority_score = problematic_elements / total_elements if total_elements > 0 else 0.0
        
        return GapAnalysis(
            framework=LegalFramework.ROME_STATUTE,
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
        
        summary = f"Rome Statute analysis of {total_elements} legal elements: "
        summary += f"{satisfied} fully satisfied, {partial} partially satisfied. "
        summary += f"Overall confidence: {overall_confidence:.1%}. "
        
        if violations:
            summary += f"Identified {len(violations)} potential violations including "
            summary += ", ".join(violations[:3])
            if len(violations) > 3:
                summary += f" and {len(violations) - 3} others"
            summary += "."
        else:
            summary += "No clear violations identified based on available evidence."
        
        return summary
    
    def _generate_recommendations(self, satisfactions: List[ElementSatisfaction],
                                gap_analysis: GapAnalysis) -> List[str]:
        """Generate recommendations for Rome Statute analysis."""
        recommendations = []
        
        # High confidence violations
        high_confidence = [s for s in satisfactions 
                          if s.status == ElementStatus.SATISFIED and s.confidence >= 0.8]
        if high_confidence:
            recommendations.append("Consider referral to ICC Prosecutor for preliminary examination")
        
        # Evidence collection recommendations  
        recommendations.extend(gap_analysis.recommendations)
        
        # Jurisdiction-specific recommendations
        recommendations.extend([
            "Verify temporal jurisdiction (crimes after 1 July 2002)",
            "Confirm territorial or personal jurisdiction",
            "Assess complementarity with national proceedings",
            "Evaluate gravity threshold for ICC intervention"
        ])
        
        return recommendations


def analyze_rome_statute_elements(evidence: List[Evidence]) -> RomeStatuteAnalysis:
    """
    Convenience function to analyze evidence against Rome Statute elements.
    
    Args:
        evidence: List of evidence to analyze
        
    Returns:
        RomeStatuteAnalysis: Detailed Rome Statute analysis
    """
    analyzer = RomeStatuteAnalyzer()
    framework_analysis = analyzer.analyze(evidence)
    
    # Extract specific findings by crime type
    genocide_findings = []
    cah_findings = []
    wc_findings = []
    
    for satisfaction in framework_analysis.element_satisfactions:
        if "genocide" in satisfaction.element_id:
            genocide_findings.append({
                "element": satisfaction.element_id,
                "status": satisfaction.status,
                "confidence": satisfaction.confidence,
                "reasoning": satisfaction.reasoning
            })
        elif "cah" in satisfaction.element_id:
            cah_findings.append({
                "element": satisfaction.element_id,
                "status": satisfaction.status, 
                "confidence": satisfaction.confidence,
                "reasoning": satisfaction.reasoning
            })
        elif "wc" in satisfaction.element_id:
            wc_findings.append({
                "element": satisfaction.element_id,
                "status": satisfaction.status,
                "confidence": satisfaction.confidence,
                "reasoning": satisfaction.reasoning
            })
    
    # Jurisdictional assessment
    jurisdictional_elements = {
        "temporal_jurisdiction": True,  # Placeholder - would need actual date checking
        "territorial_jurisdiction": True,  # Placeholder - would need location verification  
        "personal_jurisdiction": True,  # Placeholder - would need nationality checks
        "subject_matter_jurisdiction": bool(framework_analysis.violations_identified)
    }
    
    # Admissibility assessment
    admissibility_assessment = {
        "gravity": "sufficient" if framework_analysis.overall_confidence > 0.7 else "questionable",
        "complementarity": "potentially_admissible",  # Would require national proceedings analysis
        "interests_of_justice": "no_impediment"
    }
    
    return RomeStatuteAnalysis(
        crimes_analyzed=[RomeStatuteCrime.GENOCIDE, RomeStatuteCrime.CRIMES_AGAINST_HUMANITY, RomeStatuteCrime.WAR_CRIMES],
        genocide_findings=genocide_findings,
        crimes_against_humanity_findings=cah_findings,
        war_crimes_findings=wc_findings,
        jurisdictional_elements=jurisdictional_elements,
        admissibility_assessment=admissibility_assessment,
        complementarity_analysis={"status": "pending_assessment", "recommendation": "conduct_full_analysis"},
        overall_assessment=framework_analysis.summary
    )