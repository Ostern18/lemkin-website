"""
Tests for the legal taxonomy module.
"""

import pytest
from lemkin_classifier.legal_taxonomy import (
    DocumentType,
    LegalDomain,
    LegalDocumentCategory,
    CategoryHierarchy,
    get_category_hierarchy,
    get_supported_categories,
    validate_category,
    get_category_keywords,
    get_urgency_level,
    requires_human_review,
    get_sensitivity_level,
    get_categories_by_domain,
    get_high_priority_categories,
    get_categories_requiring_chain_of_custody,
    CATEGORY_DEFINITIONS,
    DOMAIN_DOCUMENT_MAPPING
)


class TestDocumentType:
    """Test DocumentType enum"""
    
    def test_document_type_values(self):
        """Test that all document types have expected values"""
        expected_types = [
            "witness_statement", "police_report", "medical_record", "court_filing",
            "government_document", "military_report", "email", "expert_testimony",
            "forensic_report", "financial_record", "other", "unknown"
        ]
        
        actual_types = [dt.value for dt in DocumentType]
        
        for expected in expected_types:
            assert expected in actual_types


class TestLegalDomain:
    """Test LegalDomain enum"""
    
    def test_legal_domain_values(self):
        """Test that all legal domains have expected values"""
        expected_domains = [
            "criminal_law", "civil_rights", "international_humanitarian_law",
            "human_rights_law", "administrative_law", "constitutional_law",
            "corporate_law", "family_law", "immigration_law", "environmental_law",
            "general"
        ]
        
        actual_domains = [ld.value for ld in LegalDomain]
        
        for expected in expected_domains:
            assert expected in actual_domains


class TestLegalDocumentCategory:
    """Test LegalDocumentCategory model"""
    
    def test_category_creation(self):
        """Test creating a legal document category"""
        category = LegalDocumentCategory(
            document_type=DocumentType.WITNESS_STATEMENT,
            legal_domain=LegalDomain.CRIMINAL_LAW,
            urgency_level="high",
            sensitivity_level="confidential"
        )
        
        assert category.document_type == DocumentType.WITNESS_STATEMENT
        assert category.legal_domain == LegalDomain.CRIMINAL_LAW
        assert category.urgency_level == "high"
        assert category.sensitivity_level == "confidential"
        assert category.keywords == []
        assert category.required_fields == []
        assert not category.requires_human_review
        assert not category.redaction_required
        assert not category.chain_of_custody_critical
    
    def test_category_with_all_fields(self):
        """Test creating category with all optional fields"""
        category = LegalDocumentCategory(
            document_type=DocumentType.POLICE_REPORT,
            legal_domain=LegalDomain.CRIMINAL_LAW,
            subcategory="incident_report",
            urgency_level="critical",
            sensitivity_level="restricted",
            keywords=["police", "report", "incident"],
            required_fields=["report_number", "officer_name"],
            typical_sources=["police_department"],
            requires_human_review=True,
            redaction_required=True,
            chain_of_custody_critical=True
        )
        
        assert category.subcategory == "incident_report"
        assert len(category.keywords) == 3
        assert len(category.required_fields) == 2
        assert category.requires_human_review
        assert category.redaction_required
        assert category.chain_of_custody_critical


class TestCategoryHierarchy:
    """Test CategoryHierarchy functionality"""
    
    def test_get_category_hierarchy(self):
        """Test getting the complete category hierarchy"""
        hierarchy = get_category_hierarchy()
        
        assert isinstance(hierarchy, CategoryHierarchy)
        assert len(hierarchy.primary_categories) > 0
        assert len(hierarchy.subcategories) > 0
        assert len(hierarchy.domain_mapping) > 0
    
    def test_hierarchy_consistency(self):
        """Test that hierarchy data is consistent"""
        hierarchy = get_category_hierarchy()
        
        # Check that all primary categories are DocumentType instances
        for doc_type, category in hierarchy.primary_categories.items():
            assert isinstance(doc_type, DocumentType)
            assert isinstance(category, LegalDocumentCategory)
            assert category.document_type == doc_type
        
        # Check that domain mapping is consistent
        for domain, doc_types in hierarchy.domain_mapping.items():
            assert isinstance(domain, LegalDomain)
            assert isinstance(doc_types, list)
            for doc_type in doc_types:
                assert isinstance(doc_type, DocumentType)


class TestTaxonomyFunctions:
    """Test taxonomy utility functions"""
    
    def test_get_supported_categories(self):
        """Test getting supported categories"""
        categories = get_supported_categories()
        
        assert isinstance(categories, list)
        assert len(categories) > 0
        assert DocumentType.WITNESS_STATEMENT in categories
        assert DocumentType.POLICE_REPORT in categories
    
    def test_validate_category_valid(self):
        """Test validating valid categories"""
        # Test valid document type
        assert validate_category("witness_statement")
        assert validate_category("police_report")
        
        # Test valid domain combination
        assert validate_category("witness_statement", "criminal_law")
        assert validate_category("medical_record", "civil_rights")
    
    def test_validate_category_invalid(self):
        """Test validating invalid categories"""
        # Test invalid document type
        assert not validate_category("invalid_document_type")
        
        # Test invalid domain
        assert not validate_category("witness_statement", "invalid_domain")
        
        # Test valid types but invalid combination
        # This depends on your domain mapping implementation
        # You may need to adjust based on actual mappings
    
    def test_get_category_keywords(self):
        """Test getting category keywords"""
        keywords = get_category_keywords(DocumentType.WITNESS_STATEMENT)
        assert isinstance(keywords, list)
        
        # Should return empty list for undefined categories
        keywords_other = get_category_keywords(DocumentType.OTHER)
        assert keywords_other == []
    
    def test_get_urgency_level(self):
        """Test getting urgency level"""
        urgency = get_urgency_level(DocumentType.WITNESS_STATEMENT)
        assert isinstance(urgency, str)
        assert urgency in ["low", "medium", "high", "critical"]
        
        # Should return default for undefined categories
        urgency_other = get_urgency_level(DocumentType.OTHER)
        assert urgency_other == "medium"
    
    def test_requires_human_review(self):
        """Test human review requirement check"""
        # This will depend on your category definitions
        review_required = requires_human_review(DocumentType.WITNESS_STATEMENT)
        assert isinstance(review_required, bool)
        
        # Should default to True for unknown types
        review_other = requires_human_review(DocumentType.OTHER)
        assert review_other is True
    
    def test_get_sensitivity_level(self):
        """Test getting sensitivity level"""
        sensitivity = get_sensitivity_level(DocumentType.WITNESS_STATEMENT)
        assert isinstance(sensitivity, str)
        assert sensitivity in ["public", "internal", "confidential", "restricted"]
        
        # Should return default for undefined categories
        sensitivity_other = get_sensitivity_level(DocumentType.OTHER)
        assert sensitivity_other == "standard"
    
    def test_get_categories_by_domain(self):
        """Test getting categories by legal domain"""
        criminal_categories = get_categories_by_domain(LegalDomain.CRIMINAL_LAW)
        assert isinstance(criminal_categories, list)
        
        # Should contain relevant categories
        category_types = [cat.document_type for cat in criminal_categories]
        assert DocumentType.POLICE_REPORT in category_types or DocumentType.WITNESS_STATEMENT in category_types
        
        # Should return empty list for unmapped domains
        general_categories = get_categories_by_domain(LegalDomain.GENERAL)
        assert isinstance(general_categories, list)
    
    def test_get_high_priority_categories(self):
        """Test getting high priority categories"""
        high_priority = get_high_priority_categories()
        assert isinstance(high_priority, list)
        
        # All returned items should be DocumentType instances
        for doc_type in high_priority:
            assert isinstance(doc_type, DocumentType)
            urgency = get_urgency_level(doc_type)
            assert urgency in ["high", "critical"]
    
    def test_get_categories_requiring_chain_of_custody(self):
        """Test getting categories requiring chain of custody"""
        custody_critical = get_categories_requiring_chain_of_custody()
        assert isinstance(custody_critical, list)
        
        # All returned items should be DocumentType instances
        for doc_type in custody_critical:
            assert isinstance(doc_type, DocumentType)


class TestCategoryDefinitions:
    """Test predefined category definitions"""
    
    def test_category_definitions_not_empty(self):
        """Test that category definitions exist"""
        assert len(CATEGORY_DEFINITIONS) > 0
    
    def test_category_definitions_structure(self):
        """Test structure of category definitions"""
        for doc_type, category in CATEGORY_DEFINITIONS.items():
            assert isinstance(doc_type, DocumentType)
            assert isinstance(category, LegalDocumentCategory)
            assert category.document_type == doc_type
            assert isinstance(category.keywords, list)
            assert isinstance(category.required_fields, list)
            assert isinstance(category.typical_sources, list)
    
    def test_domain_document_mapping_structure(self):
        """Test structure of domain document mapping"""
        assert len(DOMAIN_DOCUMENT_MAPPING) > 0
        
        for domain, doc_types in DOMAIN_DOCUMENT_MAPPING.items():
            assert isinstance(domain, LegalDomain)
            assert isinstance(doc_types, list)
            for doc_type in doc_types:
                assert isinstance(doc_type, DocumentType)
    
    def test_witness_statement_category(self):
        """Test specific witness statement category definition"""
        if DocumentType.WITNESS_STATEMENT in CATEGORY_DEFINITIONS:
            category = CATEGORY_DEFINITIONS[DocumentType.WITNESS_STATEMENT]
            assert category.document_type == DocumentType.WITNESS_STATEMENT
            assert category.legal_domain == LegalDomain.CRIMINAL_LAW
            assert category.urgency_level == "high"
            assert category.sensitivity_level == "confidential"
            assert category.requires_human_review
            assert category.redaction_required
            assert category.chain_of_custody_critical
    
    def test_police_report_category(self):
        """Test specific police report category definition"""
        if DocumentType.POLICE_REPORT in CATEGORY_DEFINITIONS:
            category = CATEGORY_DEFINITIONS[DocumentType.POLICE_REPORT]
            assert category.document_type == DocumentType.POLICE_REPORT
            assert category.legal_domain == LegalDomain.CRIMINAL_LAW
            assert category.urgency_level == "high"
            assert category.sensitivity_level == "restricted"
            assert category.requires_human_review
            assert category.redaction_required
            assert category.chain_of_custody_critical


@pytest.fixture
def sample_category():
    """Fixture for sample legal document category"""
    return LegalDocumentCategory(
        document_type=DocumentType.WITNESS_STATEMENT,
        legal_domain=LegalDomain.CRIMINAL_LAW,
        urgency_level="high",
        sensitivity_level="confidential",
        keywords=["witness", "statement", "testimony"],
        required_fields=["witness_name", "date", "location"],
        typical_sources=["police_station", "court"],
        requires_human_review=True,
        redaction_required=True,
        chain_of_custody_critical=True
    )


class TestCategoryFixture:
    """Test using the sample category fixture"""
    
    def test_sample_category_properties(self, sample_category):
        """Test sample category properties"""
        assert sample_category.document_type == DocumentType.WITNESS_STATEMENT
        assert sample_category.legal_domain == LegalDomain.CRIMINAL_LAW
        assert len(sample_category.keywords) == 3
        assert len(sample_category.required_fields) == 3
        assert sample_category.requires_human_review
        assert sample_category.redaction_required
        assert sample_category.chain_of_custody_critical