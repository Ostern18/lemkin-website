# Contributing to Lemkin Geospatial Analysis Suite

Thank you for contributing to geospatial analysis capabilities for legal investigations!

## Development Setup

```bash
git clone https://github.com/lemkin-org/lemkin-geo.git
cd lemkin-geo
pip install -e ".[dev]"
pre-commit install
```

## Key Contribution Areas

- **Coordinate Systems**: Support for new coordinate reference systems
- **Mapping Integrations**: New mapping services and satellite imagery sources
- **Geospatial Analysis**: Advanced correlation and pattern detection algorithms
- **Privacy Protection**: Location anonymization and witness protection features

## Geospatial-Specific Guidelines

### Security Considerations
- Never hardcode API keys for mapping services
- Implement coordinate obfuscation for sensitive locations
- Protect witness locations with appropriate anonymization
- Use secure protocols for satellite imagery access

### Code Standards
```python
def analyze_location_correlation(
    events: List[LocationEvent],
    radius: float = 1000.0,
    coordinate_system: str = "EPSG:4326"
) -> CorrelationResult:
    \"\"\"Correlate events by geographic proximity.

    Args:
        events: List of events with location data
        radius: Correlation radius in meters
        coordinate_system: Target coordinate reference system

    Returns:
        Correlation analysis with privacy protections applied
    \"\"\"
```

### Testing Requirements
- Use synthetic coordinates, never real sensitive locations
- Test coordinate transformations across multiple systems
- Verify privacy protection mechanisms
- Include performance tests for large datasets

## Legal and Ethical Focus

- **Witness Safety**: Prioritize protection of sensitive locations
- **Privacy**: Implement privacy-preserving geospatial analysis
- **Accuracy**: Ensure coordinate accuracy for legal evidence
- **Documentation**: Document all transformations for court admissibility

## Contact

- **Technical**: Open GitHub Issues
- **Security**: security@lemkin.org
- **Ethics**: ethics@lemkin.org

---

*Help make geospatial analysis safe and accessible for human rights investigations.*