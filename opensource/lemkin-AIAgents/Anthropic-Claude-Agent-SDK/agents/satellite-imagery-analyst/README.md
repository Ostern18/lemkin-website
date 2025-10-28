# Satellite Imagery Analyst Agent

Interprets satellite and aerial imagery for legal evidence in conflict documentation and human rights investigations.

## Capabilities

- Feature identification (buildings, vehicles, crowds, damage)
- Change detection (before/after comparison)
- Site assessment (mass graves, detention facilities)
- Crowd size estimation
- Damage assessment
- Geolocation assistance

## Usage

```python
from agents.satellite_imagery_analyst.agent import SatelliteImageryAnalystAgent

agent = SatelliteImageryAnalystAgent()

# Analyze imagery
with open("satellite_image.jpg", "rb") as f:
    image_data = f.read()

result = agent.process({
    'image_data': image_data,
    'analysis_type': 'site_assessment',
    'case_id': 'CASE-2024-001'
})

print(result['summary'])
print(result['primary_findings'])
```

## Use Cases

- Mass grave identification
- Detention facility documentation
- Battle damage assessment
- Crowd size estimation
- Change detection over time
