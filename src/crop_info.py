"""
crop_info.py
-------------
Static reference information for each crop (season, growing period, soil
type, water need, and cultivation tips). This information already existed
in the original AgriSmart repository (backend/app.py) and is reused here,
unmodified in content, purely for the terminal OUTPUT LAYER — it is
descriptive reference data, not something derived from the ML model.
"""

CROP_INFO = {
    "rice": {"season": "Kharif", "growing_period": "120-150 days", "soil_type": "Clay loam",
              "water_need": "High", "tips": "Keep fields flooded, use nitrogen-rich fertilizer, control weeds early."},
    "maize": {"season": "Kharif", "growing_period": "90-120 days", "soil_type": "Well-drained loam",
              "water_need": "Moderate", "tips": "Ensure good drainage, rotate crops, monitor for pests."},
    "chickpea": {"season": "Rabi", "growing_period": "90-120 days", "soil_type": "Well-drained loam",
              "water_need": "Moderate", "tips": "Avoid waterlogging, use disease-free seeds, inoculate with Rhizobium."},
    "kidneybeans": {"season": "Kharif", "growing_period": "80-100 days", "soil_type": "Loamy soil",
              "water_need": "Moderate", "tips": "Provide support for climbing varieties, control bean beetles."},
    "pigeonpeas": {"season": "Kharif", "growing_period": "150-180 days", "soil_type": "Deep loam",
              "water_need": "Low", "tips": "Drought-resistant, good for intercropping."},
    "mothbeans": {"season": "Kharif", "growing_period": "60-80 days", "soil_type": "Sandy loam",
              "water_need": "Low", "tips": "Excellent for arid regions, good ground cover."},
    "mungbean": {"season": "Kharif", "growing_period": "60-90 days", "soil_type": "Loamy soil",
              "water_need": "Moderate", "tips": "Fast-growing, good for crop rotation."},
    "blackgram": {"season": "Kharif", "growing_period": "80-100 days", "soil_type": "Clay loam",
              "water_need": "Moderate", "tips": "Good for rice fallows, nitrogen-fixing."},
    "lentil": {"season": "Rabi", "growing_period": "80-110 days", "soil_type": "Loamy soil",
              "water_need": "Low", "tips": "Nitrogen-fixing, good for crop rotation."},
    "pomegranate": {"season": "Year-round", "growing_period": "5-7 years to fruit", "soil_type": "Deep loam",
              "water_need": "Moderate", "tips": "Drought-tolerant, good for arid regions."},
    "banana": {"season": "Year-round", "growing_period": "9-12 months", "soil_type": "Rich loam",
              "water_need": "High", "tips": "Requires regular watering, protect from wind."},
    "mango": {"season": "Summer", "growing_period": "3-6 years to fruit", "soil_type": "Deep loam",
              "water_need": "Moderate", "tips": "Drought-tolerant, good for tropical regions."},
    "grapes": {"season": "Summer", "growing_period": "2-3 years to fruit", "soil_type": "Well-drained loam",
              "water_need": "Moderate", "tips": "Good trellis support, control pests."},
    "watermelon": {"season": "Summer", "growing_period": "70-90 days", "soil_type": "Sandy loam",
              "water_need": "High", "tips": "Requires lots of space, regular watering."},
    "muskmelon": {"season": "Summer", "growing_period": "70-90 days", "soil_type": "Sandy loam",
              "water_need": "Moderate", "tips": "Good drainage, control powdery mildew."},
    "apple": {"season": "Fall", "growing_period": "3-5 years to fruit", "soil_type": "Well-drained loam",
              "water_need": "Moderate", "tips": "Requires chilling hours, good pruning."},
    "orange": {"season": "Winter", "growing_period": "3-5 years to fruit", "soil_type": "Well-drained loam",
              "water_need": "Moderate", "tips": "Frost-sensitive, good drainage needed."},
    "papaya": {"season": "Year-round", "growing_period": "6-9 months to fruit", "soil_type": "Sandy loam",
              "water_need": "Moderate", "tips": "Fast-growing, protect from wind."},
    "coconut": {"season": "Year-round", "growing_period": "5-7 years to fruit", "soil_type": "Sandy loam",
              "water_need": "High", "tips": "Requires coastal climate, regular watering."},
    "cotton": {"season": "Kharif", "growing_period": "150-180 days", "soil_type": "Black soil",
              "water_need": "Moderate", "tips": "Good for crop rotation, control bollworms."},
    "jute": {"season": "Kharif", "growing_period": "120-150 days", "soil_type": "Alluvial soil",
              "water_need": "High", "tips": "Requires flooding, good for fiber production."},
    "coffee": {"season": "Year-round", "growing_period": "3-4 years to fruit", "soil_type": "Volcanic loam",
              "water_need": "High", "tips": "Shade-grown, regular pruning needed."},
}


def get_crop_info(crop_name: str) -> dict:
    return CROP_INFO.get(str(crop_name).lower(), {})
