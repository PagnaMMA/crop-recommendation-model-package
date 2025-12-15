# Crop Type Recommendation Model

A machine learning model package to recommend a particular crop for a specific location according to some conditions.

## Installation

Install directly from GitHub:
```bash
pip install git+https://github.com/PagnaMMA/crop-recommendation-model-package.git
```

## Usage
```python
from croptype_model import CropTypePredictor

# Initialize the predictor
predictor = CropTypePredictor()

# Make predictions
# Example feature
features = [25.5, 110.0, 6.5, 0.65, 60.0, 70.0, 65.0, 'Alkaline Soil', 1.5]  

prediction = predictor.predict_crop(features)
print(f"Predicted crop: {prediction}")
```

## Model Details

- Model type: [Gradient Boosting]
- Features: [Temperature, Rainfall, PH, Moisture, Nitrogen, Potassium, Phosphorous, Soil_Type, Carbon]
- Target: Crop type

## Version History

- 0.1.0: Initial release
```