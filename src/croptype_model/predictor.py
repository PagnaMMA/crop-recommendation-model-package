import os
import pandas as pd
import numpy as np
import joblib

class CropTypePredictor:
    """
    A class to load,make predictions with the trained crop type model.
    And finally display the results
    """
    
    def __init__(self):
        """Initialize the predictor by loading the trained model."""
        # Get the directory where the files are located
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Loadind of the model file(s)
        model_path = os.path.join(current_dir, 'crop_model_Gradient_Boosting.pkl')
        label_encoder_path = os.path.join(current_dir, 'label_encoders.pkl')

        try:
            model_path
            self.model = self._load_pickle_file(model_path)
        except FileNotFoundError:
            self.model = None
        
        try:
            self.label_encoder = self._load_pickle_file(label_encoder_path)
        except FileNotFoundError:
            self.label_encoder = None

    def _load_pickle_file(self, file_path):
        """
        Load a pickle file using joblib (handles both compressed and uncompressed).

        Parameters:
        -----------
        filepath : str
            Path to the pickle file

        Returns:
        --------
        object
            The unpickled object
        """
        return joblib.load(file_path)
    
    def predict_crop(self, input_data):
        """
        Predict crop type with probability scores.

        Parameters:
        -----------
        input_data : list

        Input features in the following order:
        temperature : float (Temperature in Celsius)
        rainfall : float (Rainfall in mm)
        ph : float (Soil pH level)
        moisture : float (Soil moisture 0-1)
        nitrogen : float (Nitrogen content)
        potassium : float (Potassium content)
        phosphorous : float (Phosphorous content)
        soil : str (Soil type: 'Loamy Soil', 'Peaty Soil', 'Neutral Soil')
        carbon : float (Carbon content)

        Returns:
        --------
        tuple : (predicted_crop, probability_dict)
        """
        # Create input dataframe
        input = pd.DataFrame([{
        'Temperature': input_data[0],
        'Rainfall': input_data[1],
        'PH': input_data[2],
        'Moisture': input_data[3],
        'Nitrogen': input_data[4],
        'Potassium': input_data[5],
        'Phosphorous': input_data[6],
        'Soil': input_data[7],
        'Carbon': input_data[8]
        }])

        # Encode categorical features
        try:
            input['Soil_Encoded'] = self.label_encoder['Soil'].transform(input['Soil'])
        except ValueError:
            raise ValueError(f"Unknown soil type: '{input['Soil']}'. Available: {list(self.label_encoder['Soil'].classes_)}")

        features = ['Temperature', 'Rainfall', 'PH', 'Moisture', 'Nitrogen', 'Potassium',
                'Phosphorous', 'Soil_Encoded', 'Carbon']
        
        X_input = input_data[features]
        
        # Get prediction and probabilities
        prediction_encoded = self.model.predict(X_input)[0]
        probabilities = self.model.predict_proba(X_input)[0]

        # Decode prediction
        crop_name = self.label_encoder['Crop'].inverse_transform([prediction_encoded])[0]
        # Create probability dictionary
        all_crops = self.label_encoder['Crop'].classes_
        prob_dict = {crop: prob for crop, prob in zip(all_crops, probabilities)}

        # Sort by probability
        prob_dict = dict(sorted(prob_dict.items(), key=lambda x: x[1], reverse=True))

        return crop_name, prob_dict

    def display_result(self, input_data):
        """
        Display the input data and prediction results.
        Parameters:
        -----------
        input_data : list
            Input features.
        predicted_fertilizer : str
            Predicted fertilizer name.
        predicted_probabilities : dict, optional
            Probability scores for each fertilizer.
            """
        
        print("\n" + "="*70)
        print("Predictions Results")
        print("="*70)
        temperature = input_data[0]
        rainfall = input_data[1]
        ph = input_data[2]
        moisture = input_data[3]
        nitrogen = input_data[4]
        potassium = input_data[5]
        phosphorous = input_data[6]
        soil = input_data[7]
        carbon = input_data[8]
        print(f"\n ==> Input Data:")
        print(f" - Temperature: {temperature} °C")
        print(f" - Rainfall: {rainfall} mm")
        print(f" - PH: {ph}")
        print(f" - Moisture: {moisture} %")
        print(f" - Nitrogen: {nitrogen} kg/ha")
        print(f" - Potassium: {potassium} kg/ha")
        print(f" - Phosphorous: {phosphorous} kg/ha")
        print(f" - Soil Type: {soil}")
        print(f" - Carbon: {carbon} %")
        print("\n" + "="*70)
        print(f"\n ==> Predictions:")
        try:
            predicted, probabilities = self.predict_crop(input_data)
            print(f"\n Predicted Crop: {predicted}")
            print(f"\n Top 3 Predictions: Confidence level")
            for i, (crop, prob) in enumerate(list(probabilities.items())[:3], 1):
                bar = "█" * int(prob * 50)
                print(f"  {i}. {crop:15s}: {bar} ==> {prob*100:.2f}%")
        except Exception as e:
            print(f" Error: {e}")

