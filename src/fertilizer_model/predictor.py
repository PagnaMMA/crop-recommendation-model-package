import pickle
import os
import numpy as np
import gzip

class FertilizerPredictor:
    """
    A class to load,make predictions with the trained fertilizer model.
    And finally display the results
    """
    
    def __init__(self):
        """Initialize the predictor by loading the trained model."""
        # Get the directory where the files are located
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Loadind of the model file(s)
        model_path = os.path.join(current_dir, 'fertilizer_model_Gradient_Boosting.pkl')
        scaler_path = os.path.join(current_dir, 'fertilizer_scaler.pkl')
        label_encoder_path = os.path.join(current_dir, 'fertilizer_label_encoder.pkl')
        soil_encoder_path = os.path.join(current_dir, 'fertilizer_soil_encoder.pkl')
        crop_encoder_path = os.path.join(current_dir, 'fertilizer_crop_encoder.pkl')

        try:
            model_path
            self.model = self._load_pickle_file(model_path)
        except FileNotFoundError:
            self.model = None
        
        try:
            scaler_path
            self.scaler = self._load_pickle_file(scaler_path)
            label_encoder_path
            self.label_encoder = self._load_pickle_file(label_encoder_path)
            soil_encoder_path
            self.soil_encoder = self._load_pickle_file(soil_encoder_path)
            crop_encoder_path
            self.crop_encoder = self._load_pickle_file(crop_encoder_path)
        except FileNotFoundError:
            self.label_encoder = None
            self.soil_encoder = None
            self.crop_encoder = None

    def _load_pickle_file(self, file_path):
        """
        Load a pickle file, handling both compressed and uncompressed formats.
        
        Parameters:
        -----------
        filepath : str
            Path to the pickle file
            
        Returns:
        --------
        object
            The unpickled object
        """
        # First, try to detect if it's compressed by reading the first bytes
        with open(file_path, 'rb') as f:
            first_bytes = f.read(2)
        
        # Check for gzip magic number (1f 8b) or zlib header (78 01, 78 9c, 78 da, etc.)
        is_compressed = (
            first_bytes[:2] == b'\x1f\x8b' or  # gzip
            (first_bytes[0] == 0x78 and first_bytes[1] in [0x01, 0x5e, 0x9c, 0xda])  # zlib
        )
        
        if is_compressed:
            # Try decompressing with gzip first
            try:
                with gzip.open(file_path, 'rb') as f:
                    return pickle.load(f)
            except (gzip.BadGzipFile, OSError):
                # If gzip fails, try raw zlib decompression
                import zlib
                with open(file_path, 'rb') as f:
                    compressed_data = f.read()
                decompressed_data = zlib.decompress(compressed_data)
                return pickle.loads(decompressed_data)
        else:
            # Not compressed, load normally
            with open(file_path, 'rb') as f:
                return pickle.load(f)
    
    def predict_fertilizer(self, input_data):
        """
        Predict fertilizer with probability scores.

        Parameters:
        -----------
        input_data : list (same format as predict_fertilizer)

        Returns:
        --------
        tuple : (predicted_fertilizer, probability_dict)
        """
        # Extract and encode features (same as above)
        temperature = input_data[0]
        rainfall = input_data[1]
        ph = input_data[2]
        moisture = input_data[3]
        nitrogen = input_data[4]
        potassium = input_data[5]
        phosphorous = input_data[6]
        soil = input_data[7]
        crop = input_data[8]

        # Encode categorical features
        try:
            soil_encoded = self.soil_encoder.transform([soil])[0]
        except ValueError:
            raise ValueError(f"Unknown soil type: '{soil}'. Available: {list(self.soil_encoder.classes_)}")

        try:
            crop_encoded = self.crop_encoder.transform([crop])[0]
        except ValueError:
            raise ValueError(f"Unknown crop: '{crop}'. Available: {list(self.crop_encoder.classes_)}")


        features = np.array([[temperature, rainfall, ph, moisture, nitrogen,
                            potassium, phosphorous, soil_encoded, crop_encoded]])

        # Get prediction and probabilities
        prediction_encoded = self.model.predict(features)[0]
        probabilities = self.model.predict_proba(features)[0]

        # Decode prediction
        fertilizer_name = self.label_encoder.inverse_transform([prediction_encoded])[0]
        # Create probability dictionary
        all_fertilizers = self.label_encoder.classes_
        prob_dict = {fert: prob for fert, prob in zip(all_fertilizers, probabilities)}

        # Sort by probability
        prob_dict = dict(sorted(prob_dict.items(), key=lambda x: x[1], reverse=True))

        return fertilizer_name, prob_dict

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
            Probability scores for each fertilizer."""
        
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
        crop = input_data[8]
        print(f"\n ==> Input Data:")
        print(f" - Temperature: {temperature} °C")
        print(f" - Rainfall: {rainfall} mm")
        print(f" - PH: {ph}")
        print(f" - Moisture: {moisture} %")
        print(f" - Nitrogen: {nitrogen} kg/ha")
        print(f" - Potassium: {potassium} kg/ha")
        print(f" - Phosphorous: {phosphorous} kg/ha")
        print(f" - Soil Type: {soil}")
        print(f" - Crop: {crop}")
        print("\n" + "="*70)
        print(f"\n ==> Predictions:")
        try:
            predicted, probabilities = self.predict_fertilizer(input_data)
            print(f"\n Predicted Fertilizer: {predicted}")
            print(f"\nTop 3 Predictions: Confidence level")
            for i, (fert, prob) in enumerate(list(probabilities.items())[:3], 1):
                print(f"  {i}. {fert}: {prob*100:.2f}%")
        except Exception as e:
            print(f" Error: {e}")

