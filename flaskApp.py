from flask import Flask, request, jsonify, make_response
import pickle
import pandas as pd

app = Flask(__name__)

# Load the saved Gradient Regression model
with open('Gradient_Regression.pkl', 'rb') as model_file:
    loaded_Gradient_Regression_model = pickle.load(model_file)

# Define the predict route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the CSV file from the request
        uploaded_file = request.files['file']

        # Validate the file
        if not allowed_file(uploaded_file.filename):
            return make_response('Invalid file format.', 400)

        # Read the CSV file into a Pandas DataFrame
        df = pd.read_csv(uploaded_file)

        # Handle missing values
        df = df.fillna(df.mean())

        # Perform prediction using the loaded model
        predictions = loaded_Gradient_Regression_model.predict(df)

        # Return the predictions in a standardized format
        return jsonify({'predictions': predictions.tolist()})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

# Check if the file is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Allowed file extensions
ALLOWED_EXTENSIONS = {'csv'}

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
