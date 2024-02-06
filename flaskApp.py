from flask import Flask, request, jsonify
import pickle
import pandas as pd


app = Flask(__name__)

# Load the saved SVM model
with open('best_svm_model.pkl', 'rb') as model_file:
    loaded_svm_model = pickle.load(model_file)

# Define the predict route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the CSV file from the request
        uploaded_file = request.files['file']

        # Read the CSV file into a Pandas DataFrame
        df = pd.read_csv(uploaded_file)

        # Perform prediction using the loaded model
        predictions = loaded_svm_model.predict(df)

        # Return the predictions in a standardized format
        return jsonify({'predictions': predictions.tolist()})

    except Exception as e:
        return jsonify({'error': str(e)}), 400  # Return an error response if something goes wrong

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)