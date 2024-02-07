from flask import Flask, request, jsonify, make_response
import pickle
import pandas as pd

app = Flask(__name__)

# Load the saved Gradient Regression model
with open('Gradient_Regression.pkl', 'rb') as model_file:
    loaded_Gradient_Regression_model = pickle.load(model_file)
    
with open('Random_forest.pkl', 'rb') as model_file:
    Random_forest = pickle.load(model_file)

with open('best_svm_model.pkl', 'rb') as model_file:
    best_svm_model = pickle.load(model_file)

with open('linear_refression_model.pkl', 'rb') as model_file:
    linear_refression_model = pickle.load(model_file)

with open('best_decision_tree_model.pkl', 'rb') as model_file:
    best_decision_tree_model  = pickle.load(model_file)    

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

        # Perform prediction using the loaded Linear Regression model
        linear_predictions = loaded_linear_regressor_model.predict(df)

        # Return both SVM and Linear Regression predictions in a standardized format
        return jsonify({'svm_predictions': svm_predictions.tolist(), 'linear_predictions': linear_predictions.tolist()})

    except Exception as e:
        return jsonify({'error': str(e)}), 400  # Return an error response if something goes wrong
        

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
