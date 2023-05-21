from flask import Flask, request, jsonify
import pandas as pd
import pickle

# Load the trained model from a pickle file
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Define the Flask app
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the input data from the request body as a pandas DataFrame
        input_data = request.get_json()
        input_df = pd.DataFrame(input_data, index=[0])
        print('dataframe :',input_df)
        # Make a prediction using the loaded model
        prediction = model.predict(input_df)
        print('at first', type(prediction))
        # Convert the prediction to a list
        prediction_list = prediction.tolist()
        print('after', type(prediction_list))
        prediction = prediction_list[0]

        # Convert the prediction to binary output
        if prediction[0] > 0.5:
            prediction = f"{1} : customer is likely to churn"
        else:
            prediction = f"{0} : customer is not likely to churn"

        # Format the prediction as a JSON response
        response = {'prediction': prediction}

        return jsonify(response)

    except Exception as e:
        error = {'error': str(e)}
        return jsonify(error)

if __name__ == '__main__':
    app.run(port=5006, debug=True)

#########-------INPUT JSON FROM BODY REQUEST #########
# {
#   "gender": 0,
#   "SeniorCitizen": 0,
#   "Partner": 1,
#   "Dependents": 0,
#   "tenure": 0.169014085,
#   "PhoneService": 1,
#   "MultipleLines": 0,
#   "OnlineSecurity": 1,
#   "OnlineBackup": 0,
#   "DeviceProtection": 0,
#   "TechSupport": 0,
#   "StreamingTV": 0,
#   "StreamingMovies": 0,
#   "PaperlessBilling": 1,
#   "MonthlyCharges": 0.33681592,
#   "TotalCharges": 0.075219248,
#   "InternetService_DSL": 1,
#   "InternetService_Fiber optic": 0,
#   "InternetService_No": 0,
#   "Contract_Month-to-month": 1,
#   "Contract_One year": 0,
#   "Contract_Two year": 0,
#   "PaymentMethod_Bank transfer (automatic)": 0,
#   "PaymentMethod_Credit card (automatic)": 0,
#   "PaymentMethod_Electronic check": 1,
#   "PaymentMethod_Mailed check": 0
# }