from flask import Flask, render_template, request, flash
import joblib
import os
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Initialize Flask application
app = Flask(__name__)
app.secret_key = 'secret12345'

# Global variables to store the model and vectorizer
model = None
vectorizer = None


def load_models():
    global model, vectorizer
    try:
        # Load the trained model and vectorizer
        model_path = os.path.join(os.path.dirname(__file__), 'spam_model.pkl')
        vectorizer_path = os.path.join(os.path.dirname(__file__), 'vectorizer.pkl')

        # Check if model files exist
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        if not os.path.exists(vectorizer_path):
            raise FileNotFoundError(f"Vectorizer file not found: {vectorizer_path}")

        # Load the model and vectorizer using joblib
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)

        print("✓ Model and vectorizer loaded successfully!")

    except Exception as e:
        print(f"✗ Error loading models: {str(e)}")
        print("Please ensure 'spam_model.pkl' and 'vectorizer.pkl' are in the project directory.")


def evaluate_model():
    # Your fixed test emails
    test_emails = [
        "Win money fast! Click here now!",
        "Meeting scheduled for next Tuesday",
        "Free iPhone! Limited time offer!",
        "Your order confirmation and tracking details",
        "Don't miss this investment opportunity!",
        "Project update: please review the attached file",
        "Lose 10 pounds in 7 days—guaranteed!",
        "Lunch with the client is confirmed at 1 PM",
        "Congratulations! You’ve been selected for a prize!",
        "Weekly report ready for your review"
    ]

    # Manually defined true labels for above emails (1 = spam, 0 = ham)
    true_labels = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]

    try:
        X_test = vectorizer.transform(test_emails)
        y_pred = model.predict(X_test)

        accuracy = accuracy_score(true_labels, y_pred)
        precision = precision_score(true_labels, y_pred, zero_division=0)
        recall = recall_score(true_labels, y_pred, zero_division=0)
        f1 = f1_score(true_labels, y_pred, zero_division=0)

        print("\nModel evaluation metrics on fixed test emails:")
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1 Score:  {f1:.4f}")
        print("-" * 40)

    except Exception as e:
        print(f"✗ Error during evaluation: {str(e)}")



def predict_spam(email_text):
    if model is None or vectorizer is None:
        raise RuntimeError("Models not loaded. Please check model files.")

    try:
        # Transform the input text using the loaded vectorizer
        text_vectorized = vectorizer.transform([email_text])

        # Make prediction using the loaded model
        prediction = model.predict(text_vectorized)[0]

        # Get prediction probabilities for confidence score
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(text_vectorized)[0]
            confidence = max(probabilities)
        else:
            confidence = None

        result = 'spam' if prediction == 1 else 'ham'

        return result, confidence

    except Exception as e:
        raise RuntimeError(f"Error during prediction: {str(e)}")


@app.route('/', methods=['GET', 'POST'])
def index():
    prediction_result = None
    confidence_score = None
    user_input = ""

    if request.method == 'POST':
        try:
            email_text = request.form.get('email_text', '').strip()
            user_input = email_text

            if not email_text:
                flash('Please enter some email text to analyze.', 'warning')
            elif len(email_text) < 3:
                flash('Please enter at least 3 characters.', 'warning')
            else:
                prediction_result, confidence_score = predict_spam(email_text)

                if prediction_result == 'spam':
                    flash('⚠️ This email appears to be SPAM!', 'danger')
                else:
                    flash('✅ This email appears to be legitimate (HAM).', 'success')

        except RuntimeError as e:
            flash(f'Error: {str(e)}', 'danger')
        except Exception as e:
            flash('An unexpected error occurred. Please try again.', 'danger')
            print(f"Unexpected error: {str(e)}")

    return render_template('index.html',
                           prediction=prediction_result,
                           confidence=confidence_score,
                           user_input=user_input)


@app.route('/health')
def health_check():
    status = {
        'status': 'healthy',
        'model_loaded': model is not None,
        'vectorizer_loaded': vectorizer is not None
    }
    return status


@app.errorhandler(404)
def not_found_error(error):
    return render_template('index.html',
                           error_message="Page not found. Redirected to home."), 404


@app.errorhandler(500)
def internal_error(error):
    return render_template('index.html',
                           error_message="Internal server error. Please try again."), 500


if __name__ == '__main__':
    load_models()

    if model is None or vectorizer is None:
        print("\n" + "=" * 50)
        print("⚠️  WARNING: Models failed to load!")
        print("Please ensure you have the following files:")
        print("- spam_model.pkl")
        print("- vectorizer.pkl")
        print("\nThe application will start but predictions won't work.")
        print("=" * 50 + "\n")
    else:
        evaluate_model()

    app.run(debug=True, host='0.0.0.0', port=5000)
