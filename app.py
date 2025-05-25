from flask import Flask, render_template, request, flash
import joblib
import os
from werkzeug.exceptions import BadRequest

# Initialize Flask application
app = Flask(__name__)
app.secret_key = 'your-secret-key-change-this'  # Change this in production

# Global variables to store the model and vectorizer
model = None
vectorizer = None


def load_models():
    """
    Load the pre-trained spam detection model and vectorizer.
    This function is called when the application starts.
    """
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


def predict_spam(email_text):
    """
    Predict whether an email text is spam or ham.

    Args:
        email_text (str): The email text to classify

    Returns:
        tuple: (prediction, confidence) where prediction is 'spam' or 'ham'
               and confidence is the probability score
    """
    if model is None or vectorizer is None:
        raise RuntimeError("Models not loaded. Please check model files.")

    try:
        # Transform the input text using the loaded vectorizer
        # This converts the text into the same format used during training
        text_vectorized = vectorizer.transform([email_text])

        # Make prediction using the loaded model
        prediction = model.predict(text_vectorized)[0]

        # Get prediction probabilities for confidence score
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(text_vectorized)[0]
            # Get the confidence of the predicted class
            confidence = max(probabilities)
        else:
            # If model doesn't support probabilities, set confidence to None
            confidence = None

        # Convert numerical prediction to readable format
        # Assuming 1 = spam, 0 = ham (adjust based on your model)
        result = 'spam' if prediction == 1 else 'ham'

        return result, confidence

    except Exception as e:
        raise RuntimeError(f"Error during prediction: {str(e)}")


@app.route('/', methods=['GET', 'POST'])
def index():
    """
    Main route that handles both displaying the form and processing submissions.
    GET: Display the email input form
    POST: Process the submitted email and show prediction results
    """
    prediction_result = None
    confidence_score = None
    user_input = ""

    if request.method == 'POST':
        try:
            # Get the email text from the form
            email_text = request.form.get('email_text', '').strip()
            user_input = email_text  # Store for displaying back to user

            # Validate input
            if not email_text:
                flash('Please enter some email text to analyze.', 'warning')
            elif len(email_text) < 3:
                flash('Please enter at least 3 characters.', 'warning')
            else:
                # Make prediction
                prediction_result, confidence_score = predict_spam(email_text)

                # Add appropriate flash message based on result
                if prediction_result == 'spam':
                    flash('⚠️ This email appears to be SPAM!', 'danger')
                else:
                    flash('✅ This email appears to be legitimate (HAM).', 'success')

        except RuntimeError as e:
            flash(f'Error: {str(e)}', 'danger')
        except Exception as e:
            flash('An unexpected error occurred. Please try again.', 'danger')
            print(f"Unexpected error: {str(e)}")

    # Render the template with results
    return render_template('index.html',
                           prediction=prediction_result,
                           confidence=confidence_score,
                           user_input=user_input)


@app.route('/health')
def health_check():
    """
    Simple health check endpoint to verify the application is running
    and models are loaded properly.
    """
    status = {
        'status': 'healthy',
        'model_loaded': model is not None,
        'vectorizer_loaded': vectorizer is not None
    }
    return status


@app.errorhandler(404)
def not_found_error(error):
    """Handle 404 errors by redirecting to home page"""
    return render_template('index.html',
                           error_message="Page not found. Redirected to home."), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors gracefully"""
    return render_template('index.html',
                           error_message="Internal server error. Please try again."), 500


if __name__ == '__main__':
    # Load the models when the application starts
    load_models()

    # Check if models were loaded successfully before starting the server
    if model is None or vectorizer is None:
        print("\n" + "=" * 50)
        print("⚠️  WARNING: Models failed to load!")
        print("Please ensure you have the following files:")
        print("- spam_model.pkl")
        print("- vectorizer.pkl")
        print("\nThe application will start but predictions won't work.")
        print("=" * 50 + "\n")

    # Start the Flask development server
    # In production, use a proper WSGI server like Gunicorn
    app.run(debug=True, host='0.0.0.0', port=5000)