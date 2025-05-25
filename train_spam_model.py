"""
Sample script to train a spam detection model and save it for use with the Flask app.
Run this script to generate the required spam_model.pkl and vectorizer.pkl files.

This script creates a simple spam detector using sample data. In a real scenario,
you would use a proper dataset like the SMS Spam Collection Dataset.
"""

import joblib
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import numpy as np


def create_sample_data():
    """
    Create sample spam and ham email data for demonstration.
    In a real scenario, you would load this from a proper dataset.
    """

    # Sample spam emails (label = 1)
    spam_emails = [
        "Congratulations! You've won $1,000,000! Click here to claim your prize!",
        "URGENT: Your account will be suspended! Verify now!",
        "Free money! No strings attached! Act now!",
        "You have been selected for a special offer! Limited time!",
        "Claim your free iPhone now! Click here immediately!",
        "WINNER! You've won the lottery! Send us your details!",
        "Free credit check! No hidden fees! Apply today!",
        "Get rich quick! Investment opportunity! Don't miss out!",
        "Pharmacy discount! Cheap medications! Order now!",
        "Work from home! Make $5000 per week! Easy money!",
        "Nigerian prince needs your help! Million dollar reward!",
        "Free vacation! All expenses paid! Claim your trip!",
        "Debt consolidation! Lower your payments! Call now!",
        "Weight loss miracle! Lose 30 pounds in 30 days!",
        "Casino bonus! Free spins! Win big today!",
        "Tax refund waiting! Claim your money back!",
        "Hot singles in your area! Meet them tonight!",
        "Free trial! No commitment! Cancel anytime!",
        "Exclusive offer! Members only! Limited availability!",
        "Cash advance! Get money fast! No credit check!"
    ]

    # Sample legitimate emails (label = 0)
    ham_emails = [
        "Meeting scheduled for tomorrow at 2 PM in Conference Room A",
        "Your order has been shipped and will arrive in 2-3 business days",
        "Quarterly report is due next Friday. Please submit on time.",
        "Thank you for your purchase. Here is your receipt.",
        "Team lunch planned for Thursday at the Italian restaurant",
        "Project deadline extended to next Monday due to holiday",
        "Your subscription will expire next month. Renewal options available.",
        "Weather alert: Heavy rain expected this afternoon",
        "Library book due date reminder: Return by end of week",
        "Doctor appointment confirmed for Tuesday at 10:30 AM",
        "School parent-teacher conference scheduled for next week",
        "Flight confirmation: Your boarding pass is ready",
        "Monthly newsletter: Updates from our development team",
        "Password reset requested for your account",
        "Invoice attached for services rendered last month",
        "Welcome to our platform! Here's how to get started",
        "System maintenance scheduled for this weekend",
        "Happy birthday! Hope you have a wonderful day",
        "Conference call details for tomorrow's client meeting",
        "Your package has been delivered to your front door"
    ]

    # Combine data and create labels
    emails = spam_emails + ham_emails
    labels = [1] * len(spam_emails) + [0] * len(ham_emails)  # 1 = spam, 0 = ham

    return emails, labels


def train_spam_detector():
    """
    Train a spam detection model using Naive Bayes and CountVectorizer
    """
    print("Creating sample training data...")
    emails, labels = create_sample_data()

    print(f"Training data: {len(emails)} emails ({sum(labels)} spam, {len(labels) - sum(labels)} ham)")

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        emails, labels, test_size=0.2, random_state=42, stratify=labels
    )

    print(f"Training set: {len(X_train)} emails")
    print(f"Testing set: {len(X_test)} emails")

    # Create and fit the CountVectorizer
    print("\nTraining CountVectorizer...")
    vectorizer = CountVectorizer(
        lowercase=True,  # Convert to lowercase
        stop_words='english',  # Remove common English stop words
        max_features=1000,  # Limit to top 1000 features
        ngram_range=(1, 2)  # Use both unigrams and bigrams
    )

    # Transform training data
    X_train_vectorized = vectorizer.fit_transform(X_train)
    X_test_vectorized = vectorizer.transform(X_test)

    print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")

    # Train the Naive Bayes classifier
    print("\nTraining Naive Bayes classifier...")
    model = MultinomialNB(alpha=1.0)  # Laplace smoothing
    model.fit(X_train_vectorized, y_train)

    # Make predictions on test set
    y_pred = model.predict(X_test_vectorized)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nModel Accuracy: {accuracy:.2%}")

    # Print detailed classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))

    # Save the trained model and vectorizer
    print("\nSaving model and vectorizer...")
    joblib.dump(model, 'spam_model.pkl')
    joblib.dump(vectorizer, 'vectorizer.pkl')

    print("✓ Model saved as 'spam_model.pkl'")
    print("✓ Vectorizer saved as 'vectorizer.pkl'")

    # Test with some example predictions
    print("\n" + "=" * 50)
    print("Testing model with sample predictions:")
    print("=" * 50)

    test_emails = [
        "Win money fast! Click here now!",
        "Meeting scheduled for next Tuesday",
        "Free iPhone! Limited time offer!",
        "Your order confirmation and tracking details"
    ]

    for email in test_emails:
        email_vectorized = vectorizer.transform([email])
        prediction = model.predict(email_vectorized)[0]
        probabilities = model.predict_proba(email_vectorized)[0]

        result = "SPAM" if prediction == 1 else "HAM"
        confidence = max(probabilities)

        print(f"\nEmail: '{email}'")
        print(f"Prediction: {result} (Confidence: {confidence:.2%})")

    print("\n" + "=" * 50)
    print("Model training completed successfully!")
    print("You can now run the Flask app with: python app.py")
    print("=" * 50)


if __name__ == "__main__":
    train_spam_detector()