
import joblib
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import numpy as np


def create_sample_data():

    # Sample spam emails (label = 1)
    spam_emails = [
        "Viiiiiiagraaaa only for the ones that want to make her scream.",
        "Yo ur wom an ne eds an escapenumber in ch ma n b e th at ma n.",
        "Start increasing your odds of success & live sexually healthy.",
        "Luckyday lottery: You’ve won a free prize! Claim now.",
        "Get free access to XXX passwords—100% guaranteed!",
        "The hottest alert symbol: UTEV—target price doubled!",
        "Buy meds online—no prescription required. Click now!",
        "Totally new dating site—tired of boring people?",
        "Dear customer, don’t trust fake pharmacies. We’re legit!",
        "Cialis soft tabs—new improved formula for maximum effect.",
        "Stock alert: This one is a sure winner by Friday!",
        "The wall I see now is a link for your fusion energy.",
        "No more tickets! Bemoan, detonate, exudate now!",
        "Everyone has heard of cheaper drugs—find reliable sellers.",
        "Illegal base of sexiest young girls—just take a look.",
        "Try our revolutionary product—soft tablets available.",
        "Biggest movie archive—get access instantly.",
        "Get access to replica watches at exquisite prices.",
        "Replica Rolex—own luxury without the cost!",
        "Lose weight fast with this new, natural formula.",
        "The gift of savings—special offer ends today!",
        "Proven formula not available in retail—click to learn more.",
        "Looking for treatments for ED? Get them online.",
        "Crowe VC alert—voca scape stock symbol VCSC.",
        "Your credit report may have errors—check now!",
        "This one is shoe-in to double—volume spike spotted.",
        "Lower cost drugs from abroad—hard to find but here!",
        "Your partner needs a boost—get free bonsayy now!",
        "Boost your income with our new forex strategy.",
        "You’ve been selected for an exclusive investment opportunity.",
        "Special situation alert—providers of broadband power.",
        "Fake university degrees—get yours in 5 days!",
        "You won’t believe the returns on this penny stock!",
        "Weight loss miracle discovered by scientists.",
        "Live longer with this herbal remedy—FDA not involved.",
        "The antidote breakthrough now available online.",
        "Safe and private online pharmacy—discreet shipping.",
        "Your profile caught someone’s eye—open now!",
        "Warning! Your PC might be infected—scan here!",
        "Adult friend finder—create a free account today!",
        "Dear investor, don’t miss this 200% return opportunity.",
        "Prolong your pleasure with one easy pill!",
        "Send flowers and win her heart—click for deals.",
        "You are a lucky winner of a Caribbean cruise!",
        "Online jobs: make $500/day from home!",
        "Meet singles in your area—free registration.",
        "No fees credit card approval—guaranteed!",
        "Congratulations! Your email was selected!",
        "Hidden cam footage—click to see more.",
        "Dear sir, I’m the son of a foreign minister...",
        "Attention: Your bank account has been flagged!",
        "Get access to software 90% off retail prices."
    ]

    # Sample legitimate emails (label = 0)
    ham_emails = [
        "Attached is the weekly deal report from 10/18/01 - 10/24/01.",
        "This is the version we created earlier using the short form agreement.",
        "Pulp writing printing paper welcome to Enron's pulp writing page.",
        "Hey there—life sounds horribly busy. I just thought I’d check in.",
        "You'd think a firewall would catch all my emails from you freaks.",
        "Author JRA: please review the attached document for corrections.",
        "Netflix stock quote notification for the week of 10/20.",
        "Justice minister Harriet Harman is announced as new Labour deputy.",
        "As per our meeting, the file summarizes the latest Enron data.",
        "Hi, I’m a grad student with a statistics question. Can you help?",
        "We shall get back to you when I return from Europe in a week.",
        "Subject: Workshop at OSU—phonology and phonetics in linguistics.",
        "Dear Shelley, sounds like a good start—feel free to add more names.",
        "Inline attachment follows. Bob Williams, e-mail chain enclosed.",
        "FYI: the report from the legal department is attached.",
        "Please clarify if the estate can do a services deal with a 3rd party.",
        "Attached: current list of Master Netting Agreement assignments.",
        "WashingtonPost: Introducing America’s new war strategy.",
        "FYI: message from Heather Alon regarding the Enron compliance team.",
        "Hey everyone, looking forward to seeing y’all tomorrow.",
        "Alert: Bush mourners attend funeral as anti-Syrian vibe grows.",
        "Dear member, please reset your NYTimes password as requested.",
        "Enron slashes profits since 1997 by 20%; Dynegy talks continue.",
        "Cary Wintz: translation of minutes from September meeting attached.",
        "Meeting scheduled for next week has been cancelled.",
        "Dear Michael, you’ve been enrolled in 'Derivatives II Energy'.",
        "Mark: I enjoyed our meeting and look forward to talking again soon.",
        "I'm happy to remove the names if we’re sure it’s the right call.",
        "They are apparently not going to roll over.",
        "FYI: David McLeroy’s email regarding legal updates.",
        "Lisa Lindsley contact list for GMT+0 recipients attached.",
        "Forwarded message from Ami Chokshi—details on conference.",
        "Dale, I’m ready to cover the cost of Steve’s Houston trip.",
        "Part of a building collapses in London—at least one casualty.",
        "Hi, that's the counter for the web page—pop-ups may occur.",
        "Begin PGP signed message—show the cursor on the screen.",
        "Damian Conway: quick reply regarding Perl documentation.",
        "Dear member, here’s the link to reset your secure password.",
        "Meeting agenda attached for the derivatives trading group.",
        "Thanks for expressing interest in the upcoming training session.",
        "Please follow up on the gas sales nomination contract terms.",
        "Jackie’s birthday is today—planning to bring a cake tomorrow.",
        "Subject: Phonetics lab meeting—new research opportunities.",
        "Enron staff update: Weekly wrap-up attached in PDF format.",
        "Conference agenda revised—please review and send feedback.",
        "Looking forward to discussing the findings tomorrow.",
        "I’m not sure if that’s the correct interpretation—let’s review.",
        "Team, great work on the project milestones this week.",
        "Reminder: our staff meeting is rescheduled to Thursday.",
        "Congrats on your recent promotion!",
        "Please submit your Q4 report by Friday noon."
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