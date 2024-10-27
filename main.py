import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import re
from urllib.parse import urlparse
from textblob import TextBlob

# Load the dataset
raw_mail_data = pd.read_csv('/content/sample_phishing_data.csv')
mail_data = raw_mail_data.fillna('')

# Check for the existence of required columns
required_columns = ['Message', 'From', 'Attachments', 'Category']
for column in required_columns:
    if column not in mail_data.columns:
        print(f"Warning: Column '{column}' not found in the dataset.")
        mail_data[column] = ''  # Create the column with empty values if missing

# Convert 'Category' column to integers (0 for spam, 1 for ham)
mail_data.loc[mail_data['Category'] == 'spam', 'Category'] = 0
mail_data.loc[mail_data['Category'] == 'ham', 'Category'] = 1
Y = mail_data['Category'].astype('int')  # Now this should work without error

# Feature extraction functions

# Check for phishing keywords in the email body
def keyword_presence(text):
    phishing_keywords = ["verify", "account", "login", "urgent", "click here", "reset password", "update", "bank"]
    return int(any(keyword in text.lower() for keyword in phishing_keywords))

# Check if the 'From' address contains suspicious domain patterns
def suspicious_from_domain(email):
    suspicious_domains = ["ru", "cn", "br", "xyz", "top"]
    match = re.search(r'@.*\.([a-z]{2,3})$', email)
    return int(match.group(1) in suspicious_domains) if match else 0

# Extract and check URLs within the email body for suspicious patterns
def suspicious_links(text):
    urls = re.findall(r'(https?://[^\s]+)', text)
    for url in urls:
        domain = urlparse(url).netloc
        if re.match(r'^\d{1,3}(\.\d{1,3}){3}$', domain) or domain.endswith((".ru", ".cn", ".xyz")):
            return 1  # Phishing likely if IP-based or unusual TLD
    return 0

# Check for dangerous attachment types
def dangerous_attachments(attachment_names):
    dangerous_types = [".exe", ".zip", ".js", ".scr"]
    return int(any(name.lower().endswith(ext) for ext in dangerous_types for name in attachment_names.split(',')))

# Detect urgency based on sentiment analysis
def detect_urgency(text):
    sentiment = TextBlob(text).sentiment
    return int(sentiment.polarity < 0 and "!" in text)  # Urgent emails often have negative polarity and exclamations

# Detect email complexity through HTML-to-text ratio
def html_text_ratio(text):
    html_tags = re.findall(r'<[^>]+>', text)
    return len(html_tags) / (len(text) + 1)  # Small denominator to avoid division by zero

# Apply feature extraction
mail_data['keyword_presence'] = mail_data['Message'].apply(keyword_presence)
mail_data['suspicious_from'] = mail_data['From'].apply(suspicious_from_domain)
mail_data['suspicious_links'] = mail_data['Message'].apply(suspicious_links)
mail_data['dangerous_attachments'] = mail_data['Attachments'].apply(dangerous_attachments)
mail_data['urgency'] = mail_data['Message'].apply(detect_urgency)
mail_data['html_text_ratio'] = mail_data['Message'].apply(html_text_ratio)

# Combine text-based feature with additional features
X_text = mail_data['Message']
additional_features = mail_data[['keyword_presence', 'suspicious_from', 'suspicious_links', 'dangerous_attachments', 'urgency', 'html_text_ratio']]

# Feature extraction using TF-IDF for text data
tfidf_vectorizer = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)
X_text_features = tfidf_vectorizer.fit_transform(X_text)

# Concatenate TF-IDF features with additional features
from scipy.sparse import hstack
X_combined_features = hstack([X_text_features, np.array(additional_features)])

# Split data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X_combined_features, Y, test_size=0.2, random_state=3)

# Train a more robust model (Random Forest)
model = RandomForestClassifier(n_estimators=100, random_state=3)
model.fit(X_train, Y_train)

# Evaluate model accuracy
train_predictions = model.predict(X_train)
train_accuracy = accuracy_score(Y_train, train_predictions)
print('Accuracy on training data:', train_accuracy)

test_predictions = model.predict(X_test)
test_accuracy = accuracy_score(Y_test, test_predictions)
print('Accuracy on test data:', test_accuracy)


# Test with an example email
input_mail = ["Urgent: Your account has been compromised. Click here to reset your password immediately to secure your account."]
input_attachments = "report.zip"  # Specify attachments if any

# Prepare the input data
input_features = {
    "Message": input_mail[0],
    "From": "no-reply@secure.com.ru",
    "Attachments": input_attachments
}
def check_emails(emails_list):
    """
    Check multiple emails for phishing attempts.

    Parameters:
    emails_list: list of dictionaries, each containing:
        - 'message': Email body text
        - 'from_address': Sender's email address
        - 'attachments': Comma-separated string of attachment names (optional)

    Returns:
    DataFrame with input details and prediction results
    """
    # Prepare input data
    input_data = []
    for email in emails_list:
        input_features = {
            "Message": email.get('message', ''),
            "From": email.get('from_address', ''),
            "Attachments": email.get('attachments', '')
        }
    input_data.append(input_features)

    input_df = pd.DataFrame(input_data)

    # Extract features
    input_df['keyword_presence'] = input_df['Message'].apply(keyword_presence)
    input_df['suspicious_from'] = input_df['From'].apply(suspicious_from_domain)
    input_df['suspicious_links'] = input_df['Message'].apply(suspicious_links)
    input_df['dangerous_attachments'] = input_df['Attachments'].apply(dangerous_attachments)
    input_df['urgency'] = input_df['Message'].apply(detect_urgency)
    input_df['html_text_ratio'] = input_df['Message'].apply(html_text_ratio)

    # Transform text features
    input_text_features = tfidf_vectorizer.transform(input_df['Message'])
    input_combined_features = hstack([
        input_text_features,
        np.array(input_df[[
            'keyword_presence', 'suspicious_from', 'suspicious_links',
            'dangerous_attachments', 'urgency', 'html_text_ratio'
        ]])
    ])

    # Make predictions
    predictions = model.predict(input_combined_features)

    # Add predictions to results
    results_df = input_df.copy()
    results_df['Prediction'] = ['Legitimate' if pred == 1 else 'Phishing' for pred in predictions]

    # Add confidence scores
    prediction_proba = model.predict_proba(input_combined_features)
    results_df['Confidence'] = [max(proba) * 100 for proba in prediction_proba]

    return results_df[['Message', 'From', 'Attachments', 'Prediction', 'Confidence']]

input_df = pd.DataFrame([input_features])
input_df['keyword_presence'] = input_df['Message'].apply(keyword_presence)
input_df['suspicious_from'] = input_df['From'].apply(suspicious_from_domain)
input_df['suspicious_links'] = input_df['Message'].apply(suspicious_links)
input_df['dangerous_attachments'] = input_df['Attachments'].apply(dangerous_attachments)
input_df['urgency'] = input_df['Message'].apply(detect_urgency)
input_df['html_text_ratio'] = input_df['Message'].apply(html_text_ratio)

# TF-IDF transformation for message
input_text_features = tfidf_vectorizer.transform(input_df['Message'])
input_combined_features = hstack([input_text_features, np.array(input_df[['keyword_presence', 'suspicious_from', 'suspicious_links', 'dangerous_attachments', 'urgency', 'html_text_ratio']])])

# Prediction
prediction = model.predict(input_combined_features)
print("Prediction:", "Legitimate" if prediction[0] == 1 else "Phishing")

test_emails = [
        {
            'message': "Urgent: Your account has been compromised. Click here to reset your password immediately to secure your account.",
            'from_address': "no-reply@secure.com.ru",
            'attachments': "report.zip"
        },
        {
            'message': "Hi team, Here's the quarterly report we discussed in the meeting. Let me know if you have any questions.",
            'from_address': "john.doe@company.com",
            'attachments': "quarterly_report.pdf"
        },
        {
            'message': "URGENT: You've won $1,000,000! Click now to claim your prize before it expires!!!",
            'from_address': "prize@winner.xyz",
            'attachments': "claim_form.exe"
        }
    ]
    
    # Check the emails
results = check_emails(test_emails)
    
    # Display results
print("\Email Analysis Results:")
print("=" * 100)
for idx, row in results.iterrows():
        print(f"\Email #{idx + 1}")
        print(f"From: {row['From']}")
        print(f"Message: {row['Message'][:100]}...")  # Show first 100 characters
        print(f"Attachments: {row['Attachments']}")
        print(f"Prediction: {row['Prediction']}")
        print(f"Confidence: {row['Confidence']:.2f}%")
        print("-" * 100)
# Example of a more clearly legitimate email
legitimate_test_emails = [
    {
        'message': """Dear Marketing Team,

I hope this email finds you well. I'm writing to share the Q4 2023 marketing performance report that we discussed in yesterday's team meeting. 

Key highlights:
- Social media engagement up 25%
- Email campaign open rates increased to 28%
- New website visits exceeded targets by 15%

The full report is attached as a PDF. Please review it before our follow-up meeting next Tuesday at 2 PM.

Best regards,
Sarah Anderson
Marketing Director""",
        'from_address': "sarah.anderson@company.com",
        'attachments': "Q4_2023_Marketing_Report.pdf"
    },
    {
        'message': """Team,

The software deployment scheduled for this weekend has been successfully completed. All systems are running normally and we've updated the documentation accordingly.

Changes implemented:
1. Security patches applied
2. Database optimization completed
3. New features added to the dashboard

Please test your applications and report any issues through the standard support channels.

Thanks,
Michael Chen
IT Operations""",
        'from_address': "michael.chen@company.com",
        'attachments': "deployment_summary.pdf"
    }
]

# Test the emails
results = check_emails(legitimate_test_emails)

# Display results
print(r"\Email Analysis Results:")
print("=" * 80)
for idx, row in results.iterrows():
    print(rf"\Email #{idx + 1}")
    print(f"From: {row['From']}")
    print(f"Message preview: {row['Message'][:200]}...")  # Show first 200 characters
    print(f"Attachments: {row['Attachments']}")
    print(f"Prediction: {row['Prediction']}")
    print(f"Confidence: {row['Confidence']:.2f}%")
    print("-" * 80)

    # Examples of highly suspicious phishing emails
phishing_test_emails = [
    {
        'message': """URGENT ACTION REQUIRED!!!

Your Bank Account Access Has Been Suspended!

We have detected suspicious login attempts from an unknown device. Your account access has been temporarily blocked for security reasons.

IMMEDIATE ACTION REQUIRED:
>> CLICK HERE TO VERIFY: http://185.32.55.98/secure-bank/login.php
>> Enter your full banking credentials
>> Update your security information

WARNING: Failure to verify within 24 hours will result in permanent account closure!

Note: Do not ignore this message as your funds may be at risk!!!

Security Department
Global Banking Authority""",
        'from_address': "security.alert@secure-bank-verify.ru",
        'attachments': "Account_Verify.exe, SecurityUpdate.zip"
    },
    {
        'message': """CONGRATULATIONS!!!

YOU HAVE WON THE INTERNATIONAL EMAIL LOTTERY!!!
Prize Amount: $5,500,000.00 USD

Your email was randomly selected as the GRAND PRIZE WINNER of our International Email Lottery Program!!!

To claim your prize, you must act NOW:
1. Send your personal details:
   - Full Name
   - Bank Account Number
   - Credit Card Information
   - Social Security Number
   - Copy of Passport

2. Pay processing fee of $99.99 via Western Union

RESPOND WITHIN 24 HOURS OR FORFEIT YOUR PRIZE!!!

Contact Agent Mr. James Williams
Email: claim-prize@winner-lottery.xyz
Tel: +234 808 555 1234

THIS IS NOT A SCAM! 100% LEGITIMATE INTERNATIONAL LOTTERY!!!""",
        'from_address': "lottery.winner@international-prize.xyz",
        'attachments': "Prize_Claim_Form.js, WinnerDocument.scr"
    },
    {
        'message': """Dear User,

We've noticed unauthorized access to your email account from the following locations:
- Beijing, China
- Lagos, Nigeria
- Moscow, Russia

To prevent account theft, please verify your identity immediately:
1. Download and run the attached security scanner
2. Enter your email password when prompted
3. Provide alternative email accounts and passwords for verification

WARNING: Account will be permanently deleted in 12 hours if not verified!!!

IT Security Team""",
        'from_address': "help@mail-security-verify.cn",
        'attachments': "Security_Scanner.exe, AccountProtect.bat"
    }
]

# Test the emails
results = check_emails(phishing_test_emails)

# Display detailed results with suspicious indicators
print("\nPhishing Email Analysis Results:")
print("=" * 100)
for idx, row in results.iterrows():
    print(f"\nEmail #{idx + 1}")
    print(f"From: {row['From']}")
    print(f"Message preview: {row['Message'][:200]}...")
    print(f"Attachments: {row['Attachments']}")
    print(f"Prediction: {row['Prediction']}")
    print(f"Confidence: {row['Confidence']:.2f}%")
    
    # Highlight suspicious elements
    print("\nSuspicious Indicators:")
    
    # Check for urgent language
    if any(word in row['Message'].lower() for word in ['urgent', 'immediate', 'warning', 'act now']):
        print("⚠ Contains urgent/threatening language")
    
    # Check for suspicious domains
    if any(domain in row['From'].lower() for domain in ['.ru', '.cn', '.xyz']):
        print("⚠ Suspicious sender domain")
    
    # Check for dangerous attachments
    if any(ext in row['Attachments'].lower() for ext in ['.exe', '.js', '.bat', '.scr']):
        print("⚠ Contains dangerous attachment types")
    
    # Check for suspicious keywords
    if any(word in row['Message'].lower() for word in ['verify', 'credentials', 'login', 'password', 'bank account', 'social security']):
        print("⚠ Requests sensitive information")
    
    # Check for suspicious URLs
    if re.search(r'http://\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', row['Message']):
        print("⚠ Contains IP-based URLs")
    
    print("-" * 100)