import pandas as pd

data = {
    "Message": [
        "Your account is compromised. Click here to reset your password now!",
        "Hey, just checking in. Let’s catch up soon!",
        "Important! Verify your account to avoid suspension!",
        "Meeting update: Tomorrow at 10 AM in Conference Room B.",
        "You won a prize! Claim your reward by clicking this link.",
        "Attached is the document you requested. Let me know if you need any further information.",
        "Suspicious login detected! Verify your account by clicking here.",
        "Please find attached the sales report for Q3.",
        "Last chance to verify your account details before suspension!",
        "Thank you for your recent purchase. Your receipt is attached."
    ],
    "From": [
        "no-reply@secure.com.ru", 
        "friend@gmail.com", 
        "support@paypal.com", 
        "colleague@company.com", 
        "win@freegift.cn", 
        "client@trustedcompany.com", 
        "security@bankalert.xyz", 
        "sales@ourcompany.com", 
        "support@secure-web.ru", 
        "support@store.com"
    ],
    "Attachments": [
        "report.zip", "", "instructions.pdf", "", "", "document.pdf", "", "report.xls", "instructions.doc", "receipt.pdf"
    ],
    "Category": [
        "spam", "ham", "spam", "ham", "spam", "ham", "spam", "ham", "spam", "ham"
    ]
}

# Convert to DataFrame and save as CSV
df = pd.DataFrame(data)
df.to_csv('sample_phishing_data.csv', index=False)
print("Sample phishing dataset saved as 'sample_phishing_data.csv'")
