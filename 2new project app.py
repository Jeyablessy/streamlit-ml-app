import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

# Email Configuration
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
SENDER_EMAIL = "blessydavisha03@gmail.com"      # Replace with your email
SENDER_PASSWORD = "ncofrhvwnhscromx"            # Use Gmail App Password
RECEIVER_EMAIL = "jeyablessy03@gmail.com"       # Organization email


# Function to Send Email Alert
def send_email_alert(supplier_names, risk_scores, new_supplier=False):

    subject = "Supplier Defect Risk Alert"
    body = "Dear Operations Team,\n\n"

    if new_supplier:
        body += "Our predictive monitoring system has detected a HIGH defect risk for a new supplier batch.\n\n"
    else:
        body += "Our predictive monitoring system has detected HIGH defect risk for the following suppliers:\n\n"

    for name, score in zip(supplier_names, risk_scores):
        body += f"Supplier: {name}\n"
        body += "Predicted Status: HIGH DEFECT RISK\n"
        body += f"Predicted Risk Score: {score:.2f}\n\n"

    body += "Recommended Actions:\n"
    body += "- Increase quality inspection\n"
    body += "- Review supplier performance\n"
    body += "- Consider alternate suppliers\n\n"
    body += "Regards,\nSupplier Monitoring System"

    msg = MIMEMultipart()
    msg['From'] = SENDER_EMAIL
    msg['To'] = RECEIVER_EMAIL
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))

    try:
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(SENDER_EMAIL, SENDER_PASSWORD)
        server.send_message(msg)
        server.quit()
        print("Email alert sent successfully!")

    except Exception as e:
        print(f"Failed to send email: {e}")


# Load Dataset
data_path = r"C:\Users\Dell\Downloads\supplier_defect_dataset.csv"
data = pd.read_csv("supplier_defect_dataset.csv")

# Features and Target
X = data[["Order_Quantity", "Delivery_Delay", "Previous_Defect_Rate"]]
y = data["Defective"]

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train Random Forest Model
model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
model.fit(X_train, y_train)

# Evaluate Model
pred_test = model.predict(X_test)
acc = accuracy_score(y_test, pred_test)
print(f"Model Accuracy: {acc:.2f}")

# Feature Importance Graph
importance = pd.Series(model.feature_importances_, index=X.columns)
importance.sort_values().plot(kind="barh", title="Feature Importance")
plt.xlabel("Importance Score")
plt.show()

# Supplier Risk Scores
data["Risk_Score"] = model.predict_proba(X)[:, 1]

# Ranking Risky Suppliers
ranking = data[["Supplier_ID", "Risk_Score"]].sort_values(by="Risk_Score", ascending=False)

print("\nTop 10 Risky Suppliers:")
print(ranking.head(10))

# Detect High Risk Suppliers
RISK_THRESHOLD = 0.7
high_risk_suppliers = data[data["Risk_Score"] > RISK_THRESHOLD]

if not high_risk_suppliers.empty:

    print("\nALERT: High-Risk Suppliers Detected!")
    print(high_risk_suppliers[["Supplier_ID", "Risk_Score"]])

    send_email_alert(
        supplier_names=high_risk_suppliers["Supplier_ID"].tolist(),
        risk_scores=high_risk_suppliers["Risk_Score"].tolist()
    )

else:
    print("\nAll suppliers are below the risk threshold.")


# Predict New Supplier Batch
new_supplier = pd.DataFrame(
    [[900, 6, 4.8]],
    columns=["Order_Quantity", "Delivery_Delay", "Previous_Defect_Rate"]
)

prediction = model.predict(new_supplier)
risk_score = model.predict_proba(new_supplier)[:, 1][0]

print(f"\nNew Supplier Prediction: {'High Risk' if prediction[0]==1 else 'Safe'}")
print(f"Predicted Risk Score: {risk_score:.2f}")

# Alert for New Supplier
if risk_score > RISK_THRESHOLD:
    send_email_alert(
        supplier_names=["New Supplier Batch"],
        risk_scores=[risk_score],
        new_supplier=True
    )
else:
    print(f"\nNew supplier batch is safe. Risk Score: {risk_score:.2f}")
