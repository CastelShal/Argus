import smtplib
import os
from dotenv import load_dotenv
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

load_dotenv()

mail = 'castelshal@gmail.com'
secret = os.getenv('GMAIL_SECRET')
smtpserver = smtplib.SMTP_SSL('smtp.gmail.com', 465)
smtpserver.ehlo()
smtpserver.login(mail, secret)

def send_alert(node_name):
    sent_from = mail
    sent_to = "castelshal@gmail.com"
    
    msg = MIMEMultipart('alternative')
    msg['Subject'] = "🚨 SECURITY ALERT - Unrecognized Individuals Detected"
    msg['From'] = sent_from
    msg['To'] = sent_to

    html = f"""
    <html>
    <body style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto; padding: 20px;">
        <div style="background-color: #f8d7da; border: 2px solid #f5c6cb; border-radius: 8px; padding: 20px; margin-bottom: 20px;">
            <h2 style="color: #721c24; margin-top: 0;">⚠️ Security Alert</h2>
            <p style="color: #721c24; font-size: 16px; line-height: 1.5;">
                Unrecognized individuals have been detected around camera node:
                <br>
                <strong style="font-size: 18px; display: block; margin: 10px 0; padding: 10px; background-color: #fff; border-radius: 4px; border: 1px solid #f5c6cb;">
                    {node_name}
                </strong>
            </p>
            <p style="color: #721c24; font-size: 16px; line-height: 1.5;">
                <strong>Immediate attention required.</strong> Please check your security system for more details.
            </p>
        </div>
        <hr style="border: none; border-top: 1px solid #ddd; margin: 20px 0;">
        <p style="color: #666; font-size: 12px; text-align: center;">
            This is an automated security message from Argus Surveillance System
            <br>
            Time of detection: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        </p>
    </body>
    </html>
    """

    html_part = MIMEText(html, 'html')
    msg.attach(html_part)

    smtpserver.sendmail(sent_from, sent_to, msg.as_string())
