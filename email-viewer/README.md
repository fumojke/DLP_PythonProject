Data Loss Prevention (DLP) System for Corporate Email
A specialized security system designed to prevent the leakage of confidential information through corporate email using Machine Learning and cryptographic protection.

Tech Stack:
Core: Python (Asynchronous processing)

Mail Server: aiosmtpd (Custom SMTP server for handling incoming/outgoing mail)

Data Processing: pandas (Analysis and processing of structured data)

Security: Fernet (Symmetric encryption from the cryptography library to ensure data integrity)

Database: MongoDB (Document-oriented storage for logs and security policies)

AI/ML: Machine Learning algorithms for content classification and leak detection.

Key Features:
Real-time email traffic monitoring via asynchronous SMTP server.

Automated classification of sensitive data using ML.

Scalable and flexible architecture adaptable to various corporate security requirements.