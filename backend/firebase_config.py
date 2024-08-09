import firebase_admin
from firebase_admin import credentials, firestore

creds = credentials.Certificate('/Users/krish/Documents/qautomator-backend/backend/serviceAccountKey.json')

firebase_admin.initialize_app(creds)

db = firestore.client()