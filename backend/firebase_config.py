import firebase_admin
from firebase_admin import credentials, firestore

creds = credentials.Certificate('serviceAccountKey.json')

firebase_admin.initialize_app(creds)

db = firestore.client()