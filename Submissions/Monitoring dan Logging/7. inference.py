import requests
import time
import random

url = "http://localhost:5000/predict"

print(">>> Feeding Inference data for Prometheus Server")
print(">>> Press Ctrl+C for stopping this program")

genders = ['Male', 'Female']

while True:
    try:
        # Dummy data
        if random.random() < 0.1:
            val_income = -50 
        else:
            val_income = random.randint(15, 140)

        payload = {
            "Gender": random.choice(genders),
            "Age": random.randint(18, 70),
            "Annual Income (k$)": val_income,
            "Spending Score (1-100)": random.randint(1, 100)
        }
        
        resp = requests.post(url, json=payload)
        
        if resp.status_code == 200:
            cluster = resp.json().get('cluster')
            print(f"[OK] Income: {payload['Annual Income (k$)']} | Score: {payload['Spending Score (1-100)']} -> Cluster: {cluster}")
        else:
            print(f"[FAIL] Error {resp.status_code}")
            
    except Exception as e:
        print(f"Connection Error: {e}")
    
    time.sleep(random.uniform(0.5, 2.0))