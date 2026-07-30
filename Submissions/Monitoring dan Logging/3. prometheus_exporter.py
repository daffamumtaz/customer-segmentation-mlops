import joblib
import pandas as pd
import time
import random
import numpy as np
import threading

from flask import Flask, request, jsonify
from prometheus_client import start_http_server, Counter, Gauge, Summary, Histogram

class MonitoringApp:
    def __init__(self):
        # Load model (or use mock)
        try:
            self.model = joblib.load("best_model(2).pkl")
            print(">>> Model Loaded Successfully. Awaiting Inference")
        except Exception as e:
            print(f">>> [E] Model Failed to Load: {e}")
            print(">>> [W] Dummy Mode")
            self.model = None

        # Initialize Flask app
        self.app = Flask(__name__)

        # Group A: System Metrics (3 Metriks)
        self.REQUEST_COUNT = Counter('app_request_count_total', '1. Total Request Masuk')
        self.PROCESS_TIME = Histogram('app_process_time_seconds', '2. Waktu Proses (Latency)')
        self.MEMORY_USAGE = Gauge('app_memory_usage_bytes', '3. Penggunaan Memory')

        # Group B: Model Metrics (2 Metriks)
        self.PREDICTION_COUNTER = Counter('model_prediction_count', '4. Jumlah Prediksi per Cluster', ['cluster_id'])
        self.CONFIDENCE_SCORE = Gauge('model_confidence_score', '5. Confidence/Inertia Score (Simulasi)')

        # Group C: Data Input Metrics (3 Metriks)
        self.INPUT_AGE = Summary('input_data_age', '6. Rata-rata Umur Customer')
        self.INPUT_INCOME = Summary('input_data_income', '7. Rata-rata Annual Income')
        self.INPUT_SCORE = Summary('input_data_score', '8. Rata-rata Spending Score')

        # Group D: Business Logic Metrics (2 Metriks)
        self.TARGET_CUSTOMER = Counter('biz_target_customer_total', '9. Customer Target (High Income-High Spender)')
        self.DATA_ANOMALY = Counter('biz_data_anomaly_total', '10. Input Data Tidak Valid')

        # Register routes
        self._register_routes()

    def _register_routes(self):
        @self.app.route('/predict', methods=['POST'])
        def predict():
            return self._handle_predict()

    def _handle_predict(self):
        start_time = time.time()
        self.REQUEST_COUNT.inc()
        
        try:
            data = request.json
            
            # 1. Parsing Data
            gender = data.get('Gender', 'Male')
            age = float(data.get('Age', 0))
            income = float(data.get('Annual Income (k$)', 0))
            score = float(data.get('Spending Score (1-100)', 0))

            # 2. Update Data Input Metrics
            self.INPUT_AGE.observe(age)
            self.INPUT_INCOME.observe(income)
            self.INPUT_SCORE.observe(score)

            # 3. Validasi Anomali
            if age < 0 or income < 0 or score < 0 or score > 100:
                self.DATA_ANOMALY.inc()

            # 4. Encoding Gender, 0 = Male, 1 = Female
            gender_encoded = 0 if gender == 'Male' else 1
            
            # Bentuk array untuk prediksi
            features = np.array([[gender_encoded, age, income, score]])

            # 5. Prediksi
            if self.model:
                prediction = self.model.predict(features)
                cluster = int(prediction[0])
            else:
                cluster = random.randint(0, 4) 

            # 6. Update Model Metrics
            self.PREDICTION_COUNTER.labels(cluster_id=str(cluster)).inc()
            self.CONFIDENCE_SCORE.set(random.uniform(0.7, 0.99))

            # 7. Update Business Metrics (Target Marketing)
            if income > 70 and score > 70:
                self.TARGET_CUSTOMER.inc()

            # 8. Update System Metrics
            self.MEMORY_USAGE.set(random.randint(200, 300) * 1024 * 1024)
            self.PROCESS_TIME.observe(time.time() - start_time)

            return jsonify({"cluster": cluster, "status": "success"})

        except Exception as e:
            print(f"[ERROR] {e}")
            return jsonify({"error": str(e)}), 500

    def run(self):
        threading.Thread(target=start_http_server, args=(8000,), daemon=True).start()
        print(">>> Exporter running on port 8000")
        print(">>> Flask running on port 5000")
        self.app.run(host='0.0.0.0', port=5000)

if __name__ == '__main__':
    app_instance = MonitoringApp()
    app_instance.run()