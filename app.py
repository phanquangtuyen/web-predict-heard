from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import joblib
import google.generativeai as genai
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

class HeartDiseaseAdviser:
    def __init__(self):
        # Tải model và pipeline đã train
        self.process_pipeline = joblib.load('process_pipeline.pkl')
        self.model = joblib.load('heart_disease_model (2).pkl')
        # Cấu hình Gemini API
        genai.configure(api_key="AIzaSyDWHxXDRmdYt1nKoXHa4VNCjiBskRIv2RE")
    
    
    def predict(self, X):
        weights = self.model.get('weights')
        bias = self.model.get('bias', 0)
        # Thực hiện dự đoán thủ công
        X = np.array(X)
        z = np.dot(X, weights) + bias
        predictions = (self.sigmoid(z) >= 0.5).astype(int)
        return predictions

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def get_prediction_result(self, prediction):
        if prediction == 1:
            return "⚠️ Bệnh nhân có khả năng mắc bệnh tim."
        else:
            return "✅ Bệnh nhân không mắc bệnh tim."

    def get_advice_with_gemini(self, user_data, prediction):
        try:
            # Cấu hình model
            model = genai.GenerativeModel('gemini-1.0-pro')

            prompt = f"""
            Là một bác sĩ chuyên khoa tim mạch, hãy phân tích và tư vấn cho bệnh nhân với các chỉ số:
            
            THÔNG TIN BỆNH NHÂN:
            - Tuổi: {user_data['age'][0]} tuổi
            - Huyết áp: {user_data['trestbps'][0]} mmHg
            - Cholesterol: {user_data['chol'][0]} mg/dl
            - Nhịp tim: {user_data['thalach'][0]}
            
            Kết quả dự đoán: {"Có" if prediction == 1 else "Không"} nguy cơ mắc bệnh tim.
            
            Hãy đưa ra:
            1. Phân tích chi tiết từng chỉ số
            2. Đánh giá mức độ nguy cơ
            3. Lời khuyên cụ thể về chế độ ăn uống và sinh hoạt
            4. Các bước cần thực hiện tiếp theo
            
            Trả lời bằng tiếng Việt, sử dụng emoji phù hợp.
            """

            # Gọi API để tạo nội dung
            response = model.generate_content(prompt)
            logging.info(f"Phản hồi API: {response}")

            return response.text if response.text else self.get_fallback_advice(prediction)

        except Exception as e:
            logging.error(f"Lỗi khi gọi API Gemini: {str(e)}")
            return self.get_fallback_advice(prediction)

    def get_advice(self, user_data, prediction):
        """Hàm chính để lấy lời khuyên"""
        return self.get_advice_with_gemini(user_data, prediction)

    def get_fallback_advice(self, prediction):
        """Lời khuyên dự phòng khi API lỗi"""
        if prediction == 1:
            return """
            ⚠️ CẢNH BÁO: Có nguy cơ mắc bệnh tim

            📌 NHỮNG VIỆC CẦN LÀM NGAY:
            1. Đặt lịch khám chuyên khoa tim mạch càng sớm càng tốt
            2. Chuẩn bị đầy đủ thông tin về tiền sử bệnh
            3. Thực hiện đầy đủ các xét nghiệm theo chỉ định

            💊 CHẾ ĐỘ SINH HOẠT:
            1. Ăn uống:
               - Giảm muối xuống dưới 5g/ngày
               - Hạn chế chất béo bão hòa
               - Tăng rau xanh và trái cây
               - Uống đủ nước (2-3 lít/ngày)

            2. Vận động:
               - Tập thể dục nhẹ nhàng 30 phút/ngày
               - Tránh vận động mạnh
               - Nghỉ ngơi khi mệt

            3. Theo dõi:
               - Đo huyết áp mỗi ngày
               - Ghi chép các triệu chứng bất thường
               - Mang theo thuốc cấp cứu nếu được kê
            """
        else:
            return """
            ✅ KẾT QUẢ TỐT: Các chỉ số trong ngưỡng an toàn

            📋 KHUYẾN NGHỊ DUY TRÌ:
            1. Chế độ ăn uống lành mạnh:
               - Đa dạng thực phẩm
               - Ưu tiên rau xanh và trái cây
               - Hạn chế đồ ăn nhanh
               - Giảm muối và đường

            2. Vận động đều đặn:
               - Tập thể dục 30-45 phút/ngày
               - Đi bộ 10.000 bước mỗi ngày
               - Tham gia các hoạt động thể thao nhẹ nhàng

            3. Theo dõi sức khỏe:
               - Khám sức khỏe định kỳ 6 tháng/lần
               - Kiểm tra huyết áp định kỳ
               - Duy trì cân nặng hợp lý
            """

    def format_advice(self, advice_text):
        """Định dạng lời khuyên để hiển thị đẹp hơn"""
        # Thêm emoji vào các tiêu đề
        advice_text = advice_text.replace('1. Phân tích', '📊 1. Phân tích')
        advice_text = advice_text.replace('2. Đánh giá', '⚠️ 2. Đánh giá')
        advice_text = advice_text.replace('3. Lời khuyên', '💡 3. Lời khuyên')
        advice_text = advice_text.replace('4. Các bước', '📋 4. Các bước')
        
        return advice_text

    def format_advice_data(self, user_data, advice_text):
        """Format dữ liệu lời khuyên thành các phần riêng biệt"""
        # Phân tích các chỉ số
        age = user_data['age'][0]
        age_comment = "Là yếu tố nguy cơ cao mắc bệnh tim." if age > 55 else "Trong độ tuổi cần theo dõi sức khỏe tim mạch."

        bp = user_data['trestbps'][0]
        bp_comment = "Bình thường, nằm trong ngưỡng 120/80 mmHg." if bp < 120 else "Cao hơn mức bình thường, cần theo dõi."

        chol = user_data['chol'][0]
        chol_comment = "Cao, trên mức khuyến nghị là dưới 200 mg/dl." if chol >= 200 else "Trong ngưỡng bình thường."

        heart_rate = user_data['thalach'][0]
        heart_rate_comment = "Nhịp tim cao, có thể là dấu hiệu của bệnh tim mạch." if heart_rate > 100 else "Nhịp tim trong ngưỡng bình thường."

        # Đánh giá nguy cơ
        risk_factors = []
        if age > 55: risk_factors.append("tuổi cao")
        if bp >= 120: risk_factors.append("huyết áp cao")
        if chol >= 200: risk_factors.append("cholesterol cao")
        if heart_rate > 100: risk_factors.append("nhịp tim nhanh")
        
        risk_assessment = f"Dựa trên các yếu tố nguy cơ ({', '.join(risk_factors)}), bệnh nhân có mức độ nguy cơ trung bình mắc bệnh tim."

        return {
            'age_comment': age_comment,
            'bp_comment': bp_comment,
            'chol_comment': chol_comment,
            'heart_rate_comment': heart_rate_comment,
            'risk_assessment': risk_assessment,
            'recommendations': advice_text,  # Phần lời khuyên từ AI
            'next_steps': """
            <ul>
                <li>Theo dõi thường xuyên huyết áp và nhịp tim</li>
                <li>Tái khám định kỳ mỗi 3-6 tháng</li>
                <li>Thực hiện các xét nghiệm theo chỉ định của bác sĩ</li>
                <li>Duy trì lối sống lành mạnh</li>
            </ul>
            """
        }

# Khởi tạo adviser
adviser = HeartDiseaseAdviser()

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    prediction_text = None
    advice = None
    
    if request.method == 'POST':
        try:
            user_data = {
                'age': [int(request.form['age'])],
                'sex': [int(request.form['sex'])],
                'trestbps': [int(request.form['trestbps'])],
                'chol': [int(request.form['chol'])],
                'thalach': [int(request.form['thalach'])],
                'oldpeak': [float(request.form['oldpeak'])],
                'cp': [int(request.form['cp'])],
                'slope': [int(request.form['slope'])],
                'thal': [int(request.form['thal'])],
                'ca': [int(request.form['ca'])],
                'exang': [int(request.form['exang'])],
                'restecg': [int(request.form['restecg'])],
                'fbs': [int(request.form['fbs'])]
            }

            df = pd.DataFrame(user_data)
            processed_data = adviser.process_pipeline.transform(df)
            prediction = adviser.predict(processed_data)[0]
            
            # Lấy kết quả dự đoán dạng text
            prediction_text = adviser.get_prediction_result(prediction)
            
            # Lấy lời khuyên
            advice = adviser.format_advice(adviser.get_advice(user_data, prediction))
            advice_data = adviser.format_advice_data(user_data, advice)

            return render_template('index.html', 
                                prediction=prediction,
                                prediction_text=prediction_text,
                                user_data=user_data,
                                **advice_data)

        except Exception as e:
            return f"Có lỗi xảy ra: {str(e)}"

    return render_template('index.html', 
                         prediction=prediction,
                         prediction_text=prediction_text, 
                         advice=advice)

if __name__ == "__main__":
    app.run(debug=True)
