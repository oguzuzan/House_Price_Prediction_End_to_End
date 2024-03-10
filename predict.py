import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from scipy.stats import skew
from sklearn.preprocessing import LabelEncoder
import joblib


model = joblib.load('new_model_xgb.joblib')
feature_names = joblib.load('feature_names.joblib')
#label_encoder = joblib.load('label_encoder.joblib')

def predict_sale_price(model, input_data):
    for feature in feature_names:
        if input_data[feature].dtype == 'object':
            label_encoder = LabelEncoder()
            input_data[feature] = label_encoder.fit_transform(input_data[feature].astype(str))

    prediction = model.predict(input_data)
    return np.expm1(prediction)  # Reverse the log transformation


# House App
def main():
    st.title("House Sale Price Prediction App")

    # Modeli Cagiriyoruz.
    trained_model = model
    feature_names = joblib.load('feature_names.joblib') 

    # Resim eklemek için st.image fonksiyonunu kullanın
    st.image("house.png", use_column_width=True)
    
    st.markdown("""Uygulamamız, ABD'nin Iowa eyaletine bağlı Ames şehrindeki konutlara ait zengin bir veri seti ile eğitilerek, size gerçekçi ve güvenilir fiyat tahminleri sunabiliyor.""")
                 
    st.markdown("""Ev alıcıları ve satıcıları için değerli bir araç olmayı hedeflemektedir. Uygulamamızdaki çeşitli parametreler, konut fiyatlarını etkileyen faktörleri anlamanıza yardımcı olacak şekilde seçilmiştir.""")

    st.markdown("""Ev Fiyat Tahmin Uygulaması sayesinde, gelecekteki konut alım veya satım kararlarınızı daha bilinçli bir şekilde planlayabilirsiniz. Uygulamayı kullanarak Ames şehrindeki konut piyasasındaki trendleri keşfedin ve potansiyel mülklerin gelecekteki değerini tahmin edin.""")
    st.markdown("""Unutmayın, bu tahminler sadece bir pusula gibidir; gerçek piyasa denizine açılmadan önce, dalgaların dansını dikkatlice izlemek önemlidir. Sizlere, emlak serüveninizde rüzgarlı denizlerde rehberlik etmeyi ve gerçek hazineyi keşfetmeyi dileriz! 🏡🌊""")


    # 2.Baslik
    st.header("Evin Bilgilerini Giriniz.")

    # Inputlari girdigimiz yerin Duzenlemesi ve Ayarlanmasi
    input_features = {}
    for feature in feature_names:
        if feature == 'OverallQual':
            Overall_key = f"{feature}_slider"
            input_features[feature] = st.slider("Genel malzeme ve kaplama kalitesi: (1-10)", 1, 10, value=10, key=Overall_key)
            puanlar_metin = {
            10: "Very Excellent",
            9: "Excellent",
            8: "Very Good",
            7: "Good",
            6: "Above Average",
            5: "Average",
            4: "Below Average",
            3: "Fair",
            2: "Poor",
            1: "Very Poor"
            }
            secilen_metin = puanlar_metin.get(input_features[feature], "Bilinmeyen Puan")
            input_features[feature] = input_features[feature]
            st.write(f"Seçilen Puan: {input_features[feature]} - {secilen_metin}")
        
        elif feature == 'ExterQual':
            ExterQual_puanlar = {
                0: "Poor",
                2: "Good",
                1: "Average/Typical"
            }
            slider_key = f"{feature}_slider"
            input_features[feature] = st.slider("Dış malzeme kalitesi", 0, 2, value=2, key=slider_key)
            ExterQual_metin = ExterQual_puanlar.get(input_features[feature])
            input_features[feature] = input_features[feature]
            st.write(f"Seçilen Puan: {input_features[feature]} - {ExterQual_metin}")

        elif feature == 'YearBuilt':
            input_features[feature] = st.number_input("Binanın Yapım Yılını Girin:", min_value=1872, max_value=2024, step=1, key="year_built")

        elif feature == 'YearRemodAdd':
            input_features[feature] = st.number_input("Tadilat Tarihi (Tadilat veya Ekleme Yoksa İnşaat Tarihiyle Aynı)?:", min_value=1872, max_value=2024, step=1, key="year_remod_built")

        elif feature == 'TotalSF':
            input_features[feature] = st.number_input("Toplam Alan (Total Square Feet):", min_value=334, max_value=11752, help="Mülkün toplam alanını girin." )
            if input_features[feature] < 334:
                st.warning("Lütfen Daha Büyük Bir Değer Girin.")
            elif input_features[feature] > 11752:
                st.warning("Lütfen Küçük Bir Değer Girin.")

        elif feature == 'GrLivArea':
            input_features[feature] = st.number_input("Ev içindeki Yaşam Alanının Toplam Square Feet Ölçümü:", min_value=334, max_value=5642, step=10)
            if input_features[feature] < 334:
                st.warning("Lütfen Daha Büyük Bir Değer Girin.")
            elif input_features[feature] > 5642:
                st.warning("Lütfen Daha Küçük Bir Değer Girin.")

        elif feature == 'KitchenQual':
            kitchen_qual_options = {
                "Excellent": 0,
                "Good": 1,
                "Typical/Average": 2,
                "Poor": 3
            }

            selected_kitchen_qual = st.selectbox("Mutfak Kalitesi:", list(kitchen_qual_options.keys()))

            input_features[feature] = kitchen_qual_options[selected_kitchen_qual]
        
        elif feature == 'Fireplaces':
            input_features[feature] = st.number_input("Şömine sayısı", min_value=0, max_value=3, value=0, step=1)

        
        elif feature == 'CentralAir':
            central_key = f"{feature}_central"
            options_central = {'Hayır': 0, 'Evet': 1}
            input_features[feature] = st.selectbox("Klima Var mi?", options_central, key=central_key)
            input_features[feature] = options_central[input_features[feature]]
        
        elif feature == 'GarageFinish':
            GarageFinish_qual_options = {
                "İyi Kalite": 0,
                "Orta Kalite": 1,
                "Garaj Yok": 3
            }
            GarageFinish_key = f"{feature}_selectbox"
            input_features[feature] = st.selectbox("Garajın İç Mekan Kalite Durumu Nasıl?", GarageFinish_qual_options, key=GarageFinish_key)
            input_features[feature] = GarageFinish_qual_options[input_features[feature]]

        elif feature == 'GarageType':
            garaj_turleri = {
                "Ev ile Bağlantılı ve Aynı Zamanda İnşa Edilmiş": 3,
                "Ev ile Bağlantılı Garaj(Sonradan Yapıldı)": 1,
                "Evden Ayrı Garaj": 5,
                "Bodrum Katında Garaj": 2,
                "Carport Tipinde Garaj": 4,
                "Birden Fazla Garaj Var": 0,
                "Garaj Yok": 6
            }
            GarageType_key = f"{feature}_selectbox"
            input_features[feature] = st.selectbox("Evinizin Garaj Türü Hangisi?", garaj_turleri, key="GarageType_key")
            input_features[feature] = garaj_turleri[input_features[feature]]

        elif feature == 'GarageQual':
            garaj_kalitesi = {
                "Mükemmel": 5,
                "İyi": 2,
                "Ortalama": 4,
                "Kötü": 3,
                "Garaj Yok": 0
            }

            input_features[feature] = st.selectbox("Garajın Kalitesini Seçin:", list(garaj_kalitesi.keys()))
            input_features[feature] = garaj_kalitesi[input_features[feature]]

        elif feature == 'GarageCars':
            input_features[feature] = st.number_input("Garaj Kapasitesi", min_value=0, max_value=4, value=0, step=1)


        elif feature == 'GarageArea':  
            input_features[feature] = st.number_input("Garaj Alanının Toplam Square Feet Ölçümü:", min_value=0, max_value=1500, step=10, value=0)
            if input_features[feature] < 0:
                st.warning("Lütfen Daha Büyük Bir Değer Girin.")
            elif input_features[feature] > 1500:
                st.warning("Lütfen Daha Küçük Bir Değer Girin.")

        elif feature == 'BsmtQual':
            bsmt_qual_siniflandirma = {
                "Mükemmel (100+ inches)": 0,
                "İyi (100+ inches)": 2,
                "Ortalama (80-89 inches)": 1,
                "Kötü (<70 inches)": 3,
                "Garaj Yok": 4
            }
            # Benzersiz bir key değeri oluşturunuz (örneğin, feature adını kullanabilirsiniz)
            selectbox_key = f"{feature}_selectbox"
            # Dış malzeme kalitesi için Streamlit selectbox'ını kullanarak değeri alınız
            input_features[feature] = st.selectbox("Bodrum Katının Kalitesini Seçin:", list(bsmt_qual_siniflandirma.keys()), key=selectbox_key)
            input_features[feature] = bsmt_qual_siniflandirma[input_features[feature]]
        
        elif feature == 'BsmtFullBath':
            BsmtFullBath_key = f"{feature}_central"
            options_BsmtFullBath = {'Hayır': 0, 'Evet': 1}
            input_features[feature] = st.selectbox("Bodrum Katında Banyo Var mı?", options_BsmtFullBath, key=BsmtFullBath_key)
            input_features[feature] = options_BsmtFullBath[input_features[feature]]

        elif feature in ["FullBath"]:
            input_features[feature] = st.number_input("Banyo Sayısı", value=0, step=1)
        elif feature == 'PavedDrive':
            options_paved = {
                'Asfaltlı Yol': 2,
                'Yarı Asfaltlı': 1,
                'Toprak Yol': 0
            }
            paved_key = f"{feature}_selectbox"
            input_features[feature] = st.selectbox("Garaj Girişinin Zemini Nasıl?", options_paved, key="paved_key")
            input_features[feature] = options_paved[input_features[feature]]
        
        
        else:
            input_features[feature] = st.number_input(f"Enter {feature}", value=0.0)

    # Girdigimiz inputlari kaydeden bir DataFrame Olusturmak icin (Opsiyonel)
    input_df = pd.DataFrame([input_features])

    # Inputlari Tek bir sirada goruntulemek icin
    st.subheader("User Input Data")
    show_input_data = st.checkbox("Show Input Data", value=False)
    if show_input_data:
        st.write(input_df)

    # Predict tusu ve Predict
    if st.button("Predict Sale Price"):
        prediction = predict_sale_price(trained_model, input_df)

        formatted_prediction = f"<h2 style='font-size: 36px; color: #0d730d;'> ${prediction[0]:,.2f}</h2>"
        st.markdown(formatted_prediction, unsafe_allow_html=True)
        st.success("Prediction successful!")
        st.balloons()

if __name__ == "__main__":
    main()