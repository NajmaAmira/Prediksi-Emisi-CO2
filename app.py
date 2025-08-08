import streamlit as st
import pandas as pd
import pickle
import os
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np

# --- Page Configuration ---
st.set_page_config(
    page_title="CO₂ Emissions Estimation",
    page_icon="🌍",
    layout="wide"
)

# --- Caching Functions for Loading Data & Models ---
@st.cache_data
def load_raw_data(file_path):
    """Loads the original, raw CSV data."""
    if not os.path.exists(file_path):
        return None
    try:
        # Based on the notebook, the delimiter is ';'
        df = pd.read_csv(file_path, delimiter=';')
        return df
    except Exception as e:
        st.error(f"Error loading raw data file ('{file_path}'): {e}")
        return None

@st.cache_data
def load_cleaned_data(file_path):
    """Loads and caches the cleaned CSV data."""
    if not os.path.exists(file_path):
        return None
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        st.error(f"Error loading cleaned data file ('{file_path}'): {e}")
        return None

@st.cache_resource
def load_model(file_path):
    """Generic function to load a .pkl model file."""
    if not os.path.exists(file_path):
        return None
    try:
        with open(file_path, 'rb') as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        st.error(f"Error loading model file {file_path}: {e}")
        return None

# --- Main Application UI ---
st.image("Header_Streamlit.png")

# --- Load Data and Models ---
MODEL_DIR = 'province_models_fixed_order' 
raw_df = load_raw_data('dataset.csv')
source_df = load_cleaned_data('cleaned_dataset.csv')
regression_model = load_model('gradient_boosting_model.pkl')
regression_url = "https://colab.research.google.com/drive/1gKiGprGtOf1U0LJpDtPk0fLd7gZTRHrb?usp=sharing"
arima_url = "https://colab.research.google.com/drive/141thc0NK_SIjM0JK--x4013xN38LXvmA?usp=sharing"

# --- Pre-calculate quantiles and define level function ---
quantiles = {}
if source_df is not None:
    cols_for_quantiles = [
        'per_capita_gdp_yuan', 'total_population_million', 'urbanization_rate_percent',
        'proportion_of_primary_industry_percent', 'proportion_of_secondary_industry_percent',
        'proportion_of_the_tertiary_industry_percent', 'coal_proportion_percent',
        'total_emissions' 
    ]
    # Update target column name to match cleaned file
    if 'total_carbon_dioxide_emissions_(_million_tons_)' in source_df.columns:
         source_df.rename(columns={'total_carbon_dioxide_emissions_(_million_tons_)': 'total_emissions'}, inplace=True)
    
    quantiles = source_df[cols_for_quantiles].quantile([0.25, 0.75]).to_dict()

def get_level(value, column_name):
    """Determines if a value is Low, Medium, or High based on pre-calculated dataset quantiles."""
    if not quantiles or column_name not in quantiles:
        return "" 
    q1 = quantiles[column_name][0.25]
    q3 = quantiles[column_name][0.75]
    if value < q1:
        return "L"
    elif value > q3:
        return "H"
    else:
        return "M"

# --- Sidebar Navigation ---
st.sidebar.title("Menu")
if 'page' not in st.session_state:
    st.session_state.page = 'Beranda'

home_btn_type = "primary" if st.session_state.page == 'Beranda' else "secondary"
est_btn_type = "primary" if st.session_state.page == 'Prediksi' else "secondary"

if st.sidebar.button("Beranda", use_container_width=True, type=home_btn_type):
    st.session_state.page = 'Beranda'
    st.rerun()
if st.sidebar.button("Prediksi", use_container_width=True, type=est_btn_type):
    st.session_state.page = 'Prediksi'
    st.rerun()

# ==========================================================================
# HOME PAGE
# ==========================================================================
if st.session_state.page == "Beranda":
    home_tab1, home_tab2 = st.tabs(["Deskripsi", "Gambaran Umum Dataset"])

    with home_tab1:
        st.header("Latar Belakang")
        st.write("""
            Di tengah urgensi krisis iklim global, emisi karbon dioksida (CO₂) menjadi perhatian utama sebagai salah satu penyumbang terbesar.
            Untuk merumuskan solusi yang efektif, pemahaman terhadap faktor-faktor yang mendorong emisi perlu dikaji secara mendalam, tidak hanya di tingkat global, tetapi juga pada skala regional di mana kebijakan dapat diterapkan secara langsung.
            
            Analisis ini secara khusus difokuskan pada provinsi-provinsi di China dengan tujuan memahami bagaimana berbagai faktor seperti pertumbuhan ekonomi, dinamika kependudukan, dan perubahan struktur industri secara bersama-sama membentuk jejak karbon suatu wilayah. Pemahaman menyeluruh terhadap keterkaitan kompleks antara faktor-faktor tersebut menjadi langkah penting dalam merancang strategi mitigasi yang lebih tepat sasaran sesuai dengan kondisi khas di masing-masing daerah.
            Pemahaman mendalam terhadap hubungan kompleks ini merupakan langkah krusial untuk merancang strategi mitigasi yang lebih tajam dengan kondisi unik setiap daerah di China.
        """)
        st.header("Deskripsi Proyek")
        st.write("""
        Aplikasi ini dirancang untuk menganalisis dan memprediksi emisi CO₂ di berbagai provinsi di China menggunakan dua pendekatan utama.
        Dengan memanfaatkan data historis dari tahun 1999 hingga 2019, aplikasi ini membangun model machine learning yang andal untuk memberikan prediksi yang akurat dan mendalam terkait tren emisi di masa depan.
        """)
        st.divider()
        st.header("Metode yang Digunakan")

        # --- Feature 1 with Image ---
        col1, col2 = st.columns([1, 4])
        with col1:
            st.image("Regression.png")
        with col2:
            st.markdown("""
            **1. Regresi:**
            - Pendekatan ini menggunakan model **Gradient Boosting Regressor**, yang telah dilatih menggunakan seluruh dataset di seluruh provinsi.
            - Model ini mempelajari hubungan kompleks antara emisi CO₂ dan berbagai faktor sosio-ekonomi seperti GDP, populasi, tingkat urbanisasi, dan struktur industri.
            - Hal ini memungkinkan Anda untuk membuat skenario “what-if” yang detail untuk melihat bagaimana perubahan kebijakan dan ekonomi tertentu dapat memengaruhi emisi.
            """)

            st.markdown("**Google Colab :** [Prediksi_Emisi_CO2_Regresi.ipynb](%s)" % regression_url)

        st.divider()

        # --- Feature 2 with Image ---
        col1, col2 = st.columns([1, 4])
        with col1:
            st.image("forecasting.jpg", )
        with col2:
            st.markdown("""
            **2. Forecasting (Time-Series):**
            - Pendekatan ini menggunakan model **ARIMA (AutoRegressive Integrated Moving Average)**.
            - Model ARIMA dilatih untuk setiap provinsi secara individual, dengan fokus pada tren emisi historisnya.
            - Hal ini memberikan perkiraan “dasar”, menunjukkan arah emisi suatu provinsi jika momentum historisnya terus berlanjut tanpa perubahan eksternal yang signifikan.
            """)

            st.markdown("**Google Colab :** [Prediksi_Emisi_CO2_Forecasting.ipynb](%s)" % arima_url)
        
        st.divider()
        st.markdown("Dengan menggunakan kedua pendekatan ini, pengguna dapat memperoleh pemahaman yang komprehensif tentang faktor-faktor mendasar yang menyebabkan emisi serta tren masa depan yang kemungkinan terjadi.")

    with home_tab2:
        st.header("Gambaran Umum Dataset")
        
        if raw_df is None or source_df is None:
            st.error("One or more dataset files (`dataset.csv`, `cleaned_dataset.csv`) not found. Cannot display overview.")
        else:
            with st.expander("Tentang Dataset", expanded=False):
                st.markdown("""
                ### Sumber Dataset
                - **Sumber**: [Data for: Spatial Characteristics and Future Forecasting of Carbon Dioxide Emissions in China: A Provincial-Level Analysis](https://data.mendeley.com/datasets/rp3f7mdjxz/1)
                - **Tanggal Publikasi**: 30 Jul 2024
                - **DOI**: [10.17632/rp3f7mdjxz.1](https://doi.org/10.17632/rp3f7mdjxz.1)
                
                ### Deskripsi Umum
                Dataset berisi data socio-ekonomi dan emisi karbon dioksida (CO₂) dari 31 provinsi di China selama periode 21 tahun (1999 hingga 2019). Secara keseluruhan, terdapat 651 baris data, yang merupakan hasil perkalian antara 31 provinsi dengan 21 tahun pengamatan.

                ### Deskripsi Kolom
                - **`Name`**: Nama provinsi di China.
                - **`Year`**: Tahun ketika data tersebut direkam (1999-2019).
                - **`per capita gdp(yuan)`**: GDP (Gross Domestic Product) per kapita, diukur dalam Yuan China. Ini adalah indikator tingkat kesejahteraan ekonomi rata-rata per orang di provinsi tersebut..
                - **`total population(million)`**: Total populasi/penduduk, diukur dalam jutaan.
                - **`urbanization rate(%)`**: Persentase penduduk yang tinggal di daerah perkotaan.
                - **`proportion of primary industry(%)`**: Persentase kontribusi sektor primer (pertanian, kehutanan, perikanan) terhadap total GDP provinsi.
                - **`proportion of secondary industry(%)`**: Persentase kontribusi sektor sekunder (manufaktur, industri, konstruksi) terhadap total GDP.
                - **`proportion of the tertiary industry(%)`**: Persentase kontribusi sektor tersier (jasa, perdagangan, pariwisata) terhadap total GDP.
                - **`coal proportion(%)`**: Persentase penggunaan batu bara dalam konsumsi energi total.
                - **`Total carbon dioxide emissions (million tons)`**: Total emisi CO2, diukur dalam juta ton. Ini adalah variabel yang akan diprediksi.
                """)

            with st.expander("1. Pratinjau Dataset Mentah", expanded=False):
                st.markdown("Berikut adalah dataset asli yang belum diolah, sebagaimana dimuat dari sumber.")
                st.dataframe(raw_df.head(10))

            with st.expander("2. Langkah Pra-pemrosesan", expanded=False):
                st.markdown("""
                Dataset mentah memerlukan beberapa langkah pembersihan dan transformasi agar sesuai untuk proses machine learning. Proses yang diterapkan pada dataset mentah adalah:
                
                #### a. Standarisasi Nama Kolom
                Nama kolom asli mengandung spasi, tanda kurung, dan campuran huruf besar dan kecil. Nama-nama tersebut telah distandarkan ke format `snake_case` yang konsisten untuk memudahkan akses.
                - **Contoh Sebelum**: `per capita gdp(yuan)`
                - **Contoh Setelah After**: `per_capita_gdp_yuan`
                - Kolom target `total_carbon_dioxide_emissions_(_million_tons_)` juga diganti menjadi `total_emissions`.
                
                #### b. Penanganan Missing Values
                Pada kolom `name` terdapat missing value (`NaN`). Data ini diisi menggunakan metode **forward-fill (`ffill`)**. Metode ini menyebarkan pengamatan terakhir yang valid ke depan, yang sesuai untuk dataset ini di mana nama-nama yang hilang terkait dengan provinsi yang disebutkan sebelumnya.
                
                #### c. Konversi Tipe Data
                Kolom numerik dimuat secara salah sebagai teks (`object`) karena menggunakan koma sebagai pemisah desimal. Setiap kolom ini dikonversi:
                1.  Koma (`,`) diganti dengan titik (`.`).
                2.  String hasil konversi dikonversi menjadi bilangan floating-point (`float`).
                
                #### d. Penanganan Extreme Outliers
                Boxplots menunjukkan satu extrem outlier pada `proportion_of_primary_industry_percent` dan satu pada `proportion_of_secondary_industry_percent`. Outlier-outlier ini dapat memengaruhi akurasi model.
                - **Strategi**: Alih-alih menghapus baris tersebut, nilai outlier diganti dengan **nilai rata-rata untuk provinsi tersebut**, yang dihitung dari data tahun-tahun lainnya. Hal ini menjaga titik data sambil memperbaiki nilai yang anomali.
                  - Untuk **Shannxi** (2007), proporsi industri primer sebesar `-477.60` diganti dengan rata-rata provinsi sebesar `10.96`.
                  - Untuk **Shannxi** (2007), proporsi industri sekunder sebesar `543.00` diganti dengan rata-rata provinsi sebesar `49.56`.
                
                Setelah langkah-langkah ini, dataset tersebut sudah bersih, lengkap, dan siap untuk dianalisis.
                """)
            
            with st.expander("3. Pratinjau Dataset yang Telah Diproses"):
                st.markdown("Berikut ini adalah pratinjau dari dataset akhir yang telah dibersihkan yang digunakan untuk semua model dan visualisasi dalam aplikasi ini.")
                st.dataframe(source_df.head(10))

            with st.expander("4. Visualisasi Data"):
                numeric_cols = source_df.select_dtypes(include=np.number).drop(columns='year')

                st.subheader("Heatmap Korelasi")
                st.markdown("Heatmap berikut menunjukkan koefisien korelasi antara berbagai variabel numerik. Nilai yang mendekati 1 (warna terang) menunjukkan korelasi positif yang kuat.")
                
                corr = numeric_cols.corr()
                fig_heatmap = go.Figure(data=go.Heatmap(
                                   z=corr.values,
                                   x=corr.index.values,
                                   y=corr.columns.values,
                                   colorscale='Viridis',
                                   colorbar=dict(title='Korelasi')))
                fig_heatmap.update_layout(title='Matriks Korelasi Antar Kolom Numerik', yaxis_autorange='reversed')
                st.plotly_chart(fig_heatmap, use_container_width=True)


                st.subheader("Tren Emisi CO₂ Nasional (1999-2019)")
                st.markdown("Grafik ini menggabungkan emisi dari semua provinsi untuk menunjukkan tren keseluruhan di China selama dua dekade.")

                total_emissions_by_year = source_df.groupby('year')['total_emissions'].sum().reset_index()
                fig_total_trend = px.line(total_emissions_by_year, x='year', y='total_emissions',
                                          title='Total Emisi CO₂ Nasional Sepanjang Waktu', markers=True,
                                          labels={'year': 'Tahun', 'total_emissions': 'Emisi Total (Juta Ton)'})
                st.plotly_chart(fig_total_trend, use_container_width=True)


                st.subheader("Tren Emisi CO₂ per Provinsi")
                st.markdown("Grafik ini membandingkan tren emisi dari semua provinsi. Anda dapat mengklik item di legenda untuk menyembunyikan atau menampilkan provinsi tertentu.")
                
                fig_all_provinces = px.line(source_df, x='year', y='total_emissions', color='name',
                                   title='Emisi CO₂ untuk Semua Provinsi (1999-2019)', markers=False,
                                   labels={'year': 'Tahun', 'total_emissions': 'Emisi Total (Juta Ton)', 'name': 'Provinsi'})
                
                fig_all_provinces.update_xaxes(
                    dtick=2,
                    tickangle=45
                )

                # fig_all_provinces.update_layout(legend=dict(
                #     orientation="h",
                #     yanchor="bottom",
                #     y=-0.4, 
                #     xanchor="right",
                #     x=1
                # ))
                st.plotly_chart(fig_all_provinces, use_container_width=True)

                st.subheader("Distribusi Fitur")
                st.markdown("Histogram-histogram ini menunjukkan distribusi setiap fitur numerik dalam dataset, membantu memahami rentang dan nilai-nilai umum yang sering muncul.")
                
                cols_to_plot = numeric_cols.columns
                fig_hist = make_subplots(rows=(len(cols_to_plot) + 2) // 3, cols=3, subplot_titles=[col.replace("_", " ").title() for col in cols_to_plot])
                for i, col in enumerate(cols_to_plot):
                    row = i // 3 + 1
                    col_num = i % 3 + 1
                    fig_hist.add_trace(go.Histogram(x=source_df[col], name=col), row=row, col=col_num)
                
                fig_hist.update_layout(height=800, showlegend=False, title_text="Distribusi Fitur Numerik")
                st.plotly_chart(fig_hist, use_container_width=True)

# ==========================================================================
# ESTIMATE PAGE
# ==========================================================================
elif st.session_state.page == "Prediksi":
    est_tab1, est_tab2 = st.tabs(["Prediksi Berdasarkan Faktor", "Prediksi Tren Historis"])

    with est_tab1:
        if regression_model is None or source_df is None:
            st.error("Regression model or source data not found. Please check your files.")
        else:
            province_names = sorted(source_df['name'].unique())
            
            with st.form(key='prediction_form'):
                st.subheader("Masukkan nilai-nilai berikut untuk memprediksi total emisi CO₂")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    province = st.selectbox("Province", options=province_names)
                    year = st.number_input("Year", min_value=2020, max_value=2050, value=2025)
                    gdp = st.number_input("GDP per Capita (yuan)", min_value=0)
                with col2:
                    population = st.number_input("Total Population (million)", min_value=0.0, format="%.2f")
                    urbanization = st.number_input("Urbanization Rate (%)", min_value=0.0, max_value=100.0, format="%.1f")
                    primary_ind = st.number_input("Proportion of Primary Industry (%)", min_value=0.0, max_value=100.0, format="%.1f")
                with col3:
                    secondary_ind = st.number_input("Proportion of Secondary Industry (%)", min_value=0.0, max_value=100.0, format="%.1f")
                    tertiary_ind = st.number_input("Proportion of Tertiary Industry (%)", min_value=0.0, max_value=100.0, format="%.1f")
                    coal_prop = st.number_input("Coal Proportion (%)", min_value=0.0, max_value=100.0, format="%.1f")
                
                submit_button = st.form_submit_button(label='Prediksi Emisi')

            if submit_button:
                st.divider()
                res_col1, res_col2 = st.columns([1, 2])
                
                with res_col1:
                    st.subheader("Hasil")
                    input_data_dict = {
                        'name': [province], 'year': [year], 'per_capita_gdp_yuan': [gdp],
                        'total_population_million': [population], 'urbanization_rate_percent': [urbanization],
                        'proportion_of_primary_industry_percent': [primary_ind], 
                        'proportion_of_secondary_industry_percent': [secondary_ind],
                        'proportion_of_the_tertiary_industry_percent': [tertiary_ind], 
                        'coal_proportion_percent': [coal_prop]
                    }
                    # Rename the columns in the prediction input to match the model's training columns
                    input_data_df = pd.DataFrame(input_data_dict)
                    
                    prediction = regression_model.predict(input_data_df)[0]
                    st.metric(label=f"Prediksi Emisi CO₂ Provinsi {province}", value=f"{prediction:,.2f} juta ton", width="content")
                
                with res_col2:
                    st.subheader("Deskripsi")
                    
                    gdp_level = get_level(gdp, 'per_capita_gdp_yuan')
                    pop_level = get_level(population, 'total_population_million')
                    urban_level = get_level(urbanization, 'urbanization_rate_percent')
                    primary_level = get_level(primary_ind, 'proportion_of_primary_industry_percent')
                    secondary_level = get_level(secondary_ind, 'proportion_of_secondary_industry_percent')
                    tertiary_level = get_level(tertiary_ind, 'proportion_of_the_tertiary_industry_percent')
                    coal_level = get_level(coal_prop, 'coal_proportion_percent')
                    prediction_level = get_level(prediction, 'total_emissions')

                    explanation_text = f"""
                    Berdasarkan data yang Anda berikan untuk **Provinsi {province}** pada tahun **{year}**:

                    - GDP per kapita sebesar **{gdp:,.0f} yuan** berada pada **Level {gdp_level}**.
                    - Total populasi penduduk sebesar **{population:,.2f} juta** berada pada **Level {pop_level}**.
                    - Tingkat urbanisasi sebesar **{urbanization:.1f}%** berada pada **Level {urban_level}**.
                    - Proporsi sektor industri primer sebesar **{primary_ind:.1f}%** berada pada **Level {primary_level}**.
                    - Proporsi sektor industri sekunder sebesar **{secondary_ind:.1f}%** berada pada **Level {secondary_level}**.
                    - Proporsi sektor industri tersier sebesar **{tertiary_ind:.1f}%** berada pada **Level {tertiary_level}**.
                    - Proporsi penggunaan batu bara sebesar **{coal_prop:.1f}%** berada pada **Level {coal_level}**.

                    ---
                    Diperkirakan faktor-faktor tersebut akan menyebabkan emisi CO₂ sebesar **{prediction:,.2f} juta ton**, yang berada pada **Level {prediction_level}** jika dibandingkan dengan data historis.
                    """
                    st.markdown(explanation_text)
                    st.caption("Level L: Rendah (di bawah persentil ke-25), M: Sedang (antara persentil ke-25 hingga ke-75), H: Tinggi (di atas persentil ke-75)")

    with est_tab2:
        if not os.path.exists(MODEL_DIR) or source_df is None:
            st.error(f"ARIMA model directory '{MODEL_DIR}' or source data not found.")
        else:
            st.subheader("Pilih provinsi dan jumlah tahun untuk memprediksi tren emisi CO₂")
            
            arima_province_names = sorted([
                f.replace('arima_model_', '').replace('.pkl', '').replace('_', ' ') 
                for f in os.listdir(MODEL_DIR) if f.endswith('.pkl')
            ])
            
            fc_province = st.selectbox("Provinsi", options=arima_province_names, key="forecast_province")
            fc_years = st.slider("Pilih jumlah tahun yang akan diprediksi", 1, 5, 3, key="forecast_years")

            safe_province_name = fc_province.replace(" ", "_")
            model_path = os.path.join(MODEL_DIR, f'arima_model_{safe_province_name}.pkl')
            arima_model = load_model(model_path)
            
            if arima_model:
                province_ts = source_df[source_df['name'] == fc_province][['year', 'total_emissions']].copy().set_index('year')
                
                forecast = arima_model.get_forecast(steps=fc_years)
                forecast_df = forecast.summary_frame(alpha=0.05)
                
                last_year = province_ts.index.max()
                forecast_df.index = range(last_year + 1, last_year + 1 + len(forecast_df))
                
                forecast_df['mean'] = forecast_df['mean'].clip(lower=0)
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=province_ts.index, y=province_ts.iloc[:, 0], mode='lines+markers', name='Historical Emissions', line=dict(color='royalblue')))
                fig.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df['mean'], mode='lines', name='Forecast', line=dict(dash='dash', color='firebrick')))
                fig.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df['mean_ci_upper'], fill='tonexty', fillcolor='rgba(255, 82, 82, 0.2)', line=dict(color='rgba(255,255,255,0)'), name='95% Confidence Interval', showlegend=True))
                fig.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df['mean_ci_lower'], fill='tonexty', fillcolor='rgba(255, 82, 82, 0.2)', line=dict(color='rgba(255,255,255,0)'), showlegend=False))
                
                fig.update_layout(
                    title=f"Prediksi Emisi CO₂ di Provinsi {fc_province}",
                    xaxis_title="Tahun",
                    yaxis_title="Emisi (Juta Ton)",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                st.plotly_chart(fig, use_container_width=True)
                
                st.subheader("Data Prediksi")
                st.dataframe(forecast_df[['mean', 'mean_ci_lower', 'mean_ci_upper']].rename(columns={
                    'mean': 'Prediksi Rata-Rata',
                    'mean_ci_lower': 'Batas Bawah 95% CI',
                    'mean_ci_upper': 'Batas Atas 95% CI'
                }).style.format("{:,.2f}"))
            else:
                st.error(f"Tidak dapat memuat model prediksi untuk {fc_province}.")