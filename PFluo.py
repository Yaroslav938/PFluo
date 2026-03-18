import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats
import plotly.express as px
import plotly.graph_objects as go
import io

# --- НАСТРОЙКИ СТРАНИЦЫ ---
st.set_page_config(page_title="Анализ данных FLUOstar", layout="wide")

# --- ФУНКЦИИ СТАТИСТИКИ ---

def grubbs_test(x, alpha=0.05):
    """
    Рекурсивный тест Граббса для поиска и удаления выбросов.
    Возвращает очищенный массив и список выбросов.
    """
    x = pd.to_numeric(x, errors='coerce')
    x = np.array(x, dtype=float)
    x = x[~np.isnan(x)]
    outliers = []
    
    while True:
        n = len(x)
        if n < 3: # Тест Граббса требует минимум 3 значения
            break
        
        mean = np.mean(x)
        std = np.std(x, ddof=1)
        if std == 0:
            break
            
        z_scores = np.abs(x - mean) / std
        max_idx = np.argmax(z_scores)
        max_z = z_scores[max_idx]
        
        # Расчет критического значения
        t_crit = stats.t.ppf(1 - alpha / (2 * n), n - 2)
        g_crit = ((n - 1) / np.sqrt(n)) * np.sqrt(t_crit**2 / (n - 2 + t_crit**2))
        
        if max_z > g_crit:
            outliers.append(x[max_idx])
            x = np.delete(x, max_idx)
        else:
            break
            
    return x, outliers

def assess_normality(x, alpha=0.05):
    """
    Оценка нормальности. 
    Шапиро-Уилк для малых выборок (<=50), Д'Агостино-Пирсон для больших.
    """
    n = len(x)
    if n < 3:
        return "Недостаточно данных", np.nan, np.nan, "N/A"
    
    # Добавление небольшого шума, если все значения одинаковые, чтобы избежать ошибок
    if np.std(x) == 0:
        return "Одинаковые значения", np.nan, np.nan, "Нет"

    if n <= 50:
        stat, p = stats.shapiro(x)
        test_name = "Шапиро-Уилк"
    else:
        # Для D'Agostino нужно минимум 8 элементов, но лучше использовать для больших
        if n >= 8:
            stat, p = stats.normaltest(x)
            test_name = "Д'Агостино-Пирсон"
        else:
            stat, p = stats.shapiro(x)
            test_name = "Шапиро-Уилк"

    is_normal = "Да" if p > alpha else "Нет"
    return test_name, stat, p, is_normal

# --- ФУНКЦИИ ОБРАБОТКИ ДАННЫХ ---

def process_dataframe(df, sample_col, time_cols, alpha_grubbs, alpha_norm):
    """
    Проходит по всем образцам и временным точкам, считает статистику.
    """
    results = []
    cleaned_data_long = []
    
    grouped = df.groupby(sample_col)
    
    for sample_name, group in grouped:
        for time_col in time_cols:
            raw_values = pd.to_numeric(group[time_col], errors='coerce').dropna().values
            
            # 1. Тест Граббса
            clean_values, outliers = grubbs_test(raw_values, alpha=alpha_grubbs)
            
            for val in clean_values:
                cleaned_data_long.append({sample_col: sample_name, 'Time': time_col, 'Value': val})
            
            # 2. Описательная статистика
            n = len(clean_values)
            mean_val = np.mean(clean_values) if n > 0 else np.nan
            std_val = np.std(clean_values, ddof=1) if n > 1 else 0
            se_val = std_val / np.sqrt(n) if n > 0 else np.nan
            cv_val = (std_val / mean_val * 100) if mean_val != 0 and not np.isnan(mean_val) else np.nan
            
            # 3. Нормальность
            test_name, stat, p_val, is_norm = assess_normality(clean_values, alpha=alpha_norm)
            
            results.append({
                'Образец': sample_name,
                'Время': time_col,
                'N (исходно)': len(raw_values),
                'N (очищ.)': n,
                'Выбросы': str(outliers) if outliers else "Нет",
                'Среднее': mean_val,
                'Ст. откл.': std_val,
                'Ст. ошибка (SE)': se_val,
                'CV (%)': cv_val,
                'Тест нормальности': test_name,
                'p-value (норм.)': p_val,
                'Распределение нормально?': is_norm
            })
            
    return pd.DataFrame(results), pd.DataFrame(cleaned_data_long)

def parse_uploaded_file(uploaded_file):
    """Загрузка файла и попытка найти таблицу с данными"""
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        # Для Excel загружаем все листы
        xl = pd.ExcelFile(uploaded_file)
        sheet_name = st.selectbox(f"Выберите лист для файла {uploaded_file.name}", xl.sheet_names)
        df = pd.read_excel(uploaded_file, sheet_name=sheet_name)
    return df

# --- ИНТЕРФЕЙС STREAMLIT ---

st.title("🔬 Анализ данных FLUOstar (Lumi / OD600)")
st.markdown("Загрузите сырые данные. Приложение автоматически удалит выбросы, проверит нормальность и построит графики.")

# БОКОВАЯ ПАНЕЛЬ
st.sidebar.header("Настройки")
file_lumi = st.sidebar.file_uploader("Загрузить файл Lumi (Excel/CSV)", type=['csv', 'xlsx'])
file_od = st.sidebar.file_uploader("Загрузить файл OD600 (Excel/CSV)", type=['csv', 'xlsx'])

st.sidebar.markdown("---")
alpha_grubbs = st.sidebar.slider("Уровень значимости (α) теста Граббса (Выбросы)", 0.01, 0.10, 0.05, 0.01)
alpha_norm = st.sidebar.slider("Уровень значимости (α) теста нормальности", 0.01, 0.10, 0.05, 0.01)

# ОСНОВНАЯ ЛОГИКА
if file_lumi is not None:
    st.header("Данные Luminescence")
    df_lumi_raw = parse_uploaded_file(file_lumi)
    
    st.subheader("Предпросмотр загруженных данных")
    st.dataframe(df_lumi_raw.head(10))
    
    # Выбор колонок
    cols = df_lumi_raw.columns.tolist()
    col1, col2 = st.columns(2)
    with col1:
        sample_col = st.selectbox("Выберите колонку с названием образца (например, 'LB+dH₂O')", cols)
    with col2:
        time_cols = st.multiselect("Выберите колонки со временем (0 min, 10 min...)", cols, default=[c for c in cols if 'min' in str(c) or 'h' in str(c) or str(c).isdigit()])

    if st.button("Рассчитать статистику Lumi") and sample_col and time_cols:
        with st.spinner('Анализ данных...'):
            res_df, clean_long_df = process_dataframe(df_lumi_raw, sample_col, time_cols, alpha_grubbs, alpha_norm)
            st.session_state['res_lumi'] = res_df
            st.session_state['clean_lumi'] = clean_long_df
            st.session_state['time_cols_lumi'] = time_cols

if 'res_lumi' in st.session_state:
    res_df = st.session_state['res_lumi']
    clean_long_df = st.session_state['clean_lumi']
    
    # Вкладки для результатов
    tab1, tab2, tab3 = st.tabs(["📊 Сводная таблица", "📈 Кинетика (Кривые роста)", "🔔 Гистограммы и Нормальность"])
    
    with tab1:
        st.dataframe(res_df)
        
        # Скачивание результатов
        csv = res_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Скачать таблицу результатов как CSV",
            data=csv,
            file_name='fluostar_results.csv',
            mime='text/csv',
        )

    with tab2:
        st.subheader("Кривые кинетики (Среднее ± Ст. Ошибка)")
        # Подготовка данных для графика
        plot_df = res_df.dropna(subset=['Среднее'])
        # Попытка преобразовать время в числа для правильной оси X
        try:
            plot_df['Время_num'] = plot_df['Время'].str.extract('(\d+)').astype(float)
            plot_df = plot_df.sort_values(by=['Образец', 'Время_num'])
            x_col = 'Время_num'
        except:
            x_col = 'Время'

        fig = px.line(plot_df, x=x_col, y='Среднее', color='Образец', markers=True, 
                      error_y='Ст. ошибка (SE)', title="Динамика Luminescence")
        fig.update_layout(xaxis_title="Время", yaxis_title="Свечение (Среднее)")
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.subheader("Оценка распределения конкретной выборки")
        col_s, col_t = st.columns(2)
        with col_s:
            sel_sample = st.selectbox("Образец", res_df['Образец'].unique())
        with col_t:
            sel_time = st.selectbox("Время", st.session_state['time_cols_lumi'])
            
        subset = clean_long_df[(clean_long_df.iloc[:,0] == sel_sample) & (clean_long_df['Time'] == sel_time)]
        
        if not subset.empty:
            vals = subset['Value'].values
            fig2 = px.histogram(subset, x="Value", marginal="box", nbins=10, 
                                title=f"Гистограмма: {sel_sample} при {sel_time} (Без выбросов)")
            st.plotly_chart(fig2, use_container_width=True)
            
            # Вывод стат. данных для выбранного среза
            stat_row = res_df[(res_df['Образец'] == sel_sample) & (res_df['Время'] == sel_time)].iloc[0]
            st.info(f"**Тест нормальности:** {stat_row['Тест нормальности']} | **p-value:** {stat_row['p-value (норм.)']:.4f} | **Нормально?** {stat_row['Распределение нормально?']}")
            if stat_row['Выбросы'] != "Нет":
                st.warning(f"Удалены выбросы (Граббс): {stat_row['Выбросы']}")
        else:
            st.warning("Нет данных для выбранной комбинации.")

# ДОПОЛНИТЕЛЬНО: Интеграция OD600
if file_od is not None and 'res_lumi' in st.session_state:
    st.markdown("---")
    st.header("Анализ Lumi / OD600")
    df_od_raw = parse_uploaded_file(file_od)
    
    st.info("Убедитесь, что названия образцов и временные точки совпадают в обоих файлах для корректного расчета соотношения.")
    
    if st.button("Свести таблицы и рассчитать Lumi/OD600"):
        # Для простоты: берем средние значения из Lumi и делим на средние из OD600
        # (в идеале нужно делить по-повторностно, но при наличии выбросов ряды могут не совпадать.
        # Поэтому деление средних — наиболее устойчивый подход).
        
        # Сначала посчитаем статистику для OD600
        with st.spinner("Считаем статистику OD600..."):
            res_od, _ = process_dataframe(df_od_raw, sample_col, st.session_state['time_cols_lumi'], alpha_grubbs, alpha_norm)
            
        # Объединение
        merged = pd.merge(st.session_state['res_lumi'], res_od, on=['Образец', 'Время'], suffixes=('_Lumi', '_OD'))
        merged['Lumi / OD600'] = merged['Среднее_Lumi'] / merged['Среднее_OD']
        
        st.subheader("Сводная таблица (Lumi/OD600)")
        st.dataframe(merged[['Образец', 'Время', 'Среднее_Lumi', 'Среднее_OD', 'Lumi / OD600']])
        
        # График отношения
        try:
            merged['Время_num'] = merged['Время'].str.extract('(\d+)').astype(float)
            merged = merged.sort_values(by=['Образец', 'Время_num'])
            x_col = 'Время_num'
        except:
            x_col = 'Время'
            
        fig3 = px.line(merged, x=x_col, y='Lumi / OD600', color='Образец', markers=True, title="Отношение Luminescence к OD600")
        st.plotly_chart(fig3, use_container_width=True)
        
        # Скачивание
        csv_merged = merged.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Скачать сводные результаты Lumi/OD",
            data=csv_merged,
            file_name='lumi_od_ratio.csv',
            mime='text/csv',
        )

elif file_lumi is None:
    st.info("Пожалуйста, загрузите файл Lumi на боковой панели, чтобы начать работу.")