import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats
import plotly.express as px
import io
import re

# --- НАСТРОЙКИ СТРАНИЦЫ ---
st.set_page_config(page_title="Анализ данных FLUOstar", layout="wide")

# --- ФУНКЦИИ СТАТИСТИКИ ---

def grubbs_test(x, alpha=0.05):
    """
    Рекурсивный тест Граббса для поиска и удаления выбросов.
    """
    x = pd.to_numeric(x, errors='coerce')
    x = np.array(x, dtype=float)
    x = x[~np.isnan(x)]
    outliers = []
    
    while True:
        n = len(x)
        if n < 3:
            break
        
        mean = np.mean(x)
        std = np.std(x, ddof=1)
        if std == 0:
            break
            
        z_scores = np.abs(x - mean) / std
        max_idx = np.argmax(z_scores)
        max_z = z_scores[max_idx]
        
        t_crit = stats.t.ppf(1 - alpha / (2 * n), n - 2)
        g_crit = ((n - 1) / np.sqrt(n)) * np.sqrt(t_crit**2 / (n - 2 + t_crit**2))
        
        if max_z > g_crit:
            outliers.append(x[max_idx])
            x = np.delete(x, max_idx)
        else:
            break
            
    return x, outliers

def assess_normality(x, alpha=0.05):
    """Оценка нормальности."""
    n = len(x)
    if n < 3:
        return "Недостаточно данных", np.nan, np.nan, "N/A"
    
    if np.std(x) == 0:
        return "Одинаковые значения", np.nan, np.nan, "Нет"

    if n <= 50:
        stat, p = stats.shapiro(x)
        test_name = "Шапиро-Уилк"
    else:
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
    """Расчет статистики по сгруппированным данным."""
    results = []
    cleaned_data_long = []
    grouped = df.groupby(sample_col)
    
    for sample_name, group in grouped:
        for time_col in time_cols:
            raw_values = pd.to_numeric(group[time_col], errors='coerce').dropna().values
            
            clean_values, outliers = grubbs_test(raw_values, alpha=alpha_grubbs)
            
            for val in clean_values:
                cleaned_data_long.append({sample_col: sample_name, 'Time': time_col, 'Value': val})
            
            n = len(clean_values)
            mean_val = np.mean(clean_values) if n > 0 else np.nan
            std_val = np.std(clean_values, ddof=1) if n > 1 else np.nan
            se_val = std_val / np.sqrt(n) if n > 1 else np.nan
            cv_val = (std_val / mean_val * 100) if n > 1 and mean_val != 0 and not np.isnan(mean_val) else np.nan
            
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

def parse_uploaded_file(uploaded_file, key_prefix):
    """Загрузка файла и умный парсинг (поддержка сырых логов FLUOstar)."""
    
    def deduplicate(columns):
        """Абсолютно надежное удаление дубликатов в названиях колонок."""
        seen = set()
        res = []
        for col in columns:
            col_str = str(col)
            new_col = col_str
            counter = 1
            while new_col in seen:
                new_col = f"{col_str}_{counter}"
                counter += 1
            seen.add(new_col)
            res.append(new_col)
        return res

    if uploaded_file.name.endswith('.csv'):
        try:
            df = pd.read_csv(uploaded_file, header=None)
        except:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, header=None, sep=';')
    else:
        xl = pd.ExcelFile(uploaded_file)
        sheet_name = st.selectbox(f"Лист ({uploaded_file.name})", xl.sheet_names, key=f"{key_prefix}_sheet")
        df = pd.read_excel(uploaded_file, sheet_name=sheet_name, header=None)
        
    df = df.dropna(how='all').dropna(axis=1, how='all').reset_index(drop=True)
    if df.empty:
        return pd.DataFrame(), False
        
    df.columns = range(df.shape[1])
    
    # Поиск маркеров сырой выгрузки FLUOstar
    well_row_idx = None
    content_row_idx = None
    
    for i in range(min(50, len(df))):
        val = str(df.iloc[i, 0]).lower()
        if 'well' in val and 'row' in val:
            well_row_idx = i
        if 'content' in val:
            content_row_idx = i
            
    if well_row_idx is not None:
        # ПАРСИНГ СЫРЫХ ДАННЫХ
        start_col = 1
        for c in range(1, df.shape[1]):
            val = str(df.iloc[well_row_idx, c]).strip()
            if val != 'nan' and val != '':
                start_col = c
                break
                
        # Названия образцов (из 'Content' или по координатам лунок A1, A2...)
        if content_row_idx is not None:
            samples = df.iloc[content_row_idx, start_col:].astype(str).values
        else:
            rows = df.iloc[well_row_idx, start_col:].astype(str).values
            cols = df.iloc[well_row_idx+1, start_col:].astype(str).values
            samples = [f"{r}{c}" for r, c in zip(rows, cols)]
            
        samples = [s.strip() if str(s).strip() not in ['nan', ''] else f'Unknown_{i}' for i, s in enumerate(samples)]
        
        data_rows = []
        times = []
        start_idx = max([well_row_idx + 2] + ([content_row_idx + 1] if content_row_idx else []))
        
        # Сбор временных точек
        for i in range(start_idx, len(df)):
            val_time = str(df.iloc[i, 0]).strip()
            if pd.isna(df.iloc[i, 0]) or val_time == '' or val_time == 'nan':
                continue
            
            # Если начинаются итоги или следующий блок таблицы - прерываем чтение
            if any(x in val_time.lower() for x in ['deviation', 'average', 'blank', 'result']):
                if len(times) > 0:
                    break
                else:
                    continue
            
            times.append(val_time)
            data_rows.append(df.iloc[i, start_col:].values)
            
        parsed_df = pd.DataFrame(data_rows, columns=samples, index=times).T
        parsed_df.reset_index(inplace=True)
        parsed_df.rename(columns={'index': 'Образец'}, inplace=True)
        parsed_df.columns = deduplicate(parsed_df.columns)
        return parsed_df, True
    else:
        # ОБЫЧНАЯ ТАБЛИЦА
        headers = [str(h).strip() if str(h).strip() not in ['nan', ''] else f"Unnamed_{i}" for i, h in enumerate(df.iloc[0])]
        df = pd.DataFrame(df.values[1:], columns=headers)
        df.columns = deduplicate(df.columns)
        return df, False


# --- ГЕНЕРАТОР ИНТЕРФЕЙСА ДЛЯ ОТДЕЛЬНОГО ФАЙЛА ---

def render_analysis_ui(file_obj, title, prefix, alpha_grubbs, alpha_norm):
    st.header(title)
    df_raw, is_raw = parse_uploaded_file(file_obj, prefix)
    
    if is_raw:
        st.success("✅ Обнаружен и автоматически разобран первый блок сырого файла FLUOstar!")
    else:
        st.info("ℹ️ Загружена стандартная таблица. Выберите колонки ниже.")
        
    st.dataframe(df_raw.head(10))
    
    cols = df_raw.columns.tolist()
    col1, col2 = st.columns(2)
    with col1:
        default_sample = 'Образец' if 'Образец' in cols else cols[0]
        sample_col_idx = cols.index(default_sample) if default_sample in cols else 0
        sample_col = st.selectbox(f"Колонка с названием образца ({prefix})", cols, index=sample_col_idx, key=f"{prefix}_sample")
    with col2:
        if is_raw:
            default_times = [c for c in cols if c != sample_col]
        else:
            default_times = [c for c in cols if 'min' in str(c).lower() or 'h' in str(c).lower() or str(c).isdigit()]
        time_cols = st.multiselect(f"Колонки со временем ({prefix})", cols, default=default_times, key=f"{prefix}_time")

    if st.button(f"Рассчитать статистику {title}") and sample_col and time_cols:
        with st.spinner('Анализ данных...'):
            res_df, clean_long_df = process_dataframe(df_raw, sample_col, time_cols, alpha_grubbs, alpha_norm)
            st.session_state[f'res_{prefix}'] = res_df
            st.session_state[f'clean_{prefix}'] = clean_long_df
            st.session_state[f'time_cols_{prefix}'] = time_cols
            st.session_state[f'sample_col_{prefix}'] = sample_col

    if f'res_{prefix}' in st.session_state:
        res_df = st.session_state[f'res_{prefix}']
        clean_long_df = st.session_state[f'clean_{prefix}']
        
        tab1, tab2, tab3 = st.tabs(["📊 Сводная таблица", "📈 Кинетика (Кривые роста)", "🔔 Гистограммы"])
        
        with tab1:
            st.dataframe(res_df)
            
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                res_df.to_excel(writer, index=False, sheet_name=f'{prefix}_Results')
            excel_data = output.getvalue()
            
            st.download_button(
                label=f"Скачать результаты {title} (Excel)",
                data=excel_data,
                file_name=f'fluostar_{prefix}_results.xlsx',
                mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                key=f"dl_{prefix}"
            )

        with tab2:
            st.subheader(f"Кривые кинетики - {title} (Среднее ± Ст. Ошибка)")
            plot_df = res_df.copy()
            
            def extract_time_val(t):
                t_str = str(t).lower().strip().replace(',', '.')
                
                # Защита от системных колонок (чтобы "OD600" не стало 600-й минутой)
                if any(w in t_str for w in ['od600', 'образец', 'sample', 'unnamed', 'blank', 'raw', 'well', 'content']):
                    return np.nan
                    
                h_match = re.search(r'(\d+[.]?\d*)\s*h', t_str)
                m_match = re.search(r'(\d+[.]?\d*)\s*m', t_str)
                
                if h_match or m_match:
                    h = float(h_match.group(1)) if h_match else 0.0
                    m = float(m_match.group(1)) if m_match else 0.0
                    return h * 60.0 + m
                    
                match = re.search(r'(\d+[.]?\d*)', t_str)
                if match:
                    return float(match.group(1))
                return np.nan

            # Приводим к числам и выбрасываем пустые
            plot_df['Время_num'] = plot_df['Время'].apply(extract_time_val)
            plot_df['Среднее'] = pd.to_numeric(plot_df['Среднее'], errors='coerce')
            plot_df['Ст. ошибка (SE)'] = pd.to_numeric(plot_df['Ст. ошибка (SE)'], errors='coerce').fillna(0)
            
            plot_df = plot_df.dropna(subset=['Время_num', 'Среднее'])
            plot_df = plot_df.sort_values(by=['Образец', 'Время_num'])
            
            if plot_df.empty:
                st.warning("⚠️ Недостаточно числовых данных для графика. Проверьте правильность выбранных временных колонок.")
            else:
                # ПАТЧ: Если суммарная ошибка равна нулю (все N=1), вообще отключаем error_y, чтобы Plotly не глючил
                err_col = 'Ст. ошибка (SE)' if plot_df['Ст. ошибка (SE)'].sum() > 0 else None
                
                fig = px.line(plot_df, x='Время_num', y='Среднее', color='Образец', markers=True, 
                              error_y=err_col, title=f"Динамика {title}")
                fig.update_layout(xaxis_title="Время (в минутах / циклах)", yaxis_title="Значение")
                st.plotly_chart(fig, use_container_width=True, key=f"plot1_{prefix}")

        with tab3:
            st.subheader(f"Оценка распределения выборки - {title} (Для групп N > 2)")
            col_s, col_t = st.columns(2)
            with col_s:
                sel_sample = st.selectbox("Образец", res_df['Образец'].unique(), key=f"hist_sample_{prefix}")
            with col_t:
                sel_time = st.selectbox("Время", st.session_state[f'time_cols_{prefix}'], key=f"hist_time_{prefix}")
                
            subset = clean_long_df[(clean_long_df.iloc[:,0] == sel_sample) & (clean_long_df['Time'] == sel_time)]
            
            if len(subset) > 2:
                fig2 = px.histogram(subset, x="Value", marginal="box", nbins=10, 
                                    title=f"Распределение: {sel_sample} при {sel_time} (Без выбросов)")
                st.plotly_chart(fig2, use_container_width=True, key=f"plot2_{prefix}")
                
                stat_row = res_df[(res_df['Образец'] == sel_sample) & (res_df['Время'] == sel_time)].iloc[0]
                st.info(f"**Тест:** {stat_row['Тест нормальности']} | **p-value:** {stat_row['p-value (норм.)']:.4f} | **Нормально?** {stat_row['Распределение нормально?']}")
                if stat_row['Выбросы'] != "Нет":
                    st.warning(f"Удалены выбросы: {stat_row['Выбросы']}")
            else:
                st.warning("Недостаточно данных для построения распределения (N <= 2). Выборка слишком мала.")


# --- ИНТЕРФЕЙС STREAMLIT ---

st.title("🔬 Анализ данных FLUOstar (Lumi / OD600)")

st.sidebar.header("Настройки")
file_lumi = st.sidebar.file_uploader("Загрузить файл Lumi (Excel/CSV)", type=['csv', 'xlsx'])
file_od = st.sidebar.file_uploader("Загрузить файл OD600 (Excel/CSV)", type=['csv', 'xlsx'])

st.sidebar.markdown("---")
alpha_grubbs = st.sidebar.slider("Уровень значимости (α) теста Граббса (Выбросы)", 0.01, 0.10, 0.05, 0.01)
alpha_norm = st.sidebar.slider("Уровень значимости (α) теста нормальности", 0.01, 0.10, 0.05, 0.01)

# БЛОК 1: LUMI
if file_lumi is not None:
    render_analysis_ui(file_lumi, "Luminescence", "lumi", alpha_grubbs, alpha_norm)
else:
    st.info("👈 Загрузите данные в боковой панели слева (Lumi, OD600 или оба).")

# БЛОК 2: OD600
if file_od is not None:
    st.markdown("---")
    render_analysis_ui(file_od, "OD600", "od", alpha_grubbs, alpha_norm)

# БЛОК 3: ИНТЕГРАЦИЯ LUMI И OD600
if 'res_lumi' in st.session_state and 'res_od' in st.session_state:
    st.markdown("---")
    st.header("🔗 Анализ отношения Lumi / OD600")
    
    # Безопасное объединение: оставляет только те точки, которые есть в ОБЕИХ таблицах
    merged = pd.merge(st.session_state['res_lumi'], st.session_state['res_od'], on=['Образец', 'Время'], suffixes=('_Lumi', '_OD'))
    
    if merged.empty:
        st.warning("⚠️ Внимание: Не найдено совпадающих образцов и временных точек между Lumi и OD600. Расчет отношения невозможен.")
    else:
        st.success(f"✅ Найдено {len(merged)} совпадающих точек для расчета отношения!")
        merged['Lumi / OD600'] = merged['Среднее_Lumi'] / merged['Среднее_OD']
        
        st.subheader("Сводная таблица (Lumi/OD600)")
        st.dataframe(merged[['Образец', 'Время', 'Среднее_Lumi', 'Среднее_OD', 'Lumi / OD600']])
        
        def extract_time_val_ratio(t):
            t_str = str(t).lower().strip().replace(',', '.')
            if any(w in t_str for w in ['od600', 'образец', 'sample', 'unnamed', 'blank', 'raw', 'well', 'content']):
                return np.nan
            h_match = re.search(r'(\d+[.]?\d*)\s*h', t_str)
            m_match = re.search(r'(\d+[.]?\d*)\s*m', t_str)
            if h_match or m_match:
                h = float(h_match.group(1)) if h_match else 0.0
                m = float(m_match.group(1)) if m_match else 0.0
                return h * 60.0 + m
            match = re.search(r'(\d+[.]?\d*)', t_str)
            if match:
                return float(match.group(1))
            return np.nan

        merged['Время_num'] = merged['Время'].apply(extract_time_val_ratio)
        merged['Lumi / OD600'] = pd.to_numeric(merged['Lumi / OD600'], errors='coerce')
        
        merged = merged.dropna(subset=['Время_num', 'Lumi / OD600'])
        merged = merged.sort_values(by=['Образец', 'Время_num'])
            
        if merged.empty:
            st.warning("⚠️ График отношения не построен (недостаточно точек).")
        else:
            fig3 = px.line(merged, x='Время_num', y='Lumi / OD600', color='Образец', markers=True, title="Отношение Luminescence к OD600")
            fig3.update_layout(xaxis_title="Время (в минутах / циклах)", yaxis_title="Отношение Lumi / OD")
            st.plotly_chart(fig3, use_container_width=True, key="plot_ratio")
        
        output_merged = io.BytesIO()
        with pd.ExcelWriter(output_merged, engine='openpyxl') as writer:
            merged.to_excel(writer, index=False, sheet_name='Lumi_OD_Ratio')
        excel_data_merged = output_merged.getvalue()
        
        st.download_button(
            label="Скачать сводные результаты Lumi/OD (Excel)",
            data=excel_data_merged,
            file_name='lumi_od_ratio.xlsx',
            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            key="dl_ratio"
        )