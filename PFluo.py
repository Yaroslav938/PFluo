import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats
import plotly.express as px
import io
import re

# --- НАСТРОЙКИ СТРАНИЦЫ ---
st.set_page_config(page_title="Анализ данных FLUOstar", layout="wide")

# --- ГЛОБАЛЬНЫЕ ФУНКЦИИ ---

def parse_time_to_minutes(t):
    """Умная функция перевода времени в числа для оси X."""
    t_str = str(t).lower().strip().replace(',', '.')
    t_str = re.sub(r'_\d+$', '', t_str).strip()
    
    if any(w in t_str for w in ['od600', 'образец', 'sample', 'unnamed', 'blank', 'raw', 'well', 'content', 'deviation', 'average', 'standard', 'result', 'mean']):
        return np.nan
        
    h_match = re.search(r'(\d+[.]?\d*)\s*(h|ч)(?:\b|\s|$)', t_str)
    m_match = re.search(r'(\d+[.]?\d*)\s*(m|min|мин)(?:\b|\s|$)', t_str)
    
    if h_match or m_match:
        h = float(h_match.group(1)) if h_match else 0.0
        m = float(m_match.group(1)) if m_match else 0.0
        return h * 60.0 + m
        
    match = re.search(r'(\d+[.]?\d*)', t_str)
    if match:
        return float(match.group(1))
        
    return np.nan

def clean_number_str(val):
    """Идеальный очиститель чисел (понимает любые запятые, пробелы и артефакты прибора)."""
    if pd.isna(val):
        return np.nan
        
    val = str(val).strip().lower()
    if val in ['nan', 'none', '', 'ovr', 'overflow']:
        return np.nan
    
    val = re.sub(r'\s+', '', val)
    val = val.replace('<', '').replace('>', '').replace('−', '-')
    
    if ',' in val and '.' in val:
        if val.rfind(',') > val.rfind('.'):
            val = val.replace('.', '').replace(',', '.')
        else:
            val = val.replace(',', '')
    else:
        val = val.replace(',', '.')
        
    try:
        return float(val)
    except ValueError:
        return np.nan

def grubbs_test(x, alpha=0.05):
    """Рекурсивный тест Граббса для поиска и удаления выбросов."""
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
            
            raw_series = group[time_col].apply(clean_number_str)
            raw_values = raw_series.dropna().values
            
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
            
    df_results = pd.DataFrame(results)
    df_cleaned = pd.DataFrame(cleaned_data_long)
    
    if df_cleaned.empty:
        df_cleaned = pd.DataFrame(columns=[sample_col, 'Time', 'Value'])
        
    return df_results, df_cleaned

def parse_uploaded_file(uploaded_file, key_prefix):
    """Мощнейший сканер блоков данных с отсечением дублирующихся таблиц."""
    
    def deduplicate(columns):
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

    def is_time_string(t_str):
        t_str = str(t_str).lower().strip().replace(',', '.')
        if t_str in ['nan', 'none', '']:
            return False
        if re.search(r'\d', t_str) and any(w in t_str for w in ['min', 'h', 'm', 'ч', 'мин', 'cycle']):
            return True
        clean_t = re.sub(r'_\d+$', '', t_str).strip()
        if re.fullmatch(r'\d+[.]?\d*', clean_t):
            return True
        return False

    if uploaded_file.name.endswith('.csv'):
        uploaded_file.seek(0)
        first_line = uploaded_file.readline().decode('utf-8', errors='ignore')
        uploaded_file.seek(0)
        delimiter = ';' if first_line.count(';') > first_line.count(',') else ','
        df = pd.read_csv(uploaded_file, header=None, sep=delimiter)
    else:
        xl = pd.ExcelFile(uploaded_file)
        sheet_name = st.selectbox(f"Лист ({uploaded_file.name})", xl.sheet_names, key=f"{key_prefix}_sheet")
        df = pd.read_excel(uploaded_file, sheet_name=sheet_name, header=None)
        
    df = df.dropna(how='all').dropna(axis=1, how='all').reset_index(drop=True)
    if df.empty:
        return pd.DataFrame(), False
        
    df.columns = range(df.shape[1])
    
    well_row_idx = None
    well_col_idx = 0
    content_row_idx = None
    
    for i in range(min(150, len(df))):
        for j in range(min(5, df.shape[1])):
            val = str(df.iloc[i, j]).lower()
            if 'well' in val and 'row' in val and well_row_idx is None:
                well_row_idx = i
                well_col_idx = j
            if 'content' in val and content_row_idx is None:
                content_row_idx = i
            
    if well_row_idx is not None:
        start_col = well_col_idx + 1
        for c in range(well_col_idx + 1, df.shape[1]):
            val = str(df.iloc[well_row_idx, c]).strip()
            if val.lower() not in ['nan', '', 'none']:
                start_col = c
                break
                
        if content_row_idx is not None:
            samples = df.iloc[content_row_idx, start_col:].astype(str).values
        else:
            rows = df.iloc[well_row_idx, start_col:].astype(str).values
            cols = df.iloc[well_row_idx+1, start_col:].astype(str).values
            samples = [f"{str(r).replace('.0', '')}{str(c).replace('.0', '')}" for r, c in zip(rows, cols)]
            
        samples = [s.strip() if str(s).strip().lower() not in ['nan', '', 'none'] else f'Unknown_{i}' for i, s in enumerate(samples)]
        
        time_rows = []
        time_col_for_row = {}
        for i in range(len(df)):
            is_parasite = False
            for c in range(start_col):
                val_str = str(df.iloc[i, c]).lower()
                if any(w in val_str for w in ['od600', 'образец', 'sample', 'unnamed', 'blank', 'raw', 'well', 'content', 'deviation', 'average', 'standard', 'result', 'mean']):
                    is_parasite = True
                    break
            if is_parasite:
                continue
                
            for c in range(start_col):
                val_str = str(df.iloc[i, c])
                if is_time_string(val_str):
                    time_rows.append(i)
                    time_col_for_row[i] = c
                    break
                
        if not time_rows:
            return pd.DataFrame(), False
            
        blocks = []
        current_block = [time_rows[0]]
        for i in range(1, len(time_rows)):
            row_idx = time_rows[i]
            prev_row_idx = time_rows[i-1]
            
            curr_t = parse_time_to_minutes(df.iloc[row_idx, time_col_for_row[row_idx]])
            prev_t = parse_time_to_minutes(df.iloc[prev_row_idx, time_col_for_row[prev_row_idx]])
            
            if row_idx - prev_row_idx > 1:
                blocks.append(current_block)
                current_block = [row_idx]
            elif pd.notna(curr_t) and pd.notna(prev_t) and curr_t <= prev_t:
                blocks.append(current_block)
                current_block = [row_idx]
            else:
                current_block.append(row_idx)
        blocks.append(current_block)
        
        best_block = None
        max_score = -float('inf')
        
        for block in blocks:
            data_matrix = df.iloc[block, start_col:].astype(str)
            try:
                num_count = data_matrix.map(lambda x: bool(re.search(r'\d', x))).sum().sum()
            except AttributeError:
                num_count = data_matrix.applymap(lambda x: bool(re.search(r'\d', x))).sum().sum()
                
            if num_count == 0:
                continue
                
            title = ""
            for r in range(block[0]-1, max(-1, block[0]-15), -1):
                row_vals = [str(x).strip().lower() for x in df.iloc[r, :start_col+2].values if pd.notna(x) and str(x).strip() != '']
                text = " ".join(row_vals)
                if "well" in text or "content" in text or "time" in text:
                    continue
                if text:
                    title = text
                    break
                    
            score = num_count
            if "blank corrected" in title:
                score += 500000
            elif "raw data" in title:
                score += 100000
                
            if score > max_score:
                max_score = score
                best_block = block
                
        if best_block is None:
            return pd.DataFrame(), False
            
        times = [str(df.iloc[i, time_col_for_row[i]]).strip() for i in best_block]
        data_rows = df.iloc[best_block, start_col:].values
        
        parsed_df = pd.DataFrame(data_rows, columns=samples, index=times).T
        parsed_df.reset_index(inplace=True)
        parsed_df.rename(columns={'index': 'Образец'}, inplace=True)
        parsed_df.columns = deduplicate(parsed_df.columns)
        return parsed_df, True
    else:
        headers = [str(h).strip() if str(h).strip().lower() not in ['nan', '', 'none'] else f"Unnamed_{i}" for i, h in enumerate(df.iloc[0])]
        df = pd.DataFrame(df.values[1:], columns=headers)
        df.columns = deduplicate(df.columns)
        return df, False


# --- ГЕНЕРАТОР ИНТЕРФЕЙСА ДЛЯ ОТДЕЛЬНОГО ФАЙЛА ---

def render_analysis_ui(file_obj, title, prefix, alpha_grubbs, alpha_norm):
    st.header(title)
    df_raw, is_raw = parse_uploaded_file(file_obj, prefix)
    
    cols = df_raw.columns.tolist() if not df_raw.empty else []
    if not cols:
        st.error(f"⚠️ Не удалось распознать структуру данных в файле '{title}'. Убедитесь, что таблица не пустая и имеет правильный формат FLUOstar.")
        return
        
    if is_raw:
        st.success("✅ Сырой файл FLUOstar отсканирован успешно!")
    else:
        st.info("ℹ️ Загружена стандартная таблица. Выберите колонки ниже.")
        
    with st.expander("Предпросмотр данных", expanded=False):
        st.dataframe(df_raw.head(10))
    
    col1, col2 = st.columns(2)
    with col1:
        default_sample = 'Образец' if 'Образец' in cols else cols[0]
        sample_col_idx = cols.index(default_sample) if default_sample in cols else 0
        sample_col = st.selectbox(f"Колонка с названием образца ({prefix})", cols, index=sample_col_idx, key=f"{prefix}_sample")
    with col2:
        if is_raw:
            default_times = [c for c in cols if c != sample_col and pd.notna(parse_time_to_minutes(c))]
        else:
            default_times = [c for c in cols if 'min' in str(c).lower() or 'h' in str(c).lower() or str(c).isdigit()]
        time_cols = st.multiselect(f"Колонки со временем ({prefix})", [c for c in cols if c != sample_col], default=default_times, key=f"{prefix}_time")

    if st.button(f"Рассчитать статистику {title}") and sample_col and time_cols:
        with st.spinner('Анализ данных...'):
            res_df, clean_long_df = process_dataframe(df_raw, sample_col, time_cols, alpha_grubbs, alpha_norm)
            res_df['Время_num'] = res_df['Время'].apply(parse_time_to_minutes)
            
            st.session_state[f'res_{prefix}'] = res_df
            st.session_state[f'clean_{prefix}'] = clean_long_df
            st.session_state[f'time_cols_{prefix}'] = time_cols
            st.session_state[f'sample_col_{prefix}'] = sample_col

    if f'res_{prefix}' in st.session_state:
        res_df = st.session_state[f'res_{prefix}']
        clean_long_df = st.session_state[f'clean_{prefix}']
        sample_col_name = st.session_state[f'sample_col_{prefix}']
        
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Сводная таблица", "📈 Кинетика", "🔔 Гистограммы", "⚖️ Пики и Сравнение (T-test)"])
        
        with tab1:
            st.subheader("Матрица средних значений (Широкий формат)")
            try:
                pivot_mean = res_df.pivot_table(index='Образец', columns='Время', values='Среднее', aggfunc='first')
                time_map = res_df.drop_duplicates(subset=['Время', 'Время_num']).set_index('Время')['Время_num']
                sorted_cols = [c for c in time_map.sort_values().index if c in pivot_mean.columns]
                pivot_mean = pivot_mean[sorted_cols].reset_index()
                st.dataframe(pivot_mean)
            except Exception as e:
                pivot_mean = pd.DataFrame()
                
            st.subheader("Полная статистика (Длинный формат)")
            st.dataframe(res_df.drop(columns=['Время_num'], errors='ignore'))
            
            if not res_df.empty:
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    if not pivot_mean.empty:
                        pivot_mean.to_excel(writer, index=False, sheet_name=f'{prefix}_Средние_Широкая')
                    res_df.drop(columns=['Время_num'], errors='ignore').to_excel(writer, index=False, sheet_name=f'{prefix}_Полная_Стат')
                excel_data = output.getvalue()
                
                st.download_button(
                    label=f"Скачать результаты {title} (Excel)",
                    data=excel_data,
                    file_name=f'fluostar_{prefix}_results.xlsx',
                    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                    key=f"dl_{prefix}"
                )

        with tab2:
            st.subheader(f"Кривые кинетики - {title}")
            plot_df = res_df.copy()
            if plot_df.empty:
                st.warning("Нет рассчитанных данных.")
            else:
                plot_df['Среднее'] = pd.to_numeric(plot_df['Среднее'], errors='coerce')
                plot_df['Ст. ошибка (SE)'] = pd.to_numeric(plot_df['Ст. ошибка (SE)'], errors='coerce').fillna(0)
                plot_df = plot_df.dropna(subset=['Время_num', 'Среднее'])
                plot_df = plot_df.sort_values(by=['Образец', 'Время_num'])
                
                if plot_df.empty:
                    st.warning("⚠️ Недостаточно данных для графика.")
                else:
                    all_samples = plot_df['Образец'].unique()
                    selected_samples = st.multiselect(
                        f"Отображаемые образцы:",
                        options=all_samples,
                        default=all_samples,
                        key=f"filter_plot_{prefix}"
                    )
                    filtered_plot_df = plot_df[plot_df['Образец'].isin(selected_samples)]
                    
                    if filtered_plot_df.empty:
                        st.info("Выберите образцы.")
                    else:
                        err_col = 'Ст. ошибка (SE)' if filtered_plot_df['Ст. ошибка (SE)'].sum() > 0 else None
                        fig = px.line(filtered_plot_df, x='Время_num', y='Среднее', color='Образец', markers=True, 
                                      error_y=err_col, title=f"Динамика {title}",
                                      color_discrete_sequence=px.colors.qualitative.Alphabet * 5)
                        fig.update_layout(xaxis_title="Время (в минутах / циклах)", yaxis_title="Значение", height=700)
                        st.plotly_chart(fig, use_container_width=True, key=f"plot1_{prefix}")

        with tab3:
            st.subheader("Гистограммы распределения")
            if not res_df.empty:
                col_s, col_t = st.columns(2)
                with col_s:
                    sel_sample = st.selectbox("Образец", res_df['Образец'].unique(), key=f"hist_sample_{prefix}")
                with col_t:
                    sel_time = st.selectbox("Время", st.session_state[f'time_cols_{prefix}'], key=f"hist_time_{prefix}")
                
                subset = pd.DataFrame() if clean_long_df.empty else clean_long_df[(clean_long_df[sample_col_name] == sel_sample) & (clean_long_df['Time'] == sel_time)]
                
                if len(subset) > 2:
                    fig2 = px.histogram(subset, x="Value", marginal="box", nbins=10, title=f"{sel_sample} при {sel_time}")
                    st.plotly_chart(fig2, use_container_width=True, key=f"plot2_{prefix}")
                    stat_row = res_df[(res_df['Образец'] == sel_sample) & (res_df['Время'] == sel_time)].iloc[0]
                    st.info(f"**Тест:** {stat_row['Тест нормальности']} | **p-value:** {stat_row['p-value (норм.)']:.4f} | **Нормально?** {stat_row['Распределение нормально?']}")
                else:
                    st.warning("Недостаточно данных для распределения (N <= 2).")

        with tab4:
            st.subheader("Максимальные значения (Поиск пиков)")
            
            # Автоматический поиск максимального значения для каждого образца
            idx_max = res_df.groupby('Образец')['Среднее'].idxmax().dropna()
            peaks_df = res_df.loc[idx_max].sort_values(by='Среднее', ascending=False)
            
            st.dataframe(peaks_df[['Образец', 'Время', 'Среднее', 'Ст. ошибка (SE)', 'CV (%)']].rename(columns={'Время': 'Время пика (T max)', 'Среднее': 'Max Значение'}))
            
            st.markdown("---")
            st.subheader("Сравнение образцов (T-критерий Стьюдента)")
            st.write("Определяет, есть ли статистически значимая разница между двумя образцами.")
            
            colA, colB, colT = st.columns(3)
            with colA:
                sampA = st.selectbox("Образец 1", res_df['Образец'].unique(), key=f"ttest_s1_{prefix}")
            with colB:
                sampB = st.selectbox("Образец 2", res_df['Образец'].unique(), key=f"ttest_s2_{prefix}")
            with colT:
                time_opts = ["Максимум (Пик)"] + list(st.session_state[f'time_cols_{prefix}'])
                time_choice = st.selectbox("Время для сравнения", time_opts, key=f"ttest_t_{prefix}")
                
            if st.button("Сравнить (T-test)", key=f"ttest_btn_{prefix}"):
                t1 = peaks_df[peaks_df['Образец'] == sampA]['Время'].values[0] if time_choice == "Максимум (Пик)" else time_choice
                t2 = peaks_df[peaks_df['Образец'] == sampB]['Время'].values[0] if time_choice == "Максимум (Пик)" else time_choice
                
                valA = clean_long_df[(clean_long_df[sample_col_name] == sampA) & (clean_long_df['Time'] == t1)]['Value']
                valB = clean_long_df[(clean_long_df[sample_col_name] == sampB) & (clean_long_df['Time'] == t2)]['Value']
                
                if len(valA) < 2 or len(valB) < 2:
                    st.warning("Недостаточно данных для T-теста (нужно минимум 2 значения в очищенной выборке для каждого).")
                else:
                    t_stat, p_val = stats.ttest_ind(valA, valB, equal_var=False) # Welch's t-test
                    st.success(f"**Результат сравнения:** {sampA} (при {t1}) **vs** {sampB} (при {t2})")
                    st.write(f"- t-статистика: **{t_stat:.4f}**")
                    st.write(f"- p-value: **{p_val:.4f}**")
                    if p_val < 0.05:
                        st.info("Вывод: Различия **статистически значимы** (p < 0.05).")
                    else:
                        st.info("Вывод: Статистически значимых различий **нет** (p >= 0.05).")


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
    
    use_mcf = st.checkbox("Пересчитать OD600 в МакФарланды (McF)", value=False)
    mcf_factor = 1.0
    if use_mcf:
        mcf_factor = st.number_input("Коэффициент: 1 McFarland = OD600 ...", min_value=0.01, value=0.50, step=0.05)
    
    res_lumi_merged = st.session_state['res_lumi'].dropna(subset=['Время_num']).copy()
    res_od_merged = st.session_state['res_od'].dropna(subset=['Время_num']).copy()
    
    res_lumi_merged['Время_num_round'] = res_lumi_merged['Время_num'].round(2)
    res_od_merged['Время_num_round'] = res_od_merged['Время_num'].round(2)
    
    merged = pd.merge(res_lumi_merged, res_od_merged, on=['Образец', 'Время_num_round'], suffixes=('_Lumi', '_OD'))
    
    if merged.empty:
        st.warning("⚠️ Внимание: Не найдено совпадающих образцов и временных точек (в минутах) между Lumi и OD600.")
    else:
        st.success(f"✅ Найдено {len(merged)} совпадающих точек для расчета!")
        
        merged['Среднее_Lumi'] = pd.to_numeric(merged['Среднее_Lumi'], errors='coerce')
        merged['Среднее_OD'] = pd.to_numeric(merged['Среднее_OD'], errors='coerce')
        
        if use_mcf:
            merged['Среднее_OD'] = merged['Среднее_OD'] / mcf_factor
            ratio_col = 'Lumi / McF'
            od_col_title = 'Среднее_McF'
            merged.rename(columns={'Среднее_OD': od_col_title}, inplace=True)
        else:
            ratio_col = 'Lumi / OD600'
            od_col_title = 'Среднее_OD'
            
        merged[ratio_col] = merged['Среднее_Lumi'] / merged[od_col_title]
        merged.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        show_cols = ['Образец', 'Время_Lumi', 'Среднее_Lumi', od_col_title, ratio_col]
        
        st.subheader(f"Матрица отношения {ratio_col} (Широкий формат)")
        try:
            pivot_ratio = merged.pivot_table(index='Образец', columns='Время_Lumi', values=ratio_col, aggfunc='first')
            time_map_ratio = merged.drop_duplicates(subset=['Время_Lumi', 'Время_num_round']).set_index('Время_Lumi')['Время_num_round']
            sorted_cols_ratio = [c for c in time_map_ratio.sort_values().index if c in pivot_ratio.columns]
            pivot_ratio = pivot_ratio[sorted_cols_ratio].reset_index()
            st.dataframe(pivot_ratio)
        except Exception as e:
            pivot_ratio = pd.DataFrame()
            
        st.subheader(f"Полная сводная таблица ({ratio_col})")
        st.dataframe(merged[show_cols].rename(columns={'Время_Lumi': 'Время'}))
        
        merged = merged.dropna(subset=['Время_num_round', ratio_col])
        merged = merged.sort_values(by=['Образец', 'Время_num_round'])
            
        if merged.empty:
            st.warning("⚠️ График не построен (недостаточно точек).")
        else:
            all_samples_ratio = merged['Образец'].unique()
            selected_samples_ratio = st.multiselect(
                "Отображаемые образцы:",
                options=all_samples_ratio,
                default=all_samples_ratio,
                key="filter_plot_ratio"
            )
            filtered_ratio = merged[merged['Образец'].isin(selected_samples_ratio)]
            
            if filtered_ratio.empty:
                st.info("Выберите образцы.")
            else:
                fig3 = px.line(filtered_ratio, x='Время_num_round', y=ratio_col, color='Образец', markers=True, 
                               title=f"Отношение {ratio_col}",
                               color_discrete_sequence=px.colors.qualitative.Alphabet * 5)
                fig3.update_layout(xaxis_title="Время (в минутах / циклах)", yaxis_title=ratio_col, height=700)
                st.plotly_chart(fig3, use_container_width=True, key="plot_ratio")
        
        output_merged = io.BytesIO()
        with pd.ExcelWriter(output_merged, engine='openpyxl') as writer:
            if not pivot_ratio.empty:
                pivot_ratio.to_excel(writer, index=False, sheet_name='Ratio_Широкая')
            merged[show_cols].rename(columns={'Время_Lumi': 'Время'}).to_excel(writer, index=False, sheet_name='Ratio_Полная')
        excel_data_merged = output_merged.getvalue()
        
        st.download_button(
            label="Скачать сводные результаты (Excel)",
            data=excel_data_merged,
            file_name='fluostar_ratio_results.xlsx',
            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            key="dl_ratio"
        )