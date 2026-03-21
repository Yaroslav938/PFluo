import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats
import plotly.express as px
import io
import re
import itertools

# --- НАСТРОЙКИ СТРАНИЦЫ ---
st.set_page_config(page_title="Анализ данных FLUOstar", layout="wide")

# --- ГЛОБАЛЬНЫЕ ФУНКЦИИ ---

def fdr_bh(pvals):
    """Поправка Бенджамини-Хохберга (FDR) для множественных сравнений."""
    pvals = np.asarray(pvals)
    n = len(pvals)
    if n == 0: return []
    sorted_indices = np.argsort(pvals)
    sorted_pvals = pvals[sorted_indices]
    fdr_pvals = sorted_pvals * n / np.arange(1, n + 1)
    fdr_pvals[::-1] = np.minimum.accumulate(fdr_pvals[::-1])
    fdr_pvals = np.minimum(fdr_pvals, 1.0)
    unsorted_pvals = np.empty(n)
    unsorted_pvals[sorted_indices] = fdr_pvals
    return unsorted_pvals

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
    """Безопасный сканер данных с системой авто-транспонирования (Fallback)."""
    
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

    def get_base_samplename(h):
        """Умное отсечение суффиксов повторностей (C1-2.1 -> C1-2, Control1 -> Control)"""
        b = re.sub(r'[._\s]\d+$', '', h)
        if b == h:
            m = re.search(r'^([a-zA-Zа-яА-Я]+)(\d)$', h)
            if m: b = m.group(1)
        return b

    def _fallback(df_in):
        """Анализирует таблицу и при необходимости автоматически её переворачивает."""
        if df_in.empty: 
            return pd.DataFrame(), False
            
        headers = [str(h).strip() if str(h).strip().lower() not in ['nan', '', 'none'] else f"Unnamed_{i}" for i, h in enumerate(df_in.iloc[0])]
        
        # Проверяем, состоят ли данные в первом столбце (под шапкой) только из цифр
        if len(df_in) > 1:
            first_col_numeric = pd.to_numeric(df_in.iloc[1:, 0], errors='coerce').notna().mean() > 0.8
        else:
            first_col_numeric = False
            
        if first_col_numeric:
            # Проверяем, является ли первая ячейка указателем на Время
            h0 = str(headers[0]).lower()
            first_col_is_time = any(w in h0 for w in ['time', 'min', 'h', 'время', 'cycle', 'цикл', 'unnamed']) or (pd.to_numeric(headers[0], errors='coerce') is not np.nan)
            
            if first_col_is_time:
                # Тип 1: Время в 1 колонке, Образцы в остальных колонках
                time_vals = df_in.iloc[1:, 0].values
                time_cols = [str(t).replace(',', '.') for t in time_vals]
                data = df_in.iloc[1:, 1:].values
                sample_headers = headers[1:]
                
                fallback_df = pd.DataFrame(data.T, columns=time_cols)
                base_samples = [get_base_samplename(h) for h in sample_headers]
                fallback_df.insert(0, 'Образец', base_samples)
                fallback_df.columns = deduplicate(fallback_df.columns)
                return fallback_df, False
                
            else:
                # Тип 2: Как в xen_data_c.csv (Времени нет, просто циклы в строках, Образцы в колонках)
                data = df_in.iloc[1:].values
                time_cols = [f"{i+1}" for i in range(data.shape[0])]
                
                fallback_df = pd.DataFrame(data.T, columns=time_cols)
                base_samples = [get_base_samplename(h) for h in headers]
                fallback_df.insert(0, 'Образец', base_samples)
                fallback_df.columns = deduplicate(fallback_df.columns)
                return fallback_df, False
                
        # Если это стандартная таблица (Образцы в строках, Время в колонках)
        fallback_df = pd.DataFrame(df_in.values[1:], columns=headers)
        fallback_df.columns = deduplicate(fallback_df.columns)
        fallback_df.rename(columns={fallback_df.columns[0]: 'Образец'}, inplace=True)
        return fallback_df, False

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
        sample_text = uploaded_file.read(2048).decode('utf-8', errors='ignore')
        uploaded_file.seek(0)
        delimiter = ';' if sample_text.count(';') > sample_text.count(',') else ','
        if sample_text.count('\t') > sample_text.count(delimiter): 
            delimiter = '\t'
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
            if 'well' in val and ('row' in val or 'col' in val) and well_row_idx is None:
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
            for c in range(start_col + 1):
                val_str = str(df.iloc[i, c])
                if is_time_string(val_str):
                    time_rows.append(i)
                    time_col_for_row[i] = c
                    break
                
        if not time_rows:
            return _fallback(df)
            
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
                if any(w in text for w in ['well', 'content', 'time', 'min', 'cycle']):
                    continue
                if text:
                    title = text
                    break
                    
            score = num_count
            if "blank corrected" in title:
                score += num_count * 2
            elif "raw data" in title:
                score += num_count * 1
            elif "average" in title or "deviation" in title or "standard" in title:
                score -= num_count * 0.9 
                
            if score > max_score:
                max_score = score
                best_block = block
                
        if best_block is None:
            return _fallback(df)
            
        times = [str(df.iloc[i, time_col_for_row[i]]).strip() for i in best_block]
        data_rows = df.iloc[best_block, start_col:].values
        
        parsed_df = pd.DataFrame(data_rows, columns=samples, index=times).T
        parsed_df.reset_index(inplace=True)
        parsed_df.rename(columns={'index': 'Образец'}, inplace=True)
        parsed_df.columns = deduplicate(parsed_df.columns)
        return parsed_df, True
    else:
        return _fallback(df) 


# --- ГЕНЕРАТОР ИНТЕРФЕЙСА ДЛЯ ОТДЕЛЬНОГО ФАЙЛА ---

def render_analysis_ui(file_obj, title, prefix, alpha_grubbs, alpha_norm):
    st.header(title)
    df_raw, is_raw = parse_uploaded_file(file_obj, prefix)
    
    cols = df_raw.columns.tolist() if not df_raw.empty else []
    if not cols:
        st.error(f"⚠️ Ошибка структуры данных. Файл `{title}` абсолютно пуст или содержит только текст без цифр.")
        try:
            st.write("Содержимое файла:")
            file_obj.seek(0)
            if file_obj.name.endswith('.csv'):
                st.dataframe(pd.read_csv(file_obj, nrows=15, header=None))
            else:
                st.dataframe(pd.read_excel(file_obj, nrows=15, header=None))
        except: pass
        return
        
    if is_raw:
        st.success("✅ Сырой файл FLUOstar отсканирован успешно!")
    else:
        st.info("ℹ️ Данные распознаны и подготовлены. (Если таблица была 'широкой', мы автоматически перевернули её и сгруппировали повторности).")
        
    with st.expander("Предпросмотр загруженных данных", expanded=False):
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
            # Для транспонированных таблиц колонки будут "1", "2", "3"...
            default_times = [c for c in cols if c != sample_col and ('min' in str(c).lower() or 'h' in str(c).lower() or str(c).replace('.','',1).isdigit())]
            
        safe_defaults = [c for c in default_times if c in cols]
        time_cols = st.multiselect(f"Колонки со временем / циклами ({prefix})", [c for c in cols if c != sample_col], default=safe_defaults, key=f"{prefix}_time")

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
        
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Сводная таблица", "📈 Кинетика", "🔔 Гистограммы", "⚖️ Био-Статистика"])
        
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
            idx_max = res_df.groupby('Образец')['Среднее'].idxmax().dropna()
            peaks_df = res_df.loc[idx_max].sort_values(by='Среднее', ascending=False)
            st.dataframe(peaks_df[['Образец', 'Время', 'Среднее', 'Ст. ошибка (SE)', 'CV (%)']].rename(columns={'Время': 'Время пика (T max)', 'Среднее': 'Max Значение'}))
            
            st.markdown("---")
            st.subheader("🧬 Полный статистический анализ (Пайплайн из R)")
            st.write("Автоматический выбор теста: Проверка нормальности $\\to$ Гомогенность дисперсий (Levene) $\\to$ ANOVA / Kruskal-Wallis $\\to$ Попарные сравнения с поправкой FDR.")
            
            time_opts = ["Максимум (Пик)"] + list(st.session_state[f'time_cols_{prefix}'])
            time_choice = st.selectbox("Выберите время для сравнения всех групп", time_opts, key=f"biostats_t_{prefix}")
            
            analysis_data = {}
            for samp in res_df['Образец'].unique():
                if time_choice == "Максимум (Пик)":
                    t_val_series = peaks_df[peaks_df['Образец'] == samp]['Время']
                    if t_val_series.empty: continue
                    t_val = t_val_series.values[0]
                else:
                    t_val = time_choice
                
                vals = clean_long_df[(clean_long_df[sample_col_name] == samp) & (clean_long_df['Time'] == t_val)]['Value'].dropna().values
                if len(vals) >= 2:
                    analysis_data[samp] = vals
                    
            if len(analysis_data) < 2:
                st.warning("Недостаточно очищенных данных для статистики. Выберите другое время или убедитесь, что есть минимум 2 образца с повторностями (N ≥ 2).")
            else:
                st.markdown("#### 1. Условия применимости параметрических тестов")
                normality_res = []
                all_normal = True
                for samp, vals in analysis_data.items():
                    if len(vals) >= 3:
                        stat_sw, p_sw = stats.shapiro(vals)
                        is_norm = p_sw > 0.05
                    else:
                        p_sw = np.nan
                        is_norm = False
                    
                    all_normal = all_normal and is_norm
                    normality_res.append({"Образец": samp, "N": len(vals), "Shapiro p-value": p_sw, "Нормально (>0.05)": "Да" if is_norm else "Нет"})
                
                col1_norm, col2_lev = st.columns(2)
                with col1_norm:
                    st.dataframe(pd.DataFrame(normality_res), height=200)
                
                with col2_lev:
                    arrays = list(analysis_data.values())
                    lev_stat, lev_p = stats.levene(*arrays)
                    is_homoscedastic = lev_p > 0.05
                    st.info(f"**Тест Левена (Дисперсии):** p-value = **{lev_p:.4f}**\n\nВывод: {'Дисперсии гомогенны (равны)' if is_homoscedastic else 'Дисперсии ГЕТЕРОГЕННЫ (не равны)'}.")
                
                st.markdown("#### 2. Глобальное сравнение (Omnibus Test)")
                if all_normal and is_homoscedastic:
                    st.success("Условия выполнены (Нормальность + Гомоскедастичность). Применяется параметрический дисперсионный анализ **One-way ANOVA**.")
                    stat_omnibus, p_omnibus = stats.f_oneway(*arrays)
                    test_name = "ANOVA"
                else:
                    st.warning("Условия НЕ выполнены (Распределение не нормально или дисперсии не равны). Применяется непараметрический **Тест Краскела-Уоллиса (Kruskal-Wallis)**.")
                    stat_omnibus, p_omnibus = stats.kruskal(*arrays)
                    test_name = "Kruskal-Wallis"
                    
                st.write(f"Значение p-value для {test_name}: **{p_omnibus:.4e}**")
                
                if p_omnibus < 0.05:
                    st.info("Различия между группами **статистически значимы** (p < 0.05). Проводим Post-hoc (попарные) сравнения с поправкой Бенджамини-Хохберга (FDR).")
                    
                    group_names = list(analysis_data.keys())
                    pairs = list(itertools.combinations(group_names, 2))
                    posthoc_res = []
                    
                    for g1, g2 in pairs:
                        v1 = analysis_data[g1]
                        v2 = analysis_data[g2]
                        if test_name == "ANOVA":
                            _, p_pair = stats.ttest_ind(v1, v2, equal_var=True)
                        else:
                            _, p_pair = stats.mannwhitneyu(v1, v2, alternative='two-sided')
                            
                        posthoc_res.append({
                            "Группа 1": g1,
                            "Группа 2": g2,
                            "Raw p-value": p_pair
                        })
                        
                    ph_df = pd.DataFrame(posthoc_res)
                    ph_df['FDR p-value (BH)'] = fdr_bh(ph_df['Raw p-value'])
                    ph_df['Значимо (FDR < 0.05)'] = ph_df['FDR p-value (BH)'] < 0.05
                    st.dataframe(ph_df.sort_values('Raw p-value').reset_index(drop=True))
                else:
                    st.write("Глобальных различий не обнаружено (p >= 0.05). Попарные сравнения не требуются.")
                    
                st.markdown("---")
                st.markdown("#### 🎯 Сравнение с Контролем (аналог R-скрипта)")
                st.write("Прямое сравнение (Тест Уилкоксона/Mann-Whitney и Welch T-test) всех образцов относительно выбранной контрольной группы.")
                
                ctrl_group = st.selectbox("Выберите контрольную группу (Control)", list(analysis_data.keys()), key=f"ctrl_group_{prefix}")
                ctrl_vals = analysis_data[ctrl_group]
                
                ctrl_res = []
                for g_name, g_vals in analysis_data.items():
                    if g_name == ctrl_group: continue
                    _, p_mwu = stats.mannwhitneyu(ctrl_vals, g_vals, alternative='two-sided')
                    _, p_ttest = stats.ttest_ind(ctrl_vals, g_vals, equal_var=False)
                    
                    ctrl_res.append({
                        "Образец": g_name,
                        "Mann-Whitney (Wilcoxon) p-value": p_mwu,
                        "Welch T-test p-value": p_ttest,
                        "Значимо (Wilcoxon < 0.05)": "Да" if p_mwu < 0.05 else "Нет"
                    })
                
                if ctrl_res:
                    st.dataframe(pd.DataFrame(ctrl_res).sort_values("Mann-Whitney (Wilcoxon) p-value").reset_index(drop=True))


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