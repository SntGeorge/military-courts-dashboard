import re
import io
import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Судебная аналитика", page_icon="⚖️", layout="wide")

st.markdown("""
<style>
    div[data-testid="stMetric"] {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        padding: 15px 20px;
        border-radius: 10px;
    }
    @media (prefers-color-scheme: dark) {
        div[data-testid="stMetric"] {
            background-color: #1e2127;
            border: 1px solid #2e333d;
        }
    }
</style>
""", unsafe_allow_html=True)

ARTICLE_LABELS = {
    "332": "332 — Неисполнение приказа",
    "333": "333 — Сопротивление начальнику",
    "334": "334 — Насильственные действия в отношении начальника",
    "335": "335 — Нарушение уставных правил",
    "336": "336 — Оскорбление военнослужащего",
    "337": "337 — Самовольное оставление части",
    "338": "338 — Дезертирство",
    "339": "339 — Уклонение от службы",
    "346": "346 — Умышленное уничтожение военного имущества",
    "105": "105 — Убийство",
    "111": "111 — Умышленное причинение тяжкого вреда здоровью",
    "158": "158 — Кража",
    "159": "159 — Мошенничество",
    "161": "161 — Грабеж",
    "162": "162 — Разбой",
    "163": "163 — Вымогательство",
    "222": "222 — Незаконный оборот оружия",
    "228": "228 — Незаконный оборот наркотиков",
    "228.1": "228.1 — Сбыт наркотиков",
    "275": "275 — Государственная измена",
    "286": "286 — Превышение должностных полномочий",
    "290": "290 — Получение взятки",
    "291": "291 — Дача взятки",
    "327": "327 — Подделка документов",
}


def format_article(code: str) -> str:
    code = str(code).strip()
    return ARTICLE_LABELS.get(code, f"ст. {code} УК РФ")


def fmt_int(n) -> str:
    """Красивое форматирование числа с пробелами-разделителями."""
    try:
        return f"{int(n):,}".replace(",", " ")
    except (TypeError, ValueError):
        return "—"


def split_articles(value) -> list[str]:
    if pd.isna(value):
        return []
    return [a.strip() for a in str(value).split(",") if a.strip()]


def filter_valid_days(series: pd.Series, upper: int = 730) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    return s[(s >= 0) & (s <= upper)]


def classify_decision(text) -> str:
    """Грубая классификация текста решения по ключевым словам."""
    if pd.isna(text):
        return "Не указано"
    t = str(text).upper()
    if "ПРЕКРАЩ" in t:
        return "Прекращено"
    if "ОПРАВД" in t:
        return "Оправдан"
    if "ПРИГОВОР" in t and "УСЛОВН" in t:
        return "Приговор (условно)"
    if "ПРИГОВОР" in t:
        return "Приговор (реально)"
    return "Другое"


# --- Выделение чистого обвиняемого из поля «Стороны» ---
SKIP_PREFIX_MARKERS = re.compile(
    r"^(?:Военн|прокурор|помощник|ВП|адвокат|защитник|потерпевш|представител|ФКУ|УФСИН|в/ч|войсковая|заявитель)",
    flags=re.IGNORECASE,
)

FIO_PATTERN = re.compile(
    r"\b([А-ЯЁ][а-яё]+(?:-[А-ЯЁ][а-яё]+)?\s+"
    r"(?:[А-ЯЁ][а-яё]+(?:-[А-ЯЁ][а-яё]+)?\s+[А-ЯЁ][а-яё]+(?:-[А-ЯЁ][а-яё]+)?|"
    r"[А-ЯЁ]\.\s*[А-ЯЁ]\.))"
)


def get_accused(parties_raw) -> str:
    """Возвращает ФИО обвиняемого из строки «Стороны» или пустую строку."""
    if pd.isna(parties_raw):
        return ""
    text = str(parties_raw).strip()
    if SKIP_PREFIX_MARKERS.search(text):
        return ""
    m = FIO_PATTERN.search(text)
    return m.group(1).strip() if m else ""


@st.cache_data
def load_data():
    use_cols = {
        "УИД", "Номер дела", "Суд", "Регион", "Тип", "Инстанция",
        "Судья", "Решение", "Статьи", "Стороны", "source_file",
        "base_case_id", "clean_articles", "Дата поступления",
        "Дата решения", "days_to_decision"
    }
    df = pd.read_csv(
        "cases_clean.csv.gz",
        usecols=lambda c: c in use_cols,
        dtype=str
    )
    
    df["Дата поступления"] = pd.to_datetime(df["Дата поступления"], errors="coerce")
    df["Дата решения"] = pd.to_datetime(df["Дата решения"], errors="coerce")
    df["days_to_decision"] = pd.to_numeric(df.get("days_to_decision"), errors="coerce")

    # Корректная конвертация булева флага (на случай строк "True"/"False")
    if "is_foreign_suspect" in df.columns:
        df["is_foreign_suspect"] = (
            df["is_foreign_suspect"]
            .astype(str).str.strip().str.lower()
            .map({"true": True, "1": True, "false": False, "0": False})
            .fillna(False).astype(bool)
        )
    else:
        df["is_foreign_suspect"] = False

    df["decision_class"] = df["Решение"].map(classify_decision)

    df["Суд"] = df["Суд"].fillna("Суд не указан")
    df["Регион"] = df["Регион"].fillna("Регион не указан")
    df["Инстанция_рус"] = df["Инстанция"].map({
        "FIRST": "1-я инстанция",
        "APPELLATION": "Апелляция",
        "CASSATION": "Кассация",
    }).fillna(df["Инстанция"])

    df["Обвиняемый"] = df["Стороны"].map(get_accused)

    return df


df = load_data()

st.title("⚖️ Аналитика судебных дел военнослужащих")

# --- БОКОВАЯ ПАНЕЛЬ ФИЛЬТРОВ ---
st.sidebar.header("🎛️ Фильтры выборки")

valid_dates = df["Дата поступления"].dropna()
if not valid_dates.empty:
    min_d, max_d = valid_dates.min().date(), valid_dates.max().date()
    date_range = st.sidebar.date_input(
        "Период поступления дела:",
        value=(min_d, max_d),
        min_value=min_d,
        max_value=max_d,
    )
else:
    date_range = None

all_articles = sorted({
    art for arts in df["clean_articles"].dropna().map(split_articles)
    for art in arts
})
selected_articles = st.sidebar.multiselect(
    "Статьи УК РФ (дело содержит):",
    options=all_articles,
    format_func=format_article,
    default=[],
)

instances = sorted(df["Инстанция_рус"].dropna().unique())
selected_instances = st.sidebar.multiselect("Инстанция:", options=instances, default=[])

regions = sorted(df["Регион"].dropna().unique())
selected_regions = st.sidebar.multiselect("Регион:", options=regions, default=[])

decisions = sorted(df["Решение"].dropna().unique())
selected_decisions = st.sidebar.multiselect("Решение суда:", options=decisions, default=[])

decision_classes = sorted(df["decision_class"].dropna().unique())
selected_classes = st.sidebar.multiselect(
    "Тип решения (классификатор):", options=decision_classes, default=[]
)

search_query = st.sidebar.text_input("🔎 Поиск (ФИО, суд, судья, № дела):")

# Ползунок срока рассмотрения
if df["days_to_decision"].notna().any():
    max_days_val = int(min(df["days_to_decision"].max(), 730))
    days_filter = st.sidebar.slider(
        "Срок рассмотрения (дней):",
        min_value=0,
        max_value=max_days_val,
        value=(0, max_days_val),
        step=5,
    )
else:
    days_filter = None

if st.sidebar.button("♻️ Сбросить фильтры"):
    st.rerun()

# --- ПРИМЕНЕНИЕ ФИЛЬТРОВ ---
filtered_df = df.copy()

if date_range:
    if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
        start_d, end_d = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
        filtered_df = filtered_df[
            (filtered_df["Дата поступления"] >= start_d) &
            (filtered_df["Дата поступления"] <= end_d)
        ]
    else:
        st.sidebar.warning(
            "Выбрана только одна дата. Выберите обе границы периода "
            "или нажмите ♻️ Сбросить фильтры."
        )

if days_filter is not None:
    filtered_df = filtered_df[
        filtered_df["days_to_decision"].between(*days_filter, inclusive="both")
        | filtered_df["days_to_decision"].isna()
    ]

if selected_articles:
    # Точное совпадение с элементом списка статей — без "228 ловит 228.1"
    art_set = set(selected_articles)
    filtered_df = filtered_df[
        filtered_df["clean_articles"].map(
            lambda v: bool(art_set & set(split_articles(v)))
        )
    ]

if selected_instances:
    filtered_df = filtered_df[filtered_df["Инстанция_рус"].isin(selected_instances)]

if selected_regions:
    filtered_df = filtered_df[filtered_df["Регион"].isin(selected_regions)]

if selected_decisions:
    filtered_df = filtered_df[filtered_df["Решение"].isin(selected_decisions)]

if selected_classes:
    filtered_df = filtered_df[filtered_df["decision_class"].isin(selected_classes)]

if search_query:
    q = search_query.strip()
    mask = (
        filtered_df["Стороны"].str.contains(q, case=False, na=False, regex=False) |
        filtered_df["Суд"].str.contains(q, case=False, na=False, regex=False) |
        filtered_df["Судья"].str.contains(q, case=False, na=False, regex=False) |
        filtered_df["Номер дела"].str.contains(q, case=False, na=False, regex=False) |
        filtered_df["УИД"].str.contains(q, case=False, na=False, regex=False)
    )
    filtered_df = filtered_df[mask]

# --- ВЕРХНИЕ МЕТРИКИ ---
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Судебных актов", fmt_int(len(filtered_df)))
c2.metric("Уникальных дел", fmt_int(filtered_df["base_case_id"].nunique()))
c3.metric("Судов в выборке", fmt_int(filtered_df["Суд"].nunique()))

valid_days = filter_valid_days(filtered_df["days_to_decision"])
median_val = int(valid_days.median()) if len(valid_days) else 0
c4.metric("Медианный срок", f"{median_val} дн.")
c5.metric("Распознано обвиняемых", fmt_int((filtered_df["Обвиняемый"] != "").sum()))

st.divider()

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Общая картина",
    "👤 Подсудимые (Статистика)",
    "🏛️ Аналитика по судам",
    "🗂️ Карточка дела (Досье)",
    "🕵️ OSINT / Другое",
    "📋 Полная таблица и выгрузка",
])


# ==================== ВКЛАДКА 1 ====================
with tab1:
    st.subheader("Динамика поступления дел по месяцам")
    ts_df = filtered_df.dropna(subset=["Дата поступления"]).copy()
    if not ts_df.empty:
        ts_df["Месяц"] = ts_df["Дата поступления"].dt.to_period("M").astype(str)
        monthly = ts_df.groupby(["Месяц", "Инстанция_рус"]).size().reset_index(name="Количество")
        fig_time = px.bar(
            monthly, x="Месяц", y="Количество", color="Инстанция_рус",
            barmode="stack", template="plotly_white",
            color_discrete_sequence=px.colors.qualitative.Set2,
            labels={"Инстанция_рус": "Инстанция"},
        )
        st.plotly_chart(fig_time, use_container_width=True)
    else:
        st.info("Нет данных с корректной датой поступления.")

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Топ-15 регионов по числу актов")
        top_reg = filtered_df["Регион"].value_counts().head(15).reset_index()
        top_reg.columns = ["Регион", "Количество"]
        fig_reg = px.bar(
            top_reg, x="Количество", y="Регион", orientation="h",
            template="plotly_white", color_discrete_sequence=px.colors.sequential.Teal_r,
        )
        fig_reg.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig_reg, use_container_width=True)

    with col_b:
        st.subheader("Распределение судебных решений")
        dec_counts = (
            filtered_df["Решение"].fillna("В процессе / Не указано")
            .value_counts().head(10).reset_index()
        )
        dec_counts.columns = ["Решение", "Количество"]
        fig_dec = px.pie(
            dec_counts, values="Количество", names="Решение", hole=0.45,
            template="plotly_white", color_discrete_sequence=px.colors.qualitative.Pastel,
        )
        st.plotly_chart(fig_dec, use_container_width=True)

    col_c, col_d = st.columns(2)
    with col_c:
        st.subheader("Топ-15 комбинаций статей в делах")
        comb = filtered_df["clean_articles"].replace("", "Не указано").value_counts().head(15).reset_index()
        comb.columns = ["Статьи", "Количество"]
        fig_comb = px.bar(
            comb, x="Количество", y="Статьи", orientation="h",
            template="plotly_white", color_discrete_sequence=px.colors.sequential.Blues_r,
        )
        fig_comb.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig_comb, use_container_width=True)

    with col_d:
        st.subheader("Длительность рассмотрения дел (дней)")
        days_plot_df = filtered_df[
            filtered_df["days_to_decision"].between(0, 365, inclusive="both")
        ]
        if not days_plot_df.empty:
            fig_days = px.histogram(
                days_plot_df, x="days_to_decision", nbins=45, color="Инстанция_рус",
                template="plotly_white", color_discrete_sequence=px.colors.qualitative.Set2,
                labels={"days_to_decision": "Дней от поступления до решения", "Инстанция_рус": "Инстанция"},
            )
            st.plotly_chart(fig_days, use_container_width=True)
        else:
            st.info("Недостаточно данных для гистограммы сроков.")

    st.divider()
    st.subheader("Динамика доли статей по годам (топ-8)")

    art_year_df = (
        filtered_df.dropna(subset=["Дата поступления"])
        .assign(
            Год=lambda d: d["Дата поступления"].dt.year,
            st=lambda d: d["clean_articles"].str.split(","),
        )
        .explode("st")
        .assign(st=lambda d: d["st"].astype(str).str.strip())
    )
    art_year_df = art_year_df[art_year_df["st"] != ""]

    if art_year_df.empty:
        st.info("Нет данных для построения трендов по статьям.")
    else:
        top_arts = (
            art_year_df.groupby("st").size().nlargest(8).index.tolist()
        )
        trend = (
            art_year_df[art_year_df["st"].isin(top_arts)]
            .groupby(["Год", "st"]).size().reset_index(name="Количество")
        )
        trend["Статья"] = trend["st"].map(format_article)
        fig_trend = px.line(
            trend, x="Год", y="Количество", color="Статья", markers=True,
            template="plotly_white",
            labels={"Год": "Год поступления", "Количество": "Число актов"},
        )
        st.plotly_chart(fig_trend, use_container_width=True)

    st.divider()
    st.subheader("Иерархия: Инстанция → Решение")

    sunburst_df = filtered_df.dropna(subset=["Инстанция_рус", "Решение"]).copy()
    if sunburst_df.empty:
        st.info("Недостаточно данных для иерархии инстанций.")
    else:
        sunburst_df["Решение_short"] = (
            sunburst_df["Решение"].str.slice(0, 40) + "…"
        )
        fig_sun = px.sunburst(
            sunburst_df,
            path=["Инстанция_рус", "Решение_short"],
            template="plotly_white",
        )
        fig_sun.update_traces(textinfo="label+percent parent")
        st.plotly_chart(fig_sun, use_container_width=True)



# ==================== ВКЛАДКА: ПОДСУДИМЫЕ ====================
with tab2:
    st.subheader("👤 Аналитика по подсудимым: динамика, повторные дела и закономерности")

    acc_all = filtered_df[filtered_df["Обвиняемый"] != ""].copy()
    hidden_df = filtered_df[filtered_df["Обвиняемый"] == ""].copy()

    total_acts = len(filtered_df)
    unique_people = acc_all["Обвиняемый"].nunique()
    hidden_acts = len(hidden_df)
    hidden_pct = (hidden_acts / total_acts * 100) if total_acts else 0

    person_cases = acc_all.groupby("Обвиняемый")["base_case_id"].nunique()
    repeat_people_names = person_cases[person_cases > 1].index
    repeat_count = len(repeat_people_names)
    repeat_pct = (repeat_count / unique_people * 100) if unique_people else 0

    # --- ВЕРХНИЕ МЕТРИКИ ---
    pm1, pm2, pm3, pm4 = st.columns(4)
    pm1.metric("Уникальных подсудимых", fmt_int(unique_people))
    pm2.metric("Повторных (2+ дела)", f"{fmt_int(repeat_count)} ({repeat_pct:.1f}%)")
    pm3.metric("Индекс анонимности (скрыто ФИО)", f"{hidden_pct:.1f}%")
    avg_c = (acc_all["base_case_id"].nunique() / unique_people) if unique_people else 0
    pm4.metric("Среднее дел на человека", f"{avg_c:.2f}")

    # --- РАСЧЕТ ЭСКАЛАЦИИ И ЗАКОНОМЕРНОСТЕЙ ---
    transitions = []
    gaps_days = []

    if repeat_count > 0:
        rep_df = (
            acc_all[acc_all["Обвиняемый"].isin(repeat_people_names)]
            .drop_duplicates(subset=["Обвиняемый", "base_case_id"])
            .sort_values(["Обвиняемый", "Дата поступления"])
        )
        for person, grp in rep_df.groupby("Обвиняемый"):
            if len(grp) < 2:
                continue
            arts_seq = grp["clean_articles"].replace("", "Не указана").tolist()
            dates_seq = grp["Дата поступления"].tolist()
            for i in range(len(grp) - 1):
                transitions.append(f"{arts_seq[i]} → {arts_seq[i+1]}")
                if pd.notna(dates_seq[i]) and pd.notna(dates_seq[i+1]):
                    diff = (dates_seq[i+1] - dates_seq[i]).days
                    if 0 <= diff <= 1825:
                        gaps_days.append(diff)

    if transitions:
        vc_tr = pd.Series(transitions).value_counts()
        top_transition = vc_tr.index[0]
        top_transition_cnt = vc_tr.iloc[0]
    else:
        top_transition = "—"
        top_transition_cnt = 0

    median_gap = int(pd.Series(gaps_days).median()) if gaps_days else None

    # --- АВТОМАТИЧЕСКИЕ ЗАКОНОМЕРНОСТИ ---
    with st.container(border=True):
        st.markdown("#### 💡 Выявленные закономерности в текущей выборке")
        insights = []
        if unique_people > 0:
            insights.append(
                f"• **Уровень повторных судов:** **{repeat_pct:.1f}%** от всех распознанных подсудимых "
                f"(**{fmt_int(repeat_count)}** чел.) проходят сразу по нескольким разным уголовным делам."
            )
        if transitions:
            insights.append(
                f"• **Главный сценарий повторных дел:** чаще всего встречается переход квалификаций "
                f"**«{top_transition}»** (**{fmt_int(top_transition_cnt)}** случаев)."
            )
        if median_gap is not None:
            insights.append(
                f"• **Скорость возврата в суд:** медианный интервал между поступлением первого и повторного дела "
                f"составляет **{median_gap} дн.** (~{median_gap / 30.4:.1f} мес.)."
            )
        if total_acts > 0:
            insights.append(
                f"• **Прозрачность судов:** в **{hidden_pct:.1f}%** актов (**{fmt_int(hidden_acts)}** из {fmt_int(total_acts)}) "
                f"данные о подсудимом скрыты или не распознаны в карточке."
            )
        st.markdown("\n\n".join(insights))

    st.markdown("---")

    # --- ГРАФИК РОСТА И ИНДЕКСА АНОНИМНОСТИ ---
    st.markdown("#### 📈 Динамика числа подсудимых и сокрытия данных по месяцам")
    ts_people = filtered_df.dropna(subset=["Дата поступления"]).copy()
    if not ts_people.empty:
        ts_people["Месяц"] = ts_people["Дата поступления"].dt.to_period("M").astype(str)
        ts_people["Статус_ФИО"] = ts_people["Обвиняемый"].apply(
            lambda x: "ФИО открыто" if x != "" else "ФИО скрыто судом"
        )
        open_m = (
            ts_people[ts_people["Статус_ФИО"] == "ФИО открыто"]
            .groupby("Месяц")["Обвиняемый"].nunique()
            .reset_index(name="Количество")
        )
        open_m["Категория"] = "Уникальные подсудимые"

        hid_m = (
            ts_people[ts_people["Статус_ФИО"] == "ФИО скрыто судом"]
            .groupby("Месяц")["base_case_id"].nunique()
            .reset_index(name="Количество")
        )
        hid_m["Категория"] = "Дела со скрытым ФИО"

        combined_m = pd.concat([open_m, hid_m], ignore_index=True)
        fig_growth = px.bar(
            combined_m, x="Месяц", y="Количество", color="Категория",
            barmode="stack", template="plotly_white",
            color_discrete_map={
                "Уникальные подсудимые": "#2b8a3e",
                "Дела со скрытым ФИО": "#adb5bd",
            },
        )
        st.plotly_chart(fig_growth, use_container_width=True)

    # --- РАЗРЕЗ ПО СТАТЬЯМ И РЕГИОНАМ В ЛЮДЯХ ---
    pc1, pc2 = st.columns(2)
    with pc1:
        st.markdown("#### Топ-10 статей по числу уникальных подсудимых")
        art_people = (
            acc_all.assign(st=acc_all["clean_articles"].str.split(","))
            .explode("st")
            .assign(st=lambda d: d["st"].astype(str).str.strip())
        )
        art_people = art_people[art_people["st"] != ""]
        if not art_people.empty:
            top_art_p = (
                art_people.groupby("st")["Обвиняемый"]
                .nunique()
                .nlargest(10)
                .reset_index(name="Подсудимых")
            )
            top_art_p["Статья"] = top_art_p["st"].map(format_article)
            fig_ap = px.bar(
                top_art_p, x="Подсудимых", y="Статья", orientation="h",
                template="plotly_white", color_discrete_sequence=px.colors.sequential.Teal_r,
            )
            fig_ap.update_layout(yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig_ap, use_container_width=True)

    with pc2:
        st.markdown("#### Топ-10 регионов по числу уникальных подсудимых")
        top_reg_p = (
            acc_all.groupby("Регион")["Обвиняемый"]
            .nunique()
            .nlargest(10)
            .reset_index(name="Подсудимых")
        )
        fig_rp = px.bar(
            top_reg_p, x="Подсудимых", y="Регион", orientation="h",
            template="plotly_white", color_discrete_sequence=px.colors.sequential.Blues_r,
        )
        fig_rp.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig_rp, use_container_width=True)

    # --- АНАЛИЗ ЭСКАЛАЦИИ СТАТЕЙ ---
    if transitions:
        st.markdown("---")
        st.markdown("#### 🔄 Эскалация и повторность: как меняются статьи от первого дела к следующему")
        ec1, ec2 = st.columns([1.2, 1])
        with ec1:
            trans_df = pd.Series(transitions).value_counts().head(12).reset_index()
            trans_df.columns = ["Переход_статей", "Случаев"]
            fig_tr = px.bar(
                trans_df, x="Случаев", y="Переход_статей", orientation="h",
                template="plotly_white", color_discrete_sequence=["#d9480f"],
            )
            fig_tr.update_layout(yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig_tr, use_container_width=True)
        with ec2:
            if gaps_days:
                fig_gap = px.histogram(
                    pd.DataFrame({"Дней_между_делами": gaps_days}),
                    x="Дней_между_делами", nbins=30,
                    title="Через сколько дней заводится повторное дело",
                    template="plotly_white", color_discrete_sequence=["#e67700"],
                )
                st.plotly_chart(fig_gap, use_container_width=True)

    # --- СВОДНЫЙ РЕЕСТР УНИКАЛЬНЫХ ПОДСУДИМЫХ ---
    st.markdown("---")
    st.markdown("#### 📋 Сводный реестр уникальных подсудимых (1 строка = 1 человек)")

    rf1, rf2 = st.columns([2, 1])
    with rf1:
        p_search = st.text_input("🔎 Поиск по фамилии подсудимого:", key="accused_registry_search")
    with rf2:
        min_deals_filter = st.number_input("Минимум дел на человека:", min_value=1, max_value=20, value=1)

    if not acc_all.empty:
        registry = (
            acc_all.sort_values("Дата поступления")
            .groupby("Обвиняемый")
            .agg(
                Уникальных_дел=("base_case_id", "nunique"),
                Всего_актов=("УИД", "count"),
                Первое_дело=("Дата поступления", "min"),
                Последнее_дело=("Дата поступления", "max"),
                Статьи=("clean_articles", lambda x: "; ".join(sorted(set(a for a in x if a)))),
                Регионы=("Регион", lambda x: ", ".join(sorted(set(r for r in x if r)))),
                Суды=("Суд", lambda x: ", ".join(sorted(set(s for s in x if s)))),
            )
            .reset_index()
            .sort_values(["Уникальных_дел", "Последнее_дело"], ascending=[False, False])
        )
        registry["Первое_дело"] = registry["Первое_дело"].dt.strftime("%Y-%m-%d")
        registry["Последнее_дело"] = registry["Последнее_дело"].dt.strftime("%Y-%m-%d")

        if min_deals_filter > 1:
            registry = registry[registry["Уникальных_дел"] >= min_deals_filter]
        if p_search:
            registry = registry[registry["Обвиняемый"].str.contains(p_search.strip(), case=False, na=False)]

        st.write(f"В реестре показано: **{fmt_int(len(registry))}** чел.")
        st.dataframe(registry.head(1000), use_container_width=True, hide_index=True)

        try:
            p_buf = io.BytesIO()
            with pd.ExcelWriter(p_buf, engine="openpyxl") as writer:
                registry.to_excel(writer, index=False, sheet_name="defendants")
            st.download_button(
                label="📥 Скачать сводный реестр подсудимых (.xlsx)",
                data=p_buf.getvalue(),
                file_name="defendants_registry.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="download_defendants_xlsx",
            )
        except ImportError:
            st.caption("Для экспорта в Excel установите пакет `openpyxl`.")

# ==================== ВКЛАДКА 2 ====================
with tab2:
    st.subheader("Детальный разбор конкретного суда")

    court_counts = filtered_df["Суд"].value_counts()
    if court_counts.empty:
        st.warning("Нет данных под выбранные фильтры.")
    else:
        court_options = [f"{name} ({cnt} дел)" for name, cnt in court_counts.items()]
        court_map = {f"{name} ({cnt} дел)": name for name, cnt in court_counts.items()}

        selected_court_label = st.selectbox("Выберите суд для анализа:", options=court_options)
        selected_court = court_map[selected_court_label]
        court_df = filtered_df[filtered_df["Суд"] == selected_court].copy()

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Всего актов в суде", len(court_df))
        m2.metric("Судей в штате (по делам)", court_df["Судья"].dropna().nunique())
        c_days = filter_valid_days(court_df["days_to_decision"])
        m3.metric("Медианный срок", f"{int(c_days.median())} дн." if len(c_days) else "—")
        verdicts = court_df["Решение"].str.contains("ПРИГОВОР", case=False, na=False).sum()
        pct = (verdicts / len(court_df) * 100) if len(court_df) else 0
        m4.metric("Вынесено приговоров", f"{verdicts} ({pct:.0f}%)")

        st.markdown("---")

        # ⭐ Сравнение с другим судом
        compare = st.checkbox("Сравнить с другим судом")
        if compare:
            other_options = [c for c in court_options if c != selected_court_label]
            if other_options:
                second_label = st.selectbox(
                    "Суд для сравнения:", options=other_options, key="cmp_court"
                )
                court_df2 = filtered_df[filtered_df["Суд"] == court_map[second_label]]

                def _metrics(d: pd.DataFrame):
                    vd = filter_valid_days(d["days_to_decision"])
                    v = d["Решение"].str.contains("ПРИГОВОР", case=False, na=False).sum()
                    p = (v / len(d) * 100) if len(d) else 0
                    return {
                        "актов": len(d),
                        "судей": d["Судья"].dropna().nunique(),
                        "медиана": f"{int(vd.median())} дн." if len(vd) else "—",
                        "приговоров": f"{v} ({p:.0f}%)",
                    }

                m_a, m_b = _metrics(court_df), _metrics(court_df2)
                cmp_table = pd.DataFrame({
                    "Метрика": ["Актов", "Судей", "Медиана срока", "Приговоров"],
                    selected_court: list(m_a.values()),
                    court_map[second_label]: list(m_b.values()),
                })
                st.dataframe(cmp_table, use_container_width=True, hide_index=True)
                st.markdown("---")

        sc1, sc2 = st.columns(2)

        with sc1:
            st.markdown("#### Нагрузка и скорость по судьям")
            judges_stats = (
                court_df.dropna(subset=["Судья"])
                .groupby("Судья")
                .agg(
                    Дел=("УИД", "count"),
                    Приговоров=("Решение", lambda x: x.str.contains("ПРИГОВОР", case=False, na=False).sum()),
                    Медиана_дней=("days_to_decision", lambda x: filter_valid_days(x).median()),
                )
                .reset_index()
                .sort_values("Дел", ascending=False)
            )
            judges_stats["Медиана_дней"] = judges_stats["Медиана_дней"].round(0)
            st.dataframe(judges_stats, use_container_width=True, hide_index=True)

        with sc2:
            st.markdown("#### Статьи и исходы в этом суде")
            c_art = court_df["clean_articles"].replace("", "Не указано").value_counts().head(10).reset_index()
            c_art.columns = ["Статьи", "Количество"]
            fig_c_art = px.bar(
                c_art, x="Количество", y="Статьи", orientation="h",
                template="plotly_white", color_discrete_sequence=["#2b8a3e"],
            )
            fig_c_art.update_layout(yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig_c_art, use_container_width=True)

        st.markdown("#### Динамика поступления дел в этот суд")
        c_ts = court_df.dropna(subset=["Дата поступления"]).copy()
        if not c_ts.empty:
            c_ts["Месяц"] = c_ts["Дата поступления"].dt.to_period("M").astype(str)
            c_monthly = c_ts.groupby("Месяц").size().reset_index(name="Дел")
            st.plotly_chart(
                px.line(c_monthly, x="Месяц", y="Дел", markers=True, template="plotly_white"),
                use_container_width=True,
            )

# ==================== ВКЛАДКА 3 ====================
with tab3:
    st.subheader("Детальный просмотр дела и движения по инстанциям")
    st.caption(
        "Выберите дело из списка ниже (или сначала сузьте поиск через фильтры слева), "
        "чтобы увидеть всю информацию и связанные апелляции/кассации."
    )
    if filtered_df.empty:
        st.warning("По текущим фильтрам дел не найдено.")
    else:
        preview_df = filtered_df.head(1000).copy()
        st.caption(
            f"Показаны первые **{len(preview_df)}** из **{len(filtered_df)}** актов. "
            "Если нужного дела нет в списке — сузьте фильтры слева."
        )

        preview_df = preview_df.reset_index().rename(columns={"index": "_row_id"})
        preview_df["_label"] = (
            preview_df["Номер дела"].fillna("Б/Н") + " | " +
            preview_df["Суд"] + " | " +
            preview_df["Стороны"].fillna("Стороны скрыты").str.slice(0, 60) +
            " #" + preview_df["_row_id"].astype(str)
        )

        selected_idx = st.selectbox(
            "Выберите судебный акт для изучения:",
            options=preview_df["_row_id"].tolist(),
            format_func=lambda i: preview_df.loc[
                preview_df["_row_id"] == i, "_label"
            ].iloc[0],
        )
        selected_row = preview_df[preview_df["_row_id"] == selected_idx].iloc[0]
        base_id = selected_row["base_case_id"]

        if pd.isna(base_id):
            related_cases = selected_row.to_frame().T
        else:
            related_cases = df[df["base_case_id"] == base_id].sort_values("Дата поступления")
            if related_cases.empty:
                related_cases = selected_row.to_frame().T

        st.markdown(f"### 📁 Базовый идентификатор дела: {base_id}")
        st.info(
            f"Найдено связанных судебных производств (инстанций) "
            f"по этому делу: {len(related_cases)}"
        )

        for _, row in related_cases.iterrows():
            with st.container(border=True):
                h1, h2, h3 = st.columns([2, 2, 1])
                h1.markdown(f"Инстанция: {row['Инстанция_рус']} ({row['Номер дела']})")
                h2.markdown(f"Суд: {row['Суд']}")
                h3.markdown(f"Регион: {row['Регион']}")

                d1, d2, d3, d4 = st.columns(4)
                d1.write(
                    "Поступило: "
                    + (row["Дата поступления"].strftime("%Y-%m-%d")
                       if pd.notna(row["Дата поступления"]) else "—")
                )
                d2.write(
                    "Решение: "
                    + (row["Дата решения"].strftime("%Y-%m-%d")
                       if pd.notna(row["Дата решения"]) else "В процессе")
                )
                days = row.get("days_to_decision")
                d3.write(
                    f"Срок: {int(days)} дн."
                    if pd.notna(days) and days >= 0 else "Срок: —"
                )
                d4.write(
                    f"Судья: {row['Судья'] if pd.notna(row['Судья']) else 'Не указан'}"
                )

                st.write(
                    f"Итог / Статус: "
                    f"{row['Решение'] if pd.notna(row['Решение']) else 'Решение еще не вынесено'}"
                )

                raw_codes = split_articles(row["clean_articles"])
                if raw_codes:
                    readable_arts = [format_article(c) for c in raw_codes]
                    st.write("Квалификация (статьи): " + " • ".join(readable_arts))
                if pd.notna(row["Статьи"]):
                    st.caption(f"Исходная строка статей: {row['Статьи']}")

                st.markdown(f"**Обвиняемый:** `{row['Обвиняемый'] or 'Скрыт судом'}`")

                st.write("Участники процесса (Стороны):")
                parties_str = (
                    str(row["Стороны"]) if pd.notna(row["Стороны"])
                    else "Данные скрыты судом"
                )
                st.code(parties_str, language=None)
                st.caption(
                    f"Полный УИД: {row['УИД']} | Источник: {row['source_file']}"
                )

# ==================== ВКЛАДКА 4: OSINT ====================
with tab4:
    st.subheader("Другое")

    osint_mode = st.radio(
        "Выберите инструмент анализа:",
        [
            "🪖 Реестр воинских частей (в/ч)",
            "🔁 Повторные фигуранты (несколько разных дел)",
            "👥 Соучастники (групповые дела)",
            "⚖️ Прокуроры и защитники",
            "🔎 Гибкий поиск по сторонам и группам (Regex)",
        ],
        horizontal=True,
    )

    st.markdown("---")

    # --- РЕЖИМ 1: ВОИНСКИЕ ЧАСТИ ---
    if osint_mode == "🪖 Реестр воинских частей (в/ч)":
        st.caption(
            "Автоматическое извлечение номеров войсковых частей из поля «Стороны» "
            "(где часть выступает потерпевшей стороной или гражданским истцом)."
        )

        vc_series = (
            filtered_df["Стороны"]
            .dropna()
            .str.findall(r"(?:в/ч|войсковая часть)\s*№?\s*(\d{4,6})", flags=re.IGNORECASE)
        )
        vc_exploded = vc_series.explode().dropna().str.strip()

        if vc_exploded.empty:
            st.info("В текущей выборке не найдено явных упоминаний номеров в/ч в столбце «Стороны».")
        else:
            vc_counts = vc_exploded.value_counts().reset_index()
            vc_counts.columns = ["Номер_вч", "Упоминаний_в_делах"]

            c_left, c_right = st.columns([1.2, 1])
            with c_left:
                fig_vc = px.bar(
                    vc_counts.head(20), x="Упоминаний_в_делах", y="Номер_вч",
                    orientation="h", title="Топ-20 воинских частей по числу дел в выборке",
                    template="plotly_white", color_discrete_sequence=px.colors.sequential.Teal_r,
                )
                fig_vc.update_layout(
                    yaxis={"categoryorder": "total ascending", "type": "category"}
                )
                st.plotly_chart(fig_vc, use_container_width=True)

            with c_right:
                st.markdown("#### Детализация по конкретной в/ч")
                selected_vc = st.selectbox(
                    "Выберите номер в/ч для просмотра всех её дел:",
                    options=vc_counts["Номер_вч"].tolist(),
                )
                vc_mask = filtered_df["Стороны"].str.contains(
                    rf"(?:в/ч|войсковая часть)\s*№?\s*{re.escape(selected_vc)}\b",
                    case=False, na=False, regex=True,
                )
                vc_cases = filtered_df[vc_mask]
                st.metric(f"Дел по в/ч {selected_vc}", len(vc_cases))
                st.write(
                    "**Основные статьи:**",
                    ", ".join(vc_cases["clean_articles"].value_counts().head(5).index.tolist()),
                )
                st.write(
                    "**Суды:**",
                    ", ".join(vc_cases["Суд"].value_counts().head(3).index.tolist()),
                )

            st.markdown(f"#### Список дел по в/ч {selected_vc}")
            st.dataframe(
                vc_cases[["Дата поступления", "Регион", "Суд", "Инстанция_рус",
                          "clean_articles", "Стороны", "Решение", "УИД"]],
                use_container_width=True, hide_index=True,
            )

    # --- РЕЖИМ 2: ПОВТОРНЫЕ ФИГУРАНТЫ (по чистому «Обвиняемый») ---
    elif osint_mode == "🔁 Повторные фигуранты (несколько разных дел)":
        st.caption(
            "Поиск обвиняемых военнослужащих, которые проходят сразу "
            "по нескольким разным делам (разные базовые УИД)."
        )
        acc_df = filtered_df[filtered_df["Обвиняемый"] != ""].copy()
        if acc_df.empty:
            st.info("В текущей выборке нет распознанных обвиняемых.")
        else:
            counts = (
                acc_df.groupby("Обвиняемый")["base_case_id"]
                .nunique()
                .reset_index(name="Уникальных_дел")
            )
            repeat_acc = (
                counts[counts["Уникальных_дел"] > 1]
                .sort_values("Уникальных_дел", ascending=False)
            )
            st.write(
                f"Найдено обвиняемых с 2 и более разными делами: "
                f"**{len(repeat_acc)}**"
            )

            if not repeat_acc.empty:
                r_col1, r_col2 = st.columns([1, 1.5])
                with r_col1:
                    st.dataframe(
                        repeat_acc.head(200),
                        use_container_width=True, hide_index=True,
                    )
                with r_col2:
                    target = st.selectbox(
                        "Выберите обвиняемого:",
                        options=repeat_acc["Обвиняемый"].head(200).tolist(),
                    )
                    p_cases = (
                        acc_df[acc_df["Обвиняемый"] == target]
                        .drop_duplicates(subset=["УИД"])
                        .sort_values("Дата поступления")
                    )
                    st.markdown(f"#### Дела обвиняемого: **{target}**")
                    st.dataframe(
                        p_cases[["Дата поступления", "Суд", "clean_articles",
                                 "Решение", "Стороны", "УИД"]],
                        use_container_width=True, hide_index=True,
                    )

    # --- РЕЖИМ 3: СОУЧАСТНИКИ (ГРУППОВЫЕ ДЕЛА) ---
    elif osint_mode == "👥 Соучастники (групповые дела)":
        st.caption(
            "Поиск дел, где в поле «Стороны» указано более одного обвиняемого. "
            "Эвристика: ищем ФИО в формате «Иванов И. И.» или «Иванов Иван Иванович» "
            "и считаем их количество в одной записи."
        )

        group_df = filtered_df.dropna(subset=["Стороны"]).copy()
        fio_pattern = r"\b([А-ЯЁ][а-яё]+(?:\s+[А-ЯЁ][а-яё.]+){1,2})\b"
        group_df["n_persons"] = group_df["Стороны"].str.findall(fio_pattern).apply(
            lambda lst: len(lst) if isinstance(lst, list) else 0
        )

        min_persons = st.slider(
            "Минимум человек в одном акте:", min_value=2, max_value=10, value=2
        )
        grouped = group_df[group_df["n_persons"] >= min_persons].copy()

        st.write(
            f"Найдено актов с {min_persons}+ фигурантами: **{len(grouped)}** "
            f"(уникальных дел: **{grouped['base_case_id'].nunique()}**)"
        )

        if grouped.empty:
            st.info("В текущей выборке групповых дел не найдено.")
        else:
            g1, g2 = st.columns(2)
            with g1:
                top_articles = (
                    grouped["clean_articles"]
                    .replace("", "Не указано")
                    .value_counts()
                    .head(10)
                    .reset_index()
                )
                top_articles.columns = ["Статьи", "Количество"]
                st.plotly_chart(
                    px.bar(
                        top_articles, x="Количество", y="Статьи", orientation="h",
                        title="Топ статей в групповых делах",
                        template="plotly_white",
                    ),
                    use_container_width=True,
                )
            with g2:
                by_region = grouped["Регион"].value_counts().head(10).reset_index()
                by_region.columns = ["Регион", "Количество"]
                st.plotly_chart(
                    px.bar(
                        by_region, x="Количество", y="Регион", orientation="h",
                        title="Топ регионов по групповым делам",
                        template="plotly_white",
                    ),
                    use_container_width=True,
                )

            st.dataframe(
                grouped[
                    ["Дата поступления", "Регион", "Суд", "clean_articles",
                     "n_persons", "Стороны", "Решение", "УИД"]
                ].head(500),
                use_container_width=True, hide_index=True,
            )

    # --- РЕЖИМ 4: ПРОКУРОРЫ И ЗАЩИТНИКИ ---
    elif osint_mode == "⚖️ Прокуроры и защитники":
        st.caption(
            "Обратный рейтинг: кто из юристов чаще появляется в делах и с каким "
            "исходом. Эвристика по ключевым словам в поле «Стороны»."
        )

        role_keywords = {
            "Прокурор": r"прокурор|гособвинит",
            "Защитник": r"защитник|адвокат|защиту",
        }
        role = st.radio("Роль:", list(role_keywords.keys()), horizontal=True)
        pattern = role_keywords[role]

        role_df = filtered_df[
            filtered_df["Стороны"].str.contains(pattern, case=False, na=False, regex=True)
        ].copy()

        if role_df.empty:
            st.info(f"В выборке не найдено упоминаний роли «{role}».")
        else:
            fio_after = role_df["Стороны"].str.findall(
                rf"(?:{pattern})[^А-ЯЁ]*([А-ЯЁ][а-яё]+\s+[А-ЯЁ]\.[\s]*[А-ЯЁ]\.?)",
                flags=re.IGNORECASE,
            )
            role_df["fio"] = fio_after.apply(
                lambda lst: lst[0] if isinstance(lst, list) and lst else None
            )
            role_df = role_df.dropna(subset=["fio"])

            if role_df.empty:
                st.info("Не удалось выделить конкретные ФИО для этой роли.")
            else:
                role_stats = (
                    role_df.groupby("fio")
                    .agg(
                        Дел=("УИД", "count"),
                        Приговоров=("Решение",
                            lambda x: x.str.contains("ПРИГОВОР", case=False, na=False).sum()),
                        Прекращено=("Решение",
                            lambda x: x.str.contains("ПРЕКРАЩ", case=False, na=False).sum()),
                    )
                    .reset_index()
                )
                role_stats["Доля приговоров, %"] = (
                    role_stats["Приговоров"] / role_stats["Дел"] * 100
                ).round(1)
                role_stats["Доля прекращений, %"] = (
                    role_stats["Прекращено"] / role_stats["Дел"] * 100
                ).round(1)
                role_stats = role_stats.sort_values("Дел", ascending=False)

                st.write(
                    f"Найдено представителей роли «{role}»: **{len(role_stats)}**"
                )
                st.dataframe(
                    role_stats.head(200),
                    use_container_width=True, hide_index=True,
                )

    # --- РЕЖИМ 5: ГИБКИЙ REGEX-ПОИСК ---
    else:
        st.caption(
            "Прямой поиск по полю «Стороны» с поддержкой регулярных выражений. "
            "Удобно для поиска конкретных подразделений, прокуроров, адвокатов "
            "или редких формулировок."
        )
        custom_regex = st.text_input(
            "Введите запрос или регулярное выражение:",
            placeholder="Например: ФСБ|Росгварди|батальон|Шторм|потерпевший",
        )
        if custom_regex:
            try:
                res_df = filtered_df[
                    filtered_df["Стороны"].str.contains(
                        custom_regex, case=False, na=False, regex=True
                    )
                ].copy()
                st.write(
                    f"Найдено судебных актов: **{len(res_df)}** "
                    f"(уникальных дел: **{res_df['base_case_id'].nunique()}**)"
                )
                st.dataframe(
                    res_df[
                        ["Дата поступления", "Регион", "Суд", "Инстанция_рус",
                         "clean_articles", "Стороны", "Решение", "УИД"]
                    ],
                    use_container_width=True, hide_index=True,
                )
            except re.error as e:
                st.error(f"Ошибка в регулярном выражении: {e}")

# ==================== ВКЛАДКА 5 ====================
with tab5:
    st.subheader("Реестр судебных актов")
    multi_counts = filtered_df["base_case_id"].value_counts()
    multi_ids = multi_counts[multi_counts > 1].index
    only_multi = st.checkbox(
        f"Показать только дела, прошедшие несколько инстанций "
        f"(нашлось {len(multi_ids)} дел)"
    )
    table_df = (
        filtered_df[filtered_df["base_case_id"].isin(multi_ids)]
        .sort_values(["base_case_id", "Дата поступления"])
        if only_multi else filtered_df
    )

    dec_query = st.text_input(
        "🔎 Поиск по тексту решения (например, 'условно', 'прекращено', 'штраф'):",
        key="dec_search",
    )
    if dec_query:
        table_df = table_df[
            table_df["Решение"].str.contains(
                dec_query, case=False, na=False, regex=False
            )
        ]

    cols_to_show = [
        "Номер дела", "Обвиняемый", "Инстанция_рус", "Дата поступления",
        "Дата решения", "days_to_decision", "Регион", "Суд", "Судья",
        "Статьи", "Решение", "Стороны", "УИД",
    ]
    st.dataframe(
        table_df[cols_to_show].head(1000),
        use_container_width=True, hide_index=True,
    )
    st.caption(
        "Отображаются первые 1000 строк для плавной работы. "
        "Полную отфильтрованную выборку можно скачать:"
    )

    csv_export = table_df[cols_to_show].to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        label="📥 Скачать выборку в CSV",
        data=csv_export,
        file_name="filtered_cases.csv",
        mime="text/csv",
        key="download_csv",
    )

    try:
        xlsx_buf = io.BytesIO()
        with pd.ExcelWriter(xlsx_buf, engine="openpyxl") as writer:
            table_df[cols_to_show].to_excel(
                writer, index=False, sheet_name="cases"
            )
        st.download_button(
            label="📥 Скачать выборку в Excel (.xlsx)",
            data=xlsx_buf.getvalue(),
            file_name="filtered_cases.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="download_xlsx",
        )
    except ImportError:
        st.caption("Для экспорта в Excel установите пакет `openpyxl`.")