import re
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
    if pd.isna(text):
        return "Не указано"
    t = str(text).upper()
    if "ПОДСУДНОСТ" in t:
        return "Передано по подсудности"
    if "ПРЕКРАЩ" in t:
        return "Прекращено"
    if "ОПРАВД" in t:
        return "Оправдан"
    if "ПРИГОВОР" in t and "УСЛОВН" in t:
        return "Приговор (условно)"
    if "ПРИГОВОР" in t:
        return "Приговор (реально)"
    return "Другое"


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
    if pd.isna(parties_raw):
        return ""
    text = str(parties_raw).strip()
    if SKIP_PREFIX_MARKERS.search(text):
        return ""
    m = FIO_PATTERN.search(text)
    return m.group(1).strip() if m else ""


def get_case_stage(num) -> str:
    if pd.isna(num):
        return "Прочее"
    s = str(num).strip()
    if s.startswith("1-"):
        return "1-я инстанция (по существу)"
    if s.startswith(("22К-", "22к-")):
        return "Апелляция (мера пресечения)"
    if s.startswith(("22-", "10-")):
        return "Апелляция (по существу)"
    if s.startswith(("7У-", "7у-")):
        return "Кассация"
    if s.startswith(("4/", "3/")):
        return "Материалы (арест, УДО, исполнение)"
    return "Прочее"


@st.cache_data
def load_data():
    use_cols = {
        "УИД", "Номер дела", "Суд", "Регион", "Тип", "Инстанция",
        "Судья", "Решение", "Статьи", "Стороны", "source_file",
        "base_case_id", "clean_articles", "Дата поступления",
        "Дата решения", "days_to_decision", "Ссылка на карточку",
        "Карточка на сайте суда", "Обвиняемый", "Тип_производства",
        "dedup_key", "vc_num",
    }
    df = pd.read_csv(
        "cases_clean.csv.gz",
        usecols=lambda c: c in use_cols,
        dtype=str,
    )

    df["Дата поступления"] = pd.to_datetime(df["Дата поступления"], errors="coerce")
    df["Дата решения"] = pd.to_datetime(df["Дата решения"], errors="coerce")
    df["days_to_decision"] = pd.to_numeric(df.get("days_to_decision"), errors="coerce")

    df["decision_class"] = df["Решение"].map(classify_decision)
    df["Суд"] = df["Суд"].fillna("Суд не указан")
    df["Регион"] = df["Регион"].fillna("Регион не указан")
    df["Инстанция_рус"] = df["Инстанция"].map({
        "FIRST": "1-я инстанция",
        "APPELLATION": "Апелляция",
        "CASSATION": "Кассация",
    }).fillna(df["Инстанция"])

    if "Обвиняемый" not in df.columns:
        df["Обвиняемый"] = df["Стороны"].map(get_accused)
    if "Тип_производства" not in df.columns:
        df["Тип_производства"] = df["Номер дела"].map(get_case_stage)
    if "dedup_key" not in df.columns:
        df["dedup_key"] = (
            df["Номер дела"].fillna("").astype(str) + "|" +
            df["Суд"].fillna("").astype(str) + "|" +
            df["Обвиняемый"].fillna("").astype(str)
        )
    if "vc_num" not in df.columns:
        df["vc_num"] = (
            df["Стороны"]
            .str.extract(r"(?:в/ч|войсковая часть)\s*№?\s*(\d{4,6})", flags=re.IGNORECASE)[0]
            .fillna("")
            .astype(str)
        )
    if "base_case_id" not in df.columns:
        df["base_case_id"] = df["УИД"].apply(
            lambda x: str(x).split("~")[0] if pd.notna(x) and "~" in str(x) else str(x)
        )

    for str_col in ["Обвиняемый", "dedup_key", "vc_num", "clean_articles",
                    "Номер дела", "УИД", "base_case_id"]:
        if str_col in df.columns:
            df[str_col] = df[str_col].fillna("").astype(str)

    cat_cols = ["Суд", "Регион", "Инстанция", "Инстанция_рус", "Судья",
                "decision_class", "Тип_производства"]
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype("category")

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

stages = sorted(df["Тип_производства"].dropna().unique())
selected_stages = st.sidebar.multiselect("Тип производства:", options=stages, default=[])

instances = sorted(df["Инстанция_рус"].dropna().unique())
selected_instances = st.sidebar.multiselect("Инстанция:", options=instances, default=[])

regions = sorted(df["Регион"].dropna().unique())
selected_regions = st.sidebar.multiselect("Регион:", options=regions, default=[])

decision_classes = sorted(df["decision_class"].dropna().unique())
selected_classes = st.sidebar.multiselect("Тип решения:", options=decision_classes, default=[])

search_query = st.sidebar.text_input("🔎 Поиск (ФИО, суд, судья, № дела):")

if st.sidebar.button("♻️ Сбросить фильтры"):
    st.rerun()

# --- ПРИМЕНЕНИЕ ФИЛЬТРОВ ---
filtered_df = df

if date_range and isinstance(date_range, (tuple, list)) and len(date_range) == 2:
    start_d, end_d = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
    filtered_df = filtered_df[
        (filtered_df["Дата поступления"] >= start_d) &
        (filtered_df["Дата поступления"] <= end_d)
    ]

if selected_articles:
    art_set = set(selected_articles)
    filtered_df = filtered_df[
        filtered_df["clean_articles"].map(lambda v: bool(art_set & set(split_articles(v))))
    ]

if selected_stages:
    filtered_df = filtered_df[filtered_df["Тип_производства"].isin(selected_stages)]

if selected_instances:
    filtered_df = filtered_df[filtered_df["Инстанция_рус"].isin(selected_instances)]

if selected_regions:
    filtered_df = filtered_df[filtered_df["Регион"].isin(selected_regions)]

if selected_classes:
    filtered_df = filtered_df[filtered_df["decision_class"].isin(selected_classes)]

if search_query:
    q = search_query.strip()
    mask = (
        filtered_df["Стороны"].fillna("").astype(str).str.contains(q, case=False, na=False, regex=False) |
        filtered_df["Суд"].fillna("").astype(str).str.contains(q, case=False, na=False, regex=False) |
        filtered_df["Судья"].fillna("").astype(str).str.contains(q, case=False, na=False, regex=False) |
        filtered_df["Номер дела"].fillna("").astype(str).str.contains(q, case=False, na=False, regex=False) |
        filtered_df["УИД"].fillna("").astype(str).str.contains(q, case=False, na=False, regex=False)
    )
    filtered_df = filtered_df[mask]

# --- ВЕРХНИЕ МЕТРИКИ ---
substantive_first = filtered_df[
    (filtered_df["Инстанция"] == "FIRST") &
    (filtered_df["Тип_производства"] == "1-я инстанция (по существу)") &
    (filtered_df["decision_class"] != "Передано по подсудности")
]

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Всего карточек (актов)", fmt_int(len(filtered_df)))
c2.metric("Дел 1-й инст. по существу", fmt_int(substantive_first["dedup_key"].nunique()))
base_uid_cnt = filtered_df["base_case_id"].nunique() if "base_case_id" in filtered_df.columns else 0
c3.metric("Уникальных базовых УИД", fmt_int(base_uid_cnt))

valid_days = filter_valid_days(filtered_df["days_to_decision"])
median_val = int(valid_days.median()) if len(valid_days) else 0
c4.metric("Медианный срок", f"{median_val} дн.")
c5.metric("Уникальных обвиняемых", fmt_int(filtered_df[filtered_df["Обвиняемый"] != ""]["Обвиняемый"].nunique()))

st.divider()

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Общая картина",
    "👤 Подсудимые (Статистика)",
    "🏛️ Аналитика по судам",
    "🗂️ Карточка дела (Досье)",
    "🕵️ OSINT / Связи",
    "📋 Полная таблица и выгрузка",
])

# ==================== ВКЛАДКА 1: ОБЩАЯ КАРТИНА ====================
with tab1:
    st.subheader("Динамика поступления дел по месяцам")
    ts_df = filtered_df.dropna(subset=["Дата поступления"]).copy()
    if not ts_df.empty:
        ts_df["Месяц"] = ts_df["Дата поступления"].dt.to_period("M").astype(str)
        monthly = (
            ts_df.groupby(["Месяц", "Инстанция_рус"], observed=True)
            .size().reset_index(name="Количество")
        )
        fig_time = px.bar(
            monthly, x="Месяц", y="Количество", color="Инстанция_рус",
            barmode="stack", template="plotly_white",
            color_discrete_sequence=px.colors.qualitative.Set2,
            labels={"Инстанция_рус": "Инстанция"},
        )
        st.plotly_chart(fig_time, use_container_width=True)

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
        st.subheader("Структура производств (по номерам дел)")
        stage_counts = filtered_df["Тип_производства"].value_counts().reset_index()
        stage_counts.columns = ["Тип", "Количество"]
        fig_st = px.pie(
            stage_counts, values="Количество", names="Тип", hole=0.45,
            template="plotly_white", color_discrete_sequence=px.colors.qualitative.Pastel,
        )
        st.plotly_chart(fig_st, use_container_width=True)

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
        days_plot_df = filtered_df[filtered_df["days_to_decision"].between(0, 365, inclusive="both")]
        if not days_plot_df.empty:
            fig_days = px.histogram(
                days_plot_df, x="days_to_decision", nbins=45, color="Инстанция_рус",
                template="plotly_white", color_discrete_sequence=px.colors.qualitative.Set2,
                labels={"days_to_decision": "Дней от поступления до решения", "Инстанция_рус": "Инстанция"},
            )
            st.plotly_chart(fig_days, use_container_width=True)


# ==================== ВКЛАДКА 2: ПОДСУДИМЫЕ ====================
with tab2:
    st.subheader("👤 Аналитика по подсудимым: чистый подсчет без дублей инстанций")
    st.caption(
        "Методика: повторными считаются отдельные уголовные дела 1-й инстанции по существу (1-...) "
        "без апелляций, материалов об аресте и передач по подсудности."
    )

    acc_all = filtered_df[filtered_df["Обвиняемый"] != ""].copy()
    clean_first = acc_all[
        (acc_all["Инстанция"] == "FIRST") &
        (acc_all["Тип_производства"] == "1-я инстанция (по существу)") &
        (acc_all["decision_class"] != "Передано по подсудности")
    ].drop_duplicates(subset=["Обвиняемый", "Номер дела"]).copy()

    total_acts = len(filtered_df)
    unique_people = acc_all["Обвиняемый"].nunique()
    hidden_acts = (filtered_df["Обвиняемый"] == "").sum()
    hidden_pct = (hidden_acts / total_acts * 100) if total_acts else 0

    transitions = []
    gaps_days = []
    real_repeat_people = set()

    if not clean_first.empty:
        cand_counts = (
            clean_first.drop_duplicates(subset=["Обвиняемый", "Номер дела"])
            .groupby("Обвиняемый")["Номер дела"].nunique()
        )
        cand_names = cand_counts[cand_counts > 1].index
        real_repeat_people = set(cand_names)
        rep_df = clean_first[clean_first["Обвиняемый"].isin(cand_names)].sort_values(
            ["Обвиняемый", "Дата поступления"]
        )

        for person, grp in rep_df.groupby("Обвиняемый"):
            if len(grp) < 2:
                continue
            arts_seq = grp["clean_articles"].replace("", "Не указана").tolist()
            dates_seq = grp["Дата поступления"].tolist()
            for i in range(len(grp) - 1):
                if pd.notna(dates_seq[i]) and pd.notna(dates_seq[i + 1]):
                    transitions.append(f"{arts_seq[i]} → {arts_seq[i+1]}")
                    diff = (dates_seq[i + 1] - dates_seq[i]).days
                    if 0 <= diff <= 1825:
                        gaps_days.append(diff)

    repeat_count = len(real_repeat_people)
    repeat_pct = (repeat_count / unique_people * 100) if unique_people else 0

    pm1, pm2, pm3, pm4 = st.columns(4)
    pm1.metric("Уникальных подсудимых", fmt_int(unique_people))
    pm2.metric("Повторных (2+ дела)", f"{fmt_int(repeat_count)} ({repeat_pct:.1f}%)")
    pm3.metric("Индекс анонимности (скрыто ФИО)", f"{hidden_pct:.1f}%")
    pm4.metric("Чистых дел 1-й инст. с ФИО", fmt_int(len(clean_first)))

    if transitions:
        vc_tr = pd.Series(transitions).value_counts()
        top_transition = vc_tr.index[0]
        top_transition_cnt = vc_tr.iloc[0]
    else:
        top_transition = "—"
        top_transition_cnt = 0
    median_gap = int(pd.Series(gaps_days).median()) if gaps_days else None

    with st.container(border=True):
        st.markdown("#### 💡 Выявленные закономерности (очищено от связанных актов)")
        insights = []
        if unique_people > 0:
            insights.append(
                f"• **Уровень повторных судов:** **{repeat_pct:.1f}%** от распознанных подсудимых "
                f"(**{fmt_int(repeat_count)}** чел.) имеют два и более отдельных уголовных дела по существу "
                f"(без учёта апелляций и материалов)."
            )
        if transitions:
            insights.append(
                f"• **Главный сценарий повторных дел:** самый частый переход квалификаций — "
                f"**«{top_transition}»** (**{fmt_int(top_transition_cnt)}** случаев с известными датами)."
            )
        if median_gap is not None:
            insights.append(
                f"• **Скорость возврата в суд:** медианный интервал (по делам с известными датами, ≤5 лет) "
                f"составляет **{median_gap} дн.** (~{median_gap / 30.4:.1f} мес.)."
            )
        if total_acts > 0:
            insights.append(
                f"• **Сокрытие данных судами:** в **{hidden_pct:.1f}%** актов (**{fmt_int(hidden_acts)}** из {fmt_int(total_acts)}) "
                f"ФИО подсудимого скрыто или отсутствует."
            )
        st.markdown("\n\n".join(insights))

    st.markdown("---")
    st.markdown("#### 📈 Динамика появления уникальных подсудимых и скрытых дел по месяцам")
    ts_people = filtered_df.dropna(subset=["Дата поступления"]).copy()
    if not ts_people.empty:
        ts_people["Месяц"] = ts_people["Дата поступления"].dt.to_period("M").astype(str)
        open_m = (
            ts_people[ts_people["Обвиняемый"] != ""]
            .groupby("Месяц")["Обвиняемый"].nunique()
            .reset_index(name="Количество")
        )
        open_m["Категория"] = "Уникальные подсудимые (ФИО открыто)"
        hid_m = (
            ts_people[ts_people["Обвиняемый"] == ""]
            .groupby("Месяц")["dedup_key"].nunique()
            .reset_index(name="Количество")
        )
        hid_m["Категория"] = "Дела со скрытым ФИО"
        combined_m = pd.concat([open_m, hid_m], ignore_index=True)
        fig_growth = px.bar(
            combined_m, x="Месяц", y="Количество", color="Категория",
            barmode="stack", template="plotly_white",
            color_discrete_map={
                "Уникальные подсудимые (ФИО открыто)": "#2b8a3e",
                "Дела со скрытым ФИО": "#adb5bd",
            },
        )
        st.plotly_chart(fig_growth, use_container_width=True)

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
                .nunique().nlargest(10).reset_index(name="Подсудимых")
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
            acc_all.groupby("Регион", observed=True)["Обвиняемый"]
            .nunique().nlargest(10).reset_index(name="Подсудимых")
        )
        fig_rp = px.bar(
            top_reg_p, x="Подсудимых", y="Регион", orientation="h",
            template="plotly_white", color_discrete_sequence=px.colors.sequential.Blues_r,
        )
        fig_rp.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig_rp, use_container_width=True)

    if transitions:
        st.markdown("---")
        st.markdown("#### 🔄 Переходы квалификаций при повторных делах")
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
                    title="Интервал между первым и повторным делом (дней)",
                    template="plotly_white", color_discrete_sequence=["#e67700"],
                )
                st.plotly_chart(fig_gap, use_container_width=True)

    st.markdown("---")
    st.markdown("#### 📋 Сводный реестр уникальных подсудимых (1 строка = 1 человек)")
    rf1, rf2 = st.columns([2, 1])
    with rf1:
        p_search = st.text_input("🔎 Поиск по фамилии подсудимого:", key="accused_registry_search")
    with rf2:
        only_real_rep = st.checkbox("Только с несколькими делами по существу (2+ дела)")

    if not acc_all.empty:
        acc_all["substantive_key"] = acc_all["Номер дела"].where(
            (acc_all["Инстанция"] == "FIRST") &
            (acc_all["Тип_производства"] == "1-я инстанция (по существу)") &
            (acc_all["decision_class"] != "Передано по подсудности")
        )
        registry = (
            acc_all.sort_values("Дата поступления")
            .groupby("Обвиняемый")
            .agg(
                Дел_по_существу=("substantive_key", "nunique"),
                Всего_карточек=("УИД", "count"),
                Первое_дело=("Дата поступления", "min"),
                Последнее_дело=("Дата поступления", "max"),
                Статьи=("clean_articles", lambda x: "; ".join(sorted(set(a for a in x if a)))),
                Регионы=("Регион", lambda x: ", ".join(sorted(set(str(r) for r in x if pd.notna(r))))),
            )
            .reset_index()
            .sort_values(["Дел_по_существу", "Последнее_дело"], ascending=[False, False])
        )
        registry["Первое_дело"] = registry["Первое_дело"].dt.strftime("%Y-%m-%d")
        registry["Последнее_дело"] = registry["Последнее_дело"].dt.strftime("%Y-%m-%d")
        if only_real_rep:
            registry = registry[registry["Обвиняемый"].isin(real_repeat_people)]
        if p_search:
            registry = registry[
                registry["Обвиняемый"].str.contains(p_search.strip(), case=False, na=False, regex=False)
            ]
        st.write(f"В реестре показано: {fmt_int(len(registry))} чел.")
        st.dataframe(registry.head(1000), use_container_width=True, hide_index=True)


# ==================== ВКЛАДКА 3: АНАЛИТИКА ПО СУДАМ ====================
with tab3:
    st.subheader("Детальный разбор конкретного суда")
    court_counts = filtered_df["Суд"].value_counts()
    if court_counts.empty:
        st.warning("Нет данных под выбранные фильтры.")
    else:
        court_options = [f"{name} ({cnt} актов)" for name, cnt in court_counts.items()]
        court_map = {f"{name} ({cnt} актов)": name for name, cnt in court_counts.items()}
        selected_court_label = st.selectbox("Выберите суд для анализа:", options=court_options)
        selected_court = court_map[selected_court_label]
        court_df = filtered_df[filtered_df["Суд"] == selected_court]

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Всего актов в суде", fmt_int(len(court_df)))
        m2.metric("Судей (по делам)", fmt_int(court_df["Судья"].dropna().nunique()))
        c_days = filter_valid_days(court_df["days_to_decision"])
        m3.metric("Медианный срок", f"{int(c_days.median())} дн." if len(c_days) else "—")
        verdicts = court_df["Решение"].str.contains("ПРИГОВОР", case=False, na=False).sum()
        pct = (verdicts / len(court_df) * 100) if len(court_df) else 0
        m4.metric("Вынесено приговоров", f"{verdicts} ({pct:.0f}%)")

        st.markdown("---")
        sc1, sc2 = st.columns(2)
        with sc1:
            st.markdown("#### Нагрузка и скорость по судьям")
            judges_stats = (
                court_df.dropna(subset=["Судья"])
                .groupby("Судья", observed=True)
                .agg(
                    Актов=("УИД", "count"),
                    Приговоров=("Решение", lambda x: x.str.contains("ПРИГОВОР", case=False, na=False).sum()),
                    Медиана_дней=("days_to_decision", lambda x: filter_valid_days(x).median()),
                )
                .reset_index()
                .sort_values("Актов", ascending=False)
            )
            judges_stats["Медиана_дней"] = judges_stats["Медиана_дней"].round(0)
            st.dataframe(judges_stats, use_container_width=True, hide_index=True)

        with sc2:
            st.markdown("#### Статьи в этом суде")
            c_art = court_df["clean_articles"].replace("", "Не указано").value_counts().head(10).reset_index()
            c_art.columns = ["Статьи", "Количество"]
            fig_c_art = px.bar(
                c_art, x="Количество", y="Статьи", orientation="h",
                template="plotly_white", color_discrete_sequence=["#2b8a3e"],
            )
            fig_c_art.update_layout(yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig_c_art, use_container_width=True)


# ==================== ВКЛАДКА 4: КАРТОЧКА ДЕЛА (ДОСЬЕ) ====================
with tab4:
    st.subheader("Детальный просмотр дела и всех связанных карточек")
    st.caption("Связывает все инстанции по базовому УИД, а также находит все дела этого же обвиняемого в базе.")
    if filtered_df.empty:
        st.warning("По текущим фильтрам дел не найдено.")
    else:
        preview_df = filtered_df.head(500).reset_index(drop=True)
        preview_df["_row_id"] = preview_df.index
        preview_df["_label"] = (
            preview_df["Номер дела"].fillna("Б/Н") + " | " +
            preview_df["Обвиняемый"].replace("", "ФИО скрыто") + " | " +
            preview_df["Суд"].astype(str) + " #" + preview_df["_row_id"].astype(str)
        )
        selected_idx = st.selectbox(
            "Выберите судебный акт (показаны первые 500 из выборки):",
            options=preview_df["_row_id"].tolist(),
            format_func=lambda i: preview_df.loc[preview_df["_row_id"] == i, "_label"].iloc[0],
        )
        selected_row = preview_df[preview_df["_row_id"] == selected_idx].iloc[0]
        base_id = selected_row["base_case_id"]
        accused_name = selected_row["Обвиняемый"]

        if accused_name:
            cond = (df["Обвиняемый"] == accused_name)
            if base_id:
                cond = cond | (df["base_case_id"] == base_id)
            related_cases = df[cond].sort_values("Дата поступления")
        elif base_id:
            related_cases = df[df["base_case_id"] == base_id].sort_values("Дата поступления")
        else:
            related_cases = df[df["УИД"] == selected_row["УИД"]].sort_values("Дата поступления")

        st.info(f"Найдено связанных карточек (по УИД и обвиняемому): {len(related_cases)}")

        for _, row in related_cases.iterrows():
            with st.container(border=True):
                h1, h2, h3 = st.columns([2, 2, 1])
                h1.markdown(f"{row['Тип_производства']} ({row['Номер дела']})")
                h2.markdown(f"Суд: {row['Суд']}")
                h3.markdown(f"Регион: {row['Регион']}")

                d1, d2, d3, d4 = st.columns(4)
                d1.write("Поступило: " + (row["Дата поступления"].strftime("%Y-%m-%d") if pd.notna(row["Дата поступления"]) else "—"))
                d2.write("Решение: " + (row["Дата решения"].strftime("%Y-%m-%d") if pd.notna(row["Дата решения"]) else "В процессе"))
                days = row.get("days_to_decision")
                d3.write(f"Срок: {int(days)} дн." if pd.notna(days) and days >= 0 else "Срок: —")
                d4.write(f"Судья: {row['Судья'] if pd.notna(row['Судья']) else 'Не указан'}")

                st.write(f"Итог: {row['Решение'] if pd.notna(row['Решение']) else 'Решение еще не вынесено'}")
                raw_codes = split_articles(row["clean_articles"])
                if raw_codes:
                    st.write("Статьи: " + " • ".join([format_article(c) for c in raw_codes]))
                st.markdown(f"Обвиняемый: {row['Обвиняемый'] or 'Скрыт судом'}")
                st.code(str(row["Стороны"]) if pd.notna(row["Стороны"]) else "Данные скрыты судом", language=None)

                links = []
                if pd.notna(row.get("Карточка на сайте суда")):
                    links.append(f"[Карточка на сайте суда]({row['Карточка на сайте суда']})")
                if pd.notna(row.get("Ссылка на карточку")):
                    links.append(f"[Decretum]({row['Ссылка на карточку']})")
                if links:
                    st.markdown(" • ".join(links))
                st.caption(f"УИД: {row['УИД']} | Источник: {row.get('source_file', '—')}")


# ==================== ВКЛАДКА 5: OSINT / СВЯЗИ ====================
with tab5:
    st.subheader("🕵️ OSINT-инструменты и связи")
    osint_mode = st.radio(
        "Выберите инструмент:",
        [
            "🪖 Реестр воинских частей (в/ч)",
            "👥 Соучастники (групповые дела)",
            "🔎 Гибкий поиск по сторонам (Regex)",
        ],
        horizontal=True,
    )
    st.markdown("---")

    if osint_mode == "🪖 Реестр воинских частей (в/ч)":
        vc_valid = filtered_df[filtered_df["vc_num"] != ""]
        if vc_valid.empty:
            st.info("В текущей выборке не найдено упоминаний номеров в/ч.")
        else:
            vc_counts = vc_valid["vc_num"].value_counts().reset_index()
            vc_counts.columns = ["Номер_вч", "Упоминаний"]
            selected_vc = st.selectbox("Выберите номер в/ч:", options=vc_counts["Номер_вч"].tolist())
            vc_cases = vc_valid[vc_valid["vc_num"] == selected_vc]
            st.write(f"Найдено карточек по в/ч {selected_vc}: {len(vc_cases)}")
            st.dataframe(
                vc_cases[["Дата поступления", "Регион", "Суд", "Обвиняемый", "clean_articles", "Решение", "УИД"]],
                use_container_width=True, hide_index=True,
            )

    elif osint_mode == "👥 Соучастники (групповые дела)":
        group_df = filtered_df.dropna(subset=["Стороны"]).copy()
        fio_pattern = r"\b([А-ЯЁ][а-яё]+(?:\s+[А-ЯЁ][а-яё.]+){1,2})\b"
        group_df["n_persons"] = group_df["Стороны"].str.findall(fio_pattern).apply(
            lambda lst: len(lst) if isinstance(lst, list) else 0
        )
        grouped = group_df[group_df["n_persons"] >= 3]
        st.write(f"Найдено карточек с 3+ персонами в строке: {len(grouped)}")
        st.dataframe(
            grouped[["Дата поступления", "Регион", "Суд", "clean_articles", "Стороны", "Решение", "УИД"]].head(500),
            use_container_width=True, hide_index=True,
        )

    else:
        custom_regex = st.text_input(
            "Введите запрос или регулярное выражение:",
            placeholder="Например: ФСБ|Росгварди|батальон|Шторм",
        )
        if custom_regex:
            try:
                res_df = filtered_df[
                    filtered_df["Стороны"].str.contains(custom_regex, case=False, na=False, regex=True)
                ]
                st.write(f"Найдено карточек: {len(res_df)}")
                st.dataframe(
                    res_df[["Дата поступления", "Регион", "Суд", "Обвиняемый", "clean_articles", "Стороны", "Решение", "УИД"]].head(500),
                    use_container_width=True, hide_index=True,
                )
            except re.error as e:
                st.error(f"Ошибка в регулярном выражении: {e}")


# ==================== ВКЛАДКА 6: ПОЛНАЯ ТАБЛИЦА ====================
with tab6:
    st.subheader("Реестр судебных актов")
    cols_to_show = [
        "Номер дела", "Тип_производства", "Обвиняемый", "Инстанция_рус",
        "Дата поступления", "Дата решения", "days_to_decision",
        "Регион", "Суд", "Судья", "Статьи", "Решение", "Стороны", "УИД",
    ]
    st.dataframe(filtered_df[cols_to_show].head(1000), use_container_width=True, hide_index=True)
    st.caption("Отображаются первые 1000 строк для быстрой работы. Полную выборку можно скачать в CSV (открывается в Excel):")
    csv_export = filtered_df[cols_to_show].to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        label="📥 Скачать выборку в CSV (для Excel)",
        data=csv_export,
        file_name="filtered_cases.csv",
        mime="text/csv",
    )