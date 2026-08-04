import itertools

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------
# Streamlit 페이지 설정
# ---------------------------------------------------------
st.set_page_config(
    page_title="KBO 팀 전력 분석 & 시즌 시뮬레이션",
    page_icon="⚾",
    layout="wide",
)


# ---------------------------------------------------------
# 데이터 불러오기 및 병합
# ---------------------------------------------------------
@st.cache_data
def load_and_merge_data(
    batting_path="kbobattingdata.csv",
    pitching_path="kbopitchingdata.csv",
    year_start=2001,
    year_end=2021,
):
    batting = pd.read_csv(batting_path, encoding="utf-8-sig")
    pitching = pd.read_csv(pitching_path, encoding="utf-8-sig")

    batting = batting[
        (batting["year"] >= year_start)
        & (batting["year"] <= year_end)
    ].copy()

    pitching = pitching[
        (pitching["year"] >= year_start)
        & (pitching["year"] <= year_end)
    ].copy()

    merged = pd.merge(
        batting,
        pitching,
        on=["year", "team"],
        suffixes=("_bat", "_pit"),
    )

    numeric_cols = merged.select_dtypes(include=[np.number]).columns
    merged[numeric_cols] = merged[numeric_cols].fillna(0)

    return merged


# ---------------------------------------------------------
# 피타고리안 승률 계산
# ---------------------------------------------------------
def add_pythagorean_win_pct(df, exponent=1.83):
    df = df.copy()

    df["runs_scored"] = df["runs_bat"]
    df["runs_allowed"] = df["runs_pit"]

    denominator = (
        df["runs_scored"] ** exponent
        + df["runs_allowed"] ** exponent
    )

    df["pyth_win_pct"] = np.where(
        denominator == 0,
        0.5,
        (df["runs_scored"] ** exponent) / denominator,
    )

    return df


# ---------------------------------------------------------
# 모델 입력 특성
# ---------------------------------------------------------
FEATURE_COLUMNS = [
    "OPS",
    "batting_average",
    "runs_per_game_bat",
    "homeruns",
    "ERA",
    "WHIP",
    "strikeouts_9",
    "walks_9",
    "runs_per_game_pit",
    "pyth_win_pct",
]


# ---------------------------------------------------------
# Random Forest 모델 학습
# ---------------------------------------------------------
@st.cache_resource
def train_power_model(df):
    df = add_pythagorean_win_pct(df)

    X = df[FEATURE_COLUMNS].copy()
    y = df["win_loss_percentage"].copy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=6,
        random_state=42,
        n_jobs=-1,
    )

    cv_scores = cross_val_score(
        model,
        X_scaled,
        y,
        cv=5,
        scoring="r2",
        n_jobs=-1,
    )

    model.fit(X_scaled, y)

    train_pred = model.predict(X_scaled)

    metrics = {
        "cv_mean": cv_scores.mean(),
        "cv_std": cv_scores.std(),
        "full_r2": r2_score(y, train_pred),
    }

    return model, scaler, df, metrics


# ---------------------------------------------------------
# 특정 팀-시즌 전력 조회
# ---------------------------------------------------------
def get_team_power(df, model, scaler, year, team):
    row = df[
        (df["year"] == year)
        & (df["team"] == team)
    ]

    if row.empty:
        raise ValueError(
            f"{year}년 {team} 데이터를 찾을 수 없습니다."
        )

    X_scaled = scaler.transform(row[FEATURE_COLUMNS])
    predicted_win_pct = model.predict(X_scaled)[0]

    # 비정상적으로 범위를 벗어나는 경우를 대비해 0~1로 제한
    predicted_win_pct = np.clip(predicted_win_pct, 0, 1)

    return {
        "year": year,
        "team": team,
        "actual_win_pct": row["win_loss_percentage"].iloc[0],
        "pyth_win_pct": row["pyth_win_pct"].iloc[0],
        "predicted_win_pct": predicted_win_pct,
    }


# ---------------------------------------------------------
# Log5 맞대결 승리 확률
# ---------------------------------------------------------
def log5_win_probability(win_pct_a, win_pct_b):
    numerator = win_pct_a - (win_pct_a * win_pct_b)

    denominator = (
        win_pct_a
        + win_pct_b
        - (2 * win_pct_a * win_pct_b)
    )

    if np.isclose(denominator, 0):
        return 0.5

    probability = numerator / denominator

    return float(np.clip(probability, 0, 1))


# ---------------------------------------------------------
# 팀별 총 경기 수가 최대한 균등한 대진표 생성
# ---------------------------------------------------------
def build_schedule(n_teams, total_games):
    if n_teams < 2:
        raise ValueError("최소 2개 이상의 팀이 필요합니다.")

    if total_games < 1:
        raise ValueError("총 경기 수는 1 이상이어야 합니다.")

    # 모든 팀이 서로 동일한 횟수만큼 경기하는 기본 대진
    base_games = total_games // (n_teams - 1)
    remainder = total_games % (n_teams - 1)

    pairs = list(itertools.combinations(range(n_teams), 2))
    schedule = {pair: base_games for pair in pairs}

    if remainder == 0:
        return {
            pair: games
            for pair, games in schedule.items()
            if games > 0
        }

    # 모든 팀이 정확히 total_games를 치르려면
    # 전체 경기 참가 횟수 n_teams * total_games가 짝수여야 함
    if (n_teams * total_games) % 2 != 0:
        raise ValueError(
            f"{n_teams}개 팀이 각각 {total_games}경기를 치르는 "
            "완전히 균등한 대진은 만들 수 없습니다. "
            "팀 수가 홀수라면 총 경기 수를 짝수로 설정해 주세요."
        )

    extra_pairs = set()

    # 팀 수가 짝수이고 remainder가 홀수이면
    # 서로 반대편 팀을 연결하는 완전 매칭을 추가
    if n_teams % 2 == 0 and remainder % 2 == 1:
        half = n_teams // 2

        for i in range(half):
            pair = tuple(sorted((i, i + half)))
            extra_pairs.add(pair)

        distance_count = (remainder - 1) // 2
    else:
        distance_count = remainder // 2

    # 원형 배치를 이용해 팀별 추가 경기 수를 균등하게 배분
    for distance in range(1, distance_count + 1):
        for i in range(n_teams):
            j = (i + distance) % n_teams

            if i != j:
                pair = tuple(sorted((i, j)))
                extra_pairs.add(pair)

    for pair in extra_pairs:
        schedule[pair] += 1

    return {
        pair: games
        for pair, games in schedule.items()
        if games > 0
    }


# ---------------------------------------------------------
# 몬테카를로 시즌 시뮬레이션
# ---------------------------------------------------------
def simulate_season(
    team_list,
    df,
    model,
    scaler,
    total_games=144,
    n_simulations=2000,
    random_state=42,
):
    n_teams = len(team_list)

    if n_teams < 2:
        raise ValueError("최소 2개 이상의 팀을 선택해야 합니다.")

    power_info = [
        get_team_power(df, model, scaler, year, team)
        for year, team in team_list
    ]

    power_df = pd.DataFrame(power_info)

    power_df["team_label"] = (
        power_df["year"].astype(str)
        + " "
        + power_df["team"]
    )

    win_pcts = power_df["predicted_win_pct"].to_numpy()
    labels = power_df["team_label"].to_numpy()

    schedule = build_schedule(n_teams, total_games)

    # 이론적 기대 승수 계산
    expected_wins = np.zeros(n_teams)
    expected_games = np.zeros(n_teams)

    for (i, j), games in schedule.items():
        p_i = log5_win_probability(
            win_pcts[i],
            win_pcts[j],
        )

        expected_wins[i] += p_i * games
        expected_wins[j] += (1 - p_i) * games

        expected_games[i] += games
        expected_games[j] += games

    # 몬테카를로 시뮬레이션
    rng = np.random.default_rng(random_state)

    mc_win_totals = np.zeros(
        (n_simulations, n_teams),
        dtype=int,
    )

    for simulation in range(n_simulations):
        wins = np.zeros(n_teams, dtype=int)

        for (i, j), games in schedule.items():
            p_i = log5_win_probability(
                win_pcts[i],
                win_pcts[j],
            )

            results = rng.random(games) < p_i
            team_i_wins = int(results.sum())

            wins[i] += team_i_wins
            wins[j] += games - team_i_wins

        mc_win_totals[simulation] = wins

    mc_mean_wins = mc_win_totals.mean(axis=0)

    # 공동 1위가 나오면 우승 횟수를 공동 1위 팀끼리 균등 배분
    champion_points = np.zeros(n_teams)

    for simulation_wins in mc_win_totals:
        max_wins = simulation_wins.max()
        champions = np.flatnonzero(
            simulation_wins == max_wins
        )

        champion_points[champions] += 1 / len(champions)

    champion_pct = (
        champion_points / n_simulations
    ) * 100

    result_df = pd.DataFrame({
        "팀(연도)": labels,
        "모델예상승률": win_pcts.round(4),
        "피타고리안승률": (
            power_df["pyth_win_pct"].to_numpy().round(4)
        ),
        "실제승률(원시즌)": (
            power_df["actual_win_pct"].to_numpy().round(4)
        ),
        "총경기수": expected_games.astype(int),
        "기대승수": expected_wins.round(1),
        "기대패수": (
            expected_games - expected_wins
        ).round(1),
        "MC평균승수": mc_mean_wins.round(1),
        "우승확률(%)": champion_pct.round(1),
    })

    result_df = result_df.sort_values(
        "기대승수",
        ascending=False,
    ).reset_index(drop=True)

    result_df.index += 1
    result_df.index.name = "예상순위"

    return result_df


# ---------------------------------------------------------
# 선택 가능한 팀 목록
# ---------------------------------------------------------
def list_available_teams(
    df,
    year=None,
    team_keyword=None,
):
    result = df[
        [
            "year",
            "team",
            "win_loss_percentage",
            "OPS",
            "ERA",
        ]
    ].drop_duplicates()

    if year is not None:
        result = result[result["year"] == year]

    if team_keyword:
        result = result[
            result["team"].str.contains(
                team_keyword,
                case=False,
                na=False,
            )
        ]

    return result.sort_values(
        ["year", "team"]
    ).reset_index(drop=True)


# ---------------------------------------------------------
# Streamlit 화면
# ---------------------------------------------------------
st.title(
    "⚾ KBO 타격·투구 통합 팀 전력 분석 및 "
    "144경기 가상 시즌 시뮬레이션"
)

st.caption(
    "2001~2021년 KBO 팀 기록 기반 · "
    "Random Forest 회귀 · 피타고리안 승률 · "
    "Log5 · 몬테카를로 시뮬레이션"
)


# 데이터 및 모델 준비
try:
    with st.spinner(
        "데이터를 불러오고 모델을 학습하는 중입니다..."
    ):
        merged_df = load_and_merge_data(
            year_start=2001,
            year_end=2021,
        )

        model, scaler, df_full, metrics = train_power_model(
            merged_df
        )

except FileNotFoundError as error:
    st.error(
        "CSV 파일을 찾을 수 없습니다. "
        "app.py와 같은 위치에 kbobattingdata.csv와 "
        "kbopitchingdata.csv가 있는지 확인해 주세요."
    )
    st.exception(error)
    st.stop()

except Exception as error:
    st.error(
        "데이터를 불러오거나 모델을 학습하는 중 "
        "오류가 발생했습니다."
    )
    st.exception(error)
    st.stop()


st.success(
    f"총 {len(merged_df)}개 팀-시즌 데이터 로드 완료 "
    "(2001~2021년)"
)


# 모델 성능 표시
col1, col2 = st.columns(2)

with col1:
    st.metric(
        "5-Fold CV R² 평균",
        f"{metrics['cv_mean']:.4f}",
        f"표준편차 ±{metrics['cv_std']:.4f}",
    )

with col2:
    st.metric(
        "전체 데이터 R²",
        f"{metrics['full_r2']:.4f}",
    )


st.divider()


# 팀 목록 조회
st.subheader("📋 팀-연도 목록 조회")

year_options = [
    int(year)
    for year in sorted(
        df_full["year"].unique(),
        reverse=True,
    )
]

year_filter = st.selectbox(
    "연도로 필터링",
    options=["전체"] + year_options,
)

year_arg = (
    None
    if year_filter == "전체"
    else int(year_filter)
)

st.dataframe(
    list_available_teams(
        df_full,
        year=year_arg,
    ),
    use_container_width=True,
    hide_index=True,
)


st.divider()


# 가상 시즌 참가 팀 선택
st.subheader("🏆 가상 시즌에 참가할 팀 선택")

team_options_df = list_available_teams(df_full)

team_options_df["label"] = (
    team_options_df["year"].astype(str)
    + " "
    + team_options_df["team"]
)

label_list = team_options_df["label"].tolist()

default_team_labels = [
    "2002 Samsung Lions",
    "2012 Samsung Lions",
    "2002 Kia Tigers",
    "2006 Kia Tigers",
]

default_selection = [
    label
    for label in default_team_labels
    if label in label_list
]

if len(default_selection) < 2:
    default_selection = label_list[:4]

selected_labels = st.multiselect(
    "2개 이상의 '연도 팀'을 선택하세요.",
    options=label_list,
    default=default_selection,
)


col3, col4 = st.columns(2)

with col3:
    total_games = st.number_input(
        "팀별 총 경기 수",
        min_value=10,
        max_value=200,
        value=144,
        step=1,
    )

with col4:
    n_simulations = st.number_input(
        "몬테카를로 시뮬레이션 횟수",
        min_value=100,
        max_value=10000,
        value=2000,
        step=100,
    )


run_button = st.button(
    "🚀 시즌 시뮬레이션 실행",
    type="primary",
    use_container_width=True,
)


if run_button:
    if len(selected_labels) < 2:
        st.error(
            "최소 2개 이상의 팀을 선택해야 합니다."
        )

    else:
        selected_teams = []

        for label in selected_labels:
            year_str, team_name = label.split(" ", 1)

            selected_teams.append(
                (int(year_str), team_name)
            )

        try:
            with st.spinner(
                "가상 시즌을 시뮬레이션하는 중입니다..."
            ):
                ranking = simulate_season(
                    team_list=selected_teams,
                    df=df_full,
                    model=model,
                    scaler=scaler,
                    total_games=int(total_games),
                    n_simulations=int(n_simulations),
                    random_state=42,
                )

        except ValueError as error:
            st.error(str(error))

        except Exception as error:
            st.error(
                "시뮬레이션 중 오류가 발생했습니다."
            )
            st.exception(error)

        else:
            # 최종 순위표
            st.subheader("📊 최종 예상 순위표")

            st.dataframe(
                ranking,
                use_container_width=True,
            )

            # 우승 확률 그래프
            st.subheader("🥇 우승 확률")

            champion_ranking = ranking.sort_values(
                "우승확률(%)",
                ascending=False,
            )

            fig_champ = px.bar(
                champion_ranking,
                x="팀(연도)",
                y="우승확률(%)",
                text="우승확률(%)",
                color="우승확률(%)",
                color_continuous_scale="Blues",
            )

            fig_champ.update_traces(
                texttemplate="%{text:.1f}%",
                textposition="outside",
            )

            fig_champ.update_layout(
                xaxis_tickangle=0,
                xaxis_title=None,
                yaxis_title="우승 확률 (%)",
                coloraxis_showscale=False,
                height=500,
                margin=dict(t=40, b=40),
                plot_bgcolor="white",
                font=dict(size=13),
            )

            st.plotly_chart(
                fig_champ,
                use_container_width=True,
            )

            # 기대 승수 그래프
            st.subheader("📈 기대 승수 비교")

            wins_ranking = ranking.sort_values(
                "기대승수",
                ascending=False,
            )

            fig_wins = px.bar(
                wins_ranking,
                x="팀(연도)",
                y="기대승수",
                text="기대승수",
                color="기대승수",
                color_continuous_scale="Oranges",
            )

            fig_wins.update_traces(
                texttemplate="%{text:.1f}",
                textposition="outside",
            )

            fig_wins.update_layout(
                xaxis_tickangle=0,
                xaxis_title=None,
                yaxis_title="기대 승수",
                coloraxis_showscale=False,
                height=500,
                margin=dict(t=40, b=40),
                plot_bgcolor="white",
                font=dict(size=13),
            )

            st.plotly_chart(
                fig_wins,
                use_container_width=True,
            )
