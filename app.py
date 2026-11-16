import json
import platform
import time
import urllib.parse
import urllib.request
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

# 웹페이지 기본 설정
st.set_page_config(page_title="AI 기반 대한민국 미래 기후 예측", page_icon="🌡️", layout="centered")

# 한글 폰트 설정
os_name = platform.system()
if os_name == 'Windows':
    plt.rc('font', family='Malgun Gothic')
elif os_name == 'Darwin':
    plt.rc('font', family='AppleGothic')
else:
    plt.rc('font', family='NanumGothic')
plt.rcParams['axes.unicode_minus'] = False

st.title("🌡️ AI 기반 대한민국 미래 기후 예측 플랫폼")
st.write("장기 기후 데이터(2000~2026)와 실시간 기상청 API 데이터를 결합해 미래 기온과 폭염을 시뮬레이션합니다.")

# 두 가지 기능을 탭으로 분리
tab1, tab2 = st.tabs(["📊 2000~2026 장기 기후 데이터 AI 예측 (CSV)", "🛰️ 실시간 기상청 API 여름 기온 예측"])

# ==============================================================================
# TAB 1: 업로드한 CSV 파일 기반 장기 기후 데이터 분석 및 AI 예측 (13개 도시)
# ==============================================================================
with tab1:
    st.subheader("전국 13개 도시 장기 기후 데이터 분석 (2000~2026년)")

    CITY_KR_MAP = {
        'Seoul': '서울', 'Incheon': '인천', 'Suwon': '수원', 'Chuncheon': '춘천',
        'Gangneung': '강릉', 'Daejeon': '대전', 'Cheongju': '청주', 'Daegu': '대구',
        'Busan': '부산', 'Ulsan': '울산', 'Gwangju': '광주', 'Jeonju': '전주', 'Jeju': '제주'
    }
    KR_TO_EN_MAP = {v: k for k, v in CITY_KR_MAP.items()}

    try:
        df = pd.read_csv("south_korea_climate_2000_2026-v2.csv")

        c1, c2, c3 = st.columns([1, 1, 1.2])
        with c1:
            selected_kr_city = st.selectbox("도시 선택 (13개 지역)", list(KR_TO_EN_MAP.keys()), key="csv_city")
        with c2:
            csv_target_year = st.number_input("예측 미래 연도", min_value=2027, max_value=2100, value=2040, step=1, key="csv_year")
        with c3:
            metric_choice = st.selectbox("예측할 기후 지표", ["연평균 기온 (℃)", "폭염일수 (일)", "연강수량 (mm)"])

        en_city = KR_TO_EN_MAP[selected_kr_city]
        city_df = df[df['City'] == en_city].sort_values('Year')

        X = city_df[['Year']].values
        if "기온" in metric_choice:
            y = city_df['Avg_Temp_C'].values
            unit, color_main = "℃", "#1f77b4"
        elif "폭염" in metric_choice:
            y = city_df['Heatwave_Days'].values
            unit, color_main = "일", "#ff7f0e"
        else:
            y = city_df['Precipitation_mm'].values
            unit, color_main = "mm", "#2ca02c"

        # AI 머신러닝 모델 학습 (다항 회귀 vs 선형 회귀 자동 앙상블/비교)
        model_linear = make_pipeline(PolynomialFeatures(1), Ridge(alpha=1.0)).fit(X, y)
        model_poly = make_pipeline(PolynomialFeatures(2), Ridge(alpha=10.0)).fit(X, y)

        future_years = np.arange(2000, csv_target_year + 1).reshape(-1, 1)
        pred_linear_curve = model_linear.predict(future_years)
        pred_poly_curve = model_poly.predict(future_years)

        val_linear = float(model_linear.predict([[csv_target_year]])[0])
        val_poly = float(model_poly.predict([[csv_target_year]])[0])
        past_avg = float(np.mean(y))
        recent_2020s = float(city_df[city_df['Year'] >= 2020][y_col := ('Avg_Temp_C' if "기온" in metric_choice else ('Heatwave_Days' if "폭염" in metric_choice else 'Precipitation_mm'))].mean())

        # 주요 지표 카드 출력
        st.divider()
        m1, m2, m3 = st.columns(3)
        m1.metric(f"{csv_target_year}년 AI 예측 (선형 추세)", f"{val_linear:.1f} {unit}", f"{val_linear - past_avg:+.1f} {unit} (27년 평균 대비)")
        m2.metric(f"{csv_target_year}년 AI 예측 (가속 추세)", f"{val_poly:.1f} {unit}", f"{val_poly - past_avg:+.1f} {unit} (27년 평균 대비)")
        m3.metric("2000~2026년 전체 평균", f"{past_avg:.1f} {unit}", f"2020년대 평균: {recent_2020s:.1f} {unit}")

        # 그래프 그리기
        fig, ax = plt.subplots(figsize=(10, 5.2))
        ax.plot(city_df['Year'], y, marker='o', linewidth=2, markersize=5, color=color_main, label=f'{selected_kr_city} 실제 관측값 (2000~2026)')
        ax.plot(future_years.flatten(), pred_linear_curve, linestyle='--', linewidth=2, color='#d62728', label='AI 선형 예측 추세')
        ax.plot(future_years.flatten(), pred_poly_curve, linestyle='-.', linewidth=2, color='#9467bd', label='AI 다항(가속) 예측 추세')

        ax.scatter(csv_target_year, val_linear, color='#d62728', s=130, zorder=5, edgecolors='black')
        ax.scatter(csv_target_year, val_poly, color='#9467bd', s=130, zorder=5, edgecolors='black')

        all_vals = np.concatenate([y, [val_linear, val_poly]])
        y_min, y_max = np.min(all_vals), np.max(all_vals)
        margin = (y_max - y_min) * 0.25 if y_max != y_min else 2.0
        ax.set_ylim(max(0, y_min - margin) if "기온" not in metric_choice else y_min - margin, y_max + margin)
        ax.set_xlim(1999, csv_target_year + max(2, int((csv_target_year - 2000) * 0.08)))

        ax.annotate(
            f'{csv_target_year}년 예측\n선형: {val_linear:.1f}{unit}\n가속: {val_poly:.1f}{unit}',
            xy=(csv_target_year, (val_linear + val_poly) / 2), xytext=(-15, 18), textcoords='offset points',
            ha='center', fontweight='bold', fontsize=9.5, color='#333333',
            bbox=dict(boxstyle='round,pad=0.35', fc='white', ec='#d62728', alpha=0.95)
        )

        ax.xaxis.get_major_locator().set_params(integer=True)
        ax.set_title(f'[{selected_kr_city}] 2000~2026년 데이터 기반 {csv_target_year}년 {metric_choice} AI 예측', fontsize=13.5, fontweight='bold', pad=15)
        ax.set_xlabel('연도')
        ax.set_ylabel(metric_choice)
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend(loc='best')
        plt.tight_layout()
        st.pyplot(fig)

        # 원본 데이터 표 미리보기 및 통계
        with st.expander(f"📋 [{selected_kr_city}] 연도별 원본 기후 데이터 표 보기 (2000~2026)"):
            display_df = city_df[['Year', 'Decade', 'Avg_Temp_C', 'Heatwave_Days', 'Precipitation_mm']].rename(
                columns={'Year': '연도', 'Decade': '연대', 'Avg_Temp_C': '연평균기온(℃)', 'Heatwave_Days': '폭염일수(일)', 'Precipitation_mm': '연강수량(mm)'}
            )
            st.dataframe(display_df, use_container_width=True, hide_index=True)

    except FileNotFoundError:
        st.error("`south_korea_climate_2000_2026-v2.csv` 파일을 찾을 수 없습니다. GitHub 저장소에 CSV 파일을 업로드해 주세요.")


# ==============================================================================
# TAB 2: 기존 실시간 기상청 API 연동 (8월 한여름 집중 분석)
# ==============================================================================
with tab2:
    st.subheader("기상청 실시간 서버 연동 8월 한여름 기온 예측")
    MY_API_KEY = '404e80f5bbda1aa9285ba87f66e0e384246caa1dbe9a7b4c14d9a1a598c81239'
    STATIONS = {
        '서울': '108', '인천': '112', '수원': '119', '대전': '133',
        '대구': '143', '광주': '156', '부산': '159', '제주': '184',
    }

    col1, col2 = st.columns(2)
    with col1:
        api_city = st.selectbox("관측소 선택", list(STATIONS.keys()), key="api_city")
    with col2:
        api_target_year = st.number_input("예측할 미래 연도", min_value=2025, max_value=2100, value=2040, step=1, key="api_year")

    if st.button("🛰️ 기상청 API 실시간 수집 및 예측 실행", type="primary", use_container_width=True):
        stn_id = STATIONS[api_city]
        base_url = 'https://apis.data.go.kr/1360000/AsosDalyInfoService/getWthrDataList'
        years, aug_avg_temps = [], []

        progress_bar = st.progress(0, text="기상청 서버에서 15년치 8월 데이터를 불러오는 중...")

        for idx, y in enumerate(range(2010, 2025)):
            params = {
                'serviceKey': MY_API_KEY, 'pageNo': '1', 'numOfRows': '31',
                'dataType': 'JSON', 'dataCd': 'ASOS', 'dateCd': 'DAY',
                'startDt': f'{y}0801', 'endDt': f'{y}0831', 'stnIds': stn_id,
            }
            full_url = f"{base_url}?{urllib.parse.urlencode(params)}"
            req = urllib.request.Request(full_url, headers={'User-Agent': 'Mozilla/5.0'})

            for attempt in range(3):
                try:
                    with urllib.request.urlopen(req, timeout=10) as response:
                        res = json.loads(response.read().decode('utf-8'))
                        if res['response']['header']['resultCode'] == '00':
                            items = res['response']['body']['items']['item']
                            daily_avgs = [float(it['avgTa']) for it in items if it.get('avgTa')]
                            if daily_avgs:
                                years.append(y)
                                aug_avg_temps.append(np.mean(daily_avgs))
                            break
                except Exception:
                    pass
                time.sleep(0.2)

            progress_bar.progress((idx + 1) / 15, text=f"{y}년 8월 데이터 수집 완료 ({idx + 1}/15)")

        progress_bar.empty()

        if len(years) < 5:
            st.error("기상청 서버 응답 지연으로 데이터를 충분히 불러오지 못했습니다. 잠시 후 다시 시도해주세요.")
        else:
            X = np.array(years).reshape(-1, 1)
            y_avg = np.array(aug_avg_temps)

            model = make_pipeline(PolynomialFeatures(1), Ridge(alpha=1.0)).fit(X, y_avg)
            future_X = np.arange(years[0], api_target_year + 1).reshape(-1, 1)
            pred_curve = model.predict(future_X)
            pred_val = float(model.predict([[api_target_year]])[0])
            past_mean = float(np.mean(y_avg))

            st.divider()
            m1, m2 = st.columns(2)
            m1.metric(f"{api_target_year}년 8월 예상 평균기온", f"{pred_val:.2f} ℃", f"{pred_val - past_mean:+.2f} ℃ (15년 평균 대비)")
            m2.metric("과거 15년(2010~2024) 8월 평균", f"{past_mean:.2f} ℃")

            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(years, y_avg, marker='o', linewidth=2, color='#1f77b4', label=f'{api_city} 실제 8월 평균기온')
            ax.plot(future_X.flatten(), pred_curve, linestyle='--', linewidth=2, color='#d62728', label='AI 예측 추세선')
            ax.scatter(api_target_year, pred_val, color='#d62728', s=160, zorder=5, edgecolors='black')

            y_min, y_max = min(np.min(y_avg), pred_val), max(np.max(y_avg), pred_val)
            margin = (y_max - y_min) * 0.25 if y_max != y_min else 1.5
            ax.set_ylim(y_min - margin, y_max + margin)
            ax.set_xlim(years[0] - 1, api_target_year + max(2, int((api_target_year - years[0]) * 0.08)))

            ax.annotate(
                f'{api_target_year}년 예측\n{pred_val:.1f}℃',
                xy=(api_target_year, pred_val), xytext=(0, 14), textcoords='offset points',
                ha='center', fontweight='bold', fontsize=11, color='#d62728',
                bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#d62728', alpha=0.9)
            )
            ax.xaxis.get_major_locator().set_params(integer=True)
            ax.set_title(f'[{api_city}] 기상청 실시간 데이터 기반 {api_target_year}년 8월 기온 예측', fontsize=13.5, fontweight='bold', pad=15)
            ax.set_xlabel('연도')
            ax.set_ylabel('8월 평균 기온 (℃)')
            ax.grid(True, linestyle=':', alpha=0.6)
            ax.legend(loc='best')
            plt.tight_layout()
            st.pyplot(fig)
