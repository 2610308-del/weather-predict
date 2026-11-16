# ==============================================================================
# 1. 필요한 라이브러리(도구) 불러오기
# ==============================================================================
import calendar  # 각 달의 마지막 날짜(28일, 30일, 31일 및 윤년)를 자동 계산하기 위한 모듈
import json  # 기상청 서버에서 받은 JSON 형식의 데이터를 파이썬 딕셔너리로 변환
import platform  # 현재 코드가 실행되는 운영체제(Windows, Mac, Linux)를 확인
import time  # API 요청 간 짧은 대기 시간(딜레이)을 주어 서버 과부하 방지
import urllib.parse  # 한글이나 특수문자가 포함된 API 주소를 인터넷 표준 규격으로 변환
import urllib.request  # 기상청 오픈 API 서버에 직접 접속해 데이터를 요청
import matplotlib.pyplot as plt  # 기온 및 강수량 변화 그래프를 그리는 시각화 라이브러리
import numpy as np  # 배열 계산, 평균, 표준편차, 사인/코사인 파동 등 수학 연산 수행
import pandas as pd  # CSV 엑셀 데이터를 표(DataFrame) 형태로 읽고 필터링하는 도구
import streamlit as st  # 파이썬 코드만으로 웹사이트 화면(버튼, 선택창, 차트)을 만드는 프레임워크
from sklearn.linear_model import Ridge  # AI 머신러닝 회귀 모델 (과적합을 막아 안정적인 미래 예측 수행)

# ==============================================================================
# 2. 웹페이지 기본 설정 및 한글 폰트 깨짐 방지
# ==============================================================================
# 브라우저 탭 제목, 아이콘, 화면 너비 레이아웃 설정
st.set_page_config(page_title="AI 기반 대한민국 미래 기후 예측", page_icon="🌡️", layout="centered")

# 그래프 내부에 한글이 네모(□)로 깨져 보이지 않도록 운영체제별 기본 한글 폰트 지정
os_name = platform.system()
if os_name == 'Windows':
    plt.rc('font', family='Malgun Gothic')  # 윈도우 환경: 맑은 고딕
elif os_name == 'Darwin':
    plt.rc('font', family='AppleGothic')   # 맥(Mac) 환경: 애플고딕
else:
    plt.rc('font', family='NanumGothic')   # Streamlit Cloud(리눅스) 환경: 나눔고딕

# 그래프 Y축 등에서 마이너스(-) 부호가 깨지는 현상 방지
plt.rcParams['axes.unicode_minus'] = False


# ==============================================================================
# 3. AI 기후 예측 핵심 알고리즘 함수 정의
# ==============================================================================
def build_climate_features(years_arr, periods=(3.5, 5.5, 11.0)):
    """
    단순 직선이 아닌 현실적인 기후 변동을 표현하기 위해 입력 데이터(특성)를 만드는 함수
    - 연도(t): 장기적인 지구 온난화 상승 기울기를 학습
    - 사인(sin)·코사인(cos) 파동: 3.5년(엘니뇨), 5.5년, 11년(태양 활동 주기) 등 자연 주기 학습
    """
    t = years_arr.reshape(-1, 1)  # 1차원 연도 배열을 머신러닝 입력용 2차원 행렬로 변환
    features = [t]  # 기본 추세(연도) 추가
    for p in periods:
        features.append(np.sin(2 * np.pi * t / p))  # 주기별 사인 파동 성분 추가
        features.append(np.cos(2 * np.pi * t / p))  # 주기별 코사인 파동 성분 추가
    return np.hstack(features)  # 모든 특성을 가로로 합쳐 하나의 학습 데이터 표로 반환


def fit_and_predict_climate(years, values, target_year):
    """
    과거 관측 데이터를 AI 모델에 학습시켜 미래 예측 곡선과 변동 범위를 계산하는 함수
    """
    # 1) 학습용 데이터 생성 및 AI 주기 모델 학습
    X_train = build_climate_features(years)
    # Ridge(alpha=3.0): 주기 파동이 너무 과격하게 출렁이지 않도록 규제를 걸어 부드럽게 학습
    cycle_model = Ridge(alpha=3.0).fit(X_train, values)

    # 2) 비교용 순수 직선(장기 평균 추세) 모델 별도 학습
    trend_model = Ridge(alpha=1.0).fit(years.reshape(-1, 1), values)

    # 3) 과거 시작 연도부터 사용자가 선택한 목표 연도(target_year)까지 연도 배열 생성
    future_years = np.arange(years[0], target_year + 1)
    X_future = build_climate_features(future_years)

    # 4) 전체 기간에 대한 예측 곡선(출렁이는 곡선 & 직선 추세) 계산
    pred_cycle_curve = cycle_model.predict(X_future)
    pred_trend_curve = trend_model.predict(future_years.reshape(-1, 1))

    # 5) 과거 실제 데이터와 추세선 간의 오차(잔차) 표준편차를 구해 '자연 변동 폭(±1σ)' 계산
    residuals = values - trend_model.predict(years.reshape(-1, 1))
    std_dev = float(np.std(residuals))

    # 6) 사용자가 지정한 특정 미래 연도(target_year)의 최종 예측값 추출
    target_cycle_val = float(cycle_model.predict(build_climate_features(np.array([target_year])))[0])
    target_trend_val = float(trend_model.predict([[target_year]])[0])

    return future_years, pred_cycle_curve, pred_trend_curve, target_cycle_val, target_trend_val, std_dev


# ==============================================================================
# 4. 웹사이트 메인 제목 및 상단 탭(Tab) 화면 구성
# ==============================================================================
st.title("🌡️ AI 기반 대한민국 미래 기후 예측 플랫폼")
st.write("기후 자연 주기(파동)와 장기 온난화 추세를 함께 학습하는 AI 모델로 미래 날씨를 시뮬레이션합니다.")

# 화면을 두 개의 탭으로 분리하여 CSV 분석과 실시간 API 분석을 각각 제공
tab1, tab2 = st.tabs(["📊 2000~2026 장기 기후 데이터 AI 예측 (CSV)", "🛰️ 실시간 기상청 API 월별 기온 예측 (27개 도시)"])


# ==============================================================================
# 5. 첫 번째 탭: CSV 파일 기반 13개 도시 장기 기후 데이터 AI 예측
# ==============================================================================
with tab1:
    st.subheader("전국 13개 도시 장기 기후 데이터 분석 (2000~2026년)")

    # CSV 파일 안의 영문 도시명과 화면에 보여줄 한글 도시명을 연결하는 딕셔너리
    CITY_KR_MAP = {
        'Seoul': '서울', 'Incheon': '인천', 'Suwon': '수원', 'Chuncheon': '춘천',
        'Gangneung': '강릉', 'Daejeon': '대전', 'Cheongju': '청주', 'Daegu': '대구',
        'Busan': '부산', 'Ulsan': '울산', 'Gwangju': '광주', 'Jeonju': '전주', 'Jeju': '제주'
    }
    # 한글 선택 시 영문 데이터로 찾기 위한 역방향 매핑 딕셔너리
    KR_TO_EN_MAP = {v: k for k, v in CITY_KR_MAP.items()}

    try:
        # GitHub 저장소에 업로드된 CSV 파일 읽어오기
        df = pd.read_csv("south_korea_climate_2000_2026-v2.csv")

        # 화면을 3개의 가로 열(Column)로 나누어 사용자 입력 UI 배치
        c1, c2, c3 = st.columns([1, 1, 1.2])
        with c1:
            selected_kr_city = st.selectbox("도시 선택 (13개 지역)", list(KR_TO_EN_MAP.keys()), key="csv_city")
        with c2:
            csv_target_year = st.number_input("예측 미래 연도", min_value=2027, max_value=2100, value=2048, step=1, key="csv_year")
        with c3:
            metric_choice = st.selectbox("예측할 기후 지표", ["연평균 기온 (℃)", "폭염일수 (일)", "연강수량 (mm)"])

        # 선택한 도시에 해당하는 데이터만 걸러내고 연도순으로 정렬
        en_city = KR_TO_EN_MAP[selected_kr_city]
        city_df = df[df['City'] == en_city].sort_values('Year')

        # 사용자가 고른 기후 지표에 맞춰 CSV 컬럼명, 단위, 그래프 색상 결정
        if "기온" in metric_choice:
            y_col, unit, color_main = 'Avg_Temp_C', "℃", "#1f77b4"
        elif "폭염" in metric_choice:
            y_col, unit, color_main = 'Heatwave_Days', "일", "#ff7f0e"
        else:
            y_col, unit, color_main = 'Precipitation_mm', "mm", "#2ca02c"

        years_arr = city_df['Year'].values  # 연도 배열 추출
        y_arr = city_df[y_col].values       # 선택한 기후 수치 배열 추출

        # 위에서 정의한 AI 예측 함수 실행
        future_years, pred_cycle, pred_trend, val_cycle, val_trend, std_dev = fit_and_predict_climate(
            years_arr, y_arr, csv_target_year
        )

        # 폭염일수와 강수량은 물리적으로 0보다 작을 수 없으므로 최소값을 0으로 보정
        if "기온" not in metric_choice:
            pred_cycle = np.maximum(0, pred_cycle)
            val_cycle = max(0.0, val_cycle)

        # 비교 기준이 될 과거 27년 전체 평균 및 최근(2020년대) 평균 계산
        past_avg = float(np.mean(y_arr))
        recent_2020s = float(city_df[city_df['Year'] >= 2020][y_col].mean())

        # 예측 결과를 3개의 요약 카드(Metric) 형태로 화면에 표시
        st.divider()
        m1, m2, m3 = st.columns(3)
        m1.metric(
            f"{csv_target_year}년 AI 주기 반영 예측",
            f"{val_cycle:.1f} {unit}",
            f"{val_cycle - past_avg:+.1f} {unit} (27년 평균 대비)"
        )
        m2.metric(
            f"{csv_target_year}년 장기 평균 추세",
            f"{val_trend:.1f} {unit}",
            f"예상 변동폭 ±{std_dev:.1f} {unit}"
        )
        m3.metric(
            "2000~2026년 전체 평균",
            f"{past_avg:.1f} {unit}",
            f"2020년대 평균: {recent_2020s:.1f} {unit}"
        )

        # --- 메인 시뮬레이션 그래프 그리기 ---
        fig, ax = plt.subplots(figsize=(10, 5.4))

        # 1) 2000~2026년 실제 관측 데이터 선 그래프
        ax.plot(years_arr, y_arr, marker='o', linewidth=2, markersize=5, color=color_main,
                label=f'{selected_kr_city} 실제 관측값 (2000~2026)')

        # 2) 2026년 이후 미래 구간만 잘라내어 AI 주기 파동 예측 선 그리기
        future_mask = future_years >= years_arr[-1]
        ax.plot(future_years[future_mask], pred_cycle[future_mask], linestyle='-', linewidth=2.2, color='#d62728',
                label='AI 주기 변동 예측 (자연 변동 반영)')

        # 3) 미래 기상 변동 범위(±1 표준편차)를 반투명 음영 영역으로 채우기
        lower_band = pred_trend[future_mask] - std_dev
        upper_band = pred_trend[future_mask] + std_dev
        if "기온" not in metric_choice:
            lower_band = np.maximum(0, lower_band)
        ax.fill_between(future_years[future_mask], lower_band, upper_band, color='#d62728', alpha=0.12,
                        label='예상 기상 변동 범위 (±1σ)')

        # 4) 기준이 되는 장기 상승 직선 추세 점선 추가
        ax.plot(future_years, pred_trend, linestyle=':', linewidth=1.6, color='#555555',
                label='장기 평균 상승 추세선')

        # 5) 사용자가 선택한 목표 연도 지점에 큰 빨간 점과 말풍선 라벨 표시
        ax.scatter(csv_target_year, val_cycle, color='#d62728', s=150, zorder=5, edgecolors='black')

        # 그래프 상하좌우 여백 자동 조절 (말풍선이 그래프 밖으로 잘리지 않게 보호)
        all_vals = np.concatenate([y_arr, pred_cycle[future_mask], upper_band, lower_band])
        y_min, y_max = np.min(all_vals), np.max(all_vals)
        margin = (y_max - y_min) * 0.22 if y_max != y_min else 2.0
        ax.set_ylim(max(0, y_min - margin) if "기온" not in metric_choice else y_min - margin, y_max + margin)
        ax.set_xlim(1999, csv_target_year + max(2, int((csv_target_year - 2000) * 0.08)))

        ax.annotate(
            f'{csv_target_year}년 AI 예측\n주기반영: {val_cycle:.1f}{unit}\n(추세평균: {val_trend:.1f}{unit})',
            xy=(csv_target_year, val_cycle), xytext=(-18, 18), textcoords='offset points',
            ha='center', fontweight='bold', fontsize=9.5, color='#333333',
            bbox=dict(boxstyle='round,pad=0.35', fc='white', ec='#d62728', alpha=0.95)
        )

        # X축 연도가 소수점(예: 2020.5)으로 나오지 않고 정수로만 표시되게 설정
        ax.xaxis.get_major_locator().set_params(integer=True)
        ax.set_title(f'[{selected_kr_city}] 2000~2026년 데이터 기반 {csv_target_year}년 {metric_choice} AI 시뮬레이션', fontsize=13.5, fontweight='bold', pad=15)
        ax.set_xlabel('연도')
        ax.set_ylabel(metric_choice)
        ax.grid(True, linestyle=':', alpha=0.6)  # 배경 격자무늬 추가
        ax.legend(loc='best', fontsize=9)        # 범례 자동 위치 배치
        plt.tight_layout()
        st.pyplot(fig)  # 완성된 그래프를 웹사이트에 출력

        # --- 하단 부가 기능 1: 여러 도시 동시 비교 차트 ---
        st.divider()
        st.subheader("🏙️ 여러 도시 동시에 비교하기")
        compare_cities = st.multiselect(
            "비교할 도시들을 선택하세요 (최대 5개 권장)",
            list(KR_TO_EN_MAP.keys()),
            default=["서울", "대구", "부산", "제주"]
        )
        if compare_cities:
            fig_comp, ax_comp = plt.subplots(figsize=(10, 4.5))
            for c_kr in compare_cities:
                c_en = KR_TO_EN_MAP[c_kr]
                sub_df = df[df['City'] == c_en].sort_values('Year')
                ax_comp.plot(sub_df['Year'], sub_df[y_col], marker='o', markersize=4, linewidth=1.8, label=c_kr)
            ax_comp.xaxis.get_major_locator().set_params(integer=True)
            ax_comp.set_title(f'주요 도시별 {metric_choice} 추이 비교 (2000~2026)', fontsize=13, fontweight='bold')
            ax_comp.set_xlabel('연도')
            ax_comp.set_ylabel(metric_choice)
            ax_comp.grid(True, linestyle=':', alpha=0.6)
            ax_comp.legend(loc='best', ncol=len(compare_cities))
            plt.tight_layout()
            st.pyplot(fig_comp)

        # --- 하단 부가 기능 2: 접이식 원본 데이터 표 ---
        with st.expander(f"📋 [{selected_kr_city}] 연도별 원본 기후 데이터 표 보기 (2000~2026)"):
            display_df = city_df[['Year', 'Decade', 'Avg_Temp_C', 'Heatwave_Days', 'Precipitation_mm']].rename(
                columns={'Year': '연도', 'Decade': '연대', 'Avg_Temp_C': '연평균기온(℃)', 'Heatwave_Days': '폭염일수(일)', 'Precipitation_mm': '연강수량(mm)'}
            )
            st.dataframe(display_df, use_container_width=True, hide_index=True)

    # CSV 파일이 GitHub 저장소에 없을 경우 친절한 안내 문구 출력
    except FileNotFoundError:
        st.error("`south_korea_climate_2000_2026-v2.csv` 파일을 찾을 수 없습니다. GitHub 저장소에 CSV 파일을 업로드해 주세요.")


# ==============================================================================
# 6. 두 번째 탭: 기상청 실시간 API 연동 월별 기온 예측 (전국 27개 도시)
# ==============================================================================
with tab2:
    st.subheader("기상청 실시간 서버 연동 월별 미래 기온 예측 (전국 27개 도시)")
    
    # 공공데이터포털 기상청 종관기상관측(ASOS) 인증키
    MY_API_KEY = '404e80f5bbda1aa9285ba87f66e0e384246caa1dbe9a7b4c14d9a1a598c81239'

    # 전국 27개 기상청 관측소 지점번호 매핑
    STATIONS = {
        '서울 (수도권)': '108', '인천 (수도권)': '112', '수원 (경기)': '119',
        '춘천 (강원)': '101', '원주 (강원)': '114', '강릉 (강원 영동)': '105', '속초 (강원 영동)': '90',
        '청주 (충북)': '131', '충주 (충북)': '127', '대전 (충남)': '133', '천안 (충남)': '232', '서산 (충남)': '129',
        '전주 (전북)': '146', '군산 (전북)': '140', '광주 (전남)': '156', '목포 (전남)': '165', '여수 (전남)': '168', '순천 (전남)': '174',
        '대구 (경북)': '143', '포항 (경북)': '138', '안동 (경북)': '136',
        '부산 (경남)': '159', '울산 (경남)': '152', '창원 (경남)': '155', '진주 (경남)': '192',
        '제주 (제주)': '184', '서귀포 (제주)': '189',
    }

    # 도시, 예측 연도, 예측 월(1~12월) 선택 위젯 배치
    col1, col2, col3 = st.columns([1.3, 1, 0.9])
    with col1:
        selected_station_label = st.selectbox("관측 도시 선택", list(STATIONS.keys()), key="api_city")
    with col2:
        api_target_year = st.number_input("예측할 미래 연도", min_value=2025, max_value=2100, value=2040, step=1, key="api_year")
    with col3:
        selected_month = st.selectbox("예측할 월 선택", list(range(1, 13)), index=7, format_func=lambda m: f"{m}월", key="api_month")

    # 버튼을 눌렀을 때만 기상청 서버 통신을 시작하도록 구성
    if st.button(f"🛰️ 기상청 API 수집 및 {api_target_year}년 {selected_month}월 AI 예측 실행", type="primary", use_container_width=True):
        stn_id = STATIONS[selected_station_label]
        city_clean_name = selected_station_label.split()[0]  # '서울 (수도권)'에서 '서울'만 추출
        base_url = 'https://apis.data.go.kr/1360000/AsosDalyInfoService/getWthrDataList'
        years, month_avg_temps, month_max_temps = [], [], []

        # 데이터 수집 진행률을 보여주는 프로그레스 바 생성
        progress_bar = st.progress(0, text=f"기상청 서버에서 [{city_clean_name}] 최근 15년 {selected_month}월 데이터를 불러오는 중...")

        # 2010년부터 2024년까지 15년 치 데이터를 연도별로 순회하며 수집
        for idx, y in enumerate(range(2010, 2025)):
            # 선택한 연도와 월의 마지막 날짜 계산 (예: 2월 윤년 29일 자동 처리)
            last_day = calendar.monthrange(y, selected_month)[1]
            start_dt = f"{y}{selected_month:02d}01"
            end_dt = f"{y}{selected_month:02d}{last_day:02d}"

            # 기상청 API 서버에 전달할 요청 파라미터 구성
            params = {
                'serviceKey': MY_API_KEY, 'pageNo': '1', 'numOfRows': str(last_day),
                'dataType': 'JSON', 'dataCd': 'ASOS', 'dateCd': 'DAY',
                'startDt': start_dt, 'endDt': end_dt, 'stnIds': stn_id,
            }
            full_url = f"{base_url}?{urllib.parse.urlencode(params)}"
            req = urllib.request.Request(full_url, headers={'User-Agent': 'Mozilla/5.0'})

            # 일시적인 네트워크 끊김에 대비해 연도당 최대 3번까지 재시도
            for attempt in range(3):
                try:
                    with urllib.request.urlopen(req, timeout=10) as response:
                        res = json.loads(response.read().decode('utf-8'))
                        # 응답 코드가 '00'(정상)일 때만 일별 평균기온과 최고기온 추출
                        if res['response']['header']['resultCode'] == '00':
                            items = res['response']['body']['items']['item']
                            daily_avgs = [float(it['avgTa']) for it in items if it.get('avgTa')]
                            daily_maxs = [float(it['maxTa']) for it in items if it.get('maxTa')]
                            if daily_avgs:
                                years.append(y)
                                month_avg_temps.append(np.mean(daily_avgs))  # 해당 월의 평균기온 저장
                                month_max_temps.append(np.mean(daily_maxs))  # 해당 월의 낮 최고기온 평균 저장
                            break
                except Exception:
                    pass
                time.sleep(0.2)

            # 진행률 게이지 업데이트
            progress_bar.progress((idx + 1) / 15, text=f"{y}년 {selected_month}월 데이터 수집 완료 ({idx + 1}/15)")

        progress_bar.empty()  # 수집이 끝나면 진행률 바 숨기기

        if len(years) < 5:
            st.error("기상청 서버 응답 지연으로 데이터를 충분히 불러오지 못했습니다. 잠시 후 다시 시도해주세요.")
        else:
            years_arr = np.array(years)
            y_avg = np.array(month_avg_temps)
            y_max = np.array(month_max_temps)

            # 수집된 15년 월별 데이터를 AI 모델에 학습시켜 미래 예측 수행
            future_years, pred_cycle, pred_trend, val_cycle, val_trend, std_dev = fit_and_predict_climate(
                years_arr, y_avg, api_target_year
            )
            _, _, _, val_max_cycle, _, _ = fit_and_predict_climate(
                years_arr, y_max, api_target_year
            )
            past_mean = float(np.mean(y_avg))

            # 요약 수치 카드 출력
            st.divider()
            m1, m2, m3 = st.columns(3)
            m1.metric(f"{api_target_year}년 {selected_month}월 AI 주기 예측", f"{val_cycle:.2f} ℃", f"{val_cycle - past_mean:+.2f} ℃ (15년 평균 대비)")
            m2.metric(f"{api_target_year}년 {selected_month}월 예상 낮 최고", f"{val_max_cycle:.2f} ℃", f"추세 평균: {val_trend:.2f} ℃")
            m3.metric(f"과거 15년 {selected_month}월 평균", f"{past_mean:.2f} ℃", f"예상 변동폭 ±{std_dev:.2f} ℃")

            # 실시간 수집 데이터 및 AI 예측 그래프 그리기
            fig, ax = plt.subplots(figsize=(10, 5.2))
            ax.plot(years_arr, y_avg, marker='o', linewidth=2, color='#1f77b4', label=f'{city_clean_name} 실제 {selected_month}월 평균기온')

            future_mask = future_years >= years_arr[-1]
            ax.plot(future_years[future_mask], pred_cycle[future_mask], linestyle='-', linewidth=2.2, color='#d62728',
                    label='AI 주기 변동 예측')
            ax.fill_between(future_years[future_mask],
                            pred_trend[future_mask] - std_dev,
                            pred_trend[future_mask] + std_dev,
                            color='#d62728', alpha=0.12, label='예상 기상 변동 범위 (±1σ)')
            ax.plot(future_years, pred_trend, linestyle=':', linewidth=1.6, color='#555555', label='장기 평균 상승 추세선')

            ax.scatter(api_target_year, val_cycle, color='#d62728', s=150, zorder=5, edgecolors='black')

            all_vals = np.concatenate([y_avg, pred_cycle[future_mask], pred_trend[future_mask] + std_dev, pred_trend[future_mask] - std_dev])
            y_min, y_max_val = np.min(all_vals), np.max(all_vals)
            margin = (y_max_val - y_min) * 0.25 if y_max_val != y_min else 1.5
            ax.set_ylim(y_min - margin, y_max_val + margin)
            ax.set_xlim(years_arr[0] - 1, api_target_year + max(2, int((api_target_year - years_arr[0]) * 0.08)))

            ax.annotate(
                f'{api_target_year}년 {selected_month}월 예측\n평균 {val_cycle:.1f}℃ (최고 {val_max_cycle:.1f}℃)',
                xy=(api_target_year, val_cycle), xytext=(-15, 16), textcoords='offset points',
                ha='center', fontweight='bold', fontsize=10, color='#
