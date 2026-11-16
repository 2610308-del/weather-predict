import json
import platform
import time
import urllib.parse
import urllib.request
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

# 웹페이지 기본 설정
st.set_page_config(page_title="전국 미래 기온 예측", page_icon="🌡️", layout="centered")

# 한글 폰트 설정 (로컬 및 리눅스 배포 환경 대응)
os_name = platform.system()
if os_name == 'Windows':
    plt.rc('font', family='Malgun Gothic')
elif os_name == 'Darwin':
    plt.rc('font', family='AppleGothic')
else:
    plt.rc('font', family='NanumGothic')
plt.rcParams['axes.unicode_minus'] = False

# 1. 인증키 및 관측소 정보
MY_API_KEY = '404e80f5bbda1aa9285ba87f66e0e384246caa1dbe9a7b4c14d9a1a598c81239'
STATIONS = {
    '서울': '108', '인천': '112', '수원': '119', '대전': '133',
    '대구': '143', '광주': '156', '부산': '159', '제주': '184',
}

# 2. 웹사이트 화면 구성
st.title("🌡️ 기상청 데이터 기반 미래 기온 예측")
st.write("최근 10년(2015~2024) 8월 1일 실제 기상청 데이터를 바탕으로 미래 기온을 시뮬레이션합니다.")

col1, col2 = st.columns(2)
with col1:
    city = st.selectbox("분석할 도시를 선택하세요", list(STATIONS.keys()))
with col2:
    target_year = st.number_input("예측할 미래 연도", min_value=2025, max_value=2100, value=2050, step=1)

# 3. 예측 실행 버튼
if st.button("기온 예측 실행하기", type="primary", use_container_width=True):
    stn_id = STATIONS[city]
    base_url = 'https://apis.data.go.kr/1360000/AsosDalyInfoService/getWthrDataList'
    years, temps = [], []

    progress_bar = st.progress(0, text="기상청 서버에서 데이터를 불러오는 중...")

    for idx, y in enumerate(range(2015, 2025)):
        params = {
            'serviceKey': MY_API_KEY, 'pageNo': '1', 'numOfRows': '1',
            'dataType': 'JSON', 'dataCd': 'ASOS', 'dateCd': 'DAY',
            'startDt': f'{y}0801', 'endDt': f'{y}0801', 'stnIds': stn_id,
        }
        full_url = f"{base_url}?{urllib.parse.urlencode(params)}"
        req = urllib.request.Request(full_url, headers={'User-Agent': 'Mozilla/5.0'})

        for attempt in range(3):
            try:
                with urllib.request.urlopen(req, timeout=10) as response:
                    res = json.loads(response.read().decode('utf-8'))
                    if res['response']['header']['resultCode'] == '00':
                        avg_temp = float(res['response']['body']['items']['item'][0]['avgTa'])
                        years.append(y)
                        temps.append(avg_temp)
                        break
            except Exception:
                pass
            time.sleep(0.3)

        progress_bar.progress((idx + 1) / 10, text=f"{y}년 데이터 수집 완료 ({idx + 1}/10)")
        time.sleep(0.1)

    progress_bar.empty()

    if len(years) < 2:
        st.error("데이터를 충분히 불러오지 못했습니다. 잠시 후 다시 시도해주세요.")
    else:
        years = np.array(years)
        temps = np.array(temps)

        slope, intercept = np.polyfit(years, temps, 1)
        predicted_temp = slope * target_year + intercept
        temp_diff = predicted_temp - temps[-1]

        st.divider()
        m1, m2 = st.columns(2)
        m1.metric(f"{target_year}년 8월 1일 예상 기온", f"{predicted_temp:.2f} ℃")
        m2.metric(f"{years[-1]}년({temps[-1]}℃) 대비 변화", f"{temp_diff:+.2f} ℃")

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(years, temps, marker='o', linewidth=2, markersize=7, color='#1f77b4',
                label=f'{city} 실제 기온 ({years[0]}~{years[-1]}년 8월 1일)')

        all_years = np.arange(years[0], target_year + 1)
        all_trend = slope * all_years + intercept
        ax.plot(all_years, all_trend, linestyle='--', linewidth=1.8, color='#d62728', alpha=0.8,
                label=f'기온 변화 추세선 (연평균 {slope:+.2f}℃)')

        ax.scatter(target_year, predicted_temp, color='#d62728', s=180, zorder=5, edgecolors='black')

        y_min, y_max = min(np.min(temps), predicted_temp), max(np.max(temps), predicted_temp)
        y_margin = (y_max - y_min) * 0.25 if y_max != y_min else 2.0
        ax.set_ylim(y_min - y_margin, y_max + y_margin)
        ax.set_xlim(years[0] - 1, target_year + max(2, int((target_year - years[0]) * 0.08)))

        ax.annotate(
            f'{target_year}년 예측\n{predicted_temp:.1f}℃',
            xy=(target_year, predicted_temp), xytext=(0, 14), textcoords='offset points',
            ha='center', fontweight='bold', fontsize=11, color='#d62728',
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#d62728', alpha=0.9)
        )

        ax.xaxis.get_major_locator().set_params(integer=True)
        ax.set_title(f'[{city}] {target_year}년 8월 1일 기온 예측 시뮬레이션', fontsize=14, fontweight='bold', pad=15)
        ax.set_xlabel('연도')
        ax.set_ylabel('평균 기온 (℃)')
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend(loc='best')
        plt.tight_layout()

        st.pyplot(fig)
