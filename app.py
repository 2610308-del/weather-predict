import json
import platform
import time
import urllib.parse
import urllib.request
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

# 웹페이지 기본 설정
st.set_page_config(page_title="AI 기반 전국 미래 기온 예측", page_icon="🌡️", layout="centered")

# 한글 폰트 설정
os_name = platform.system()
if os_name == 'Windows':
    plt.rc('font', family='Malgun Gothic')
elif os_name == 'Darwin':
    plt.rc('font', family='AppleGothic')
else:
    plt.rc('font', family='NanumGothic')
plt.rcParams['axes.unicode_minus'] = False

MY_API_KEY = '404e80f5bbda1aa9285ba87f66e0e384246caa1dbe9a7b4c14d9a1a598c81239'
STATIONS = {
    '서울': '108', '인천': '112', '수원': '119', '대전': '133',
    '대구': '143', '광주': '156', '부산': '159', '제주': '184',
}

st.title("🌡️ AI 기반 대한민국 미래 여름 기온 예측")
st.write("기상청 최근 15년(2010~2024) 8월 전체 실제 관측 데이터를 머신러닝(AI) 모델에 학습시켜 미래 여름 기온을 예측합니다.")

col1, col2, col3 = st.columns([1, 1, 1.2])
with col1:
    city = st.selectbox("분석할 도시", list(STATIONS.keys()))
with col2:
    target_year = st.number_input("예측 미래 연도", min_value=2025, max_value=2100, value=2040, step=1)
with col3:
    ai_model_type = st.selectbox("AI 예측 모델 선택", ["AI 다항 회귀 (온난화 가속 반영)", "AI 선형 회귀 (안정적 추세)"])

if st.button("🤖 AI 기온 예측 실행하기", type="primary", use_container_width=True):
    stn_id = STATIONS[city]
    base_url = 'https://apis.data.go.kr/1360000/AsosDalyInfoService/getWthrDataList'
    years, aug_avg_temps, aug_max_temps = [], [], []

    progress_bar = st.progress(0, text="기상청 서버에서 15년치 8월 전체 데이터를 수집 중...")

    # 하루가 아닌 8월 전체(31일) 데이터를 수집하여 이상 기후 노이즈 제거 (2010~2024년)
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
                        daily_maxs = [float(it['maxTa']) for it in items if it.get('maxTa')]
                        if daily_avgs:
                            years.append(y)
                            aug_avg_temps.append(np.mean(daily_avgs))
                            aug_max_temps.append(np.mean(daily_maxs))
                        break
            except Exception:
                pass
            time.sleep(0.2)

        progress_bar.progress((idx + 1) / 15, text=f"{y}년 8월 전체 데이터 학습 준비 완료 ({idx + 1}/15)")

    progress_bar.empty()

    if len(years) < 5:
        st.error("데이터를 충분히 불러오지 못했습니다. 잠시 후 다시 시도해주세요.")
    else:
        X = np.array(years).reshape(-1, 1)
        y_avg = np.array(aug_avg_temps)
        y_max = np.array(aug_max_temps)

        # AI 머신러닝 모델 학습 (Scikit-Learn)
        degree = 2 if "다항" in ai_model_type else 1
        model_avg = make_pipeline(PolynomialFeatures(degree), Ridge(alpha=1.0))
        model_max = make_pipeline(PolynomialFeatures(degree), Ridge(alpha=1.0))

        model_avg.fit(X, y_avg)
        model_max.fit(X, y_max)

        # 미래 예측
        future_X = np.arange(years[0], target_year + 1).reshape(-1, 1)
        pred_avg_curve = model_avg.predict(future_X)
        pred_max_curve = model_max.predict(future_X)

        pred_target_avg = float(model_avg.predict([[target_year]])[0])
        pred_target_max = float(model_max.predict([[target_year]])[0])

        past_mean = float(np.mean(y_avg))
        diff_from_past_mean = pred_target_avg - past_mean

        # 결과 지표 출력 (과거 15년 평균 대비 변화량으로 비교)
        st.divider()
        m1, m2, m3 = st.columns(3)
        m1.metric(f"{target_year}년 8월 예상 평균기온", f"{pred_target_avg:.2f} ℃", f"{diff_from_past_mean:+.2f} ℃ (15년 평균 대비)")
        m2.metric(f"{target_year}년 8월 예상 낮 최고기온", f"{pred_target_max:.2f} ℃")
        m3.metric(f"과거 15년 8월 평균기온", f"{past_mean:.2f} ℃")

        # 그래프 시각화
        fig, ax = plt.subplots(figsize=(10, 5.2))
        ax.plot(years, y_avg, marker='o', linewidth=2, color='#1f77b4', label=f'{city} 실제 8월 평균기온 (2010~2024)')
        ax.plot(future_X.flatten(), pred_avg_curve, linestyle='--', linewidth=2.2, color='#d62728', label=f'AI 예측 추세 곡선 ({ai_model_type.split()[1]})')
        ax.axhline(past_mean, color='gray', linestyle=':', alpha=0.7, label=f'과거 15년 평균 ({past_mean:.1f}℃)')

        ax.scatter(target_year, pred_target_avg, color='#d62728', s=180, zorder=5, edgecolors='black')

        y_min, y_max_val = min(np.min(y_avg), pred_target_avg), max(np.max(y_avg), pred_target_avg)
        margin = (y_max_val - y_min) * 0.25 if y_max_val != y_min else 1.5
        ax.set_ylim(y_min - margin, y_max_val + margin)
        ax.set_xlim(years[0] - 1, target_year + max(2, int((target_year - years[0]) * 0.08)))

        ax.annotate(
            f'AI 예측 ({target_year}년)\n평균 {pred_target_avg:.1f}℃\n(최고 {pred_target_max:.1f}℃)',
            xy=(target_year, pred_target_avg), xytext=(0, 14), textcoords='offset points',
            ha='center', fontweight='bold', fontsize=10, color='#d62728',
            bbox=dict(boxstyle='round,pad=0.35', fc='white', ec='#d62728', alpha=0.95)
        )

        ax.xaxis.get_major_locator().set_params(integer=True)
        ax.set_title(f'[{city}] AI 머신러닝 기반 {target_year}년 8월 기온 예측', fontsize=14, fontweight='bold', pad=15)
        ax.set_xlabel('연도')
        ax.set_ylabel('8월 평균 기온 (℃)')
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend(loc='best')
        plt.tight_layout()

        st.pyplot(fig)
        st.caption("※ 기상청 ASOS 일자료 8월 전체(31일) 평균 데이터를 Scikit-Learn 회귀 모델에 학습시킨 시뮬레이션 결과입니다.")
