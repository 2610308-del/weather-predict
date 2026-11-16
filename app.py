import json
import platform
import time
import urllib.error
import urllib.parse
import urllib.request
import matplotlib.pyplot as plt
import numpy as np

# 1. 인증키 (Decoding 키 사용 - 따옴표 유지 필수)
MY_API_KEY = '404e80f5bbda1aa9285ba87f66e0e384246caa1dbe9a7b4c14d9a1a598c81239'

# 2. 기상청 ASOS 전국 주요 관측 지점 코드
STATIONS = {
    '서울': '108',
    '인천': '112',
    '수원': '119',
    '대전': '133',
    '대구': '143',
    '광주': '156',
    '부산': '159',
    '제주': '184',
}

print('=== 기상청 실제 데이터 기반 전국 미래 기온 예측 프로그램 ===')
print('선택 가능한 지역:', ', '.join(STATIONS.keys()))

while True:
    city = input('분석할 도시 이름을 입력하세요 (예: 서울, 부산, 제주): ').strip()
    if city in STATIONS:
        stn_id = STATIONS[city]
        break
    print('목록에 없는 지역입니다. 위 제시된 도시 이름 중에서 정확히 입력해주세요.')

target_year = int(input('예측하고 싶은 미래 연도를 입력하세요 (예: 2030, 2050): '))

# 3. 기상청 API 호출 (일시적 서버 오류 시 자동 재시도 기능 포함)
base_url = 'https://apis.data.go.kr/1360000/AsosDalyInfoService/getWthrDataList'
years = []
temps = []

print(f'\n기상청 서버에서 [{city}] 지역의 최근 10년 실제 기온 데이터를 불러오는 중입니다...')

for y in range(2015, 2025):
    params = {
        'serviceKey': MY_API_KEY,
        'pageNo': '1',
        'numOfRows': '1',
        'dataType': 'JSON',
        'dataCd': 'ASOS',
        'dateCd': 'DAY',
        'startDt': f'{y}0801',
        'endDt': f'{y}0801',
        'stnIds': stn_id,
    }
    
    query_string = urllib.parse.urlencode(params)
    full_url = f"{base_url}?{query_string}"
    
    req = urllib.request.Request(
        full_url,
        headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    )
    
    # 연도별 최대 3번까지 재시도
    success = False
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=10) as response:
                raw_data = response.read().decode('utf-8')
                res = json.loads(raw_data)
                
                # 응답 코드가 정상('00')인지 확인
                result_code = res['response']['header']['resultCode']
                if result_code == '00':
                    avg_temp = float(res['response']['body']['items']['item'][0]['avgTa'])
                    years.append(y)
                    temps.append(avg_temp)
                    print(f" - {y}년 8월 1일 평균 기온: {avg_temp}℃ 수집 완료")
                    success = True
                    break
        except Exception:
            pass
        
        # 실패 시 0.5초 대기 후 재시도
        time.sleep(0.5)
    
    if not success:
        print(f" ! {y}년 데이터는 기상청 서버 일시 오류로 건너뜁니다.")
    
    # 연속 요청 부하 방지를 위한 짧은 대기
    time.sleep(0.15)

if len(years) < 2:
    print("\n데이터를 충분히 불러오지 못했습니다. 잠시 후 다시 시도해주세요.")
else:
    years = np.array(years)
    temps = np.array(temps)

    # 4. 상승 추세선 계산 및 예측
    slope, intercept = np.polyfit(years, temps, 1)
    predicted_temp = slope * target_year + intercept
    temp_diff = predicted_temp - temps[-1]

    print(f'\n[{target_year}년 8월 1일 {city} 기온 예측 결과]')
    print(f'- 예상 평균 기온: {predicted_temp:.2f}℃')
    print(f'- 최근 데이터({years[-1]}년, {temps[-1]}℃) 대비 약 {temp_diff:+.2f}℃ 변화 예상')

    # 5. 운영체제 자동 감지 후 한글 폰트 설정
    os_name = platform.system()
    if os_name == 'Windows':
        plt.rc('font', family='Malgun Gothic')
    elif os_name == 'Darwin':
        plt.rc('font', family='AppleGothic')
    plt.rcParams['axes.unicode_minus'] = False

    # 6. 그래프 그리기 (시각화 개선 버전)
    plt.figure(figsize=(11, 5.5))

    # 과거 실제 기온 선 + 점
    plt.plot(
        years,
        temps,
        marker='o',
        linewidth=2,
        markersize=7,
        color='#1f77b4',
        label=f'{city} 실제 기온 ({years[0]}~{years[-1]}년 8월 1일)',
    )

    # 과거~미래 전체 기간을 1년 단위로 연결한 추세선
    all_years = np.arange(years[0], target_year + 1)
    all_trend = slope * all_years + intercept
    plt.plot(
        all_years,
        all_trend,
        linestyle='--',
        linewidth=1.8,
        color='#d62728',
        alpha=0.8,
        label=f'기온 변화 추세선 (연평균 {slope:+.2f}℃)',
    )

    # 미래 예측 지점 강조
    plt.scatter(target_year, predicted_temp, color='#d62728', s=180, zorder=5, edgecolors='black')

    # 텍스트가 그래프 밖으로 잘리지 않도록 상하좌우 여백 자동 조정
    y_min = min(np.min(temps), predicted_temp)
    y_max = max(np.max(temps), predicted_temp)
    y_margin = (y_max - y_min) * 0.25 if y_max != y_min else 2.0
    plt.ylim(y_min - y_margin, y_max + y_margin)
    plt.xlim(years[0] - 1, target_year + max(2, int((target_year - years[0]) * 0.08)))

    plt.annotate(
        f'{target_year}년 예측\n{predicted_temp:.1f}℃',
        xy=(target_year, predicted_temp),
        xytext=(0, 14),
        textcoords='offset points',
        ha='center',
        fontweight='bold',
        fontsize=11,
        color='#d62728',
        bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#d62728', alpha=0.9)
    )

    # X축 연도를 정수로 깔끔하게 표시
    plt.gca().xaxis.get_major_locator().set_params(integer=True)

    plt.title(f'기상청 API 연동 [{city}] {target_year}년 8월 1일 기온 예측', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('연도', fontsize=11)
    plt.ylabel('평균 기온 (℃)', fontsize=11)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='best', frameon=True)
    plt.tight_layout()
    plt.show()
