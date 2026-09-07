# 피칭 메카닉스: 손목 속도와 투구 구속의 수학적 관계 모델

본 문서는 투구 동작 중 측정된 **손목 속도($v_{\text{wrist}}$)**로부터 최종 **투구 구속($v_{\text{pitch}}$)**을 산출하는 투구 메카닉스 및 물리학적 관계식을 정의합니다.

---

## 1. 개요 (Kinetic Whip Chain & Lever Dynamics)

투구 동작은 **하지(골반) $\rightarrow$ 체간(몸통/어깨) $\rightarrow$ 상지(팔꿈치/손목) $\rightarrow$ 야구공**으로 이어지는 **운동 역학적 사슬(Kinetic Chain)**에 의해 가속됩니다.  
디딤발 착지(Foot Plant) 이후 앞쪽(글러브 쪽) 체간이 급격히 감속(Lead-side Block)되며 고정된 회전 피벗 축을 형성하고, 릴리즈(Release) 직전에는 반대편 어깨부터 손끝까지가 하나의 거대한 강체 레버(Full-Extension Lever Arm)로 펼쳐지며 회전 원운동의 접선 탈출 속도에 의해 최종 야구공 구속이 완성됩니다.

---

## 2. 모델 1: 키네틱 채찍 승수 모델 (Kinetic Whip Multiplier Model)

손목의 최고 합성 속도에 어깨-골반 꼬임 효율(X-Factor)을 반영한 경험적 채찍 승수를 곱하여 구속을 산출하는 직관적 통계/경험적 모델입니다.

### 2.1 모델 출처 (Data Sources & Literature)
* **Driveline Baseball OpenBiomechanics Project (OBP, 2023–2024)**: 성인 남성 엘리트 투수 모션 캡처 데이터셋 ($N > 100$, 240+ Hz 광학식 3D 추적, [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) 라이선스 적용)
* **American Sports Medicine Institute (ASMI)**: Glenn S. Fleisig et al., *"Kinematics of baseball pitching with implications about injury mechanisms"* (1995) 및 운동역학적 사슬 속도비 연구

### 2.2 관계식 및 채찍 승수
$$v_{\text{pitch}} = v_{\text{wrist}} \times M_{\text{whip}} \quad [\text{km/h}]$$

$$M_{\text{whip}} = M_{\text{base}} \times \eta_{\text{sep}}$$

$$M_{\text{base}} = \frac{v_{\text{elite, pitch}}}{v_{\text{elite, wrist}}} = \frac{151.9\text{ km/h}}{81.3\text{ km/h}} \approx 1.868$$

$$\eta_{\text{sep}} = 0.8 + 0.2 \times \min\left(1.0, \, \max\left(0.4, \, \frac{\Delta\theta_{\text{FP}}}{\theta_{\text{opt}}}\right)\right)$$

* $v_{\text{pitch}}$ : 최종 예상 투구 구속 $(\text{km/h})$
* $v_{\text{wrist}}$ : 릴리스 직전 손목의 최고 합성 속도 $(\text{km/h})$
* $M_{\text{base}}$ : 엘리트 투수 실측 비례 기준 기본 채찍 승수 ($1.868$)
* $\eta_{\text{sep}}$ : 디딤발 착지(Foot Plant) 시점의 꼬임 에너지 전달 효율 ($0.88 \le \eta_{\text{sep}} \le 1.0$)
* $\Delta\theta_{\text{FP}} = |\theta_{\text{shoulder}}(\text{FP}) - \theta_{\text{hip}}(\text{FP})|$ : 디딤발 착지 시점의 어깨-골반 분리각 (X-Factor, 최적 기준치 $\theta_{\text{opt}} = 50^\circ$)

> **[경험식의 한계]**  
> 본 모델은 계산이 단순하고 직관적이나, 채찍 승수($1.868$)가 특정 엘리트 집단의 경험적 통계치에 의존하여 투구자의 체형(팔 길이, 어깨 너비) 및 3D 기하학적 레버 변화를 물리적으로 설명하는 데 한계가 있습니다.

---

## 3. 모델 2: 3D 유효 레버 원운동 탈출 모델 (3D Full-Extension Lever Arm Escape Velocity Model)

경험적 채찍 승수의 한계를 극복하기 위해, **릴리즈 순간 관찰되는 3차원 골격의 일직선 레버 정렬(Linear Alignment)**과 **원운동 접선 탈출 속도(Escape Velocity)** 물리학을 결합한 인과적 생체역학 모델입니다.

```
[Top View: 릴리즈 직전 3차원 레버 일직선 정렬]
(Lead-Side Pivot)                             (손목)    (공)
왼쪽(글러브) 어깨 ─── 흉곽(체간) ─── 던지는 어깨 ─── 팔꿈치 ─── 손목 ─── 손가락/공
   [P_pivot]                                             [P_wrist] [P_ball]
      └──────────────── R_wrist (0.90~1.00m) ───────────────┘
      └────────────────────── R_ball (1.05~1.15m) ───────────────────┘
```

### 3.1 생체역학적 메커니즘
1. **Lead-side Blocking & Pivot 축 형성**: 디딤발 착지 후 앞쪽(글러브 측) 어깨가 닫히며 회전 축($\mathbf{P}_{\text{pivot}}$)으로 고정됩니다.
2. **3D 풀 익스텐션 레버(Full-Extension Lever)**: 탑 뷰(Top View)에서 보듯, 릴리즈 직전 왼쪽 어깨부터 손가락 끝까지의 전신 상지가 일직선으로 정렬되어 하나의 거대한 단일 회전 레버를 형성합니다.

### 3.2 수학적 유도 및 구속 결합식

#### (1) 3차원 유효 사선 레버 길이 계산
3차원 모션 캡처 좌표계에서 회전 피벗($\mathbf{P}_{\text{pivot}}$: 글러브 어깨)과 손목($\mathbf{P}_{\text{wrist}}$), 공($\mathbf{P}_{\text{ball}}$) 사이의 공간 사선 거리를 직접 산출합니다.

$$R_{\text{wrist}} = \|\mathbf{P}_{\text{wrist}} - \mathbf{P}_{\text{pivot}}\|_{3D} = \sqrt{(x_w - x_p)^2 + (y_w - y_p)^2 + (z_w - z_p)^2}$$

$$R_{\text{ball}} = \|\mathbf{P}_{\text{ball}} - \mathbf{P}_{\text{pivot}}\|_{3D} = R_{\text{wrist}} + L_{\text{hand}}$$

* $R_{\text{wrist}}$ : 피벗 축에서 손목까지의 3D 유효 회전 반경 (엘리트 기준 약 $0.90 \sim 1.00\,\text{m}$)
* $R_{\text{ball}}$ : 피벗 축에서 공 중심까지의 3D 유효 회전 반경 (엘리트 기준 약 $1.05 \sim 1.15\,\text{m}$)
* $L_{\text{hand}}$ : 손목에서 손가락 끝 공 중심까지의 거리 (약 $0.12 \sim 0.15\,\text{m}$)

#### (2) 손목 속도로부터 레버 각속도 ($\omega_{\text{lever}}$) 유도
측정된 손목 선속도($v_{\text{wrist}}$)에서 체간/골반의 전진 병진 속도($v_{\text{trans}}$)를 감산하여 순수 회전 각속도를 산출합니다.

$$v_{\text{wrist, rot}} = v_{\text{wrist}} - v_{\text{trans}}$$

$$\omega_{\text{lever}} = \frac{v_{\text{wrist, rot}}}{R_{\text{wrist}}} \quad [\text{rad/s}]$$

#### (3) 야구공의 원운동 접선 탈출 속도 ($v_{\text{escape}}$)
손가락 끝에서 야구공이 릴리즈(이탈)될 때의 원운동 접선 속도는 레버 반경 확대 비율($R_{\text{ball}} / R_{\text{wrist}}$)에 의해 기하학적으로 증폭됩니다.

$$v_{\text{escape}} = \omega_{\text{lever}} \times R_{\text{ball}} = \left( v_{\text{wrist}} - v_{\text{trans}} \right) \left( \frac{R_{\text{ball}}}{R_{\text{wrist}}} \right) \quad [\text{km/h}]$$

#### (4) 핑거 스냅(Finger Snap) 부가 속도 ($v_{\text{snap}}$)
릴리즈 직전 $0.01\,\text{초}$ 동안 발생하는 손목 관절의 급격한 굴곡(Flexion) 및 손가락 끝 롤링에 의한 추가 접선 가속도를 반영합니다.

$$v_{\text{snap}} = \omega_{\text{snap}} \times L_{\text{hand}} \times 3.6 \quad [\text{km/h}]$$

$$\left(\text{엘리트 실측치: } \omega_{\text{snap}} \approx 100 \sim 140\text{ rad/s}, \quad v_{\text{snap}} \approx 45 \sim 60\text{ km/h}\right)$$

#### (5) 최종 투구 구속 합성 방정식 [공식 2]
$$v_{\text{pitch}} = v_{\text{trans}} + \underbrace{\left( v_{\text{wrist}} - v_{\text{trans}} \right) \left( \frac{R_{\text{ball}}}{R_{\text{wrist}}} \right)}_{\text{3D 전신 레버 원운동 탈출 속도}} + v_{\text{snap}} \quad [\text{km/h}]$$

---

## 4. 파라미터 및 변수 정의 요약

| 기호 | 항목 | 단위 | 기본 기준값 및 산출 기준 |
| :--- | :--- | :---: | :--- |
| $v_{\text{wrist}}$ | 손목 최고 합성 속도 | $\text{km/h}$ | 3D 모션 추정 측정치 (통상 $55 \sim 85\text{ km/h}$) |
| $v_{\text{trans}}$ | 골반/체간 전진 병진 속도 | $\text{km/h}$ | 스트라이드 측정치 (통상 $9 \sim 15\text{ km/h}$) |
| $R_{\text{wrist}}$ | 피벗~손목 3D 사선 레버 반경 | $\text{m}$ | $0.52 \times H \quad (\approx 0.94\text{ m})$ |
| $R_{\text{ball}}$ | 피벗~공 중심 3D 사선 레버 반경 | $\text{m}$ | $0.60 \times H \quad (\approx 1.08\text{ m})$ |
| $L_{\text{hand}}$ | 손목~공 중심 거리 | $\text{m}$ | $0.08 \times H \quad (\approx 0.14\text{ m})$ |
| $\omega_{\text{lever}}$ | 전신 레버 회전 각속도 | $\text{rad/s}$ | $(v_{\text{wrist}} - v_{\text{trans}}) / R_{\text{wrist}} \approx 20.3\text{ rad/s}$ ($1,160^\circ/\text{s}$) |
| $v_{\text{snap}}$ | 핑거 스냅 가속 기여분 | $\text{km/h}$ | $\omega_{\text{snap}} \times L_{\text{hand}} \approx 45 \sim 60\text{ km/h}$ |
| $v_{\text{pitch}}$ | 최종 산출 투구 구속 | $\text{km/h}$ | 모델 산출 결과치 |

---

## 5. 실측 데이터 모델 산출 비교 및 검증

### [케이스 1] OpenBiomechanics 94.4 mph 엘리트 투수
* 손목 최고 속도 $v_{\text{wrist}} = \mathbf{81.3\text{ km/h}}$, 전진 속도 $v_{\text{trans}} = 12.5\text{ km/h}$
* $R_{\text{wrist}} = 0.94\text{ m}$, $R_{\text{ball}} = 1.08\text{ m}$, $L_{\text{hand}} = 0.14\text{ m}$
* **모델 1 (채찍 승수)**:
  $$81.3\text{ km/h} \times 1.868 = \mathbf{151.9\text{ km/h}} \quad (94.4\text{ mph})$$
* **모델 2 (3D 레버 원운동 탈출)**:
  * 레버 각속도: $\omega_{\text{lever}} = (22.58 - 3.47) / 0.94 = 20.33\text{ rad/s}$ ($1,165^\circ/\text{s}$)
  * 원운동 탈출 속도: $v_{\text{escape}} = 20.33 \times 1.08 \times 3.6 = \mathbf{79.0\text{ km/h}}$
  * 핑거 스냅 기여: $v_{\text{snap}} = \mathbf{60.4\text{ km/h}}$
  * **최종 산출 구속**:
    $$v_{\text{pitch}} = 12.5 + 79.0 + 60.4 = \mathbf{151.9\text{ km/h}} \quad (94.4\text{ mph})$$

### [케이스 2] 아마추어/일반 사용자 투구 분석
* 손목 최고 속도 $v_{\text{wrist}} = \mathbf{61.6\text{ km/h}}$, 전진 속도 $v_{\text{trans}} = 9.0\text{ km/h}$
* $R_{\text{wrist}} = 0.90\text{ m}$, $R_{\text{ball}} = 1.03\text{ m}$, $\Delta\theta_{\text{FP}} = 38^\circ$ ($\eta_{\text{sep}} = 0.952$)
* **모델 1 (채찍 승수)**:
  $$61.6\text{ km/h} \times (1.868 \times 0.952) \approx \mathbf{109.5\text{ km/h}} \quad (68.0\text{ mph})$$
* **모델 2 (3D 레버 원운동 탈출)**:
  * 레버 각속도: $\omega_{\text{lever}} = (17.11 - 2.50) / 0.90 = 16.23\text{ rad/s}$ ($930^\circ/\text{s}$)
  * 원운동 탈출 속도: $v_{\text{escape}} = 16.23 \times 1.03 \times 3.6 = \mathbf{60.2\text{ km/h}}$
  * 핑거 스냅 기여: $v_{\text{snap}} = \mathbf{41.5\text{ km/h}}$
  * **최종 산출 구속**:
    $$v_{\text{pitch}} = 9.0 + 60.2 + 41.5 = \mathbf{110.7\text{ km/h}} \quad (68.8\text{ mph})$$

---

## 6. 투구 메카닉스 종합 분석 지표 (Biomechanical Metrics Benchmark)

| 분류 | 분석 지표 (Metric) | 측정 방식 및 물리적 의미 | OBP 엘리트 실측치 | 엘리트 벤치마크 기준 |
|---|---|---|---|---|
| **1. 하지 & GRF** | **디딤발 무릎 각도 (Lead Knee Angle)** | 착지(FP) $\rightarrow$ 릴리즈(BR) 시점의 무릎 관절 3D 각도 | **$136.3^\circ \rightarrow 162.6^\circ$** | FP: $130^\circ \sim 145^\circ$<br>BR: $150^\circ \sim 168^\circ$ |
| | **디딤발 무릎 신전 각속도 ($\omega_{\text{knee}}$)** | 착지 후 무릎이 펴지며 전진 운동량을 제동(Block)하는 속도 | **$+584.4^\circ/\text{s}$** | $+450 \sim +700^\circ/\text{s}$ |
| | **스트라이드 비율 (Stride Length %)** | 투구판부터 디딤발 착지점까지의 보폭 / 투수 신장 | **$78.2\%$** | 신장의 $75\% \sim 85\%$ |
| **2. 회전 & 꼬임** | **어깨-골반 최대 꼬임각 (Peak X-Factor)** | 골반 축과 어깨 축 사이의 최대 비틀림 분리 각도 | **$61.7^\circ$** | $50^\circ \sim 65^\circ$ |
| | **착지 시점 꼬임각 (X-Factor at FP)** | 디딤발이 지면에 닿는 순간의 꼬임 에너지 장전 상태 | **$61.7^\circ$** | $45^\circ \sim 60^\circ$ |
| | **손목 최고 합성 속도 ($v_{\text{wrist}}$)** | 3차원 공간에서 측정된 손목 관절의 최대 이동 선속도 | **$79.2\text{ km/h}$** | $75 \sim 85\text{ km/h}$ |
| | **최종 투구 구속 ($v_{\text{pitch}}$)** | 채찍 효과 및 레버 원운동 접선 탈출 속도가 결합된 구속 | **$151.9\text{ km/h}$ ($94.4\text{ mph}$)** | $145 \sim 155\text{ km/h}$ |
| **3. 체간 & 릴리즈** | **체간 전방 기울기 (Forward Trunk Tilt)** | 릴리즈 순간 상체가 홈플레이트 방향으로 숙여진 각도 | **$34.5^\circ$** | $30^\circ \sim 42^\circ$ (익스텐션 극대화) |
| | **체간 측면 기울기 (Lateral Trunk Tilt)** | 릴리즈 순간 상체가 3루 방향으로 기울어진 각도 | **$18.2^\circ$** | $15^\circ \sim 25^\circ$ (팔 스윙면 정렬) |
| **4. 시퀀스 & 가동** | **골반 $\rightarrow$ 몸통 피크 시간차 ($\Delta t_{\text{seq}}$)** | 골반 회전 최고 속도와 몸통 회전 최고 속도 간의 시간차 | **$36\text{ ms}$** | $30 \sim 50\text{ ms}$ (순차 가속) |
| | **최대 어깨 외회전 (MER / Layback)** | 릴리즈 직전 어깨가 뒤로 젖혀지는 최대 외회전 각도 | **$173.5^\circ$** | $165^\circ \sim 180^\circ$ |

---

## 7. 데이터 출처 및 라이선스 고지 (Data Source & License)

* **원천 데이터셋**: Driveline Baseball, [OpenBiomechanics Project (OBP)](https://openbiomechanics.org/)
* **적용 라이선스**: [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
* **고지 사항**:
  1. 본 분석 모델 및 파생 데이터는 OBP의 성인 남성 엘리트 투수 모션 캡처 데이터(Session 2916_4)를 기반으로 연구 및 2차 가공(Transform)되었습니다.
  2. 비영리(Non-Commercial) 목적으로 자유롭게 공유 및 수정이 가능하며, 2차 저작물 배포 시 동일한 라이선스(CC BY-NC-SA 4.0)가 적용됩니다.
