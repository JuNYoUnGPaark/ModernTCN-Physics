# Physics Loss 관련 논문 정리

# 1. “고전적” 물리 정보 신경망 (실습용 only) 

- 핵심 아이디어: `Total_Loss = Loss_data + Loss_physics`
    - `Loss_data` : y(Target)과 y_pred의 차이 (현재 코드: `loss_ce`)
    - `Loss_physics` : y_pred를 미분하여 물리방정식 (ex: $f(\hat y, \frac {d\hat y}{dt})=0$)에 넣었을 때 그 결과가 0으로부터 얼마나 벗어나는지 ⇒ ‘Pure Physics Loss 형태’

***논문 1)"Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations" (Raissi, Perdikaris, & Karniadakis, 2019)***

- PINN 분야를 개척한 논문(필독). 딥러닝으로 어떻게 미분방정식을 풀고, 데이터 부족 영역을 물리 법칙으로 채워 넣을 수 있는지에 대한 idea 제공

**논문 2) *"Physics-guided deep learning: A review of recent advances and applications" (Willard et al., 2020/2022)***

- PINN을 포함하여 물리를 딥러닝에 통합하는 다양한 방법론 (loss function, architecture, hybrid)을 총 정리한 논문.


# IMU 센서의 물리적 특성 활용
- 지금의 물리손실은 “가속도/자이로를 그대로 회귀하여 L1으로 맞추는” 수준
- 다양한 HAR 데이터셋에 공통으로 사용 가능한 (`acc`, `gyro`)만 사용해서 사용가능한 물리손실 업그레이드

## 1. 보완필터/중력추정: `G_Estimate.ipynb`

### 1) 왜 중력을 추정해야할까?

- IMU 센서로 사람의 움직임을 인식할 때, 가속도 센서가 측정하는 값은 **순수한 움직임만이 아닌 중력까지 포함**됨.

<aside>
🔥

측정된 가속도 = 실제 몸의 움직임(`acc_body`) + 지구 중력(`acc_gravity`)

</aside>

- ex) 스마트폰을 가만히 세워둬도 가속도 센서 = (0, 0, 9.8)을 측정
- 따라서 순수한 움직임만 보고 싶다면, 현재 중력이 어느 방향으로 작용하는지를 알아야 가능하다.

### 2) 가속도계와 자이로스코프의 특징

- **Accelerometer**
    - 장점: 중력 방향을 직접적으로 감지(기울기를 알 수 있다) → **정적일 때만 기울기를 정확히 알려준다.**
        - ex) 스마트폰을 눞이면 z축 값이 줄고 x,y축 값이 생김.
    - 단점: **움직임이 섞이면 중력과 가속도를 구분하기 어렵다.**
- **Gyroscope**
    - 장점: **회전 속도를 매우 정확히 예측** → 각속도를 t에 대해서 적분하면 각도가 나온다.
    - 단점: 시간이 지나면 **오차(drift)가 누적되어 방향이 틀어진다.**

⇒ acc는 느리게 변하는 ‘기울기’정보 강함 + gyro는 빠른 감지에 강함 (상호 보완 가능)

### 3) 보완 필터의 핵심 아이디어

<aside>
🔥

가속도는 저주파(느린 변화), 자이로는 고주파(빠른 변화) 정보를 적극 활용하자. 

</aside>

- acc는 Low-pass filter로 중력 방향만 남기고,
- gyro는 High-pass filter로 회전 성분만 남긴다.

- 이 두 정보를 가중 평균처럼 합치면 안정적이고 지속적인 중력 방향을 얻을 수 있다.

$$
g\_estimate = \alpha * (gyro\_gradient)+(1-\alpha)*(acc\_gradient)
$$

- acc는 “중력 방향 측정”, gyro 적분은 “방향 예측 역할

### 4) 요약

```css
[자이로만 쓰면]
  ↳ 처음엔 정확하지만 시간이 지나면 점점 틀어짐

[가속도만 쓰면]
  ↳ 정지 상태에서는 정확하지만, 움직일 때 막 흔들림

[보완필터]
  ↳ 자이로 예측을 따르되,
     천천히 변하는 가속도 중력 성분으로 방향을 꾸준히 바로잡음
```

### 5) 적용 방법

- **L_grav**: acc에서 low-pass filter를 통과해서 얻은 중력 방향 = 모델이 예측한 중력 방향(`g_pred`)
- **L_comp**: gyro가 예측한 방향 + 가속도로 고정된 방향 = 모델이 예측한 방향
- **acc_gating**: 특정 조건일 때만 acc 관련 제약을 킨다. → acc를 믿을 수 있을 때만 ON 해주는 스위치 역할
    - ex) 회전, 가속이 거의 없을 때 게이트 ON
- **정적 구간만 acc로** 자세/중력을 믿고, **동적 구간에선 주로 gyro**(적분)을 믿자.
