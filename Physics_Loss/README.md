# Physics Loss 관련 논문 정리

# “고전적” 물리 정보 신경망 (실습용 only) 

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
가속도는 저주파(느린 변화), 자이로는 고주파(빠른 변화) 정보를 적극 활용하자. 
</aside>

- acc는 Low-pass filter로 중력 방향만 남기고,
- gyro는 High-pass filter로 회전 성분만 남긴다.

- 이 두 정보를 가중 평균처럼 합치면 안정적이고 지속적인 중력 방향을 얻을 수 있다.

$$
g\\_estimate = \alpha * (gyro\\_gradient)+(1-\alpha)*(acc\\_gradient)
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
---

1. gravity_head 추가

```python
class PhysicsModernTCNHAR(BaseModernTCNHAR):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        hidden_dim = self.head.in_features
        # 기존 physics_head 유지(원시 acc/gyro 회귀가 필요하면 사용)
        self.physics_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 6)
        )
        # ✅ 중력 방향 예측 헤드 추가 (타임스텝별 3D, 후단에서 unit_norm)
        self.gravity_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 3)
        )
```

- 작은 MLP를 붙여 중력방향을 예측

---

1. 사전 준비
    1. 가속도 분리(`a_lp`, `a_hp`)
        1. `a_lp` : “중력 방향” 후보
        2. `a_hp` : 사용자의 순수 움직임
    2. `gate`
        1. 사람이 가만히 있거나 천천히 움직일 때만 1인 되는 스위치
    
2. `L_grav` : 방향 정렬 손실
    1. 모델이 예측한 중력방향과 센서의 중력 방향이 일치해야한다. 
        1. `a_lp` 의 방향 `g_from_acc` 를 구한다.
        2. 모델이 예측한 중력방향(`g_pred`)와 `g_from_acc` 사이의 각도를 잰다.
        3. 이 각도가 클수록 손실이 커진다. 

1. `L_gmag` : 중력 크기 손실
    1. 사람이 가만히 있을 대 총 가속도의 크기는 1.0이어야 한다. (정규화 기준)
        1. 조용할 때 스위치(`gate`)가 켜진 순간만 이용
        2. 전체 가속도(`acc`)의 크기가 1.0과 얼마나 차이나는지 계산한다. 

1. `L_comp` : 보완 필터 손실
    1. 보완 필터가 계산한 중력 방향과 모델의 예측이 일치해야한다.
        1. 자이로 기반 예측 `g_gyro` : 이전 중력 방향을 현재 회전 `gyro` 값으로 돌려서 다음 중력 방향을 예측한다.  
        2. 가속도 기반 예측 `g_acc` : 현재 가속도 방향을 중력으로 본다.
        3. . `g_comp` : 위 두 값을 특정 비율로 섞어서 보완필터의 중력 방향을 만든다. 
        4.  모델 예측 `g_pred` 와 `g_comp` 의 각도 차이를 손실로 본다. 

1. `L_bias` : 자이로 바이어스 손실
    1. 사람이 가만히 있을 때, 자이로 센서의 평균값은 0이 되어야한다. 
        1. 조용할 때 스위치 (`gate`)가 켜진 순간만 본다.
        2.  자이로의 값의 평균(`gyro_m`)이 0이 아니면 손실을 준다. 

1. `L_smooth` : 스무딩 손실
    1. 센서 신호는 부드럽게 변해야 한다. 
        1. 가속도의 순간 변화량(`da`)와 각속도의 순간 변화량(`dw`)를 계산한다.
        2.  변화량이 크면 손실을 준다. 
    2. 한마디로 물리적으로 불가능한 순간이동 같은 변화를 예측하지 않도록 신호를 매끄럽게 만드는 것. 

1. `L_split` : 분해 일관성 손실
    1. 총 가속도 = 순수 신체 움직임 + 중력 가속도. 이때 순수 신체 움직임의 단기 평균은 0에 가까워야 한다. 
        1. `a_body = acc - g0 * g_pred` 의 단기 평균 (`a_body_m`)을 계산한다.
        2.  이 평균값이 0이 아니면 손실을 준다. 

1. `L_pinn` : 미분 운동학 손실
    1. 중력 방향의 변화율(dg/dt)은 자이로 회전과 정확히 일치해야한다. 
        1. `dg` : 모델이 예측한 중력(`g_pred`)의 시간에 따른 변화율을 구한다.
        2.  `w_cross_g` : 현재 회전(`gyro`)과 현재 중력(`g_pred`)을 외적하여 회전으로 인해 중력 방향이 얼마나 변해야 하는지 물리 법칙으로 계산한다.
        3.   `dg` 와 `w_cross_g` 는 부호만 다르고 같아야한다.
        4.  이 둘의 차이가 0이 아니면 손실을 준다. 

- 최종 결합

`L = (가중치1 * L_grav) + (가중치2 * L_gmag) + ... + (가중치7 * L_pinn)`

- 최종 손실

`Total_Loss = loss_ce + (lambda_phys * L)`

---
