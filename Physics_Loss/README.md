# Physics Loss 정리 
<img width="2182" height="3086" alt="image" src="https://github.com/user-attachments/assets/97ce1011-1cf6-4f98-a86c-a52d6214a473" />

## Why Physics Loss?

- DL 모델은 입력(sensor data)와 출력(Activiy label) 사이의 패턴, 상관관계를 찾는다.
- 하지만, 여기서 모델이 잘못된 정보로 학습할 수 있다.
    - “신호가 많이 흔들리면 무조건 Walking이다”
    - “Z축 acc가 1G 근처에서 안정적이면 무조건 Sitting이다”
- 여기서 `physics_loss` 를 적용한 모델은 활동을 정확히 분류하면서 모델이 보는 신호가 물리법칙에 어긋나는지를 해석한다.

1. 강력한 규제의 효과 발생 
    - 위의 예시처럼 모델이 훈련 데이터에 과적합 되거나 과도한 편향이 생기지 않도록 한다.
2. 일반화 성능 향상
    - 새로운 데이터에 대해서도 물리 법칙은 항상 동일하게 적용되기 때문에 보다 ‘보편적인 법칙’을 학습할 수 있다.
3. 특징 분리
    - 모델이 “이 신호는 중력 방향 때문”이고 “이 신호는 사용자 움직임 때문”이라는 것을 스스로 분리하도록 강제할 수 있다. 즉, 물리적으로 타당하고 올바른 특징을 기반으로 분류를 수행하도록 하는 장치가 되는 것.

## Codes

---

```python
def physics_loss_upgraded(
    X_raw,          # (B,T,9): [:,:,:3]=acc, [:,:,3:6]=gyro
    g_pred,         # (B,T,3): gravity unit vector (모델 예측)
    lambdas,        # dict: 각 항 가중치
    params          # dict: 하이퍼파라미터
):
    acc = X_raw[:, :, 0:3]
    gyro = X_raw[:, :, 3:6]
    eps = 1e-8
```

---

### 0-1. 필요한 함수 정의

1. Low-Pass Filter 함수

```python
def fir_lpf_hann_bt3(x, K=31):
    """
    x: (B,T,3)  -> a_lp, a_hp (둘 다 (B,T,3))
    간단 Hann 창 평균 기반 LPF. HPF = x - LPF
    """
    assert x.dim() == 3 and x.size(-1) == 3
    B, T, C = x.shape
    xc = x.transpose(1, 2)  # (B,3,T)
    w = torch.hann_window(K, dtype=xc.dtype, device=xc.device)
    w = (w / w.sum()).view(1,1,-1).expand(C,1,-1)  # (3,1,K)
    a_lp = F.conv1d(xc, w, padding=K//2, groups=C).transpose(1, 2)
    a_hp = x - a_lp
    return a_lp, a_hp
```

2. 미분 함수

```python
def diff1(x):
    # x: (B,T,D) -> same shape with zero-pad at t=0
    d = x[:, 1:] - x[:, :-1]
    pad = torch.zeros(x.size(0), 1, x.size(2), device=x.device, dtype=x.dtype)
    return torch.cat([pad, d], dim=1)
```

3. 정규화 함수

```python
def unit_norm(v, eps=1e-8):
    return v / (v.norm(dim=-1, keepdim=True) + eps)
```

4. 측정 간격 설정 함수 

```python
@torch.no_grad()
def estimate_dt_from_freq(default_dt=1/50):
    # UCI-HAR≈50Hz, MHEALTH≈50Hz, WISDM≈20Hz, PAMAP2≈100Hz
    return default_dt
```

---

### 0-2. data 준비

1. Acc를 LPF/HPF로 신호 분리

```python
		# 1) LPF로 중력후보, HPF로 동적성분
    a_lp, a_hp = fir_lpf_hann_bt3(acc, K=params.get('K', 31))
    g_from_acc = unit_norm(a_lp, eps=eps)
```

- `a_lp` (Low-Pass): 느리게 변하는 성분, “중력 방향”으로 사용한다.
- `a_hp` (High-Pass): 빠르게 변하는 성분, “순수 사용자의 움직임”으로 사용한다.
    - 회전도 없으면서 (high-pass를 통과시켜서 얻어낸) 움직임도 없을때가 정적구간이라고 판단하는 gate를 만들기 위해서 high-pass를 통과
- `g_from_acc`: `a_lp` 의 크기를 1로 정규화하여 “방향”정보만 뽑아낸다.

2. 정적 구간 스위치 (Gate)

```python
		# 2) 게이팅: 정적/완만 구간만 신뢰
    tau_w = params.get('tau_w', 1.0)
    tau_a = params.get('tau_a', 0.5)
    gate = ((gyro.norm(dim=-1) < tau_w) & (a_hp.norm(dim=-1) < tau_a)).float() # (B,T)
```

- 사용자의 움직임 정도를 판단하는 스위치
    1. `gyro.norm(dim=-1) < tau_w`: 회전(`gyro`)이 거의 없고,
    2. `a_hp.norm(dim=-1) < tau_a`: 순수 움직임(`a_hp`)이 거의 없을 때.
    
    `gate` 값이 1.0이 되고 아니면 0.0이 된다. 
    

3. 모델 예측값 정규화

```python
		# 3) g_pred 정규화
    g_pred = unit_norm(g_pred, eps=eps)
```

- 모델의 중력 예측값(`g_pred`)도 1로 정규화하여 “방향”정보만 사용.

---

### 1. `L_grav`: 방향 정렬 손실

```python
		# (a) 방향 정렬 (LPF(acc)와 정렬)
    cos_sim = (g_from_acc * g_pred).sum(dim=-1).clamp(-1+1e-6, 1-1e-6)
    L_grav = torch.acos(cos_sim).mean()
```

- Acc가 말하는 “중력 방향”과 모델이 “예측한 중력 방향”이 같아야 한다.
    - `g_from_acc` : Acc LPF로 만든 “중력 방향”
    - `g_pred` : 모델이 “예측한 중력 방향”
    - `(g_from_acc * g_pred).sum(dim=-1)`: 두 벡터의 내적(Dot Product)
        
        (결과 = 코사인 값)
        
    - `torch.acos(cos_sim).mean()`: 코사인 값을 각도(Radian)로 변환
- `g_pred` 와 `g_from_acc` 사이의 평균 각도 차이를 손실로 계산한다.

---

### 2. `L_gmag`: 중력 크기 손실

```python
		# (b) 중력 크기 (정규화된 입력이면 g0≈1.0)
    g0 = params.get('g0', 1.0)
    L_gmag = (gate * (acc.norm(dim=-1) - g0).pow(2)).mean()
```

- 정적 구간은 총 가속도 크기가 1G여야한다.
    - `g0`: 1G (목표값), params Dict에 `g0`가 없으면 기본값으로 `1.0`을 사용
    - `acc.norm(dim=-1)`: 총 가속도 신호의 크기 ($\sqrt{X^2+Y^2+Z^2}$)
    - `( ... - g0).pow(2)`: 이 크기가 `g0`(1.0)과 얼마나 차이 나는지 제곱
    - `gate * ( ... )`: 이 손실을 `gate` 스위치가 켜진 '정적 구간'에만 적용. 움직이는 구간(`gate=0`)에서는 0이 되니까

---

### 3. `L_comp`: 보완 필터 손실 *(고전적인 방법)*

```python
		# (c) 보완필터 잔차: g_comp vs g_pred
    alpha_c = params.get('alpha_comp', 0.97)
    dt = params.get('dt', estimate_dt_from_freq())
    g_prev = torch.roll(g_pred, shifts=1, dims=1)
    g_gyro = unit_norm(g_prev - dt * torch.cross(gyro, g_prev, dim=-1), eps=eps)
    g_acc  = unit_norm(acc, eps=eps)
    g_comp = unit_norm(alpha_c * g_gyro + (1 - alpha_c) * g_acc, eps=eps)
    L_comp = torch.acos((g_comp * g_pred).sum(dim=-1).clamp(-1+1e-6, 1-1e-6)).mean()
```

- 모델이 “애측한 중력 방향”이 전통적인 보완 필터의 답인 “중력 방향”(`g_comp`)와 일치해야한다.
    - `alpha_c`: gyro를 얼마나 신뢰할지 결정하는 가중치
    - `dt`: 센서 데이터의 측정 간격 시간 (UCI-HAR=1/50)
    - `g_prev`: 이전 스텝의 `g_pred` (자이로 적분의 시작점)
        - [g_t0, g_t1, g_t2, ..., g_t127]
            
            → [g_t127, g_t0, g_t1, ..., g_t126]
            
    - `g_gyro`: '자이로 적분'으로 계산한 방향 (빠르지만 Drift)
        
        $g_{\text{new}} = g_{\text{old}} - dt \times (\omega \times g_{\text{old}})$
        
    - `g_acc`: 가속도계가 말하는 방향 (느리지만 안정적)
    - `g_comp`: `g_gyro`와 `g_acc`를 97:3으로 섞은 **"모범 답안"**
    - `L_comp`: `g_pred`와 `g_comp` 사이의 **평균 각도 차이**를 손실로 계산.
        - `g_comp * g_pred` : 두 텐서의 element-wise 곱 (내적인데 왜?)
            
            → `g_comp`와 `g_pred`를 이미 `unit_norm`으로 정규화했기 때문에 내적공식이  $1 \times 1 \times cos(\theta) = cos(\theta)$가 됐다. 
            
        - `.sum(dim=-1)` : 마지막 차원의 값(X, Y, Z) 값들을 모두 더하기
        - `torch.acos(...)`: 각도로 변환
        - `.clamp(...)`: crash 방지, 안정성 확보용

---

### 4. `L_bias`: 자이로 바이어스 손실

```python
		# (d) 자이로 바이어스(창 평균≈0)
    win = params.get('win_mean', 16)
    gyro_m = fir_lpf_hann_bt3(gyro, K=max(3, 2* (win//2)+1))[0] # 창 평균 대용
    L_bias = (gate * gyro_m.pow(2).sum(dim=-1)).mean()
```

- 정적구간의 자이로값은 0이어야한다.
    - `gyro_m`: 자이로 값의 단기 평균 (LPF 사용)
        - 단기 평균: 짧은 시간 구간에 대해서만 계산한 평균값
        - 여기서, LPF는 “빠른 변화 신호를 걸러낸다”라는 관점에서 단기 평균과 작동 원리가 정확히 동일하다.
    - `gyro_m.pow(2).sum(-1)`: 평균값이 0이 아니면 커지는 벌점.
    - `gate * ( ... )`: 이 손실도 `gate` 스위치가 켜진 '정적 구간'에만 적용

---

### 5. `L_smooth`: 스무딩 손실

```python
		# (e) 스무딩(jerk/ω̇)
    da = diff1(acc);  dw = diff1(gyro)
    L_smooth = (da.pow(2).sum(dim=-1) + dw.pow(2).sum(dim=-1)).mean()
```

- 신호는 물리적으로 부드럽게 변해야한다. (순간이동 금지)
    - `da = diff1(acc)`: 가속도의 변화율 (Jerk)
    - `dw = diff1(gyro)`: 각속도의 변화율 (Angular Acceleration)
    - `L_smooth`: 이 변화율들이 너무 크지 않도록(신호가 널뛰지 않도록) 제어

---

### 6. `L_split`: 분해 일관성 손실

```python
		# (f) 분해 일관성: a_total = a_body + g0*ĝ → a_body 창 평균이 0 근처
    a_body = acc - g0 * g_pred
    a_body_m = fir_lpf_hann_bt3(a_body, K=max(3, 2* (win//2)+1))[0] # 로우패스=윈도 평균
    L_split = a_body_m.norm(dim=-1).mean()
```

- 총 가속도에서 중력을 뺀 “순수한 움직임”은 단기 평균이 0이어야한다. (특징 분리)
    - `a_body = acc - g0 * g_pred`: 총 가속도(`acc`)에서 모델이 예측한 중력(`g_pred`)을 뺀다. `a_body`는 모델이 생각하는 “순수 신체 움직임”
    - `a_body_m`: 이 '순수 움직임'의 단기 평균(LPF)
    - `L_split`: 이 평균값이 0이 아니라면(즉, 모델이 중력을 잘못 분리해서 '순수 움직임' 성분에 중력 성분이 새어 나갔다면) 손실로 계산.

---

### 7. `L_pinn`: 미분 운동학 손실

```python
		# (g) 미분운동학: dg/dt ≈ -ω×g
    dg = diff1(g_pred) / max(dt, 1e-3)
    w_cross_g = torch.cross(gyro, g_pred, dim=-1)
    L_pinn = (dg + w_cross_g).pow(2).sum(dim=-1).mean()
```

- 중력 방향의 실제 변화율(`dg/dt`)은 자이로로 계산한 물리 법칙(`-wxg`)과 같아야한다.
    - `dg = diff1(g_pred) / dt`: 모델이 예측한 `g_pred`가 **실제로** 얼마나 변했는지 계산 (미분).
    - `w_cross_g = torch.cross(gyro, g_pred)`: 자이로(`ω`)와 중력(`g`)으로 **물리 공식상** 변했어야 하는 값 (`ω×g`)을 계산.
    - `L_pinn`: `dg/dt`와 `w_cross_g`는 같아야 하므로, `(dg + w_cross_g)`는 0에 가까워야 합니다. 0이 아니면 벌점을 줍니다.

---

### 최종 결합

```python
L = (
        lambdas.get('grav', 0.10)   * L_grav   +
        lambdas.get('gmag', 0.05)   * L_gmag   +
        lambdas.get('comp', 0.10)   * L_comp   +
        lambdas.get('bias', 0.02)   * L_bias   +
        lambdas.get('smooth', 0.02) * L_smooth +
        lambdas.get('split', 0.05)  * L_split  +
        lambdas.get('pinn', 0.03)   * L_pinn
    )
    ...
    return L, stats
```

- 7가지 물리 손실을 `lambdas`라는 가중치로 섞어서 최종 물리손실 `L` 생성

- $L = (\lambda_{\text{grav}} \times L_{\text{grav}}) + (\lambda_{\text{gmag}} \times L_{\text{gmag}}) + (\lambda_{\text{comp}} \times L_{\text{comp}}) + (\lambda_{\text{bias}} \times L_{\text{bias}}) + (\lambda_{\text{smooth}} \times L_{\text{smooth}}) + (\lambda_{\text{split}} \times L_{\text{split}}) + (\lambda_{\text{pinn}} \times L_{\text{pinn}})$
  
- `L`이 `train_physics`함수의 `L_phys`가 되어 `loss = loss_ce + phys_scale * L_phys` 형태로 최종 손실이 정의된다.

---
