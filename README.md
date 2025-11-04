## Base Idea: "Modern" TCN
- Depthwise Separable Conv: 표준 Conv 대비 params. FLOPs를 크게 감소시킨 효율적인 방식
- Multi-Scale Conv: Kernel_size=[3, 7]을 병렬로 처리 후 융합하여, 모델이 당야한 시간적 길이의 패턴을 동시 포착
- Large Kernel Conv: 모델의 시작과 끝에 큰 커널(k=19)을 사용하여 넓은 범위의 Context를 한번에 집약
- Squeeze-Excitation(SE): 어떤 채널 정보가 현재 분류에 더 중요한지 동적으로 weight를 학습하는 attention 메커니즘 

## 1. 전체 흐름 (BaseModernTCNHAR)

```python
class BaseModernTCNHAR(nn.Module):
    def __init__(self, input_dim=9, hidden_dim=128, n_layers=4, n_classes=6,
                 kernel_sizes=[3, 7], large_kernel=21, dropout=0.1, use_se=True):
        super().__init__()
        self.input_proj = nn.Conv1d(input_dim, hidden_dim, 1)
        self.large_kernel_conv = LargeKernelConv1d(hidden_dim, large_kernel)
        self.tcn_blocks = nn.ModuleList()
        for i in range(n_layers):
            dilation = 2 ** i
            self.tcn_blocks.append(
                ModernTCNBlock(
                    hidden_dim, hidden_dim,
                    kernel_sizes=kernel_sizes,
                    dilation=dilation,
                    dropout=dropout
                )
            )
        self.final_large_kernel = LargeKernelConv1d(hidden_dim, large_kernel)
        self.use_se = use_se
        if use_se:
            self.se = SqueezeExcitation1d(hidden_dim)
        self.norm_final = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, n_classes)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.input_proj(x)
        x = self.large_kernel_conv(x)
        x = F.gelu(x)
        for block in self.tcn_blocks:
            x = block(x)
        x = self.final_large_kernel(x)
        x = F.gelu(x)
        if self.use_se:
            x = self.se(x)
        x = F.adaptive_avg_pool1d(x, 1).squeeze(-1)
        x = self.norm_final(x)
        return self.head(x)
```

1. `x ∈ (B,T,9)` → `transpose(1,2)` → `(B,9,T)` 
2. `input_proj: Conv1d(9→hidden_dim, k=1)` : 채널 임베딩
3. `large_kernel_conv(k=21, depthwise)` : **장주기 패턴** 포착 (길이 유지)
4. `tcn_blocks × n_layers` : **멀티스케일 분기 + dilation**으로 수용영역 확장
5. `final_large_kernel(k=21)` : 한 번 더 장주기 정제
6. `SE(1D)` : 채널 중요도 재가중( squeeze: T→1, excite: 채널 게이팅 )
7. `AdaptiveAvgPool1d(T→1)` 후 `squeeze` → `(B, hidden_dim)` 
8. `LayerNorm(hidden_dim)` → `Linear(hidden_dim→n_classes)`

## 2. MultiScaleConvBlock

```python
class MultiScaleConvBlock(nn.Module):
    def __init__(self, channels, kernel_sizes=[3, 5, 7], dilation=1, dropout=0.1):
        super().__init__()
        self.branches = nn.ModuleList()
        for k in kernel_sizes:
            padding = ((k - 1) * dilation) // 2
            branch = nn.ModuleDict({
                'conv': DepthwiseSeparableConv1d(channels, channels, k, dilation, padding),
                'norm': nn.BatchNorm1d(channels),
                'dropout': nn.Dropout(dropout)
            })
            self.branches.append(branch)
        self.fusion = nn.Conv1d(channels * len(kernel_sizes), channels, 1)

    def forward(self, x):
        outputs = []
        target_length = x.size(2)
        for branch in self.branches:
            out = branch['conv'](x)
            if out.size(2) != target_length:
                out = out[:, :, :target_length]
            out = branch['norm'](out)
            out = F.gelu(out)
            out = branch['dropout'](out)
            outputs.append(out)
        multi_scale = torch.cat(outputs, dim=1)
        return self.fusion(multi_scale)
```

- 병렬 분기: `kernel_sizes=[3, 7]`
    
    각 분기: DepthwiseSeparbleConv1d → BN → GELU → Dropout
    
- 모든 분기 Convat 후 1X1 Conv로 융합 → 채널 수 복원
- 다양한 패턴 동시에 포착, DS-Conv로 params. 절약, 1X1에서 Channel 상호작용

## 3. ModernTCNBlock

```python
class ModernTCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes=[3, 7], dilation=1, dropout=0.1):
        super().__init__()
        
        # NOTE: kernel_sizes가 [7]처럼 단일 리스트로 들어오면 Single-scale이 됨
        self.multi_conv1 = MultiScaleConvBlock(
            in_channels if in_channels == out_channels else out_channels,
            kernel_sizes, dilation, dropout
        )
        
        # NOTE: kernel_sizes 중 가장 큰 값을 기준으로 padding
        max_k = max(kernel_sizes) if isinstance(kernel_sizes, list) else kernel_sizes
        padding = ((max_k - 1) * dilation) // 2
        
        self.conv2 = DepthwiseSeparableConv1d(
            out_channels, out_channels, max_k, dilation, padding
        )
        self.norm2 = nn.BatchNorm1d(out_channels)
        self.dropout2 = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None

    def forward(self, x):
        residual = x
        target_length = x.size(2)
        if self.downsample is not None:
            x = self.downsample(x)
            residual = x
        
        out = self.multi_conv1(x)
        if out.size(2) != target_length:
            out = out[:, :, :target_length]
        
        out = self.conv2(out)
        if out.size(2) != target_length:
            out = out[:, :, :target_length]
        out = self.norm2(out)
        out = F.gelu(out)
        out = self.dropout2(out)
        return F.gelu(out + residual)
```

- 흐름: `multi_conv1` → 가장 큰 kernel_size 기준으로 DS-Conv → BN → GELU → Dropout → residual add + GELU

---

**Q. 현재 모델에서 가장 알맞는 `n_layers` 선택 근거는?**

$$
ERF=1+2\cdot(k\_large)+2\cdot(k\_max)\cdot1+2+4+...+2^{(L-1)}
$$

여기서 `k_large=21`, `k_max=7`, `L=n_layers`.

- UCI-HAR의 적절 TimeStep=128
    - `n_layers=2` → ERF = 약 77
    - `n_layers=3` → ERF = 약 125
    - `n_layers=4` → ERF = 약 221

따라서 n_layers를 증가시킨다고 반드시 좋아지는 것이 아닌 Dataset에 맞는 값을 사용하는 것이 중요하다.

```python
def rf(L, k_max=7, k_large=21):
    return 1 + (k_large-1) + 2*(k_max-1)*sum(2**i for i in range(L)) + (k_large-1)

print(rf(2))  # 77
print(rf(3))  # 125
print(rf(4))  # 221 
```

- n_layer=3 일때의 결과가 가장 좋았던 이유는 Data의 TimeStep에 맞춰서 Receptive Field를 설계해준 것이 이점을 가져다 준것이라고도 볼 수 있다.
- 주의할점: 이론적 RF와 실제 RF는 다를 수 있기에 보통 RF를 T의 0.7~1.2배 구간에 맞추면 대개 안정적이다.

---

## 4. DepthwiseSeparbleConv1d

```python
class DepthwiseSeparableConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, padding=0):
        super().__init__()
        self.depthwise = nn.Conv1d(
            in_channels, in_channels, kernel_size,
            padding=padding, dilation=dilation, groups=in_channels
        )
        self.pointwise = nn.Conv1d(in_channels, out_channels, 1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x
```

- Depthwise: 채널별 1D-conv (groups=channels)
- Pointwise: 1×1로 채널 혼합
- 표준 Conv 대비 **파라미터/연산 대폭 절감**.

## 5. LargeKernelConv1d(k=21, depthwise)

```python
class LargeKernelConv1d(nn.Module):
    def __init__(self, channels, kernel_size=21):
        super().__init__()
        padding = kernel_size // 2
        self.depthwise = nn.Conv1d(
            channels, channels, kernel_size,
            padding=padding, groups=channels
        )
        self.norm = nn.BatchNorm1d(channels)
    def forward(self, x):
        out = self.depthwise(x)
        out = self.norm(out)
        return out
```

- 장주기 패턴(저주파 성분)을 한 번에 훑는 **롱컨텍스트 필터**.
- 앞/뒤로 한 번씩 사용해 초기/말기 특징 안정화 (우선 이 코드는!)

## 6. PhysicsModernTCNHAR (물리 헤드)

```python
class PhysicsModernTCNHAR(BaseModernTCNHAR):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        hidden_dim = self.head.in_features
        self.physics_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 6)
        )

    def forward(self, x, return_physics=False):
        x = x.transpose(1, 2)
        x_feat = self.input_proj(x)
        x_feat = self.large_kernel_conv(x_feat)
        x_feat = F.gelu(x_feat)
        for block in self.tcn_blocks:
            x_feat = block(x_feat)
        x_feat = self.final_large_kernel(x_feat)
        x_feat = F.gelu(x_feat)
        if self.use_se:
            x_feat = self.se(x_feat)
        
        # 1. 분류 헤드
        pooled = F.adaptive_avg_pool1d(x_feat, 1).squeeze(-1)
        pooled = self.norm_final(pooled)
        logits = self.head(pooled)

        if return_physics:
            # 2. 물리 헤드
            x_feat_transposed = x_feat.transpose(1, 2)
            physics = self.physics_head(x_feat_transposed)
            return logits, physics

        return logits
```

- `head`(분류)와 병렬로 `physics_head(MLP)` 추가:
    - 입력: `x_feat ∈ (B,C,T)` → `(B,T,C)`로 전치 → per-timestep MLP
    - 출력: `(B,T,6)` = `[acc_x,y,z, gyro_x,y,z]`
- physics_loss: 예측 6채널을 원 입력 X의 앞 6채널(body_acc/gyro)과 Smooth L1로 회귀
    - 총손실: `CE + λ·physics` (코드 기본 λ=0.05)
    - 역할: 특징이 물리 신호 복원에 유의미하도록 유도(보조감독)
    - 현재는 physics 예측이 **정규화된 스케일**(정규화 적용 시)에서 비교됨. 
    “물리 단위 복원”을 원하면 loss 쪽에서 **inverse-transform** 필요.

---

**Q. Physics Loss란?** 

분류를 잘하는 것뿐 아니라 Feature Map이 실제 센서 신호를 다시 그려낼 수 있어야 한다는 보조 목표를 주는 것. (입력 신호 복원: Smooth L1)

- 흐름
    - 백본이 만든 특징 `x_feat` : `(B, C, T)`
    - `physics_head`(작은 MLP)가 시간축별로 6개 값을 예측:
        - `x_feat.transpose(1,2)` → `(B, T, C)`
        - `physics_head` → **`physics_pred` = `(B, T, 6)`**
            
            (순서: `[acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z]`)
            
    - 원래 입력 `X_raw`에서 앞 6채널(= body_acc 3 + body_gyro 3)을 꺼냄:
        - **`X_raw[:, :, :6]` = `(B, T, 6)`**
    - 두 텐서를 **시간축-채널별로** 비교해 오차를 계산:
        - `Smooth L1(physics_pred, X_raw[:, :, :6])` → 스칼라
        - 전체 손실: `loss = CE(logits, y) + λ * physics_loss` (코드에서 λ=0.05)

- 구현 방법 = Smooth L1/Huber 사용
    
    ```python
    def physics_loss(physics_pred, X_raw):
        acc_pred = physics_pred[:, :, :3]
        gyro_pred = physics_pred[:, :, 3:6]
        acc_true = X_raw[:, :, 0:3]
        gyro_true = X_raw[:, :, 3:6]
        return F.smooth_l1_loss(acc_pred, acc_true) + F.smooth_l1_loss(gyro_pred, gyro_true)
    ```
    

- 수식
    
    오차 $x=input-target$에 대해
    
    - $|x|< \beta$ 이면 L2처럼: $0.5 \cdot x^2/\beta$
    - $|x|>=\beta$ 이면 L1처럼: $|x|-0.5\beta$
    
    즉, 작은 오차엔 부드럽게 큰 오차엔 크게 반응.
    
    *여기서 L1, L2는 가중치에 거는 규제가 아닌 “예측 - 정답”으로 생기는 오차를 계산할 때 사용하는 손실임. L1(MAE), L2(MSE), Smooth L1/Huber*
    

---

## 7. GELU 사용

<img width="850" height="564" alt="image" src="https://github.com/user-attachments/assets/9deadcb6-88a9-4254-b9b1-2765875ecc27" />

- 미분 가능 영역이 넓고, 깊은 네트워크에서 수렴이 매끄럽다(최근 ConvNext/Transformer 계열 추세).

## 9. Optimizer, Regularization, Scheduler

```python
AdamW(lr=5e-4, weight_decay=0.01)
```

- weight_decay는 L2 규제

```python
Gradient Clip (1.0)
```

- 폭주 방지. dilation + large kernel 조합에서 안정화

```python
CosineAnnealingLR(T_max=epochs)
```

- 전체 epoch 스케줄에 **강하게 의존**. epoch 바꾸면 궤적 달라짐.
- *개선 팁*: **Warmup + Cosine**가 일반적으로 더 안정적

## 10. 손실 구성,

- Base: `loss = CE(logits, y)`
- Physics: `loss = CE + λ·SmoothL1(physics_pred, X[:,:,:6])` (λ=0.05)

---

## 11. 개선해볼 사항 정리

1. ~~적절한 n_layers 설정 기준 세우기~~ 
2. 입력 정규화(train 통계로 z-score) → physics 회귀 스케일 일치시키기 
3. large_kernel 앞 or 뒤쪽을 제거해보기
4. **Batch size:** DS-Conv + large-kernel은 메모리 널널함 → 가능하면 ↑
5. Warmup 적용해보기 
6. Physics Loss 개선 및 발전시켜보기 
7. 블록 시작점에 LayerNorm 또는 BatchNorm으로 Pre-Norm
8. Stochastic Depth: 깊은 레이어 일부만 확률적으로 생략 → Train 과적합 상태
9. λ 값 탐색 (0.01, 0.02, 0.05, 0.1, 0.15)
10. AdamW Weight Decay 탐색 → 5e-3 ~ 1e-3 
11. EMA 적용 (아마 Bad) 

---
