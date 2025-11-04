# CE & Physics Loss의 Scale 맞추기
---
- **기존 코드는 정규화를 따로 진행하지 않았다.**
- **문제점**)
    1. 두 Loss가 같은 scale에서 비교될 수 없다.
    2. 원본 데이터는 축/센서마다 분산이 다른 일정하지 않은 scale → 이로 인해 Physics Loss가 과대/과소되기 쉬워 수렴을 방해한다. 
    
- **해결책**: Train data로 정규화해서 두 Loss가 같은 scale에서 경쟁할 수 있도록 한다. + Physics Loss의 안정화를 기대할 수 있다.

---

**Q. CE는 정규화 이전에도 큰 문제가 없었다. 왜 Scale에 덜 민감한가?**

A. 모델이 학습하면서 **가중치 스케일을 조정**해 입력 스케일의 변화를 대부분 흡수할 수 있고 CE는 **Softmax 확률(로짓의 상대적 차이)**에 의해 결정되기 때문. 데이터가 10배 커져도 학습이 진행되면 보통 비슷한 예측 확률 분포를 만들 수 있다. 

CE 자체는 평행이동 불변이라 (모든 로짓에 같은 상수를 더하는 격) 입력의 물리 단위에 직접 묶여있지 않는다. 

반대로, Physics Loss는 Target 단위에 직결된다. 정답 데이터 자체와 직접 비교하기 때문. 입력 스케일이 크면 오차도 커지고 SmoothL1 값이 스케일에 비례(제곱 비례)한다. 

(→ Smooth L1 수식 참고)

타깃이 원 단위라 데이터 자체의 스케일에 직접적으로 민감해서 정규화가 없으면 CE와의 균형이 깨지기 쉽다. 

---

- 구현 방법

```python
def compute_train_stats(ds):
  # ds.X: numpy (N, T, C)
  X = torch.from_numpy(ds.X)             # (N, T, 9) float64일 수 있음
  X = X.float()
  mean = X.mean(dim=(0,1), keepdim=True) # (1,1,C)
  std  = X.std(dim=(0,1), keepdim=True).clamp_min(1e-6)
  return mean, std

def make_collate_with_norm(mean, std):
  # mean/std는 (1,1,C) CPU tensor (no grad)
  def collate_norm(batch):
      X, y, s = zip(*batch)
      X = torch.stack(X, dim=0).float()      # (B, T, C)
      X = (X - mean) / std                   # z-score
      y = torch.tensor(y, dtype=torch.long)
      s = torch.tensor(s, dtype=torch.long)
      return X, y, s
  return collate_norm
  

 mean, std = compute_train_stats(train_ds)              # train 통계
collate_norm = make_collate_with_norm(mean, std)

g = torch.Generator()
g.manual_seed(42)

train_loader = DataLoader(
  train_ds, batch_size=64, shuffle=True,
  num_workers=0, drop_last=True, pin_memory=torch.cuda.is_available(),
  collate_fn=collate_norm, generator=g
)
test_loader = DataLoader(
  test_ds, batch_size=64, shuffle=False,
  num_workers=0, pin_memory=torch.cuda.is_available(),
  collate_fn=collate_norm
)
```

- 결과 비교: Full Model (Physics Loss 사용 모델)
  - 기존: 0.9752
  - 적용: 0.9774 (+0.022)
  - 정규화를 적용해서 두 Loss가 같은 scale에서 경쟁할 수 있도록 하여 λ 튜닝을 정확히 할 수 있다. 
