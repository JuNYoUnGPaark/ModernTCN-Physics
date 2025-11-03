def rf_tcn(n_layers=3, k_max=7, k_large_front=21, use_back_large=True, k_large_back=21):
    s = sum(2**i for i in range(n_layers))    # 1+2+...+2^(L-1)
    rf = 1 + (k_large_front - 1) + 2*(k_max - 1)*s
    if use_back_large:
        rf += (k_large_back - 1)
    return rf

print(rf_tcn(3,7,21,True,21)) # 125
print(rf_tcn(3,7,43,False))    # 127