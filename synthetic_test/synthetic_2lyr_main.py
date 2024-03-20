# %%
from synthetic_2lyr import synthetic_2lyr
# %%
# Create a map to set resistivity values in the appropriate regions
# [[regionNumber, resistivity], [regionNumber, resistivity], [...]
rhomap = [[1, 1500.],
          [2, 500.],
          [3, 100.]]
test_name = 'synthetic_2lyr_hsr'
lam=100
plot_result = True
save_plot = True

mgr2, mgr3 = synthetic_2lyr(rhomap, test_name, lam, plot_result, save_plot)