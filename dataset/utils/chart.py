import matplotlib.pyplot as plt
import numpy as np
from h5py import File

data = File("../clean_data/balanced_data.h5")

feedbacks = data["feedbacks"][:]
yAxes = [f[0] for f in feedbacks]
bins =  np.linspace(np.min(yAxes), np.max(yAxes), 26)

fig, axs = plt.subplots()
axs.hist(yAxes, bins=bins)
plt.show()
plt.xticks(bins)

print(np.min(yAxes), np.max(yAxes))
print(f"Total: {len(feedbacks)}")
