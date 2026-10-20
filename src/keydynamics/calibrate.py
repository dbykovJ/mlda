import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from keydynamics.model import KeyModel

import matplotlib.pyplot as plt

path = "./out_test.csv"

model = KeyModel.from_csv(path)

# for i, sample in enumerate(model.sample_props):
#     dwell_time = sample.dwell_time
#     press_press = sample.press_press
#     press_release = sample.press_release
#     release_press = sample.release_press
#     release_release = sample.release_release
#     # scatter all

#     # five scatter plots for each feature
#     plt.figure(figsize=(10, 6))
#     plt.subplot(2, 3, 1)
#     plt.scatter(i, dwell_time, c='r', marker='o')
#     plt.title('Dwell Time')
#     plt.subplot(2, 3, 2)
#     plt.scatter(i, press_press, c='g', marker='o')
#     plt.title('Press-Press')
#     plt.subplot(2, 3, 3)
#     plt.scatter(i, press_release, c='b', marker='o')
#     plt.title('Press-Release')
#     plt.subplot(2, 3, 4)
#     plt.scatter(i, release_press, c='c', marker='o')
#     plt.title('Release-Press')
#     plt.subplot(2, 3, 5)
#     plt.scatter(i, release_release, c='m', marker='o')
#     plt.title('Release-Release')
#     plt.tight_layout()

# plt.show()

model1 = KeyModel.from_csv("./out_test.csv")
model2 = KeyModel.from_csv("./out_emil.csv")
model3 = KeyModel.from_csv("./out_duna.csv")

X1 = model1.to_np()
X2 = model2.to_np()
X3 = model3.to_np()
X_all = np.vstack([X1, X2, X3])


scaler = StandardScaler().fit(X_all)
pca = PCA(n_components=3).fit(scaler.transform(X_all))

c1 = pca.transform(scaler.transform(X1))
c2 = pca.transform(scaler.transform(X2))
c3 = pca.transform(scaler.transform(X3))

print(pca.explained_variance_ratio_)

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.scatter(c1[:, 0], c1[:, 1], c1[:, 2], c="r", marker="o", label="test")
ax.scatter(c2[:, 0], c2[:, 1], c2[:, 2], c="b", marker="s", label="emil")
ax.scatter(c3[:, 0], c3[:, 1], c3[:, 2], c="g", marker="^", label="duna")
ax.legend()
plt.title("PCA of Keystroke Dynamics")
plt.show()