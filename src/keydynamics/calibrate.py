import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from keydynamics.models.GlobalModel import globalModel
from keydynamics.models.model import KeyModel
from keydynamics.models.UserModel import UserModel


import matplotlib.pyplot as plt

# path = "./out_test.csv"
#
# model = KeyModel.from_csv(path)

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

model1 = KeyModel.from_csv("./outputs/out_test.csv").clean()
model2 = KeyModel.from_csv("./outputs/output.csv").clean()
model3 = KeyModel.from_csv("./outputs/out.csv").clean()

X1 = np.array(model1)
for sample in X1[:-2]:
    globalModel.add_sample(sample, 4)
X2 = np.array(model2)
for sample in X2[:-2]:
    globalModel.add_sample(sample, 5)
X3 = np.array(model3)
for sample in X3[:-2]:
    globalModel.add_sample(sample, 6)
X_all = np.vstack([X1, X2, X3])


for sample in np.vstack([X1[-2:], X2[-2:], X3[-2:]]):
    user_id = globalModel.check_sample(sample, k=5)
    print(f"Sample {sample} is classified as user {user_id}")


scaler = StandardScaler().fit(X_all)
pca = PCA(n_components=3).fit(scaler.transform(X_all))

c1 = pca.transform(scaler.transform(X1))
c2 = pca.transform(scaler.transform(X2))
c3 = pca.transform(scaler.transform(X3))

print(pca.explained_variance_ratio_)

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.scatter(c1[:, 0], c1[:, 1], c1[:, 2], c="r", marker="o", label="lorenz")
ax.scatter(c2[:, 0], c2[:, 1], c2[:, 2], c="b", marker="s", label="dima")
ax.scatter(c3[:, 0], c3[:, 1], c3[:, 2], c="g", marker="^", label="emil")
ax.legend()
plt.title("PCA of Keystroke Dynamics")
plt.show()


lorenz = UserModel.buildModel(X1, 4)
dima = UserModel.buildModel(X2, 5)
emil = UserModel.buildModel(X3, 6)


for sample in np.vstack([X1, X2, X3]):
    if not dima.is_user(sample):
        print("False user detected!! Not dima")
    else:
        print("Dima detected!!")

for sample in np.vstack([X2,X3]):
    if not lorenz.is_user(sample):
        print("False user detected!! Not lorenz")
    else:
        print("Lorenz detected!!")

for sample in np.vstack([X1,X2, X3]):
    if not emil.is_user(sample):
        print("False user detected!! Not emil")
    else:
        print("Emil detected!!")

#
# users = {"lorenz": X1, "dima": X2, "emil": X3}
# models = {name: UserModel.buildModel(X, i) for i, (name, X) in enumerate(users.items())}
#
# for name, m in models.items():
#     genuine = UserModel._loo_scores(users[name])
#     print(f"\n{name}: threshold={m.threshold:.2f}, "
#           f"genuine LOO range=[{genuine.min():.2f}, {genuine.max():.2f}]")
#     for other, X in users.items():
#         if other == name:
#             continue
#         scores = np.array([m.compare_sample(s) for s in X])
#         print(f"  vs {other}: min={scores.min():.2f}, "
#               f"accepted={np.mean(scores <= m.threshold):.2f}")
# model = KeyModel.from_csv("./outputs/output.csv")
# model = model.clean()
# for row in model.iterrows():
#     print(row)
#
# scaler = StandardScaler()
# pca = PCA(n_components=3)
#
# c1 = pca.fit_transform(scaler.fit_transform(model))
#
# fig = plt.figure()
# ax = fig.add_subplot(111, projection="3d")
# ax.scatter(c1[:, 0], c1[:, 1], c1[:, 2], c="r", marker="o", label="test")
# ax.legend()
# plt.title("PCA of Keystroke Dynamics")
# plt.show()