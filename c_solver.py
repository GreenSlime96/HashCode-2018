import numpy as np

with open('b_should_be_easy.in') as fp:
    data = np.fromiter(fp.read().split(), int).reshape(-1, 6)

    R, C, F, N, B, T = data[0]
    a, b, x, y, s, f = data[1:].T

    dists = np.empty((N + 1, N))
    dists[:-1] = np.abs(x[:, None] - a) + np.abs(y[:, None] - b)
    dists[-1] = a + b

    scores = dists.diagonal().astype(np.uint16)
    np.fill_diagonal(dists, np.inf)

    del data


# mat = np.empty((N + F,) * 2)
# mat[:N, :N] = distances + scores[:, None]
# mat[N:, :N] = (a + b)
# mat[N:, N:] = np.inf
# mat[:N, N:] = 0

# np.fill_diagonal(mat[:N, :N], np.inf)

# row_min = mat.min(axis=1)
# mat -= row_min[:, None]

# col_min = mat.min(axis=0)
# mat -= col_min

# print(row_min.sum() + col_min.sum())
# print(T)

"""
C: s = f = T, no bonus
only constraint is t < T

strategy:
1. find 2 shortest neighbours
"""

jobs = [[0, [-1]] for _ in range(F)]

queue = np.full(N, True)

while queue.any():
    # get the shortest node to the current position

    tot = queue.sum()

    for i, (t, job) in enumerate(jobs):
        pos = job[-1]

        dist = dists[pos, queue]
        score = scores[queue]

        mask = np.argwhere(t + dist + score < T)

        if not len(mask):
            continue

        i_idx = dists[pos, queue][mask].argmin()
        e_idx = np.where(queue)[0][i_idx]

        jobs[i][0] += dists[pos, e_idx] + scores[e_idx]
        job.append(e_idx)

        queue[e_idx] = False

    if queue.sum() == tot:
        print("FINISHED WITH {}".format(scores[~queue].sum()))
        break

    print("{} / {} \t {} / {}".format(queue.sum(),
                                      N, scores[queue].sum(), scores.sum()))

for _, job in jobs:
    job[0] = len(job) - 1
    print(' '.join(str(_) for _ in job))
