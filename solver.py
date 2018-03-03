import numpy as np

with open('d_metropolis.in') as fp:
    data = np.fromiter(fp.read().split(), int).reshape(-1, 6)

    R, C, F, N, B, T = data[0]
    a, b, x, y, s, f = data[1:].T

    dists = np.empty((N + 1, N))
    dists[:-1] = np.abs(x[:, None] - a) + np.abs(y[:, None] - b)
    dists[-1] = a + b

    scores = dists.diagonal().astype(np.uint16)
    np.fill_diagonal(dists, np.inf)

    del data

m_dist = np.median(dists[:-1], axis=1)

jobs = [[0, [-1]] for _ in range(F)]

queue = np.full(N, True)
queue[dists[-1] + scores > f] = False

print("impossible? {}".format((queue == False).sum()))

while queue.any():
    # get the shortest node to the current position

    tot = queue.sum()

    for i, (t, job) in enumerate(jobs):
        pos = job[-1]

        t2r = dists[pos, queue]
        t2w = s[queue] - (t + t2r)

        bonus = B * (t2w >= 0)
        score = scores[queue]

        t2w = np.maximum(0, t2w)
        t2c = t + score + t2r + t2w

        mask = np.argwhere(t2c < f[queue])

        if not len(mask):
            continue

        gain = score + bonus - t2w - t2r
        # gain = -t2r

        i_idx = gain.argmax()
        e_idx = np.where(queue)[0][i_idx]

        # if t2c[i_idx] > f[e_idx]:
        #     continue

        jobs[i][0] = t2c[i_idx]
        job.append(e_idx)

        queue[e_idx] = False

    if queue.sum() == tot:
        break

    print("{} / {} \t {} / {}".format(queue.sum(),
                                      N, scores[queue].sum(), scores.sum()))

print("FINISHED WITH {}".format(scores[~queue].sum()))


for _, job in jobs:
    job[0] = len(job) - 1
    print(' '.join(str(_) for _ in job))
