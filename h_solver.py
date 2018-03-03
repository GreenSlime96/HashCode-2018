import numpy as np

from _hungarian import linear_sum_assignment as hungarian

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

v_p = np.empty(F, dtype=np.int16)
v_t = np.empty(F, dtype=np.uint16)

time = 0
jobs = [[] for _ in range(F)]

work = np.full(N, True)
work[dists[-1] + scores > f] = False


def first_assign():
    task = np.where(work)[0]

    t2r = dists[-1, task]
    t2w = s[task] - t2r

    bonus = B * (t2w >= 0)
    score = scores[task]

    t2w = np.maximum(0, t2w)
    t2c = score + t2r + t2w

    gain = score + bonus - t2r - t2w
    gain[t2c > f[task]] = -np.inf

    i_idx = np.argsort(-gain)[:F]
    e_idx = task[i_idx]

    work[e_idx] = False

    v_t[:] = t2c[i_idx]
    v_p[:] = e_idx


first_assign()
time = v_t.min()

while time <= T and work.any():
    cars = np.where(v_t == time)[0]
    task = np.where(work)[0]

    t2r = dists[v_p[cars]][:, task]
    t2w = s[task] - (time + t2r)

    bonus = B * (t2w >= 0)
    score = scores[task]

    t2w = np.maximum(0, t2w)
    t2c = time + score + t2r + t2w

    # gain = np.full((len(cars), len(task)), np.inf)
    mask = t2c > f[task]

    gain = score + bonus - t2w - t2r - f[task]
    gain[mask] = -np.inf

    print(mask.all(axis=1))

    if mask.all():
        print("decomissioned: {}".format(cars))
        v_t[cars] = T
    else:
        # step 1 -- subtract maximum value
        gain = gain.max() - gain

        # calculate outputs
        i_idx = hungarian(gain)[1]
        e_idx = task[i_idx]

        work[e_idx] = False

        v_t[cars] = t2c[np.arange(len(t2c)), i_idx]
        v_p[cars] = e_idx

    time = v_t.min()
    work[time > f] = False

    print(scores[~work].sum())

print(done)

# for t in range(T):
#     work[t > f] = False

#     cars = np.where(v_t == 0)[0]
#     task = np.where(work)[0]

#     if not len(task):
#         print("finished at time: {}".format(t))
#         break

#     if len(cars):
#         t2r = dists[v_p[cars]][:, task]
#         t2w = s[task] - (t + t2r)

#         bonus = B * (t2w >= 0)
#         score = scores[task]

#         t2w = np.maximum(0, t2w)
#         t2c = t + score + t2r + t2w

#         # gain = np.full((len(cars), len(task)), np.inf)
#         gain = score + bonus - t2w - t2r - f[task]
#         gain[t2c > f[task]] = -np.inf

#         # step 1 -- subtract maximum value
#         gain = gain.max() - gain

#         # calculate outputs
#         i_idx = hungarian(gain)[1]
#         e_idx = task[i_idx]

#         work[e_idx] = False

#         v_t[cars] = t2c[np.arange(len(t2c)), i_idx]
#         v_p[cars] = e_idx

#     print(t)
#     v_t[v_t > 0] -= 1


# for _, job in jobs:
#     job[0] = len(job) - 1
#     print(' '.join(str(_) for _ in job))
