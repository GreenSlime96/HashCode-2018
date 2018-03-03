import numpy as np

with open('c_no_hurry.in') as fp:
    data = np.fromiter(fp.read().split(), int).reshape(-1, 6)

    R, C, F, N, B, T = data[0]
    a, b, x, y, s, f = data[1:].T

    del data

distances = np.abs(x[:, None] - a) + np.abs(y[:, None] - b)
scores = distances.diagonal()

queue = np.full(N, True)
jobs = []

while queue.any():
    v_pos = (0, 0)
    work = []

    travelled = 0
    rests = 0

    c = t = 0
    while True:
        # time to reach (to start), current pos - proposed pos
        t2r = np.abs(v_pos[0] - a[queue]) + np.abs(v_pos[1] - b[queue])

        # time to wait (if not started), or time past start
        t2w = s[queue] - (t + t2r)

        # bonus for those we can reach on time
        bonus = B * (t2w >= 0)
        dists = scores[queue]

        # time to complete: now + dist + reach + wait
        t2c = t + dists + t2r + np.maximum(0, t2w)

        # choose only those that are possible
        mask = np.where(t2c < f[queue])[0]

        # exit if vechicle cannot do any more
        if not len(mask):
            break

        # # gain assigns a profit to tasks
        # gain = dists + bonus - t2w - t2r

        # get the closest car
        gain = -t2r

        # choose the highest gain
        i_idx = np.argmax(gain[mask])
        e_idx = np.where(queue)[0][i_idx]

        # print("=" * 40)
        # print("current time:\t{}".format(t))
        # print("current loc:\t{}".format(v_pos))
        # print("chosen dest:\t{}".format(e_idx))
        # print("dest loc:\t{}".format((a[e_idx], b[e_idx])))
        # print("distance:\t{}".format(t2r[i_idx]))
        # print("arrive at:\t{}".format(t + t2r[i_idx]))
        # print("start time:\t{}".format(s[e_idx]))
        # print("wait time:\t{}".format(t2w[i_idx]))

        # increase time appropriately
        c += bonus[i_idx] + dists[i_idx]
        t = t2c[i_idx]

        travelled += t2r[i_idx] + np.maximum(0, t2w)[i_idx] + dists[i_idx]
        rests += t2r[i_idx] + np.maximum(0, t2w)[i_idx]

        # update search state
        v_pos = (x[e_idx], y[e_idx])
        queue[e_idx] = False
        work.append(e_idx)

    print(travelled, rests)
    jobs.append((c, work))

jobs.sort(key=lambda x: -x[0])

print("gain: {}".format(sum(x[0] for x in jobs[:F])))
print("loss: {}".format(sum(x[0] for x in jobs[F:])))

print(len(jobs))
for score, job in jobs[:F]:
    print("{} {}".format(len(job), ' '.join(str(_) for _ in job)))
