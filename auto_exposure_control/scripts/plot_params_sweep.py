# -*- coding: utf-8 -*-

import rospkg
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

def parseTrace(trace_name, delim=' '):
    trace = open(trace_name)

    # first line
    header = trace.readline()

    assert header[0] == '#'

    log_names = (header.split(delim))[1:-1]

    print("Log names found: {0}".format(log_names))

    # data
    data = []
    for line in trace:
        line_values = [float(v) for v in (line.split(delim))[0:-1]]
        data.append(line_values)

    data = np.array(data)

    print("Read {0} records for {1} log names.".format(data.shape[0],
          data.shape[1]));

    res = {}
    for i, name in enumerate(log_names):
        res[name] = data[:, i]

    return res

def joinDicts(traces):
    keys = traces[0].keys()
    trace_num = len(traces)

    merged = {}
    for k in keys:
        s = []
        for i in range(trace_num):
            s.append(traces[i][k])
        merged[k] = np.hstack(s)

    return merged


if __name__ == '__main__':
    rp = rospkg.RosPack()
    trace_dir = rp.get_path('auto_exposure_control') + '/tests/trace'

    # parsing
    traces = []
    trace_fs = [
                'sweep_office1.txt',
                'sweep_office2.txt',
                'sweep_indoor_switch.txt',
                'sweep_outdoor1.txt',
                'sweep_outdoor2.txt',
                'sweep_outdoor3.txt'
                ]
    for f in trace_fs:
        trace_f = os.path.join(trace_dir, f)
        print("The trace is {0}".format(trace_f))
        trace = parseTrace(trace_f)
        traces.append(trace)

    trace = joinDicts(traces)


    # filtering and pre-processing
    max_ga_rate = 100
    trace['ga_rate'] = trace['exp_update'] / trace['wg_deriv']
    valid_idx = (trace['exp_update'] != 0) & \
                (trace['ga_rate'] < max_ga_rate)
    print("Threre are {0} valid indices.".format(valid_idx.sum()))
    for key in trace.keys():
        trace[key] = trace[key][valid_idx]

    # plot
    fig = plt.figure()
    ax = fig.add_subplot(311)
    plt.scatter(trace['med_irad'], trace['ga_rate'], c=np.log(np.abs(trace['wg_deriv'])))
    plt.colorbar()
    ax.set_xlabel('med irad')

    ax = fig.add_subplot(312)
    plt.scatter(np.abs(trace['wg_deriv']), trace['ga_rate'], c=(trace['med_irad']))
    plt.colorbar()
    ax.set_xlabel('wg_deriv')

    ax = fig.add_subplot(313)
    plt.scatter(trace['wg'], trace['ga_rate'])
    ax.set_xlabel('wg')

    fig = plt.figure()
    ax = fig.add_subplot(211)
    plt.scatter(np.abs(trace['wg_deriv']), trace['med_irad'], c=np.log(trace['ga_rate']))
    plt.colorbar()
    ax.set_xlabel('wg_deriv')
    ax.set_ylabel('med irad')
    ax = fig.add_subplot(212, projection='3d')
    ax.scatter(trace['med_irad'], np.abs(trace['wg_deriv']), np.log(trace['ga_rate']))