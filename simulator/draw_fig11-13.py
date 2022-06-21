# This file can draw the subfigures for Figure 11-13 in our paper.
# Usage: 
# 1. Change the raw values accordingly (existing numbers are our test results)
# 2. python3 draw_fig11-13.py
import matplotlib.pyplot as plt
import numpy as np

# ====== raw data start here ======
# Fig 11 algorithm design
# [trace1, trace2, trace3, trace4]
Fig11_Muri_L_JCT_raw = [1072009.71,2544642.351,88686.97313,429174.8679]
Fig11_Muri_L_w_worstordering_JCT_raw = [1482465.078,3274256.088,146324.4084,614727.4696]
Fig11_Muri_L_wo_blossom_JCT_raw = [1192784.955,2771637.972,96016.95407,487577.6359]
Fig11_Muri_L_Makespan_raw = [18593245.706217,32826803.046468,8174135.28,10918239.071764]
Fig11_Muri_L_w_worstordering_Makespan_raw = [21797275.497465,38550614.779056,8174135.28,11968562.600926]
Fig11_Muri_L_wo_blossom_Makespan_raw = [19653377.905142,34035577.294260,8174135.28,11323021.724947]

# Fig 12 number of jobs in one group
# [trace1_pr, trace2_pr, trace3_pr, trace4_pr]
Fig12_antman_JCT_raw = [9808250.677,18163389.280198,2674393.018,4676092.801]
Fig12_Muri_L_2_JCT_raw = [2624732.58,5243437.378,1348968.422,2170262.136]
Fig12_Muri_L_3_JCT_raw = [2738962.215,5078885.537,1242056.648,1994350.657]
Fig12_Muri_L_4_JCT_raw = [2141936.535,4152485.536,921094.7449,1488033.596]
Fig12_antman_Makespan_raw = [20285085.619635,48327650.120433,5799634.744,11793469.427804]
Fig12_Muri_L_2_Makespan_raw = [18845825.639380,36514321.551734,5244800.59,11141527.953587]
Fig12_Muri_L_3_Makespan_raw = [17773101.828110,36971546.424126,5175874.982,10836071.391281]
Fig12_Muri_L_4_Makespan_raw = [16639500.393368,31782578.758801,4403652.203,10104721.762665]

# Fig 13 workload distributions
# [job_type_1, job_type_2, job_type_3, job_type_4]
Fig13_SRTF_raw = [192543.8569,192494.5694,192277.9859,193081.796]
Fig13_Muri_S_raw = [182804.0161,135643.09,90861.63341,85502.55387]
Fig13_Tiresias_raw = [346383.4174,346280.513,345883.8473,347213.2481]
Fig13_Muri_L_raw = [342921.6979,233034.5164,101383.3536,88686.97313]
# ====== raw data stop here ======

# calculated ratio
# Fig 11 algorithm design
Fig11_Muri_L_JCT = [1,1,1,1]
Fig11_Muri_L_w_worstordering_JCT = [a/b for a,b in zip(Fig11_Muri_L_w_worstordering_JCT_raw, Fig11_Muri_L_JCT_raw)]
Fig11_Muri_L_wo_blossom_JCT = [a/b for a,b in zip(Fig11_Muri_L_wo_blossom_JCT_raw, Fig11_Muri_L_JCT_raw)]
Fig11_Muri_L_Makespan = [1,1,1,1]
Fig11_Muri_L_w_worstordering_Makespan = [a/b for a,b in zip(Fig11_Muri_L_w_worstordering_Makespan_raw, Fig11_Muri_L_Makespan_raw)]
Fig11_Muri_L_wo_blossom_Makespan = [a/b for a,b in zip(Fig11_Muri_L_wo_blossom_Makespan_raw, Fig11_Muri_L_Makespan_raw)]

# Fig 12 number of jobs in one group
Fig12_antman_JCT = [a/b for a,b in zip(Fig12_antman_JCT_raw, Fig12_Muri_L_4_JCT_raw)]
Fig12_Muri_L_2_JCT = [a/b for a,b in zip(Fig12_Muri_L_2_JCT_raw, Fig12_Muri_L_4_JCT_raw)]
Fig12_Muri_L_3_JCT = [a/b for a,b in zip(Fig12_Muri_L_3_JCT_raw, Fig12_Muri_L_4_JCT_raw)]
Fig12_Muri_L_4_JCT = [1,1,1,1]
Fig12_antman_Makespan = [a/b for a,b in zip(Fig12_antman_Makespan_raw, Fig12_Muri_L_4_Makespan_raw)]
Fig12_Muri_L_2_Makespan = [a/b for a,b in zip(Fig12_Muri_L_2_Makespan_raw, Fig12_Muri_L_4_Makespan_raw)]
Fig12_Muri_L_3_Makespan = [a/b for a,b in zip(Fig12_Muri_L_3_Makespan_raw, Fig12_Muri_L_4_Makespan_raw)]
Fig12_Muri_L_4_Makespan = [1,1,1,1]

# Fig 13 workload distributions
Fig13_SRTF = [a/b for a,b in zip(Fig13_SRTF_raw, Fig13_Muri_S_raw)]
Fig13_Tiresias= [a/b for a,b in zip(Fig13_Tiresias_raw, Fig13_Muri_L_raw)]


legendfontsizeValue=30
fontsizeValue=40
legendfontsizeValueLarge=60
fontsizeValueLarge=88
def draw1(title:str, name_lists:list, lists:list, color:str, ncol=2):
    plt.clf()
    plt.rc('font',**{'size': 42, 'family': 'Arial' })
    plt.rc('pdf',fonttype = 42)
    trace = [str(i) for i in range(1, len(lists[0]) + 1)]
    # xt = ["0", "0.2", "0.4", "0.6", "0.8", "1"]
    xt = [1,2,3,4]
    x = np.arange(len(trace))  # the label locations
    total_width, n = 1, 2
    width = total_width / n
    x = x - (total_width - width) / 2
    fig, (ax1) = plt.subplots(1, 1, sharex=True)
    plt.figure(figsize=(16, 8))
    plt.grid(b=True, axis='y')
    l1 = plt.bar(x, lists[0], width, color=color, edgecolor='black', linewidth=2.0)
#    l2 = plt.bar(x+0.5*width, lists[1], width, edgecolor='black', linewidth=2.0)
    plt.xticks(x, xt, fontsize=fontsizeValueLarge)
    plt.xlabel("Profiling Noise", fontsize=fontsizeValueLarge)
    plt.ylabel(title, fontsize=fontsizeValueLarge)
    plt.ylim(0, max([max(i) for i in lists])*3/2) # 3.2/2 for fig 10; otherwise 5/4
    if max([max(i) for i in lists])*3/2>4:
        y_major_locator = plt.MultipleLocator(2)
        ax = plt.gca()
        ax.yaxis.set_major_locator(y_major_locator)
    else:
        y_major_locator = plt.MultipleLocator(1)
        ax = plt.gca()
        ax.yaxis.set_major_locator(y_major_locator)
    plt.tick_params(axis='both', which='major', labelsize=fontsizeValueLarge)
    plt.legend([l1], name_lists, loc = 'upper right', fontsize=legendfontsizeValueLarge, ncol=ncol, frameon=False)

def draw3(title:str, name_lists:list, lists:list, ncol=3):
    plt.clf()
    plt.rc('font',**{'size': 42, 'family': 'Arial' })
    plt.rc('pdf',fonttype = 42)
    trace = [str(i) for i in range(1, len(lists[0]) + 1)]
    x = np.arange(len(trace))  # the label locations
    total_width, n = 0.45, 3
    width = total_width / n
    x = x - (total_width - width) / 2
    fig, (ax1) = plt.subplots(1, 1, sharex=True)
    plt.figure(figsize=(16, 8))
    plt.grid(b=True, axis='y')
    l1 = plt.bar(x-1.0*width, lists[0], width, edgecolor='black', linewidth=2.0)
    l2 = plt.bar(x          , lists[1], width, edgecolor='black', linewidth=2.0)
    l3 = plt.bar(x+1.0*width, lists[2], width, edgecolor='black', linewidth=2.0)
    plt.xticks(x, trace, fontsize=fontsizeValue)
    plt.xlabel("Trace ID", fontsize=fontsizeValue)
    plt.ylabel(title, fontsize=fontsizeValue)
    plt.ylim(0, max([max(i) for i in lists])*3/2) # 3.2/2 for fig 10; otherwise 5/4
    if max([max(i) for i in lists])*3/2>4:
        y_major_locator = plt.MultipleLocator(2)
        ax = plt.gca()
        ax.yaxis.set_major_locator(y_major_locator)
    else:
        y_major_locator = plt.MultipleLocator(1)
        ax = plt.gca()
        ax.yaxis.set_major_locator(y_major_locator)
    plt.tick_params(axis='both', which='major', labelsize=fontsizeValue)
    plt.legend([l1, l2, l3], name_lists, loc = 'upper right', fontsize=legendfontsizeValue, ncol=ncol, frameon=False)

def draw4(title:str, name_lists:list, lists:list, ncol=4):
    plt.clf()
    plt.rc('font',**{'size': 42, 'family': 'Arial' })
    plt.rc('pdf',fonttype = 42)
    trace = [str(i) for i in range(1, len(lists[0]) + 1)]
    x = np.arange(len(trace))  # the label locations
    total_width, n = 0.45, 3
    width = total_width / n
    x = x - (total_width - width) / 2
    fig, (ax1) = plt.subplots(1, 1, sharex=True)
    plt.figure(figsize=(16, 8))
    plt.grid(b=True, axis='y')
    l1 = plt.bar(x-1.5*width, lists[0], width, edgecolor='black', linewidth=2.0)
    l2 = plt.bar(x-0.5*width, lists[1], width, edgecolor='black', linewidth=2.0)
    l3 = plt.bar(x+0.5*width, lists[2], width, edgecolor='black', linewidth=2.0)
    l4 = plt.bar(x+1.5*width, lists[3], width, edgecolor='black', linewidth=2.0)
    plt.xticks(x, trace, fontsize=fontsizeValue)
    plt.xlabel("Trace ID", fontsize=fontsizeValue)
    plt.ylabel(title, fontsize=fontsizeValue)
    plt.ylim(0, max([max(i) for i in lists])*4/3)
    if max([max(i) for i in lists])*4/3>4:
        y_major_locator = plt.MultipleLocator(2)
        ax = plt.gca()
        ax.yaxis.set_major_locator(y_major_locator)
    else:
        y_major_locator = plt.MultipleLocator(1)
        ax = plt.gca()
        ax.yaxis.set_major_locator(y_major_locator)
    plt.tick_params(axis='both', which='major', labelsize=fontsizeValue)
    plt.legend([l1, l2, l3, l4], name_lists, loc = 'upper right', fontsize=legendfontsizeValue, ncol=ncol, frameon=False)


draw3('Normalized\nAverage JCT', ['Muri-L', 'Muri-L w/ worst ordering', 'Muri-L w/o Blossom'], [Fig11_Muri_L_JCT, Fig11_Muri_L_w_worstordering_JCT, Fig11_Muri_L_wo_blossom_JCT], 1)
plt.savefig("Fig11a.pdf", bbox_inches='tight')
draw3('Normalized\nMakespan', ['Muri-L', 'Muri-L w/ worst ordering', 'Muri-L w/o Blossom'], [Fig11_Muri_L_Makespan, Fig11_Muri_L_w_worstordering_Makespan, Fig11_Muri_L_wo_blossom_Makespan], 1)
plt.savefig("Fig11b.pdf", bbox_inches='tight')
draw4('Normalized\nAverage JCT', ['AntMan', 'Muri-L-2', 'Muri-L-3', 'Muri-L-4'], [Fig12_antman_JCT,Fig12_Muri_L_2_JCT,Fig12_Muri_L_3_JCT,Fig12_Muri_L_4_JCT], 2)
plt.savefig("Fig12a.pdf", bbox_inches='tight')
draw4('Normalized\nMakespan', ['AntMan', 'Muri-L-2', 'Muri-L-3', 'Muri-L-4'], [Fig12_antman_Makespan,Fig12_Muri_L_2_Makespan,Fig12_Muri_L_3_Makespan,Fig12_Muri_L_4_Makespan], 2)
plt.savefig("Fig12b.pdf", bbox_inches='tight')

draw1('Speedup of\nAverage JCT', ['Speedup w.r.t. SRTF'], [Fig13_SRTF],'C0', 1)
plt.savefig("Fig13a.pdf", bbox_inches='tight')
draw1('Speedup of\nAverage JCT', ['Speedup w.r.t. Tiresias'], [Fig13_Tiresias],'C1', 1)
plt.savefig("Fig13b.pdf", bbox_inches='tight')
