# This file can draw the subfigures for Figure 9&10 in our paper.
# Usage: 
# 1. Change the raw values accordingly (existing numbers are our test results)
# 2. python3 draw_fig9-10.py
import matplotlib.pyplot as plt
import numpy as np

# ====== raw data start here ======
# [trace1, trace1_pr, trace2, trace2_pr, trace3, trace3_pr, trace4, trace4_pr]
SRTF_JCT_raw = [1370603.198, 2303543.816, 2665607.375, 4079154.454, 193081.796, 844394.78, 581982.9002, 1430856.446]
SRSF_JCT_raw = [980140.5699, 1801859.995, 2213949.687, 3384600.437, 185021.8802, 798057.6829, 487167.1688, 1218501.968]
Muri_S_JCT_raw = [867705.7941, 1467191.905, 1342400.019, 2325962.044, 85502.55387, 537776.6163, 365467.8911, 954225.2866]
Tiresias_L_JCT_raw = [3330371.307, 6921682.793, 7396368.514, 10555502.040026, 347213.2481, 2213110.128, 1319382.936, 2674022.636]
antman_JCT_raw = [5476391.317, 9808250.677, 15647814.621178, 18163389.280198, 305453.9407, 2674393.018, 2212462.719, 4676092.801]
themis_JCT_raw = [1823179.078, 3314303.178, 4628171.542, 12529490.82, 397058.802, 1502768.219, 994970.736, 2271053.053]
Muri_L_JCT_raw = [1072009.71, 2141936.535, 2544642.351, 4152485.536, 88686.97313, 921094.7449, 429174.8679, 1488033.596]

SRTF_Makespan_raw = [26045465.532945, 24513543.937825, 48847005.197635, 48697594.999386, 8174135.28, 7261795.828, 14787990.59235, 14272665.007687]
SRSF_Makespan_raw = [26065759.044845, 24272529.59311, 49026465.31107, 47972347.554251, 8174135.28, 6907790.403, 14887908.86639, 14426755.371188]
Muri_S_Makespan_raw = [18446535.222645, 16596420.784624, 34163947.020445, 32250093.334499, 8174135.28, 4400674.99, 11029124.28025, 9990192.513]
Tiresias_L_Makespan_raw = [23556410.217355, 21990477.824095, 48117093.307735, 47686997.556889, 8174226.28, 6761641.42, 13073838.61827, 12584603.56365]
antman_Makespan_raw = [21788668.549359, 20285085.619635, 48176959.246035, 48327650.120433, 8174140.28, 5799634.744, 12292270.736001, 11793469.427804]
themis_Makespan_raw = [24658268.63, 23369493.47, 47292466.46, 49310174.59, 8172104.51, 6760041.974, 13793094.42, 13471484.42]
Muri_L_Makespan_raw = [18593245.706217, 16639500.393368, 32826803.046468, 31782578.758801, 8174135.28, 4403652.203, 10918239.071764, 10104721.762665]

SRTF_99JCT_raw = [16647148.09074, 19871388.29092, 33128484.196403, 36640840.003736, 4538890.607, 6363099.549, 7768533.005, 9443500.203]
SRSF_99JCT_raw = [15529913.227905, 18267307.81118, 31241250.835485, 34656644.081924, 4189699.177, 6228160.594, 7372568.184, 9227908.326]
Muri_S_99JCT_raw = [10618148.302723, 13395942.722876, 19994293.572932, 23106016.887981, 992405.062, 4105513.658, 3734840.823, 6575927.929]
Tiresias_L_99JCT_raw = [14739851.5052, 21765400.784095, 37913098.47143, 44369455.221804, 4983397.254, 6739568.999, 8663311.856, 11219877.952119]
antman_99JCT_raw = [13690482.492206, 20130532.648104, 38571847.994149, 45081678.610968, 3904420.793, 5772254.906, 6438230.798, 9940651.791]
themis_99JCT_raw = [17159064.05, 20555281.24, 37238062.11, 44858584.44, 2967333.366, 6496197.725, 7968471.987, 10157394.49]
Muri_L_99JCT_raw = [11322989.635491, 13214336.412037, 22192059.12894, 24767218.772922, 927167.0483, 4167478.347, 4719318.299, 6383955.683]
# ====== raw data stop here ======

SRTF_JCT = [a/b for a,b in zip(SRTF_JCT_raw, Muri_S_JCT_raw)]
SRSF_JCT = [a/b for a,b in zip(SRSF_JCT_raw, Muri_S_JCT_raw)]
Muri_S_JCT = [1, 1, 1, 1, 1, 1, 1, 1]
Tiresias_L_JCT = [a/b for a,b in zip(Tiresias_L_JCT_raw, Muri_L_JCT_raw)]
antman_JCT = [a/b for a,b in zip(antman_JCT_raw, Muri_L_JCT_raw)]
themis_JCT = [a/b for a,b in zip(themis_JCT_raw, Muri_L_JCT_raw)]
Muri_L_JCT = [1, 1, 1, 1, 1, 1, 1, 1]

SRTF_Makespan = [a/b for a,b in zip(SRTF_Makespan_raw, Muri_S_Makespan_raw)]
SRSF_Makespan = [a/b for a,b in zip(SRSF_Makespan_raw, Muri_S_Makespan_raw)]
Muri_S_Makespan = [1, 1, 1, 1, 1, 1, 1, 1]
Tiresias_L_Makespan = [a/b for a,b in zip(Tiresias_L_Makespan_raw, Muri_L_Makespan_raw)]
antman_Makespan = [a/b for a,b in zip(antman_Makespan_raw, Muri_L_Makespan_raw)]
themis_Makespan = [a/b for a,b in zip(themis_Makespan_raw, Muri_L_Makespan_raw)]
Muri_L_Makespan = [1, 1, 1, 1, 1, 1, 1, 1]

SRTF_99JCT = [a/b for a,b in zip(SRTF_99JCT_raw, Muri_S_99JCT_raw)]
SRSF_99JCT = [a/b for a,b in zip(SRSF_99JCT_raw, Muri_S_99JCT_raw)]
Muri_S_99JCT = [1, 1, 1, 1, 1, 1, 1, 1]
Tiresias_L_99JCT = [a/b for a,b in zip(Tiresias_L_99JCT_raw, Muri_L_99JCT_raw)]
antman_99JCT = [a/b for a,b in zip(antman_99JCT_raw, Muri_L_99JCT_raw)]
themis_99JCT = [a/b for a,b in zip(themis_99JCT_raw, Muri_L_99JCT_raw)]
Muri_L_99JCT = [1, 1, 1, 1, 1, 1, 1, 1]

legendfontsizeValue=45
fontsizeValue=70
def draw3_new(title:str, name_lists:list, lists:list, ncol=3):
    plt.clf()
    plt.rc('font',**{'size': 42, 'family': 'Arial' })
    plt.rc('pdf',fonttype = 42)
#    trace = [str(i) for i in range(1, len(lists[0]) + 1)]
    trace = ["1", "1'", "2", "2'", "3", "3'", "4", "4'"]
    x = np.arange(len(trace))  # the label locations
    total_width, n = 0.6, 3
    width = total_width / n
    x = x - (total_width - width) / 2
    fig, (ax1) = plt.subplots(1, 1, sharex=True)
    plt.figure(figsize=(20, 10))
    plt.grid(b=True, axis='y')
    l1 = plt.bar(x-1.0*width, lists[0], width, edgecolor='black', linewidth=2.0)
    l2 = plt.bar(x          , lists[1], width, edgecolor='black', linewidth=2.0)
    l3 = plt.bar(x+1.0*width, lists[2], width, edgecolor='black', linewidth=2.0)
    plt.xticks(x, trace, fontsize=fontsizeValue)
    plt.xlabel("Trace ID", fontsize=fontsizeValue)
    plt.ylabel(title, fontsize=fontsizeValue)
    plt.ylim(0, max([max(i) for i in lists])*5/4)
    if max([max(i) for i in lists])*5/4>3:
        y_major_locator = plt.MultipleLocator(2)
        ax = plt.gca()
        ax.yaxis.set_major_locator(y_major_locator)
#        ax.axhline(1, linestyle='--', color='k')
    plt.tick_params(axis='both', which='major', labelsize=fontsizeValue)
    plt.legend([l1, l2, l3], name_lists, loc = 'upper right', fontsize=legendfontsizeValue, ncol=ncol, frameon=False)

def draw4_new(title:str, name_lists:list, lists:list, ncol=2):
    plt.clf()
    plt.rc('font',**{'size': 42, 'family': 'Arial' })
    plt.rc('pdf',fonttype = 42)
#    trace = [str(i) for i in range(1, len(lists[0]) + 1)]
    trace = ["1", "1'", "2", "2'", "3", "3'", "4", "4'"]
    x = np.arange(len(trace))  # the label locations
    total_width, n = 0.6, 3
    width = total_width / n
    x = x - (total_width - width) / 2
    fig, (ax1) = plt.subplots(1, 1, sharex=True)
    plt.figure(figsize=(20, 10))
    plt.grid(b=True, axis='y')
    l1 = plt.bar(x-1.5*width, lists[0], width, edgecolor='black', linewidth=2.0)
    l2 = plt.bar(x-0.5*width, lists[1], width, edgecolor='black', linewidth=2.0)
    l3 = plt.bar(x+0.5*width, lists[2], width, edgecolor='black', linewidth=2.0)
    l4 = plt.bar(x+1.5*width, lists[3], width, edgecolor='black', linewidth=2.0)
    plt.xticks(x, trace, fontsize=fontsizeValue)
    plt.xlabel("Trace ID", fontsize=fontsizeValue)
    plt.ylabel(title, fontsize=fontsizeValue)
    plt.ylim(0, max([max(i) for i in lists])*4.3/3)
    if max([max(i) for i in lists])*4.3/3>3:
        y_major_locator = plt.MultipleLocator(2)
        ax = plt.gca()
        ax.yaxis.set_major_locator(y_major_locator)
#    ax.axhline(1, linestyle='--', color='k')
    plt.tick_params(axis='both', which='major', labelsize=fontsizeValue)
    plt.legend([l1, l2, l3, l4], name_lists, loc = 'upper right', fontsize=legendfontsizeValue, ncol=ncol, frameon=False)

draw3_new('Normalized\nAverage JCT', ['SRTF', 'SRSF', 'Muri-S'], [SRTF_JCT, SRSF_JCT, Muri_S_JCT])
plt.savefig("Figure9a.pdf", bbox_inches='tight')
draw4_new('Normalized\nAverage JCT', ['Tiresias', 'AntMan', 'Themis', 'Muri-L'], [Tiresias_L_JCT, antman_JCT, themis_JCT, Muri_L_JCT])
plt.savefig("Figure10a.pdf", bbox_inches='tight')

draw3_new('Normalized\nMakespan', ['SRTF', 'SRSF', 'Muri-S'], [SRTF_Makespan, SRSF_Makespan, Muri_S_Makespan])
plt.savefig("Figure9b.pdf", bbox_inches='tight')
draw4_new('Normalized\nMakespan', ['Tiresias', 'AntMan', 'Themis', 'Muri-L'], [Tiresias_L_Makespan, antman_Makespan, themis_Makespan, Muri_L_Makespan])
plt.savefig("Figure10b.pdf", bbox_inches='tight')

draw3_new('Normalized\n99th%-ile JCT', ['SRTF', 'SRSF', 'Muri-S'], [SRTF_99JCT, SRSF_99JCT, Muri_S_99JCT])
plt.savefig("Figure9c.pdf", bbox_inches='tight')
draw4_new('Normalized\n99th%-ile JCT', ['Tiresias', 'AntMan', 'Themis', 'Muri-L'], [Tiresias_L_99JCT, antman_99JCT, themis_99JCT, Muri_L_99JCT])
plt.savefig("Figure10c.pdf", bbox_inches='tight')
