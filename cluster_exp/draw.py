import csv
import matplotlib.pyplot as plt

SRTF = {"time":[], "queue_length":[], "blocking_index":[], "gpu_util":[], "cpu_util":[], "io_read_speed":[]}
SRSF = {"time":[], "queue_length":[], "blocking_index":[], "gpu_util":[], "cpu_util":[], "io_read_speed":[]}
Muri_S = {"time":[], "queue_length":[], "blocking_index":[], "gpu_util":[], "cpu_util":[], "io_read_speed":[]}
Tiresias_L = {"time":[], "queue_length":[], "blocking_index":[], "gpu_util":[], "cpu_util":[], "io_read_speed":[]}
Themis = {"time":[], "queue_length":[], "blocking_index":[], "gpu_util":[], "cpu_util":[], "io_read_speed":[]}
Muri_L = {"time":[], "queue_length":[], "blocking_index":[], "gpu_util":[], "cpu_util":[], "io_read_speed":[]}

def read_csv(file:str, result:dict):
    with open(file) as f:
        reader = csv.reader(f, delimiter=',')
        header = next(reader)
        count = 0
        for row in reader:
            count += 1
            if count % 50 != 0:
                continue
            result["time"].append(float(row[0]) / 3600)
            result["queue_length"].append(int(row[1]))
            result["blocking_index"].append(float(row[2]))
            result["gpu_util"].append(float(row[3]))
            result["cpu_util"].append(float(row[4])/8*96/6/8)
            result["io_read_speed"].append(result["io_read_speed"][-1] if float(row[5]) > 1000000 else float(row[5]))
        for i in range(len(result["io_read_speed"])):
            result["io_read_speed"][i] /= 1024

read_csv("results/SRTF/cluster.csv", SRTF)
read_csv("results/SRSF/cluster.csv", SRSF)
read_csv("results/Muri-S/cluster.csv", Muri_S)
read_csv("results/tiresias/cluster.csv", Tiresias_L)
read_csv("results/themis/cluster.csv", Themis)
read_csv("results/Muri-L/cluster.csv", Muri_L)


def aware(x:str, ax=None):
    plt.plot(SRTF["time"], SRTF[x], '', label="SRTF")
    plt.plot(SRSF["time"], SRSF[x], '', label="SRSF")
    plt.plot(Muri_S["time"], Muri_S[x], '', color = 'green', label="Muri-S")

    if x == "queue_length":
        plt.ylabel("Queue Length")
    elif x == "blocking_index":
        plt.ylabel("Blocking Index")
    elif x == "gpu_util":
        plt.ylabel("GPU Util (%)")
    elif x == "cpu_util":
        plt.ylabel("CPU Util (%)")
    elif x == "io_read_speed":
        plt.ylabel("IO speed (MB/s)")

def unaware(x:str, ax=None):
    plt.plot(Tiresias_L["time"], Tiresias_L[x], '', label="Tiresias")
    plt.plot(Themis["time"], Themis[x], '', label="Themis")
    plt.plot(Muri_L["time"], Muri_L[x], '', color = 'green',label="Muri-L")

    if x == "queue_length":
        plt.ylabel("Queue Length")
    elif x == "blocking_index":
        plt.ylabel("Blocking Index")
    elif x == "gpu_util":
        plt.ylabel("GPU Util (%)")
    elif x == "cpu_util":
        plt.ylabel("CPU Util (%)")
    elif x == "io_read_speed":
        plt.ylabel("IO speed (MB/s)")
    # plt.yaxis.set_ticks_position('left')

    # ax.spines['top'].set_color('none')
    # ax.spines['right'].set_color('none')

    # plt.savefig("unaware_" + x + ".pdf", bbox_inches='tight')

def draw_all(x:list):
    plt.clf()
    plt.rc('font',**{'size': 36, 'family': 'Arial' })
    plt.rc('pdf',fonttype = 42)
    fig=plt.figure(figsize=(8, 25))
    plt.subplot(5,1,1)
    aware(x[0])
    plt.legend(loc='upper right', fontsize=24)
    ax1 = plt.subplot(5,1,2)
    aware(x[1], ax1)
    plt.subplot(5,1,3)
    aware(x[2])
    plt.subplot(5,1,4)
    aware(x[3])
    plt.subplot(5,1,5)
    aware(x[4])
    plt.xlabel('Time (h)')
    fig.align_labels()
    plt.savefig("Figure8a.pdf", bbox_inches='tight')

    plt.clf()
    plt.rc('font',**{'size': 36, 'family': 'Arial' })
    plt.rc('pdf',fonttype = 42)
    fig=plt.figure(figsize=(8, 25))
    plt.subplot(5,1,1)
    unaware(x[0])
    plt.legend(loc='upper right', fontsize=24)
    ax1 = plt.subplot(5,1,2)
    unaware(x[1], ax1)
    plt.subplot(5,1,3)
    unaware(x[2])
    plt.subplot(5,1,4)
    unaware(x[3])
    plt.subplot(5,1,5)
    unaware(x[4])
    plt.xlabel('Time (h)')
    fig.align_labels()
    plt.savefig("Figure8b.pdf", bbox_inches='tight')

if __name__ == "__main__":
    draw_all(["queue_length", "blocking_index", "gpu_util", "cpu_util", "io_read_speed"])
