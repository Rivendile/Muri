import csv
import sys

log_path = sys.argv[1]

csv_reader = csv.reader(open(log_path+"/job.csv"))
jct_sum = 0
makespan = 0
cnt = 0
jct_list = []
for line_id,line in enumerate(csv_reader):
    if line_id > 0:
        jct_sum += float(line[-5])
        makespan = max(makespan, float(line[5]))
        cnt += 1
        jct_list.append(float(line[-5]))

jct_list.sort()

print("Total jobs: %d, avg JCT: %.6f, makespan: %.6f, 99th JCT: %.6f" % (cnt, jct_sum/cnt, makespan, jct_list[int(cnt*0.99)]))