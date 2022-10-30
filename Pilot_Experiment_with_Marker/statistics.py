
file = open("rest16_appendix.txt", "r")

line = file.readline()
all_results = {}
line_num = 1
order = ""
while line:
    if line_num % 4 == 1:
        seed, order = line.split(" ------------------- ")
        order = order.strip()
        if order not in all_results:
            all_results[order] = []
    elif line_num % 4 == 2 or line_num % 4 == 0:
        pass
    elif line_num % 4 == 3:
        results = line.split(" ")
        res = {"pre": float(results[1]),
               "rec": float(results[3]),
               "f1": float(results[6].strip())}
        all_results[order].append(res)
    line = file.readline()
    line_num += 1

print(all_results)

file = open("final_result16_format.txt", "a+")
file.write("Order & Precision & Recall & F1 \n")

for each_order in all_results:
    total = all_results[each_order]
    num = len(total)
    pre, rec, f1 = 0, 0, 0
    for each in total:
        pre += each["pre"]
        rec += each['rec']
        f1 += each['f1']

    pre = pre / num
    rec = rec / num
    f1 = f1 / num

    file.write(each_order + " & " + str("%.6f"%pre) + " & " + str("%.6f"%rec) + " & " + str("%.6f"%f1) + "\n")


