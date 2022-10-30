import numpy as np
a = "[SP] positive [AC] food [OT] very delicious [AT] resuauratn quality"

index_ac = a.index("[AC]")
index_sp = a.index("[SP]")
index_at = a.index("[AT]")
index_ot = a.index("[OT]")

combined_list = [index_ac, index_sp, index_at, index_ot]
print(combined_list)
arg_index_list = list(np.argsort(combined_list))#.tolist()
print(arg_index_list)

result = []
for i in range(len(combined_list)):
    start = combined_list[i] + 4
    sort_index = arg_index_list.index(i)
    if sort_index < 3:
        next_ = arg_index_list[sort_index+1]
        re = a[start: combined_list[next_]]
    else:
        re = a[start:]
    result.append(re.strip())
print(result)


x = ["[AC]", "[SP]", "[AT]", "[OT]"]
from itertools import permutations

result = permutations(x)
for each in result:
    print(each)
