def findXNumber(t: str, x: str) -> int:
    res = [1, 1]
    info = t.split()
    m = int(info[0])
    n = int(info[1])
    model = int(info[2])
    num = int(x)
    temp = int(x)
    while temp > 2:
        if model == 0:
            res.append(m*res[num-temp+1]+n*res[num-temp])
        else:
            res.append(m*res[num-temp+1]-n*res[num-temp])
        temp = temp - 1

    return res[-1]


# t = eval(input())
# while t:
#     x = eval(input())
#     break

res = findXNumber("2 3 1", "4")

print(res)
