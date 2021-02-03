import matplotlib.pyplot as plt

# times to run
xmalloc = []
mem = []
trainerrorrate = []
testerrorrate = []

# run 0
xmalloc.append(58.355)
trainerrorrate.append(0.6125306)
testerrorrate.append(0.59482155)
mem.append(10674)

xmalloc.append(60.686)
trainerrorrate.append(0.5600)
testerrorrate.append(0.5598)
mem.append(14693)

# run 2
xmalloc.append(84.33)
trainerrorrate.append(0.5950)
testerrorrate.append(0.6298)
mem.append(10673)

# run 3
xmalloc.append(72.70)
trainerrorrate.append(0.542527)
testerrorrate.append(0.7347795)
mem.append(10673.88)

# run 4
xmalloc.append(82.3824)
trainerrorrate.append(0.6825)
testerrorrate.append(0.411987)
mem.append(10673.88)

# run 5
xmalloc.append(81.187)
trainerrorrate.append(0.7525)
testerrorrate.append(0.3149)
mem.append(14547.17)

# run 6
xmalloc.append(79.499)
trainerrorrate.append(0.5075)
testerrorrate.append(0.6298)
mem.append(14693)

# run 7
xmalloc.append(63.115)
trainerrorrate.append(0.560028)
testerrorrate.append(0.6998)
mem.append(10673)

# run 8
xmalloc.append(57.83475)
trainerrorrate.append(0.5950)
testerrorrate.append(0.6298)
mem.append(10657)

# run 9
xmalloc.append(82.35)
trainerrorrate.append(0.4900)
testerrorrate.append(0.73478)
mem.append(14547)

trainerrorrate.append(0.525)
testerrorrate.append(0.7697)
trainerrorrate.append(0.4900)
testerrorrate.append(0.5948)
trainerrorrate.append(0.5775)
testerrorrate.append(0.6648)
trainerrorrate.append(0.5950)
testerrorrate.append(0.6648)
trainerrorrate.append(0.6125)
testerrorrate.append(0.5948)
trainerrorrate.append(0.542527)
testerrorrate.append(0.8747)
trainerrorrate.append(0.6300)
testerrorrate.append(0.5598)
trainerrorrate.append(0.595)
testerrorrate.append(0.6298)
trainerrorrate.append(0.61253)
testerrorrate.append(0.59482)
trainerrorrate.append(0.4900)
testerrorrate.append(0.804758)
x = list(range(1, len(xmalloc)+1))
avtime = sum(xmalloc)/len(xmalloc)
avmem = sum(mem)/len(mem)
y = []
time = []
memory = []
yt = []
yz = []
ye = trainerrorrate + testerrorrate
ya = sum(ye)/len(ye)
ytr = sum(trainerrorrate)/len(trainerrorrate)
yte = sum(testerrorrate)/len(testerrorrate)
# print("training error", ytr)
# print("testing error", yte)
# print("overall error", ya)
for i in range(len(xmalloc)):
    memory.append(avmem)
    time.append(avtime)
    y.append(ya)
    yt.append(ytr)
    yz.append(yte)
plt.plot(x, mem)
plt.plot(x, memory)
print(avmem)
# plt.plot(x, testerrorrate)
# plt.plot(x, y)
# plt.plot(x, yt)
# plt.plot(x, yz)
# plt.legend(["training Error", "Testing Error", "Average Error", "Average Training Error", "Average Testing Error"])
plt.xlabel("Trials")
# plt.legend("Peak memory usage")
plt.ylabel("Peak memory usage in MB")
# plt.ylabel("Error Rates as Percentage of Incorrect Labels")
plt.show()
