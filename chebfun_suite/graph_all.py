import numpy as np
from matplotlib import pyplot as plt

def stacktoarray(list):
    list = list.replace(" ","")
    list = list.replace("*^","e")
    list = list.split("\n")
    while (list[-1] == ''):
        list.pop()
    for i in range(len(list)):
        list[i] = abs(float(list[i]))
    return list



args = ["times","avg_resids","max_resids"]
arg_names = dict()
arg_names["times"] = "Times"
arg_names["avg_resids"] = "Average Residuals"
arg_names["max_resids"] = "Maximum Residuals"
names = ["1.1","1.2","1.3","1.4","1.5","2.1","2.2","2.3","2.4","2.5","3.1","3.2","4.1","4.2","5.1","6.1","6.2","6.3","7.1","7.2","7.3","7.4","8.1","8.2","9.1","9.2","10.1"]


for arg in args:
    YR = np.load("yroots_results/2d_{}.npy".format(arg),allow_pickle=True)

    JR = open("jroots_results/2d_{}.txt".format(arg))
    JR = stacktoarray(JR.read())

    JR2 = open("jroots_results2/2d_{}.txt".format(arg))
    JR2 = stacktoarray(JR.read())

    # NR = open("NSolveValues_results/2d_{}.txt".format(arg))
    # NR = stacktoarray(NR.read())

    # RR = open("reduce_results/2d_{}.txt".format(arg))
    # RR = stacktoarray(RR.read())

    # CR = open("chebfun_results/2d_{}.txt".format(arg))
    # CR = stacktoarray(CR.read())

    line_args = [YR, JR, JR2]

    labels = ["Yroots","Jroots", "Chebfun","Reduce","NSolveValues"]
    colors = ["royalblue", "tab:purple", "tab:orange","tab:green","tab:red"]
    
    fig, ax = plt.subplots(layout='constrained', figsize=(5.5, 3.5), dpi=300)
    ax.set_yscale("log")
    x = np.arange(len(names)) * 1.2
    width = 0.2  
    multiplier = 0

    for i in range(len(labels)):
        offset = width * multiplier
        ax.bar(x + offset, line_args[i], width, label=labels[i],color=colors[i])
        multiplier += 1
    ax.set_xticks(x + 1.5*width, names,rotation=45)

    if (arg == "times"):
        ax.set_ylabel("Time Log Scale")
    else:
        ax.set_ylim(1e-17,1e-4)
        ax.set_ylabel("Residual")
    ax.set_title("Chebfun Suite {}".format(arg_names[arg]))
    ax.legend()
    plt.savefig("graphs/temp_chebfun_suite_{}.png".format(arg), dpi=300,bbox_inches="tight")
    plt.clf()
