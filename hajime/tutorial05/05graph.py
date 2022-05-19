from matplotlib import pyplot as plt

train_num = [1, 2, 3, 5, 10, 20, 30, 50, 100]
normal = [90.967056, 91.781792, 92.773645, 91.852639,
          93.446688, 93.234148, 92.950762, 93.269571, 93.552958]
bigram = [90.223167, 91.888062, 92.100602, 92.667375,
          92.667375, 92.773645, 92.773645, 92.773645, 92.773645]
prep = [90.719093, 83.882395, 92.171449, 92.879915,
        93.198725, 93.021608, 93.304995, 93.304995, 93.304995]

plt.plot(train_num, normal, label="1-gram", ls=":", marker="o")
plt.plot(train_num, bigram, label="2-gram", ls=":", marker="o")
plt.plot(train_num, prep, label="1-gram+stopwords", ls=":", marker="o")
plt.ylim(82.5, 95)
plt.xscale('log')
plt.xlabel("training iteration")
plt.ylabel("accuracy (%)")
plt.legend(loc="best")

# plt.show()
plt.savefig("accuracy05.png")
