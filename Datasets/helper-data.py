import random
import pandas as pd

dataset = {"Expression" : [], "Help" : [], "Label" : []}

for i in range(0,100):
    for j in range(i,100):
        s = ""
        flag = random.randint(0,1);
        if flag == 1:
            x = str(i)
            y = str(j)
        else:
            x = str(j)
            y = str(i)
        if len(x) == 1:
            x = '0' + x
        if len(y) == 1:
            y = '0' + y
        s += (x + '*' + y)
        dataset["Expression"].append(s);
        s = "";
        K=0
        for r in y[::-1]:
            e = int(r)
            for k in range(K):
                e*=10
            u = str(e*int(x))
            while len(u) < 4:
                u = '0' + u
            s += (u + '\n')
            K+=1
        v = str(int(x)*int(y))
        while(len(v)<4):
            v = '0' + v
        dataset["Help"].append(s[:-1])
        dataset["Label"].append(v)
comb = list(zip(dataset["Expression"], dataset["Help"], dataset["Label"]));
random.shuffle(comb)
dataset["Expression"], dataset["Help"], dataset["Label"] = zip(*comb)
print(len(dataset["Expression"]), len(dataset["Help"]), len(dataset["Label"]))
train_dataset = {"Expression" : dataset["Expression"][:len(dataset["Expression"])//2], "Help" : dataset["Help"][:len(dataset["Help"])//2], "Label" : dataset["Label"][:len(dataset["Label"])//2]}
test_dataset = {"Expression" : dataset["Expression"][len(dataset["Expression"])//2:], "Help" : dataset["Help"][len(dataset["Help"])//2:], "Label" : dataset["Label"][len(dataset["Label"])//2:]}
df_train = pd.DataFrame(train_dataset)
df_train.to_csv('product_algo-train.csv', index = False)
df_test = pd.DataFrame(test_dataset)
df_test.to_csv('product_algo-test.csv', index = False)
print(train_dataset["Expression"][:1], train_dataset["Help"][:1], train_dataset["Label"][:1]);
