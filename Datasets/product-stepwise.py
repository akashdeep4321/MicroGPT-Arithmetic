import random
import pandas as pd

dataset = {"Expression" : []}

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
        s += (x + '*' + y + ':')
        for r in y[::-1]:
            for t in x[::-1]:
                u = str(int(r)*int(t))
                if len(u) == 1:
                    u = '0' + u
                s += (r + '*' + t + '=' + u + ',')
        v = str(int(x)*int(y))
        while(len(v)<4):
            v = '0' + v
        dataset["Expression"].append(s[:-1] + ':' + v)
random.shuffle(dataset["Expression"])
train_dataset = {"Expression" : dataset["Expression"][:len(dataset["Expression"])//2]}
test_dataset = {"Expression" : dataset["Expression"][len(dataset["Expression"])//2:]}
df_train = pd.DataFrame(train_dataset)
df_train.to_csv('product_algo-train.csv', index = False)
df_test = pd.DataFrame(test_dataset)
df_test.to_csv('product_algo-test.csv', index = False)
                
