import random
import pandas

data = {"Expression" : [], "Result" : []}

for i in range(100):
    for j in range(i,100):
        t = random.randint(0,1)
        if t == 1:
            data["Expression"].append(f"{i}*{j}")
        else:
            data["Expression"].append(f"{j}*{i}")
random.shuffle(data["Expression"]);

for i in data["Expression"] :
    x,y = i.split("*");
    x = int(x);
    y = int(y);
    data["Result"].append(x*y);

df = pandas.DataFrame(data);
df.to_csv("products.csv", index = False)
