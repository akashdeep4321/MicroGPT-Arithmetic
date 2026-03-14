import random
import pandas

random.seed(42)

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

train = {"Expression" : data["Expression"][:len(data["Expression"])//2], "Result" : data["Result"][:len(data["Result"])//2]};
test = {"Expression" : data["Expression"][len(data["Expression"])//2:], "Result" : data["Result"][len(data["Result"])//2:]};

df = pandas.DataFrame(train);
df.to_csv("2-digit-products-train.csv", index = False)

dff = pandas.DataFrame(test);
dff.to_csv("2-digit-products-test.csv", index = False)

    
