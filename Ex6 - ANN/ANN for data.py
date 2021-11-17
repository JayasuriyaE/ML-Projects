import pandas as pd 

data = {'x1':[0.5,0.1,0.4],
'x2':[0.2,0.2,0.3],
'target':[0,1,0]}
df = pd.DataFrame(data)
# print(data['x1'][1])

epoch=int(input("Enter the no. of epoch :\t"))
w0 = float(input("Enter the weight for bias :\t"))
w1 = float(input("Enter the weight for x1 :\t"))
w2 = float(input("Enter the weight for x2 :\t"))
lr = float(input("Enter the Learning rate : \t"))
x0 = float(1)


print("\nx1","x2","T","w0","w1","w2",sep="    ")
for i in range(0,epoch,1):
    rows = len(df.index)
    print()
    for row in range(0,rows):
        x1=float(data['x1'][row])
        x2=float(data['x2'][row])
        target =int(data['target'][row])
        # print(x1," ",x2," ",target)
        sum = ((x0*w0)+(x1*w1)+(x2*w2))
#applying activation function
        if(sum>= 0):
            sum = 1
        else:
            sum = 0
        
        if(sum == target):
            print(x1," ",x2," ",target," ",w0," ",w1," ",w2)
        else:
            # w = w + learning_rate * (expected - predicted) * x
            w0 = w0 +((lr*(target-sum))*x0)
            w1 = w1 +((lr*(target-sum))*x1)
            w2 = w2 +((lr*(target-sum))*x2)
            w0 = round(w0,2)
            w1 = round(w1,2)
            w2 = round(w2,2)
            print(x1," ",x2," ",target," ",w0," ",w1," ",w2)
