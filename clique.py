import  pandas as pd
import  numpy as np
pd.set_option('display.height',1000)
pd.set_option('display.max_rows',2000)
pd.set_option('display.max_columns',500)
pd.set_option('display.width',1000)
x = pd.read_csv("多维单元聚类分析测试数据.CSV",index_col=0)

def meshing(d1,d2,d3,d4,d5,d6,x):
    x['GR']=(x.ix[:,0] - x.ix[:,0].min())//d1
    x['SP']=(x.ix[:,0] - x.ix[:,0].min())//d2
    x['ILD']=(x.ix[:,0] - x.ix[:,0].min())//d3
    x['DT']=(x.ix[:,0] - x.ix[:,0].min())//d4
    x['NPHI']=(x.ix[:,0] - x.ix[:,0].min())//d5
    x['DEN']=(x.ix[:,0] - x.ix[:,0].min())//d6
    return x



def clustering(x):
    grouped = x.groupby(level = 0)
    a = list()
    for name,group in grouped:
        list.append(a,name)
    z = 0
    new_index = np.array(x.index.get_level_values(0))
    yijulei = list([500])
    for i in range(len(a)):
        print((i+1)*100/len(a),'%')
        b = grouped.get_group(a[i])
        #b.index = b.index.droplevel()
        if i in yijulei:
            continue
        z+=1
        for s in range(len(new_index)):
            if new_index[s] == a[i]:
                new_index[s] = z
        for j in range(i+1,len(a)):
            if j in yijulei:
                continue
            else:
                c = grouped.get_group(a[j])
                #c.index = c.index.droplevel()
                fa = 0
                fb = 0
                bt = b.drop_duplicates()
                for k in range(len(bt)):
                    for n in range(len(c)):
                        if all(bt.values[k]==c.values[n]):
                            fa = fa+1
                        else:
                            fa = fa
                ct = c.drop_duplicates()
                for k in range(len(ct)):
                    for n in range(len(b)):
                        if all(ct.values[k] == b.values[n]):
                            fb = fb + 1
                        else:
                            fb = fb
                ra = fb/b.shape[0]
                rb = fa/c.shape[0]
                r = max(ra,rb)
                if r>q:
                    list.append(yijulei,j)
                    for s in range(len(new_index)):
                        if new_index[s]==a[j]:
                            new_index[s]=z
    x['class'] = new_index
    x = x.set_index(['class',x.index])
    return x

d1 = 2#float(input("Please input distance for GR:\n"))
d2 = 2#float(input("Please input distance for SP:\n"))
d3 = 2#float(input("Please input distance for ILD:\n"))
d4 = 2#float(input("Please input distance for DT:\n"))
d5 = 0.3#float(input("Please input distance for NPHI:\n"))
d6 = 0.3#float(input("Please input distance for DEN:\n"))
q = 0.5#float(input("Please input distance for partition coefficient :\n"))
x = meshing(d1,d2,d3,d4,d5,d6,x)

old_class = x.index.get_level_values(0)
x = clustering(x)
new_class = x.index.get_level_values(0)
while all(old_class == new_class):
    break
else:
    x = clustering(x)
    old_class = new_class
    new_class = x.index.get_level_values(0)



# x['class2'] = range(1787,0,-1)
# x = x.set_index(['class2',x.index])


# print(x)
print(x)
print(x.index)
# print(z)
#print(yijulei)

# b = grouped.get_group(a[0])
# print(b.shape[0])

