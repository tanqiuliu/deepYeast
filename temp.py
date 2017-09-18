from util import *
"""将positive data中各种size的cell均衡化,避免某一类cell过多"""
NB_EACH_BIN = 1000

filename = './data/rect_data.csv'
data = np.loadtxt(filename, dtype=np.int32, comments='#', delimiter=' ')
np.random.shuffle(data)
X = data[:,4:]
y = data[:,0:4]
area = np.array([(yy[1]-yy[0])*(yy[3]-yy[2]) for yy in list(y)])
sepes = np.histogram(area, bins=20)[1]
counts = np.histogram(area, bins=20)[0]

data2 = list()
for idx in range(20):
    data2.append([])
for idx in range(data.shape[0]):
    for ii in range(20):
        if(sepes[ii] <= area[idx] < sepes[ii+1]):
            data2[ii].append(data[idx])

data3 = []
for idx_bin in range(20):
    if(len(data2[idx_bin]) != 0):
        data3.append(np.vstack(tuple(data2[idx_bin])))

for idx_bin in range(len(data3)):
    if(data3[idx_bin].shape[0] > NB_EACH_BIN):
        np.random.shuffle(data3[idx_bin])
        data3[idx_bin] = data3[idx_bin][0:NB_EACH_BIN, :]
data4 = np.vstack(tuple(data3))
np.savetxt('./data/rect_data2.csv', data4, fmt='%d', header = 'rect_examples, first 4 cols are target')
