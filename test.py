path = '/Users/xuetaozhang/PycharmProjects/AndrOpGAN/target_dict.txt'
f = open(path,'r')
lines = f.readlines()
lllist = []
for line in lines:
    lllist.append(line.split('-')[0].split('/')[0].split('\\')[0].split(' ')[0])
lllist = list(set(lllist))
print(len(lllist))

for i in lllist:
    print(i)