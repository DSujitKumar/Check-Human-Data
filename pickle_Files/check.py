import pickle
images={}

pickle_in=open("images.pickle","rb")

images=pickle.load(pickle_in)
for i,keys in enumerate(images,start=1):
    print(keys,end=' : ')
    print(images[keys][1])
print(i)


