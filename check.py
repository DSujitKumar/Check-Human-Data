import numpy as np
import pickle
images={}
img={}
imagesSort={}
pickle_in=open("images.pickle","rb")

# images=pickle.load(pickle_in)
# for i,keys in enumerate(images,start=1):
#     print(keys,end=' : ')
#     #a=images[keys][0]

#     if (images[keys][1]):
#     	print(images[keys][1])
#     else:
#     	print("not Available")
#     # for key in images[keys][0]:
#     # 	if(key["rect"]):
#     # 		print(key["rect"])
#     # 	else:
#     # 		print("Not Avail")
# print(i)
# #print(a)

images=pickle.load(pickle_in)
for i,keys in enumerate(images,start=1):
	#print(keys,end=' : ')
	for key in images[keys][0]:
		if(key["rect"]):
			#print(key["rect"])
			img.update({keys:key["rect"]})
#imagesSort=sorted(img.values(),reverse=False, key = lambda kv:(kv[1], kv[0]))
imagesSort=[[k, img[k]] for k in sorted(img, key=img.get, reverse=True)]
for k in imagesSort:
	print(k[0],end=':')
	print(k[1])
print(type(imagesSort))
