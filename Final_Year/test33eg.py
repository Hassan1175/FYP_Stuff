file =  open("O:\\Nama_College\\FYP\\Final_Year\\dlib_predicted_labels.txt","r")
fil2 =  open("O:\\Nama_College\\FYP\\Final_Year\\dlib_testing_labels.txt","r")
kk = file.read()
kk2 = fil2.read()

lo =(kk.split())
lo2 = kk2.split()
new_list2  = []


new_list = []
for item in lo:
    new_str = ""
    for item2 in item:
        if item2.isalpha() == True:
            new_str = new_str + item2
    new_list.append(new_str)
print(new_list)
print(type(new_list))

for mm in lo2:
    new_str2 =""
    for mm2 in mm:
        if mm2.isalpha()==True:
            new_str2 = new_str2 + mm2
    new_list2.append(new_str2)
# print(new_list[0])
print(new_list2)
print(type(new_list2))

print(len(new_list))
print(len(new_list2))



count = 0.0
correct = 0.0
for i in range(len(new_list)):
   count+=1
   if new_list[i]==new_list2[i]:
     correct+=1



print(count)
print(correct)
M = ((correct/count))
Accuracy = M*100



print("Auuracy is ",Accuracy ," PERCENT")
