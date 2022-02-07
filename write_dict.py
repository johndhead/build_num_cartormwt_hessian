#
# example to read and write dictionary to file
#
exam_dict = {"name": "bill", "pseudo": "william","numa": 100,"numb":6.22}

file_name = "test_dict_rd_wr" + ".dnpy"
with open(file_name,"w") as dict_file:
    dict_file.write("%d: items in %s - dictionary starts on next line\n" % (len(exam_dict),file_name))
    for key, value in exam_dict.items():
        dict_file.write('%s:%s\n' % (key, value))
    dict_file.close()

# read back file 
new_dict = {}

with open(file_name,"r") as dict_file:
    line = dict_file.readline()
    print("line = %s"% line)
    print("split line: ",line.split(":",2))
    no_items = line.split(":",2)
    print("dictionary has %s items \n Comment = %s" % (no_items[0],no_items[1]))
    for dict_lines in range(int(no_items[0])):
        line = dict_file.readline()
        print("split line = ", line.split(":",2))
        [key,value] = line.split(":",2)
        new_dict[key] = value
    dict_file.close()

print(" Finished reading %s file" % file_name)
print(" New dictionary --> \n",new_dict)
