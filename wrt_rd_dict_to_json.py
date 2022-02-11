#
# example to read and write dictionary to file
# example of writing and reading a python dictionary to a json file
#

import os,sys

# with open(file_name,"r") as json_file:
#     new_dict = json_file.read()
#     json_file.close()

def json_wrt_rd_dict(wrd,job_name,json_file,optdict=None):
    """
    routine to create (write) or read .json file containing dict for psi4 options
    :param wrd: 'read' or 'write'
    :param job_name: dat file were .npy and .json file created
    :param json_file:
    :param optdict: dictionary with options included in the dat file
    :return: dictionary when wrd == 'read'
    """
    import json
    if wrd == "write":
        npy_file = f"{json_file}.npy"
        print("WRITING npy calc options to %s.json" % json_file)
        if os.path.isfile(npy_file):
            optdict["formed_by_job"] = job_name + ".dat"
            optdict["npy_file"] = npy_file
            with open(f"{json_file}.json","w") as jsf:
                json.dump(optdict,jsf)
            #    jsf.close()
            print("optdict: ",optdict)
            print("json_dumped to: ",json_file)
        else:
            print("ERROR: hessian file %s does not exist - exit" % npy_file)
            sys.exit()

        print("++++completed dict dump and exiting json_wrt_rd_dict++++")
        return

    elif wrd == "read":
        jsonf_w_dict = f"{json_file}.json"
        print("READING npy calc opt dictionary from %s" % jsonf_w_dict)
        with open(f"{json_file}.json","r") as jsf:
            opt_dict = json.loads(jsf.read())
            # opt_dict = json.loads(jsf)
        #    jsf.close()
        print("type(opt_dict): ",type(opt_dict), "    opt_dict: ",opt_dict)
        #opt_dict = json.loads(opt_dict)
        #print("new type(opt_dict): ",type(opt_dict), "    opt_dict: ",opt_dict)
        print("successfully read from: ",jsonf_w_dict)
        return opt_dict


if __name__ == "__main__":
    # test of json_rd_wri program
    # look for a npy file
    print(" Program to test reading/writing a dictionary from a .json file")
    print(" Need to find a .npy file first")
    # create a junk .npy file
    tfile = "junk_for_pgm_test.npy"
    with open(tfile,"w") as junkf:
        print("This is a junk npy file for test of json wrt_rd pgm",file=junkf)
    junkf.close()
    print("Found a .npy file: ",tfile)
    json_file = tfile[:-4]
    exam_dict = {'name':'bill',"numa":6.22}

    job_name = json_file+"_init_geom"
    print("Going to write exam_dict: ",exam_dict)
    print("to json file: %s.json \n" % json_file)
    # write dictionary first
    json_wrt_rd_dict("write",job_name,json_file,exam_dict)
    print("Finished writing exam_dict to json file: ",json_file,"\n\n\n")

    # Now check dictionary is read back in correctly
    print("Read in psi4 options dictionary from json file: ",json_file,"\n")
    new_dict = json_wrt_rd_dict("read",job_name,json_file)

    print("Finished reading new_dict from json_file: ",json_file,"\n")
    print("Check type of new_dict is dict")
    if type(new_dict) == 'dict':
        print('new_dict type is "dict"',type(new_dict))
    else:
        print("we have a problem - new_dict type is not dict but = ",type(new_dict))
    print("new dict: ",new_dict)


    #  statement needs more work: assert exam_dict == new_dict, "old_dict not the same as new_dict"

    print("====== End of writing and reading python dictionary to the %s file ======" % json_file)
