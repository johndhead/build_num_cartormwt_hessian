# -*- coding: utf-8 -*-
"""
===== SETUP_PSI_NPY_FILE for use with jdhd BUILD_HESS program ======
Developing this version started on 7-Feb-2022

Previous version was arg_parse.py with the intention
to facilitate running different num_hess calculations

"""
import psi4
import numpy as np
import sys
import wrt_rd_dict_to_json as sav_psi4opt
import build_num_hess_main as jdh_bh

# to save psi4 opts use: sav_psi4opt.wrt_rd_dict_to_json(wrd,job_name,json_file,optdict=None)

# The following block could be included if the "__name__ == '__main__' is moved to end of code
# def build_num_hess(hess_type_param,jopts=None):
#     """  build_num_hess routine to
#
#     parameters
#     ==========
#     hess_type_param - parameters to define type of hessian being calculated from "npy_file"
#     jopts:  parameters for performing energy calculations
#
#     # TODO: fix jopts
#     # if jopts == None - read in XXXXXX.jopts where XXXXXX is the same as for the npy file
#
#     return:
#     =======
#     zero if calculation complete
#
#     """
#
#     print("==============================================================================")
#     print("==================== Welcome to BUILD_NUM_HESS routine =======================")
#     print("==============================================================================")
#
#     print("\n +++ hess_type_param -->\n",hess_type_param)
#
#     print("jopts = ",jopts)
#
#     print("++++++++ EXITING BUILD_NUM_HESS ++++++++++\n")
#
#     return 0

######################################################################################################
#
# Possible modification - move this initial "__name__ == '__main'" block to end of file
#
#######################################################################################################
if __name__ == "__main__":
    print("===============================================================")
    print("====== Start of find tree of folders_files_names program ======")
    print("=========== Setting up input for build_hess program ===========")
    print("===============================================================")
    
    import argparse
    import os
    parser = argparse.ArgumentParser(
            description="""
                        Program to build hessian matrices using either cart or mwt coordinates
                        1) Pick a molecule and check for a data file 'molecule.dat'
                        2) Generate a psi4 wavefn file at starting or optimized equil geometry
                        3) Set up size of displacement (-d disp) for numerical hessian calc
                        4) Decide to use cart (-g cart) or mwt (-g mwt) coordinates
                        5) Choose between Bohr or Angstrom for coord units in xyz file
                        6) If wavefn file already exists start building hessian matrix
                        """)
    parser.add_argument('-m','--mol_name', default = None,
                        help = 'file with molecule name - default None')
    parser.add_argument('-g','--geom',default='equil',
                        help = 'geom used to build hessian - options "equil" or "init_pt"')
    parser.add_argument('-d','--disp',default= 0.01,
                            help = 'numerical displace for finite differentiation')
    parser.add_argument('-c','--coord',default='cart',
                        help='coordinate type used to form hessian - options "cart" or "mwt"')
    parser.add_argument('-u','--coord_unit',default='bohr',
                        help='coords units - options "bohr" or "angstrom"')
    args = parser.parse_args()

    print("type for args: ",type(args))

    print("args",args)

    # get working directory and look for files with mol_name
    work_dir = os.getcwd()

    # gather argparse options for type of hessian build - save options in run_type_op
    run_type_op ={}
    mol_nm = args.mol_name
    print("molecule name = %s" % mol_nm)
    # print out other parameters
    mol_geom = args.geom
    print("working with %s %s geometry" % (mol_geom,mol_nm))
    disp = args.disp
    coord_type = args.coord
    coord_unit = args.coord_unit
    print('build hessian will displace atoms by %7f bohr using coord_type = %s' % (disp,coord_type))

    run_type_op = {'mol_nm':mol_nm, 'mol_geom': mol_geom, 'disp':disp, 'coord_type': coord_type, 'coord_unit': coord_unit}

    print('arg_parse parameters converted to a dictionary \n',run_type_op)
    ######################################################################################################
    #
    # Possible modification - move the above initial block to end of file
    #
    #  declare various parameters by using parameters in run_type_op dictionary:
    #    mol_geom = run_type_op["mol_geom], disp = run_type_op["disp"], coord_type =..., coord_unit...
    #
    #######################################################################################################

    #######################################################################################################
    #  start setup of the psi4 wfn calculation at 'equil' or 'init_pt' geometry
    ###############################################################################

    mol_files = []
    for tfile in os.listdir():
        if os.path.isfile(tfile):
            if tfile.find(mol_nm) >= 0:
                mol_files.append(tfile)
    if len(mol_files) > 0:
        print("%d molecule files found" % len(mol_files), " -->\n",mol_files)
    else:
        print("no files containing molecule %s found" % mol_nm)
        print("CHECK YOU HAVE THE CORRECT mol_name?? exit")
        sys.exit()

    # Check to see if mol_nm*.npy files exist
    calc_hess = False
    if mol_geom == "equil":
        # does MOLNAME_opt_wfn.npy exist
        build_hess = mol_nm+'_opt_wfn'
    elif mol_geom == "init_pt":
        build_hess = mol_nm+'_init_geom_wfn'
    else:
        print("ERROR - mol_geom = %s and not allowed value" % mol_geom)
        sys.exit()

    ready_to_build_hess = False
    if mol_files.count(build_hess+'.npy') == 1:
        print(f"Hessian wfn file: {build_hess}.npy already exists -- ready to run build_hess pgm")
        ready_to_build_hess = True
    else:
    # elif mol_files.count(build_hess + '.npy') == 0:

        print(f"++++ hessian/wfn file {build_hess}.npy does not exist")
        print(f"need to run: 'psi4 -i {build_hess}.dat' first to form {build_hess}.npy file")
        if mol_files.count(build_hess+'.dat') == 1:
            # run the psi4 job to form the npy job
            print(f"++++ {build_hess}.dat exists - ready to running psi4 job")
            pass

        elif mol_files.count(mol_nm + ".xyz") == 0:
            print(f"****ERROR**** file {mol_nm}+.xyz DOES NOT EXIST - fix and try again ****")
            sys.exit()

        elif mol_files.count(mol_nm+".xyz") == 1:
            #
            # make build_hess+'.dat' file
            # form hessian at initial geom
            # get xyz_coords from .xyz file
            # fxyz = open(mol_nm+".xyz",'rt')
            # xyz_lines =fxyz.readlines()
            with open(mol_nm+".xyz",'rt') as fxyz:
                xyz_lines = [line.rstrip() for line in fxyz]
            print("number of lines in %s.xyz = %d" %(mol_nm,len(xyz_lines)))

            # TODO: the following can probably be deleted
            # set up beginning of dat file
            #  get MOLNAME wfn at above geom
            # E_int,h2co_init_geom_wfn = energy('scf',return_wfn=True)
            # MOLNAME_init_geom_wfn.to_file("MOLNAME_init_geom_wfn.npy")
            #

            # # get MOLNAME_opt_wfn at equilbrium geom
            #
            # E_opt,MOLNAME_opt_wfn = optimize('scf',return_wfn=True)
            #
            # MOLNAME_opt_wfn.to_file("MOLNAME_opt_wfn.npy")
            #
            # E, MOLNAME_hess_wfn = frequency('scf',return_wfn=True)
            #
            # print(MOLNAME_hess_wfn.frequencies().get(0,0))
            # hess_wfn.hessian().print_out()
            # end of TODO: block probably for deletion

            # set up "init_pt" or "equili" geom dat file
            f_init_geom = open(build_hess + '.dat','wt')

            if mol_geom == "init_pt":
                print(f"# psi4 calc hessian file for {mol_nm} at init geom",file=f_init_geom)

            elif mol_geom == "equil":
                print(f"# psi4 calc hessian file for {mol_nm} at optimized geom",file=f_init_geom)

            print("\nmemory 500 mb",file=f_init_geom)

            # set up molecoordinates
            print(f"molecule {mol_nm}" +" {",file=f_init_geom)
            print("  0  1",file=f_init_geom)

            print("# add in xyz coords here",file=f_init_geom)
            line0 = int(xyz_lines[0])
            print("no atoms in  %s = %d" % (mol_nm,line0))
            # for line in xyz_lines[2:2+line0]:
            #     print(str(line),file=f_init_geom)
            f_init_geom.write('\n'.join(xyz_lines[2:2+line0]))
            # print("#------ finished printing coords -----#",file=f_init_geom)

            # now add rest of coord options include coord_units: angstrom or bohr
            print(f"\nsymmetry c1\nno_reorient\nno_com\nunits {coord_unit}",file=f_init_geom)
            print("  }\n", file=f_init_geom)

            num_thread = 6 # parameter for num of thread being used
            print(f"set_num_threads({num_thread})\n",file=f_init_geom)

            # start setting up other job options
            # basis aug-cc-pvdz or 6-31g**
            basis = "6-31g**"
            # set up dictionary jopts with energy calc options
            # modify jopts to run different types of energy calcs

            jopts = {'basis': basis,
                     'maxiter': 100,
                     'reference': "rhf",
                     'scf_type': "direct",
                     'd_convergence': 10,
                     'e_convergence': 10,
                     'ints_tolerance': 10,
                     'print_mos': False,
                     'geom_maxiter': 50
                     }
            print("\n### start of set options block", file=f_init_geom)
            # compare local options in jopts with psi4 global options
            # goal of jopts dictionary is to same psi4 options when building john's num hessian
            print("set {", file=f_init_geom)
            for key in jopts.keys():
                print(f"  {key} {jopts[key]}", file=f_init_geom)
                # glob_op = core.get_global_option(key)
                print(f"# key: {key}  from global_opt {psi4.core.get_global_option(key)} reset to {jopts[key]}", file=f_init_geom)
            print("  }\n", file=f_init_geom)

            if mol_geom == "init_pt":
                #  get MOLNAME wfn at initial geom
                print(f"# get {mol_nm} wfn at initial geom",file=f_init_geom)
                # E_int,h2co_init_geom_wfn = energy('scf',return_wfn=True)
                print(f"E_int,{mol_nm}_init_geom_wfn = energy('scf',return_wfn=True)",file=f_init_geom)
                print(f"{mol_nm}_init_geom_wfn.to_file('{mol_nm}_init_geom_wfn.npy')",file=f_init_geom)
                print(f"# calc {mol_nm} frequencies at initial geom for comparision with buildhess freq",
                      file=f_init_geom)
                print(f"E, {mol_nm}_hess_wfn = frequency('scf',return_wfn=True)",file=f_init_geom)

            elif mol_geom == "equil":
                # get MOLNAME_opt_wfn at equilbrium geom
                print(f"# calc {mol_nm} opt_wfn at equil_geom",file=f_init_geom)
                #
                print(f"E_opt,{mol_nm}_opt_wfn = optimize('scf',return_wfn=True)",file=f_init_geom)
                # E_opt,MOLNAME_opt_wfn = optimize('scf',return_wfn=True)
                #
                print(f"{mol_nm}_opt_wfn.to_file('{mol_nm}_opt_wfn.npy')",file=f_init_geom)
                # MOLNAME_opt_wfn.to_file("MOLNAME_opt_wfn.npy")
                #
                print(f"# calc {mol_nm} frequencies at optimized geom for comparision with buildhess freq",
                      file=f_init_geom)
                print(f"E, {mol_nm}_hess_wfn = frequency('scf',return_wfn=True)",file=f_init_geom)
                #
                # print(MOLNAME_hess_wfn.frequencies().get(0,0))
                # hess_wfn.hessian().print_out()

            f_init_geom.close()
            print(f"finished forming {build_hess}.dat")

        print(f"++++ start running: psi4 -i {build_hess}.dat to form npy file")
        os.system("psi4 -i "+build_hess+".dat")

        print(f"finished running: psi4 -i {build_hess}.dat ------")
        if os.path.isfile(f"{build_hess}.npy"):
            ready_to_build_hess = True
            # save the jopts dictionary to a json file

            # sav_psi4opt.json_wrt_rd_dict(wrd, job_name, json_file, optdict=None)
            # TODO: add total energy and max force of current geometry to jopts
            #jj
            sav_psi4opt.json_wrt_rd_dict("write", build_hess, build_hess, optdict=jopts)

        else:
            print(f"****At end of arg_parse pgm - hessian {build_hess}.npy does not exist****")
            print("NEED TO FIX PROBLEM")
            ready_to_build_hess = False
            sys.exit()


    if ready_to_build_hess:
        print(f"++++ ready to run build_hess pgm using {build_hess}.npy file")
        # call jdh_build_num_hess routine
        # hess_done = build_num_hess(run_type_op,jopts)
        run_type_op['npy_file'] = build_hess + ".npy"

        # TODO: decide whether to jdh_build_hess as a psi4 job
        # OR: make jdh_build_hess a callable fn
        
        # HERE IS A CALLABLE jdhd_build_num_hess function
        # import build_num_hess_main as jdh_bh
        # run_type_op = {'mol_nm':mol_nm, 'mol_geom': mol_geom, 'disp':disp,
        # 'coord_type': coord_type, 'coord_unit': coord_unit}

        # hess_done = jdh_bh.jdh_build_num_hess(mol_nm, run_type_op)

        print("JDH Num hess calc all pau - hess_done should be zero = %d" % hess_done)

    print("ALL PAU")

    ################################################################################
    #
    #    could add the "__name__ == '__main__'" block here
    #
    #################################################################################

