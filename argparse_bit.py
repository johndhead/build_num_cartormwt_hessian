    
    import argparse
    import os
    parser = argparse.ArgumentParser(
            description="""
                        Program to build num jdh_hessian matrices using either cart or mwt coordinates
                        1) Use setup_psi4_npy_file to created appropriate wavefn.npy file for some molecule
                        2) Program checks for existence of wavefn.npy and wavefn.json files
                        3) The psi4 wavefn files can be generated at the molecule's 
                           starting or optimized equil geometry
                        """)
    parser.add_argument('-d','--disp',default= 0.01,
                            help = 'num displacement in the finite differentiation')
    parser.add_argument('-c','--coord',default='cart',
                        help='coordinate type used to form hessian - options "cart" or "mwt"')
    # parser.add_argument('-u','--coord_unit',default='bohr',
    #                     help='coords units - options "bohr" or "angstrom"')
    parser.add_argument('npy_file',help='Name of wavefn file - leave off .npy and .json - NEEDED')
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
