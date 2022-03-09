#   set up main program to test optimizer code
#   set up jdh_build_hess.py to calc a numerical hessian matrix using
#   cart or masswt coordinates to do the atom displacements
#
#   This version started as a copy of the build_num_hess_main.py file from 11-Feb-2022
#

import psi4
import numpy as np
import hess_freq as hsf
import hess_setup_anal as hsa
import wrt_rd_dict_to_json as sav_psi4opt
import os, sys

################################################################
class Pcprint:
    """
    sets up printing for jdh_build_hess.py
    """
    #def __init__(self,prt_out_fname=None):
    def __init__(self):
        self._prt_out_fname = None
        self._psi4_out_close = None

    @property
    def prt_out_fn(self):
        return self._prt_out_fname
    @prt_out_fn.setter
    def prt_out_fn(self,prt_out_fname):
        self._prt_out_fname = prt_out_fname
        # TODO: add more here?
        # test open_file
        print('prt_out_fname = %s' % prt_out_fname)
        #self._psi4_outf = open(self._prt_out_fname,"w")
        self._psi4_out_close = "open"
        ###### PROBLEM HERE: return self._psi4_outf
        # with psi4 file name
        # psi4.come_set_output(self._prt_out_fname)
        # psi4.core.set_output_file(self._prt_out_fname)
        # return "opened psi4 output file"
    @property
    def prt_out_close(self):
        return self._psi4_out_close
    @prt_out_close.setter
    def prt_out_close(self,prt_out_close):
        if prt_out_close == "close":
            self._psi4_out_close = prt_out_close
            return "closed psi4_output_file"
        else:
            return "GOTTA PROBLEM: psi4 output file was not closed"
    def pcprint(self,prt_txt,file=None):
        if file is None:
            # need to used psi4 print_out
            if self._psi4_out_close =="open":
                # print("PSI4out:"+prt_txt)
                psi4.core.print_out(prt_txt)
            elif self._psi4_out_close == "close":
                # print("psi4_prt_cls STD_OUT:"+prt_txt)
                psi4.print_stdout("\npsi4_prt_cls STD_OUT:"+prt_txt)
            else:
                psi4.print_stdout("\npsi4_prt_not_open_yet STD_OUT:"+prt_txt)
        else:
            print(prt_txt,file=file)



# modify print commands so general print uses psi4.core.print_out()
# def pcprint(prt_txt,file=None):
#     """
#     modifies regular prints to psi4.core.print_out()
#     :param prt_txt: "text to be printed"
#     :param file: == None if no file - otherwise name of file to get output
#     :return:
#     """
#     if file is None:
#         psi4.core.print_out(prt_txt)
#     else:
#         print(prt_txt,file=file)
#     # all done

# set up energy_grad function
def hess_en_gr_fun(coords, *args):
    """ hess_en_gr_fun - calcs energy and gradient for mol
        coords = coords to calculate the energy - initially set to zero
        mol_coords = init_coords + coords

    args = (mol, eg_opts, init_coords, init_com, atom_mass, mass_detscl, coord_type)
    where:
        mol = molecule class name
        eg_opts = options for energy and grad calcs
        init_coords such that coords are zero initially
        init_com = initial center of mass when coords = zero
        inv_sqrt_mass = 1/sqrt(atom_mass) - used when coord_type = 'atmwt' or 'utmwt'
        coord_type posibilities so far: 'cart','masswt'

    function returns scf_e and grad
        """

    #print("<<<<<<<< hess_en_gr_fun coords: ",coords)
    #print("no of args in *args = %d" % len(args))

    psi4.core.print_out("disp coords for hess_en_gr_fun -->")
    psi4.core.print_out(str(coords))
    if len(args) == 7:
        (mol,eg_opts,init_coords,init_com,atom_mass,mass_detscl,coord_type) = args
        #print("mol =",mol)
        print("in hess_en_gr_fun: mol =",mol)
        #print("dir(mol): ->\n",dir(mol))
        #print("init_coords =",init_coords)
        #print("inv_sqrt_mass =",inv_sqrt_mass)
        #print("coord_type = %s"% coord_type)

    nat = mol.natom()
    if coord_type == 'cart':
    # if coord_type == 'cart' or coord_type == "mwtcart":
        # coords equal linear array len 3*mol.natom()
        #debug -- print("cart disp coords: ",coords)
        pass

    elif coord_type == 'masswt':
    # elif tmasswt(coord_type):
        # TODO: need to check mass weighting correct
        # coords are mass weighted - convert to cartessian
        inv_sqrt_mass = 1./np.sqrt(atom_mass)
        coords = coords * inv_sqrt_mass # cartesian displacment

        #debug -- print("masswt disp coords: ",coords)

    else:
        print("*** Error not set up for coord_type = %s" % coord_type)
        sys.exit()


    geom = np.reshape(coords,(nat,3)) + init_coords
    print("hess_en_gr_fun: mol geom ->\n",geom)
    # calc com
    tot_m = np.sum(atom_mass[::3])
    test_com = np.dot(geom.T,atom_mass[::3])
    print("mol mass = %10.3f  test_com = " % tot_m, test_com/tot_m)

    # fix center of mass and orientation
    mol.fix_com(True)
    mol.fix_orientation(True)
    print("%s com_fixed = %s  orientation_fixed = %s" % (mol.name(),
                                                         mol.com_fixed(), mol.orientation_fixed()))

    psi4.core.print_out("\n Skip printing initial coordinates in hess_en_gr_fun()")
    #print_mol_coord_sum(mol,opt_stage="Initial")

    mol.set_full_geometry(psi4.core.Matrix.from_array(geom))

    # the next line causes the center-of-mass to move - see what happens
    # without it
    mol.update_geometry()

    #psi4.core.print_out("new mol geom: \n")
    # print_mol_coord_sum(mol,opt_stage="New CART")

    #for iat in range(mol.natom()):
    #print("atom %d %3s %9.5f xyz coord = " % (iat,mol.symbol(iat),mol.mass(iat)),mol.xyz(iat))

    #print("\n===== co bond distance = %10.5f a.u." % (mol.z(1)-mol.z(0)))

    cxcom = mol.center_of_mass()[0]
    cycom = mol.center_of_mass()[1]
    czcom = mol.center_of_mass()[2]
    #print("cxcom,cycom,czcom: ",cxcom,cycom,czcom)
    current_com = np.array([cxcom,cycom,czcom],dtype=float)
    com_dif = current_com - init_com
    psi4.core.print_out("\n         ++++ current com = %18.10f  %18.10f  %18.10f"
                        % (current_com[0],current_com[1],current_com[2]))
    psi4.core.print_out("\n ++++  diff = curr - init = %18.10f  %18.10f  %18.10f a.u.\n"
                        % (com_dif[0],com_dif[1],com_dif[2]))

    # get inertia tensor and rotational consts
    # inert_ten = np.array(mol.inertia_tensor())
    # cur_rotc = np.array(mol.rotational_constants())
    # psi4.core.print_out("\ncurrent rot consts:  %15.9f  %15.9f  %15.9f" % (cur_rotc[0],cur_rotc[1],cur_rotc[2]))
    # psi4.core.print_out("\ninert_ten -->\n")
    # psi4.core.print_out(str(inert_ten))
    # # calc evals and evecs for inertia_tensor
    # teval,tevec = np.linalg.eigh(inert_ten)
    # psi4.core.print_out("\n  Eigen vals and vecs from inertia tensor")
    # for ivec in range(3):
    #     psi4.core.print_out("\neval[%d] = %12.8f  vec = (%11.8f, %11.8f, %11.8f)"
    #                         % (ivec,teval[ivec],tevec[ivec,0],tevec[ivec,1],tevec[ivec,2]))
    #
    scf_e,wavefn = psi4.energy(eg_opts,return_wfn=True)
    psi4.core.print_out("\n++++++++ scf_e in en_fun = %18.9f\n" % scf_e)
    #print("++++++++ scf_e in en_fun = %18.9f" % scf_e)

    G0 = psi4.gradient(eg_opts,ref_wfn=wavefn)
    gvec = np.array(G0)
    #jdhd - usually comment out this line 21-dec-2019
    #print("+=+=+=+=+ Cart gradient vector: \n", gvec)

    grad = np.reshape(gvec,(len(coords),))

    if coord_type == "masswt":
    # if tmasswt(coord_type):
        grad *= inv_sqrt_mass
        #print("=+=+=+=+ Mass wt grad vector: \n",grad)



    #print("+=+=+=+ grad as a linear array -->",grad)
    #print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    return scf_e, grad

####################### end hess_en_gr_fun #################################


def jdh_build_num_hess(mol_nm,run_type_op,jopts,pc= None):
    """  function to build jdh numerical hessian
    :parameter mol_nm: molecule name
    :parameter run_type_op: dictionary with the parameters for hessian build types
    :parameter jopts: dictionary with the psi4 options for the energy/grad calcs

    remember psi4 options are read from the numpy.json file
    """

    # check that pc is an instance of the Pcp class
    if isinstance(pc,Pcprint):
        pc.pcprint("\n pc class is instance of Pcp class and is defined")
    else:
        # Define pc
        pc = Pcprint()
        pc.pcprint("\n Created pc = Pcprint() class  for print_out - presumably need to open outfile")
        pc.pcprint(f"\n pc.prt_out_fn = {pc.prt_out_fn} -- pc.prt_out_close = {pc.prt_out_close}")

    # Memory specification
    psi4.set_memory(int(5e8))
    numpy_memory = 2

    # Set output file
    # TODO: originally dat --> out
    #psi4.core.set_output_file('output.out', False)

    # Define Physicist's water -- don't forget C1 symmetry!
    # mol = psi4.geometry("""
    # O
    # H 1 1.1
    # H 1 1.1 2 104
    # symmetry c1
    # """)


    ##########################################################################
    # start of setting up atomic coords, masses for different molecules
    # use_psi4_data = True if obtaining molecular info via psi4

    # in current program inithess_type - testing program with some simple molecules
    # set use_psi4_data = False
    #########################################################################

    #### check out using __file__

    print("check __file__ ",__file__)
    print("get __file__ abspath: ",os.path.abspath(__file__))

    # use_psi4_data = False  # set false when not using psi4 data
    use_psi4_data = True
    # (a) set up psi4 data
    # below is the psi4 setup
    if use_psi4_data:

        # list contents of the 'run_type_op" dictionary

        print("=========== Parameters used in the jdh_build_num_hess pgm =============")
        print("run_type_op = ",run_type_op)

        npy_file = f"{run_type_op['npy_file']}.npy"
        json_file = f"{run_type_op['npy_file']}.json"

        nthread = 6
        psi4.set_num_threads(nthread)

        # setup np print options
        print("get_print_options: ", np.get_printoptions())
        np.set_printoptions(precision=6, linewidth=120)

        #wfn_file_name = "h2co_scf_wfn"
        wfn_file_name = "h2co_opt_wfn_dir"
        wfn_file_name = run_type_op['npy_file']
        file_wfn = psi4.core.Wavefunction.from_file(wfn_file_name + '.npy')


        #  new output file naming setup
        #  add extension to out file name = "_{geom_type}_{disp_type}_{disp}
        # original: out_ext = f"_{run_type_op['mol_geom'][:4]}_{run_type_op['coord_type'][0:4]}_{int(100*run_type_op['disp'])}"
        out_ext = f"_{run_type_op['coord_type'][0:4]}_{int(100*run_type_op['disp'])}"
        print("file_name extension =",out_ext)
        mol_nm = run_type_op['mol_nm']
        output_file = run_type_op['npy_file']+out_ext
        psi4.core.set_output_file(output_file +'.out', False)
        pc.prt_out_fn = output_file + '.out'

        # set up energy and grad summary files
        en_sum = open("en_sum_"+output_file+".txt","w")
        grad_sum = open("gr_sum_"+output_file+".txt","w")

        pc.pcprint("\n++++ WFN Molecule name: %s   WFN Energy %28.20f \n" %
              (file_wfn.molecule().name(),file_wfn.energy()), file=en_sum)
        pc.pcprint("\n RUN_TYPE_OP:-->\n"+str(run_type_op)+"\n",file=en_sum)
        #pc.pcprint("\n RUN_TYPE_OP: ",run_type_op,"\n",file=en_sum)
        pc.pcprint("\n++++ WFN Molecule name: %s   WFN Energy %28.20f \n" %
              (file_wfn.molecule().name(),file_wfn.energy()), file=grad_sum)
        pc.pcprint("\n RUN_TYPE_OP:-->\n"+str(run_type_op)+"\n",file=grad_sum)
        #######################################################################

        psi4.core.print_out("\n ============ Get initial molecule info using "
                            "%s.npy "
                            "data\n" % wfn_file_name)
        psi4.core.print_out("Initial molecular geom obtained from %s.npy file\n"
                            % wfn_file_name)

        # get current module and print out options
        # print("current_module_name: %s" % current_module_name)
        # curr_mod_name = psi4.core.get_current_module()
        #curr_mod_name = psi4.core.Options.get_current_module()
        # --or--  ...get_current_module(file_wfn)
        # print("Options for curr_mod_name = %si -->" % curr_mod_name)
        pc.pcprint("dir(psi4.core.Options) -->\n")
        pc.pcprint(str(dir(psi4.core.Options))+"\n")
        pc.pcprint("dir(psi4.core.Wavefunction) -->\n")
        pc.pcprint(str(dir(psi4.core.Wavefunction))+"\n")

        # try printing out some options
        # pc.pcprint("List of options -->\n",psi4.core.get_options())

        # get list of info available from wavefunction file
        # pc.pcprint("\n======== type(file_wfn) --> ",type(file_wfn))
        # pc.pcprint("dir(file_wfn) -->\n",dir(file_wfn))



        # Let us see if file_wfn has frequencies
        # NOT WORKING ??
        # print("\nFor file_wfn - lets see if Freq exist")
        # file_wfn_freq = np.asarray((file_wfn.frequencies()))
        # print("\ndir(file_wfn.frequencies) -->\n",dir(file_wfn.frequencies))
        # print("\nfile_wfn_freq = ",file_wfn_freq)
        #
        # wfn_print_freq = file_wfn.frequencies().print_out()
        # print("\n freq print again: ",wfn_print_freq)
        # print("\n freq print again as np.array: ",np.asarray(wfn_print_freq))

        file_mol = file_wfn.molecule()
        pc.pcprint("\n======== type(file_mol) -> "+str(type(file_mol))+"\n")
        pc.pcprint("\n======== dir(file_mol) --> \n"+str(dir(file_mol))+"\n")
        # print("\n++++ Check if file_mol_freq exist",np.asarray(file_mol.frequencies()))
        # pc.pcprint( "\n\n=============================================================")
        # pc.pcprint("\n======== type(file_wfn.molecule()) -> ",type(file_wfn.molecule))
        # pc.pcprint("\n======== dir(file_wfn.molecule()) --> ",dir(file_wfn.molecule()))

        # check that mol_nm == mol_nm
        mol_name = file_mol.name()
        if mol_nm == mol_name:
            pc.pcprint(f"YEAH mol_nm = {mol_nm} matches with mol_name = {mol_name} in file_wfn ++++++")
        else:
            pc.pcprint(f"ERROR mol_nm = {mol_nm} NOT == mol_name = {mol_name} in file_wfn ++++++")
            sys.exit()
        num_file_at = file_mol.natom()
        file_geom = np.asarray(file_mol.geometry())
        pc.pcprint("no_ats in %s molecule = %d   file_geom.shape = %s \n" % (
            mol_name, num_file_at,str(file_geom.shape)))

        npmass = np.asarray([file_mol.mass(iat) for iat in range(num_file_at)])

        pc.pcprint("\n=========================================================")

        pc.pcprint("  %s  --- Units = %s" % (
            file_wfn.molecule().name(), file_wfn.molecule().units()))
        pc.pcprint("            x            y             z       mass")

        #  included atom symbol (label) in print out
        at_label = []
        # set up a coordinate string
        opt_mol_geom_setup = ""
        for iat in range(num_file_at):
            at_label.append(file_wfn.molecule().label(iat))
            pc.pcprint("%3d %2s %12.7f %12.7f %12.7f %12.7f" % (iat, at_label[iat],
                                                           file_geom[iat, 0],
                                                           file_geom[iat, 1],
                                                           file_geom[iat, 2],
                                                           npmass[iat]))
            atom_str = "  %2s  %20.12f  %20.12f  %20.12f\n" % \
                       (at_label[iat], file_geom[iat,0],
                        file_geom[iat,1], file_geom[iat,2])
            opt_mol_geom_setup += atom_str
        opt_mol_geom_setup += "\n no_com \n no_reorient \n symmetry c1 \n " \
                              "units bohr"

        pc.pcprint("opt_mol_geom_setup -->\n"+opt_mol_geom_setup)
        pc.pcprint("\n=========================================================\n")

        pc.pcprint("\nPsi4 %s center of mass = %s" % (file_wfn.molecule().name(),
              str(file_wfn.molecule().center_of_mass())))
        pc.pcprint("\nPsi4 %s rotational consts =" % file_wfn.molecule().name())
        pc.pcprint("\n"+ str(file_wfn.molecule().rotational_constants().np))
        pc.pcprint("\nand inertia tensor =>\n" + str(file_wfn.molecule().inertia_tensor().np))
        pc.pcprint("\nPsi4 fixed com = %s   fixed orientation = %s" % (
            file_wfn.molecule().com_fixed(),
            file_wfn.molecule().orientation_fixed()))

        # In[8]:

        # get list of info available from wavefunction file
        #print("dir(file_wfn) -->\n",dir(file_wfn))
        # print("dir(genmol_wfn) -->\n",dir(genmol_wfn))
        #print("\n======== dir(file_wfn.molecule()) --> ",dir(file_wfn.molecule()))

        # pc.pcprint(" Name of molecule = file_wfn.molecule.name()? = %s" %
        #     file_wfn.molecule().name())


        pc.pcprint("\nfile_wfn.basisset().name() = %s " % file_wfn.basisset().name())
        pc.pcprint("\nfile_wfn.basisset().nbf() = %d "% file_wfn.basisset().nbf())
        pc.pcprint("\nfile_wfn.nirrep() = %d" % file_wfn.nirrep())

        psi4.core.print_out("\nMolecule name: %s" % file_wfn.molecule().name())
        psi4.core.print_out("\n Energy = %21.14f" % file_wfn.energy())

        pc.pcprint("=========== End of working with numpy nphess <= file_wfn.hess ====")



        pc.pcprint("\n=========================================================")

        # set up opt_mol - separate class to molecule in hess file

        opt_mol = psi4.geometry(opt_mol_geom_setup)
        opt_mol.set_name(mol_name)

        # Computation options
        # psi4.set_options({'basis': 'aug-cc-pvdz',
        #                  'scf_type': 'df',
        # psi4.set_options({'basis': '6-31g',
        # check to see if optim converges in 1 step with aug-cc-pvdz basis
        # psi4.set_options({'basis': 'aug-cc-pvdz',
        # 6-31g basis takes a few steps to converge
        # psi4.set_options({'basis': '6-31g',

        # # check options before resetting them again
        # test_opts =['basis','reference','scf_type']
        # ################ TODO: play with this???
        # test_opts = [psi4.core.get_options()]
        # print("List psi4 options before reseting below -->\n",test_opts)
        # print("dir(test_opts): ",dir(test_opts))

        pc.pcprint("End of psi4_options ============================")

        # # TODO: add jdh options here ------------------
        # # compare local options in jopts with psi4 global options
        # # goal of jopts dictionary is to same psi4 options when building john's num hessian
        # # print("set {", file=f_init_geom)
        # invalid_jopt_keys = ['formed_by_job', 'npy_file'] # add other keys which are NOT valid psi4 options
        # for key in invalid_jopt_keys:
        #     #try:
        #         psi4.core.print_out(f"\ndeleting jopts[{key}] = {jopts[key]} from jopts")
        #         del jopts[key]
        #     #except KeyError as k:
        #     #    psi4.core.print_out("key {key} not in jopts ",k)
        # psi4.core.print_out(f"\njopts has been pruned: {jopts}")

        # find invalid psi4 keys in jopts dictionary and save them in a list: invalid_jopts_keys
        invalid_jopts_keys = []
        psi4.core.print_out("\n### start of comparing psi4 global options with jdh options")
        for key in jopts.keys():
            psi4.core.print_out(f"\n  {key} {jopts[key]}")
            try:
                glob_op = psi4.core.get_global_option(key)
                if key == "geom_maxiter":
                    local_scf_op = glob_op
                    psi4.core.print_out("\nInvalid SCF local key skipped")
                else:
                    local_scf_op = psi4.core.get_local_option('SCF',key)
                psi4.core.print_out(f"\njopts key = {key} - local_scf_op = {local_scf_op} - global_op = {glob_op}")
                if glob_op != jopts[key]:
                    psi4.core.print_out(f"\njopts['{key}'] != glop_op - resetting glob_op")
                    psi4.core.set_global_option(key,jopts[key])
                    new_glob_op = psi4.core.get_global_option(key)
                    if key == "geom_maxiter":
                        new_local_scf_op = new_glob_op
                        psi4.core.print_out("\nInvalid SCF local key skipped")
                    else:
                        new_local_scf_op = psi4.core.get_local_option('SCF',key)
                    psi4.core.print_out(f"\nNOW: local_scf_op = {new_local_scf_op} - global_op = {new_glob_op}")
                # check for option change
            except RuntimeError as kerr:
                psi4.core.print_out(f"\n{kerr}: jopts key {key} not a valid key")
                invalid_jopts_keys.append(key)

        psi4.core.print_out("\n++++++  End of comparing psi4 global options with jdh options")

        # See what happens if we use the jopts dictionary to set the options???

        # psi4.set_options(jopts)

        #
        # # for to in test_opts:
        # # print(f"Option --{to} =",psi4.core.get_option(to))
        #
        # # psi4.set_options({'basis': 'aug-cc-pvdz',
        # psi4.set_options({ 'basis': '6-31g**',
        #                   'reference': 'rhf',
        #                   'scf_type': 'direct',
        #                   'e_convergence': 10,
        #                   'd_convergence': 10,
        #                   'ints_tolerance': 10})
        #
        # #                  'print_mos': True})
        #
        # # probably show check energy type and list options later??

        # Get the SCF wavefunction & energies for H2O
        # scf_e0, scf_wfn = psi4.energy('scf', return_wfn=True)
        # print("A float and a Wavefunction object returned:", scf_e0, scf_wfn)

        # setup energy_gradient options
        # eg_opts = 'scf'

        # print("energy/gradient options: %s" % eg_opts)

    # put fixed geom data here

    ####################################################################
    #  start of some simple molecules to test lindh approx hessian idea

    else:
        # case 0 - set up molecular data for H-Be-H
        mol_name = "H-Be-H"

        # Setup atoms: at_labels, coordinates(mol_geom) and their masses (npmass)
        at_label = ['H', 'Be', 'H']

        d = 2.1  # Be-H bondlength in atomic units (need to check)
        mol_geom = np.array(
            [[0., 0., -d], [0., 0., 0., ], [0., 0., d]], dtype=float)

        # orig  Be =4 huh? # npmass = np.array([1., 4., 1.], dtype=float)
        npmass = np.array([1., 9., 1.], dtype=float)
        num_at = len(npmass)
        units = "Angstrom"

        ############ end-of-case 0 ################

        pc.pcprint("\n++++++++++++++++++++++ Molecular data for %s ++++++++++++++++++++++"
                % mol_name)

        pc.pcprint("====================================================================\n")

        pc.pcprint("num_at in %s molecule = %d   mol_geom.shape = " %
              (mol_name, num_at), mol_geom.shape)

        print("\n=========================================================")

        # print("  %s  --- Units = %s" % (file_wfn.molecule().name(),
        # file_wfn.molecule().units()))
        pc.pcprint("  %s  --- Units = %s" % (mol_name, units))
        pc.pcprint("            x            y             z       mass")

        #  included atom symbol (label) in print out
        for iat in range(num_at):
            pc.pcprint("%3d %2s %12.7f %12.7f %12.7f %12.7f" % (iat, at_label[iat],
                                                       mol_geom[iat, 0],
                                                       mol_geom[iat, 1],
                                                       mol_geom[iat, 2],
                                                       npmass[iat]))

        pc.pcprint("\n=========================================================")



    # calc first energy and gradient here
    # ref_scf_e,ref_wavefn = psi4.energy(eg_opts, return_wfn=True)
    # psi4.core.print_out("\n++++++++ ref_scf_e in main = %24.14f\n" % ref_scf_e)
    # G0 = psi4.gradient(eg_opts,ref_wfn=ref_wavefn)
    # gvec = np.array(G0)
    #psi4.core.print_out("\n ref grad vec ->")
    #psi4.core.print_out(str(gvec))

    # now set up args to call hess_en_gr_fun function to get energy and gradient
    # check that mol = active molecule
    # mol = file_mol
    mol = file_wfn
    if mol == psi4.core.get_active_molecule():
        psi4.core.print_out("\n mol = active mol: name = %s" % mol.name())
        pc.pcprint("\n mol = active mol =",mol)
    else:
        mol = psi4.core.get_active_molecule()
        psi4.core.print_out("\n mol set to active molecule: name = %s" % mol.name())
        pc.pcprint("\n mol set to active molecule = %s" % mol)
    eg_opts = 'scf'
    init_coords = file_geom
    init_com = np.asarray([mol.center_of_mass()[ix] for ix in range(3)])
    pc.pcprint("\n in args - init_com =" + str(init_com))
    num_at = len(npmass)
    atom_mass = np.ones(3*num_at, dtype=float)
    for ix in range(3):
        atom_mass[ix::3] =npmass
    mass_detscl = 1.

    args = (mol, eg_opts, init_coords, init_com, atom_mass, mass_detscl, coord_type)

    pc.pcprint("\n args --> \n"+ str(args))

    #  xdisp is the coordinate displacement (in mass wt atomic units?)
    #  set xdisp to zero initially

    # set up energy and grad summary files
    # en_sum = open("en_opt_sum.txt","w")
    # grad_sum = open("grad_opt_sum.txt","w")

    # print("\n++++ WFN Molecule name: %s   WFN Energy %28.20f \n" %
    #       (file_wfn.molecule().name(),file_wfn.energy()), file=en_sum)
    # print("\n++++ WFN Molecule name: %s   WFN Energy %28.20f \n" %
    #       (file_wfn.molecule().name(),file_wfn.energy()), file=grad_sum)

    # set up np arrays with energies, grad_vectors, disp_labels for ref_geom and 6*num_at displacements
    nrows = 6*num_at + 1
    en_array = np.zeros((nrows,2),dtype=float)
    # gr_array stores the init geom gradient (ref_grad) vectors for ref geom
    # and then the change in gradient for displacements (dis_grad - ref_grad)
    #
    gr_array = np.zeros((nrows,3*num_at),dtype=float)
    dis_label = np.zeros((nrows,3),dtype=int)
    fin_dif_2deriv = np.zeros((3*num_at,3*num_at),dtype=float)

    #  get ref energy and gradient vec at initial geom
    xref = np.zeros(len(atom_mass))
    ref_e,ref_grad = hess_en_gr_fun(xref, *args)

    # set ref en and grad values in np arrays
    en_array[0,1]=ref_e
    gr_array[0,:] = ref_grad

    psi4.core.print_out("\n++++++++ ref_scf_e in main = %24.14f\n" % ref_e)

    psi4.core.print_out("\n ref grad vec ->")
    psi4.core.print_out(str(ref_grad))

    pc.pcprint("ref energy = %24.15f" % ref_e,file = en_sum)
    pc.pcprint("ref grad = ",file=grad_sum)
    for jat in range(num_at):
        jat3 = 3 * jat
        pc.pcprint("at %2d G: %14.6e %14.6e %14.6e DG: %14.6e %14.6e %14.6e "
              % (jat, ref_grad[jat3], ref_grad[jat3 + 1], ref_grad[jat3 + 2],
                 0.,0.,0.,),
              file=grad_sum)

    psi4.core.print_out("\n\n++++++ Start of doing coord displacements ++++++\n\n")

    # set up coordinate displacement
    coor_disp = run_type_op["disp"]


    pc.pcprint(f"\n+++ coor_disp = {coor_disp}  disp_type = {run_type_op['coord_type']}  coord_unit = {run_type_op['coord_unit']} \n", file=en_sum)
    pc.pcprint(f"\n+++ coor_disp = {coor_disp}  disp_type = {run_type_op['coord_type']}  coord_unit = {run_type_op['coord_unit']} \n", file=grad_sum)

    #  now displace each atom in turn and calc energy and grad
    plus_min = [1.,-1.]
    row_cnt = 0
    for iat in range(num_at):
        iat3 = 3*iat
        for icor in range(3):
            for pm in plus_min:
                row_cnt += 1

                pc.pcprint("\n calc disp %3d iat = %2d  ic =%d pm = %3f" % (row_cnt, iat,icor,
                                                                   pm))
                xdisp = np.zeros_like(xref)
                xdisp[iat3+icor] += coor_disp*pm
                dis_e, dis_grad = hess_en_gr_fun(xdisp, *args)

                del_e = dis_e - ref_e
                del_grad = dis_grad - ref_grad
                dis_label[row_cnt,:] = np.array([iat,icor,int(pm)],dtype=int)
                en_array[row_cnt, 0] = del_e
                en_array[row_cnt, 1] = dis_e
                gr_array[row_cnt,:] = del_grad
                # gr_array[row_cnt, 3:6] = ref_grad

                # form 2 deriv matrix
                if pm > 0:
                    fin_dif_2deriv[iat3+icor,:] = dis_grad
                else:
                    fin_dif_2deriv[iat3+icor,:] -= dis_grad
                    fin_dif_2deriv[iat3+icor] *= 0.5/coor_disp

                pc.pcprint("at %2s%2d ic %d pm %3.0f E = %24.14f DE = %24.14f"
                      % (at_label[iat],iat,
                         icor,pm, dis_e,del_e), file=en_sum)

                pc.pcprint("at %2d ic %d pm %3.0f " % (iat,icor,pm), file=grad_sum)
                for jat in range(num_at):
                    jat3 = 3*jat
                    pc.pcprint("at %2s%2d G: %14.6e %14.6e %14.6e DG: %14.6e %14.6e %14.6e "
                          % (at_label[jat],jat,
                             dis_grad[jat3], dis_grad[jat3+1], dis_grad[jat3+2],
                                  del_grad[jat3], del_grad[jat3+1], del_grad[ jat3+2]),
                                    file=grad_sum)
                    #print("at %2d ic %d " % (iat,icor),del_grad,file=grad_sum)

    # en_sum.close()
    # grad_sum.close()

    pc.pcprint("\n\n================== Finished doing all atom displacements "
          "==================")
    psi4.core.print_out("\n\n================== Finished doing all atom "
                 "displacements ==================")

    # do sort on en_array
    sorted_en = np.argsort(en_array[:,0],)   #  ,axis=2)
    en_mean = np.mean(en_array[1:,0])
    en_std = np.std(en_array[1:,0])
    gnorm_mean = 0.
    # print("sorted_en -->\n",sorted_en)
    pc.pcprint("",file=en_sum)
    pc.pcprint("+-----------------------------------------------------------+",file=en_sum)
    pc.pcprint("|      Sorted energies for different atom displacements     |",file=en_sum)
    pc.pcprint("\n|          coord_type = %6s    disp = %7.3f     |" %
          (coord_type, coor_disp))
    pc.pcprint("|     First record should be for the reference geom         |",file=en_sum)
    pc.pcprint("+-----------------------------------------------------------+\n",file=en_sum)
    pc.pcprint(f"\n+++ coor_disp = {coor_disp}  disp_type = {run_type_op['coord_type']}  coord_unit = {run_type_op['coord_unit']} \n", file=en_sum)
    sord = 0
    for sen in sorted_en:
        sord += 1
        gnorm = np.sqrt(np.dot(gr_array[sen,:],gr_array[sen,:]))
        if sen == 0:
            pc.pcprint(" found ref molecule - skip adding grad norm to total gnorm",file=en_sum)
            pc.pcprint("sen = %d  and sord = %d" % (sen,sord),file=en_sum)
        else:
            gnorm_mean += gnorm
        pc.pcprint("%2d at %2s%2d xyz %d pm %2d     DE = %18.14f E = %20.14f  |grad| = %15.10f"
              % (sord, at_label[dis_label[sen,0]],
                 dis_label[sen,0],dis_label[sen,1],dis_label[sen,2],
                                         en_array[sen,0],en_array[sen,1],gnorm),file=en_sum)
        # print("order = ",sord,dis_label[sen,0],dis_label[sen,1],dis_label[sen,2])

    pc.pcprint("+-----------------------------------------------------------+",file=en_sum)
    pc.pcprint("|  en_mean = %12.9f  en_std = %12.9f    gnorm_mean = %12.9f" %
          (en_mean,en_std,gnorm_mean/(6*num_at)),file=en_sum)

    en_sum.close()
    grad_sum.close()


    # print_hess to check it looks OK
    # ph.print_hess(fin_dif_2deriv,prnt=True)
    hsf.print_hess(fin_dif_2deriv,prnt=True)

    # symmetrize hessian
    for iind in range(3*num_at):
        for  jjnd in range(iind):
            ave_hess_ij = 0.5 * (fin_dif_2deriv[iind,jjnd] + fin_dif_2deriv[jjnd,iind])
            pc.pcprint("\n iind,jjnd = %d,%d  hess[ii,jj] = %15.9f   hess[jj,ii] = %15.9f   ave_hess = %15.9f"
                  % (iind,jjnd,fin_dif_2deriv[iind,jjnd],fin_dif_2deriv[jjnd,iind],ave_hess_ij))
            fin_dif_2deriv[iind,jjnd] = ave_hess_ij
            fin_dif_2deriv[jjnd,iind] = ave_hess_ij


    calc_opt_freq = True

    if calc_opt_freq:
        # TODO: trying to write freq to different file - needs more work on print statements before doing this
        # psi4.core.close_outfile()
        # output_file += "_freq"
        # psi4.core.set_output_file(output_file +'.out', False)
        # psi4.core.print_out(f"\n This is the frequency calc continuation for {output_file[:5]}.out")

        # (0) calc frequencies from psi4 hessian wfn
        # set up and analyze traditional mass_wt hessian


        #   add in traditional ehess frequency analysis here

        pc.pcprint("\n++++ (0) Traditional atomic mass weighted freq calc using numerical diff ehess ++++\n")
        # second derivative matrix  nphess -> from above file_wfn read

        nphess = fin_dif_2deriv
        # add in ehess_type for cart or mwt_hess
        if coord_type == 'cart':
            ehess_type = 'cart'
        elif coord_type == 'masswt':
            # for mwt_hess: ehess_type should == mhess_type
            # originally had:  ehess_type = 'mhess'
            # probably should change this
            ehess_type = 'mhess'
        else:
            pc.pcprint("***ERROR*** coord_type = %s and not valid"% coord_type)
            sys.exit()

        mwt_hess, umass, atmass_gmean, inv_hess, ret_freq_type, anal_freq, \
                anal_evec = hsa.hess_setup_anal(
                mol_name, at_label, npmass, file_geom, nphess,
                tran_rot_v=None,
                hess_type='ehess',
                approx_type=None,
                ehess_type=ehess_type,
                mhess_type='atmwt',
                inv_hess=False,
                get_unproj_freq=True,
                get_proj_freq=True,
                anal_end_freq=True,
                prnt_mol_info=False)
        num_at = num_file_at
        mol_geom = file_geom
        units = file_mol.units()
        # print("  %s  --- Units = %s" % (file_wfn.molecule().name(),
        # file_wfn.molecule().units()))

        pc.pcprint('numerical frequencies - ret_freq_type = %s\n' % ret_freq_type)
        pc.pcprint(str(anal_freq) + "\n")

        pc.pcprint("\n      ======= End of (0) %s frequencies from psi4 hess "
              "wavefn========\n\n" % mol_name)

        ####################################################################
    pc.pcprint("\n++++++++++++++++++++++ Molecular data for %s ++++++++++++++++++++++"
          % mol_name)

    pc.pcprint("\n====================================================================\n")

    pc.pcprint("num_at in %s molecule = %d   mol_geom.shape = %s" %
          (mol_name, num_file_at,str(mol_geom.shape)))


    pc.pcprint("\n=========================================================")

    #print("  %s  --- Units = %s" % (file_wfn.molecule().name(),
    # file_wfn.molecule().units()))
    pc.pcprint("\n  %s  --- Units = %s" % (mol_name, units))
    pc.pcprint("\n            x            y             z       mass")

    #  included atom symbol (label) in print out
    for iat in range(num_file_at):
        pc.pcprint("\n%3d %2s %12.7f %12.7f %12.7f %12.7f" % (iat, at_label[iat],
                                                       mol_geom[iat, 0], mol_geom[iat, 1], mol_geom[iat, 2], npmass[iat]))

    pc.pcprint("\n=========================================================")
    psi4.core.close_outfile()
    pc.prt_out_close = "close"

################################################################################
#
#  Start of main routine of jdh_build_hess - for testing program
#
################################################################################

if __name__ == "__main__":
    import argparse
    import os

    pc = Pcprint()
    pc.pcprint("\n+-----------------------------------------------------------+")
    pc.pcprint("\n|     Start of main pgm to run/test jdh_build_hess.py       |")
    pc.pcprint("\n+-----------------------------------------------------------+\n")

    # set up starting program
    
    parser = argparse.ArgumentParser(
            description="""
                        Program to build num jdh_hessian matrices using either cart or masswt coordinates.
                        
                        \n 1) Use setup_psi4_npy_file to create appropriate wavefn.npy file for some molecule.
                        \n 2) Program checks for existence of both wavefn.npy and wavefn.json files.
                        \n 3) The psi4 wavefn files can be generated at the molecule's 
                           starting or optimized equil geometry.
                        """)
    parser.add_argument('-g','--geom',default='equil',
                        help = 'geom used to build hessian - options "equil" (def) or "init_pt"')
    parser.add_argument('-d','--disp',type=float, default = 0.01,
                            help = 'num displacement in the finite differentiation (def 0.01)')
    parser.add_argument('-c','--coord',default='cart',
                        help='coord type "cart" (def) or "masswt" used to form hessian')
    parser.add_argument('-u','--coord_unit',default='angstrom',
                        help=' "angstrom" (def) or "bohr"')
    parser.add_argument('npy_file',help='Name of wavefn file - BUT leave off .npy and .json extensions')
    args = parser.parse_args()

    pc.pcprint("\ntype for args: "+ str(type(args)))

    pc.pcprint("\nargs" + str(args))

    # get working directory and look for files with mol_name
    work_dir = os.getcwd()

    # check that npy_files npy_file.npy and npy_file.json exist

    build_hess = args.npy_file[:-4] if args.npy_file[-4:] == '.npy' else args.npy_file
    fjson = build_hess+'.json'
    fnpy = build_hess +'.npy'
    pc.pcprint(f"debug build_hess = {build_hess} fnpy = {fnpy} and fjson = {fjson}")

    # read in jdh psi4 options dictionary from build_hess
    # TODO: add test that fnpy and fjson exist

    # TODO: add total energy and max force of current geometry to jopts
    jopts = sav_psi4opt.json_wrt_rd_dict("read", build_hess, build_hess)

    pc.pcprint("\njopts dictionary - type = "+str(type(jopts))+" -->: \n" + str(jopts))


    # gather argparse options for type of hessian build - save options in run_type_op
    run_type_op ={}
    # mol_nm = args.mol_name
    # mol_nm = "CH3NH2"  # TODO: get mole name from wavefn or npy_file name
    mol_nm = build_hess.split("_",1)[0]
    run_type_op['mol_nm'] = mol_nm
    pc.pcprint("molecule name = %s" % mol_nm)
    # print out other parameters
    pc.pcprint("\n working with %s %s geometry" % (args.geom, mol_nm))
    disp = args.disp
    # rescale disp if disp > 1.
    if disp >= 1.:
        disp /= 100.
        pc.pcprint(f"args.goem = {args.disp} - disp reset to {disp}")
    coord_type = args.coord
    coord_unit = args.coord_unit
    pc.pcprint('\n build hessian will displace atoms by %7f bohr using coord_type = %s' % (disp,coord_type))

    # run_type_op = {'mol_nm':mol_nm, 'mol_geom': mol_geom, 'disp':disp,
    run_type_op = {'mol_nm': mol_nm, 'mol_geom': args.geom, 'disp':disp, 'coord_type': args.coord,
                    'coord_unit': coord_unit, 'npy_file': build_hess}

    pc.pcprint('\narg_parse parameters converted to the run_type_op dictionary \n'+ str(run_type_op))

    # call jdh_build
    jdh_build_num_hess(mol_nm, run_type_op, jopts, pc=pc)

    #
    pc.pcprint("\n+++++++ Finished test of jdh_build_num_hess +++++++\n")
