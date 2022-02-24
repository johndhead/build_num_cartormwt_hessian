#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 27 15:23:33 2020
Latest update: 7-Sept-2020

@author: johnh
"""

import sys
import psi4
import numpy as np

# add in jdhd routines
import hess_freq as hsf
import tran_rot_comp as ctr

def hess_setup_anal(molname, atsymbol, atmass, molgeom, hess=None,
                    tran_rot_v=None,
                    hess_type ='ehess',  # or "approx"
                    approx_type = 'ident',  # or 'bfgs' or 'lindh'
                    ehess_type ="cart",  # or 'mhess' or 'utmwt' or 'atmwt'
                    mhess_type ='atmwt',  # or 'utmwt'
                    inv_hess = False,  # into routine -> inv_hess_out on exit
                    get_unproj_freq = False,  # True when testing geom opt
                    get_proj_freq = True,  #
                    do_invert = False,  # set to True to do psuedo inv
                    anal_end_freq =True,  # check non_zero end freq from pure
                    prnt_mol_info = False,  # print starting mol info
                    idk=None
                    ):
    """ routine to set up various types of hessians and analyze their freq etc
    parameter
    ---------
    mol_name: str
    name of molecule
    atsymbol: list
    list of atom symbols in molecule
    atmass: ndarray
    atom masses
    molgeom: ndarray
    atom coordinates for molecules
    hess: ndarray
    usually the energy 2nd derivatives for the molecule
    
    Returns
    -------
    mwt hessian and type
    Full list: hess or invhess, scl_mass, mass_detscl,
    inv_hess_out (type True or False), ret_freq_type, freq, evec

    Possible ret_freq_type = ['no_freq',
    'cart_ident', 'mwt_ident', 'inv_mwt_ident', 'lindh_inv_hess',
    'unprj_freq', 'proj_freq', 'inv_hess_freq']

    Example:
    Set up for inverted masswt hessian starting from approx ="ident",
    note _mhess_type value determined by coord_type

            if coord_type == "masswt":
                _mhess_type = "atmwt"
            elif coord_type == "utmwt":
                _mhess_type = "utmwt"


            inv_hess, scl_mass, mass_detscl, inv_hess_type, ret_freq_type, \
            ih_freq, ih_evec = hess_setup_anal(
                mol_name, atsymbol, mol_mass, init_coords,
                hess_type="approx",
                approx_type="ident",
                ehess_type="cart",
                mhess_type=_mhess_type,
                inv_hess=False,
                get_unproj_freq=True,
                get_proj_freq=True,
                do_invert=True,
                anal_end_freq="True")
    
    """
    
    psi4.core.print_out("\n\n====++++ Entered hess_setup_anal routine for molecule: %s ++++"
           % molname)

    psi4.core.print_out("\ninitially hess_type = %s  and inv_hess = %s" % (hess_type,inv_hess))
    if hess_type == "approx":
        psi4.core.print_out("\n  approx_type = %s" % approx_type)

    psi4.core.print_out("\n++++initially ehess_type = %s ++++ mhess_type = %s ++++" %
        (ehess_type, mhess_type))
    if ehess_type == "cart" or ehess_type == "mhess":
        psi4.core.print_out("\n++++ ehess_type = %s ++++ mhess_type = %s ++++" %
              (ehess_type, mhess_type))
    elif ehess_type == "atmwt" or ehess_type == "utmwt":
        psi4.core.print_out("\nehess reset to 'mhess' and mhess_type = %s" % ehess_type)
        mhess_type = ehess_type
        ehess_type = "mhess"
    else:
        psi4.core.print_out("\n***ERROR*** ehess_type = %s is not an allowed type" % ehess_type)
        sys.exit("ehess_type error")

    # print out final ehess_type and mhess_type
    psi4.core.print_out("\n++++Final ehess_type = %s ++++ mhess_type = %s ++++" %
          (ehess_type,mhess_type))

    # print other options

    psi4.core.print_out("\nget_unproj_freq = %s  get_proj_freq = %s  do_invert = %s"
          % (get_unproj_freq,get_proj_freq,do_invert))

    inv_hess_out = inv_hess
    if do_invert:
        psi4.core.print_out("\n do_invert = True")
        inv_hess_out = not inv_hess_out
    psi4.core.print_out("\nOn entry inv_hess = %s and on exit inv_hess_out = %s" %
          (inv_hess,inv_hess_out))

    if prnt_mol_info:
        psi4.core.print_out("\nAtoms and their geometry used to form mass weighted hessian for molecule: %s"
                   % molname)
        psi4.core.print_out("\nprint more stuff")
    else:
        psi4.core.print_out("\nForming mass weighted hessian for molecule: %s" % molname)
 
    # set_up some constants
    no_atom = len(atmass)
    hess_dim = 3*no_atom
    ret_freq_type = 'no_freq'
    
    # (a) generate mwt trans and rot modes
    
    if tran_rot_v is None:
        tran_rot_v = ctr.gen_mwt_trans_rots(molgeom,atmass)
    else:
        psi4.core.print_out("\ntran_rot_v passed to hess_setup_anal routine")

    # (a.b) set up mass type "utmwt" or "atmwt"
    if mhess_type == "atmwt":
        psi4.core.print_out("\nmhess_type = 'atmwt' - using standard atomic masses")
    elif mhess_type == "utmwt":
        psi4.core.print_out("\nmhess_type = 'utmwt': unit scaling used for masses")
    else:
        psi4.core.print_out("\nERROR: mhess_type = %s - program not setup for this mass type"
              % mhess_type)
        sys.exit("mhess_type error")
    # setup scaled masses and mass_detscl
    scl_mass,mass_detscl = ctr.sclmass_set(atmass, mhess_type)

    # (b) check for hess and setup intype
    psi4.core.print_out("\nA REPEAT: hess_type = %s - ehess_type = %s - mhess_type = %s "
          % (hess_type, ehess_type, mhess_type))
    psi4.core.print_out("\ninv_hess = %s - EVENTUALLY DEL 2 PRINTS"
          % (inv_hess))
    
    if hess is None:
        if hess_type == "ehess":
            psi4.core.print_out("\nSince hess_type = %s - need to setup read of ehess" % hess_type)
            psi4.core.print_out("\nFix program")
            sys.exit("No ehess matrix")
        elif hess_type == "approx":
            psi4.core.print_out("\nNeed to get approx hessian")
            #approx_type = hess_type
            psi4.core.print_out("\napprox_type = %s  ehess_type = %s  inv_hess = %s"
                  % (approx_type, ehess_type, inv_hess))
            if approx_type == "ident":
                #print("At approx_type hess_dim = %d  no_atom = %d" % (hess_dim,no_atom))
                hess = np.identity(hess_dim,dtype=float)
                if idk is None:
                    print(f"\nDiag scale of inithess - idk = {idk} -> "
                          f"hence not scaling diag hess elements")
                else:
                    print(f"idk = {idk} -> scaling init ident matrix by "
                          f"{idk/10}")
                    hess *= float(idk)/10.
                if ehess_type == "cart":
                    ret_freq_type = "cart_ident"
                elif ehess_type == "mhess":
                    if do_invert:
                        ret_freq_type = "inv_mwt_ident"
                    else:
                        ret_freq_type = "mwt_ident"
                        psi4.core.print_out("\n***WARNING: forming an approx mwt_ident hessian -why? ")
                else:
                    psi4.core.print_out("\n***ERROR ehess_type = %s which is not allowed" %
                          ehess_type)
                    sys.exit("Problem with invalid ehess_type")

        elif approx_type == "lindh":
            psi4.core.print_out("\napprox_type = %s: forming jdh-lindh approx hess" % approx_type)
            if ehess_type == "mhess":
                psi4.core.print_out("\nehess_type = %s and mhess_type = %s" %
                  (ehess_type, mhess_type))
            else:
                psi4.core.print_out("\nin lindh approx - ehess_type = %s is WRONG" % ehess_type)
                sys.exit("Problem with ehess_type in lindh")

            import lindh_approx_fconst as laf
            hess = laf.gen_lindh_approx_hess(molname,atsymbol,scl_mass[0,::3],
                                     molgeom, tran_rot_v, hess_max_dist=5.0, k_r=0.45)
            psi4.core.print_out("\n\n\n======= Finished forming mwtLindh hess ======\n")
            psi4.core.print_out("\nhess_type = %s, ehess_type = %s and mhess_type = %s"
              % (hess_type,ehess_type,mhess_type))
            # reset hess_type and mhess_type
            psi4.core.print_out("\nRESET: hess_type = %s, ehess_type = %s and mhess_type = %s"
              % (hess_type,ehess_type, mhess_type))
            ret_freq_type = "lindh_inv_hess"

        elif approx_type == "bfgs":
            if ehess_type == "cart":
                psi4.core.print_out("\ncalc freq for a cart bfgs hessian")
                psi4.core.print_out("\nNeed to mass wt hess and calc freq")
            elif ehess_type == "mhess":
                psi4.core.print_out("\ncalc freq for a mhess bfgs hessian")
                psi4.core.print_out("\nwork still needs to be done")

        else:
            psi4.core.print_out("\nError: program not set up for approx_type = %s - Exit"
                  % approx_type)
            sys.exit("hess approx_type error")
    else:  #
        psi4.core.print_out("\n A hessian has been passed to hess_setup_anal routine")
        psi4.core.print_out("\nhess.shape = " + str(hess.shape))
        # TODO: need to fix up next few statements
        if hess_type == "ehess":
            pass
        elif hess_type == "approx":
            psi4.core.print_out("\nhess_type = %s - might want to check approx_type = %s"
                  % (hess_type, approx_type))
        # TODO: end of needed fixup

    # (c) now mass wt hessian
 
    psi4.core.print_out("\nREADY TO Mass wt hessian: ehess_type = %s  mhess_type = %s"
              % (ehess_type,mhess_type))
    #print("hess array id:",id(hess))
    #print("hess -->\n",hess)
    if ehess_type == "cart":
        # now mwt hessian
        mwt_hess = hsf.mwt_ehess(molname, hess, scl_mass, mhess_type, mass_detscl=mass_detscl)
        #print("after mwt hess - mwt_hess id:",id(mwt_hess))
        #print("hess array id:", id(hess))
        #print("hess -->\n", hess)
        ehess_type == mhess_type
        psi4.core.print_out("\n\n==== Finished mwt hess -- ehess_type = %s ===" % ehess_type)
        # in mwt_ehess - just return the mwt_hess
    else:
        psi4.core.print_out("\nehess is already mass weighted - ehess_type = %s" % ehess_type)
        # need to change this
        # if ehess_type == mhess_type:
        if ehess_type == "mhess" and (mhess_type == 'atmwt' or mhess_type == 'utmwt'):
            mwt_hess = hess.copy()
            psi4.core.print_out("\nhess copied to mwt_hess")
        else:
            psi4.core.print_out("\nproblem with mwt hess: ehess_type = %s mhess_type = %s should both involve masses"
                  % (ehess_type, mhess_type))
            psi4.core.print_out("\n+++ NOTE ALSO +++: inv_hess = %s"
                  % (inv_hess))
            sys.exit("ehess_type/out problem")

    # (d) invert direct hessian?
    if do_invert:
        psi4.core.print_out("\ndo_invert = %s -- inv_hess = %s" % (do_invert, inv_hess))
        psi4.core.print_out("\ninverting hess set up below")
        #sys.exit(" Need to set up inversion of hess")

    # (e) now start freq analysis
    #  at this stage should have mwt_hess with ??
    # (e1) test mwt_hess to see vib modes are mixed with tran/rot modes
    zero_tran_rot_vcomp = hsf.test_mwthess_projd(mwt_hess, tran_rot_v,
                                                 mass_detscl, inv_hess)
    psi4.core.print_out("\nUnproj hessian vibrational modes are pure = %s" % zero_tran_rot_vcomp)
    # set do_freq_calc = False before doing any freq calcs    
    do_unprj_freq_calc = False
    
    # (e2) get unproj_frequencies such as in geom opt with ident approx hessian
    if get_unproj_freq:
        psi4.core.print_out("\n======= Calc unproj frequencies =======")
        unprj_ret_code,unprj_freq,unprj_evec = hsf.freq_calc(mwt_hess, mass_detscl, inv_hess=inv_hess)
        if ret_freq_type == "no_freq":
            ret_freq_type = 'unprj_freq'
        do_unprj_freq_calc = True
        # analyze trans_rot comp in un_projected vib modes
        comp_code = ctr.overall_tran_rot_vibcomp(unprj_freq, unprj_evec, tran_rot_v, inv_hess=inv_hess)
        psi4.core.print_out("\nFinished calc unproj freq and tran_rot_vibcomp comp_code = %d" % comp_code)
        
    
    # (e3) project trans/rots modes out vib modes in hessian
    psi4.core.print_out("\n\n=== Now work on projecting Hessian ===")
    psi4.core.print_out("\n=== get_proj_freq = %s  do_invert = %s " % (get_proj_freq, do_invert))
    if get_proj_freq or do_invert:
        psi4.core.print_out("\n+++Start of projection: get_proj_freq = %s  do_invert = %s "
            % (get_proj_freq, do_invert))
        psi4.core.print_out("\nBefore if: zero_tran_rot_vcomp = %s" % zero_tran_rot_vcomp)
        if not zero_tran_rot_vcomp:
            psi4.core.print_out("\nIn if block - doing projection")
            # project non zero trans/rot comp out of hessian
            proj_ret_code, proj_hess = hsf.proj_trans_rots_frm_hess(mwt_hess, tran_rot_v,
                                                                    detscl=mass_detscl, inv_hess=inv_hess)
        else:
            psi4.core.print_out("\nMissed if and in else block - hess projection not done")
            proj_hess = mwt_hess
            
        # do freq calc
        proj_ret_code,proj_freq,proj_evec = hsf.freq_calc(proj_hess, mass_detscl, inv_hess=inv_hess)
        # analyze trans_rot comp in un_projected vib modes - eventually comment out the following line
        ret_freq_type = 'proj_freq'
        comp_code = ctr.overall_tran_rot_vibcomp(proj_freq, proj_evec, tran_rot_v, inv_hess=inv_hess)
        #
        # add in printing eigenvalues
        hsf.prnt_vib_evec(proj_evec,proj_freq,atmass)
        
        # (d) continued - pseudo invert 
        chk_inv_freq = True # set True if you want to check the freq of the inverse hessian
        if do_invert:
            psi4.core.print_out("\n++==++ start inverting hessian - inv_hess = %s" % inv_hess)
            ntran = 3
            nrot = tran_rot_v.shape[1] - ntran
            inverted_hess, inv_hess = hsf.pseudo_invert_hess(
                proj_freq, proj_evec, ntran, nrot, detscl=mass_detscl, inv_hess=inv_hess)
            
            if chk_inv_freq:
                psi4.core.print_out("\n++==++ Chk freq and tran/rot comp in inverse matrix - inverse matrix type = %s" % inv_hess)
                inv_chk_code,inv_hess_freq,inv_hess_evec = hsf.freq_calc(
                    inverted_hess, mass_detscl, inv_hess=inv_hess)
                ret_freq_type = 'inv_hess_freq'
                comp_code = ctr.overall_tran_rot_vibcomp(inv_hess_freq,
                                                         inv_hess_evec,
                                                         tran_rot_v,
                                                         inv_hess=inv_hess)
            
            psi4.core.print_out("\n++==++ returning projected pseudo inverse matrix - "
                  "inv_hess = %s" % inv_hess)
            if ret_freq_type == 'inv_hess_freq':
                psi4.core.print_out("\n**** finished forming proj_inv_hess inhess_setup_anal - "
                      "ret_freq_type = %s " % ret_freq_type)
                return inverted_hess, scl_mass, mass_detscl, inv_hess_out,\
                       ret_freq_type,inv_hess_freq,inv_hess_evec
            else:
                 psi4.core.print_out("\nret_freq_type = %s when returning pseudo inv mat - NO freq or evec returned"
                       % ret_freq_type)
                 return inverted_hess, scl_mass, mass_detscl, inv_hess_out, \
                        ret_freq_type,0.,0.

        else:
            psi4.core.print_out("\nret_freq_type = %s when returning projected hess matrix - NO freq or evec returned"
                      % ret_freq_type)
            return proj_hess, scl_mass, mass_detscl, inv_hess_out, \
                       ret_freq_type, proj_freq, proj_evec

        # (e2a) quit analyzing hessian for standard bfgs geom opt
    else:
        psi4.core.print_out("\n***EXIT hess_setup_anal without proj trans/rots out of hessian \n --> probably to analyze hessian during a geometry optimization")
        psi4.core.print_out("\nAfter unprj_freq calc: get_proj_freq = %s" % get_proj_freq)
        # psi4.core.print_out("\ntype(mwt_hess), type(scl_mass), type(mass_detscl),  "
        #       "type(inv_hess) = "
        #       , type(mwt_hess), type(scl_mass), type(mass_detscl),
        #       type(inv_hess))
        psi4.core.print_out("\nREADY TO EXIT: ret_freq_type = %s" % ret_freq_type)
        psi4.core.print_out("\n*** hess_set_anal - returning unprojected hess matrix - "
              "inv_hess = %s" % inv_hess)
        if ret_freq_type == 'no_freq':
            psi4.core.print_out("\n**** hess_setup_anal - returning %s hess" %
                  ret_freq_type)
            return mwt_hess, scl_mass, mass_detscl, inv_hess_out, ret_freq_type,\
                   0., 0.,
        elif ret_freq_type == "cart_ident":
            psi4.core.print_out("\n**** hess_setup_anal - returning %s hess" %
                  ret_freq_type)
            return hess, scl_mass, mass_detscl, inv_hess_out, \
                   ret_freq_type, 0., 0.
        elif ret_freq_type == "proj_freq":
            psi4.core.print_out("\n*** hess_setup_anal - returning projected hess matrix - "
                  "inv_hess = %s" % inv_hess)
            return proj_hess, scl_mass, mass_detscl, inv_hess_out, \
                    ret_freq_type, proj_freq, proj_evec
        else:
            psi4.core.print_out("\nret_freq_type = %s and not == no_freq for ret unproj "
                  "hess matrix" % ret_freq_type)
        return mwt_hess, scl_mass, mass_detscl, inv_hess_out, ret_freq_type, \
               0., 0.,

############## End of hess_setup_anal routine ##############################
