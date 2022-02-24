
# tran_rot_comp.py  - uses molecule geom and masses to obtain mass weighted trans and rot vib modes

import numpy as np
import psi4
import sys

import scipy.linalg
import scipy.stats

def ck_print(*args,turn_on=False):
    """ function ck_print(*args,turn_on=False)
        selectively reduces the number of prints
        set turn_on = True to check project of trans and rot modes
                      working correctly
    """
    if turn_on:
        print(*args)

             

# set up unitary masses array
def sclmass_set(atmass,mwt_type="utmwt"):
    """ sclmass_set converts molecules atomic mass to unitary masses
        such that the product of the unitary masses = 1 and geom mean of umass = 1
        
        Parameter
        ---------
        atmass: ndarray
        atomic masses
        mwt_type: str
        "atmwt" or "utmwt"
        Returns
        -------
        umass: ndarray shape [3,3*num_at]
        umass[0,:] = np.repeat(atmass,3)/atmass_gmean
        umass[1,:] = np.sqrt(umass[0,:])   #  umass[0,:]^+half
        umass[2,:] = 1./umass[1,:]         #  umass[0,:]^-half
        mass_scl: number
        when mwt_type="utmwt" mass_scl = geom mean of molecule's atomic masses
        and when mwt_type="atmwt" mass_scl = 1
        where atmass_gmean = geom mean of molecules atomic masses - det(mass matrix) = atmass_gmean**(3*num_at)
    """
    atmass_gmean = scipy.stats.gmean(atmass)
    
    #print("molecule %s molecular weight %15.6f and geometric mean of atomic masses = %15.8f" 
    #     % (file_wfn.molecule().name(),np.sum(atmass),atmass_gmean))
    psi4.core.print_out("\n\n++umass_set: molecular weight %.7f and geometric mean of atomic masses = %.8f" \

          % (np.sum(atmass),atmass_gmean))

    mass_scl = atmass_gmean  # set for mass_type = "utmwt"
    if mwt_type == "atmwt":
        mass_scl = 1.
        psi4.core.print_out("\nFor trad masses - mass_scl = 1")
    umass = np.repeat(atmass,3)/mass_scl
    # umass^half
    umass5 = np.sqrt(umass)
    # umass^-half
    umassm5=1./umass5
    
        
    psi4.core.print_out("\numass**one gmean = %.10f products = %15.9e atom sum = %.10f umass[::3] ->\n"
          % (scipy.stats.gmean(umass)
              , np.prod(umass), np.sum(umass)/3.) + str(umass[::3]))
    psi4.core.print_out("\numass**half gmean %.12f prod = %12.5e  atom sum = %.10f umass5 ->\n" %
        (scipy.stats.gmean(umass5),np.prod(umass5),np.sum(umass5)/3.) + str(umass5[::3]))
    psi4.core.print_out("\numass**-half gmean %.12f  prod = %12.5e  atom sum = %.10f umassm5 ->\n" %
        (scipy.stats.gmean(umassm5),np.prod(umassm5), np.sum(umassm5)/3.) + str(umassm5[::3]))
    
    return np.array([umass, umass5, umassm5]),mass_scl

# form the mass weighted normalized trans and rotational vectors
# inithess_type 3  from 1-may-2020

def gen_mwt_trans_rots(init_xyz_array,atmass):
    """ (2.1) Form mass weighted trans and rotational vectors
    init_xyz_array and atmass are np darray objects """

    psi4.core.print_out("\n\n\n========================Start of gen_mwt_trans_rots============================\n\n")

    mat_dim = 3*len(atmass)
    # find center of coord in mol_xyz
    #xyz_array = np.asarray(mol_xyz)
    #print("xyz_array = ",xyz_array.shape,"\n",xyz_array)

    # find center of mass
    cofm = np.dot(atmass,init_xyz_array)
    #print(" init cofm = ",cofm)
    tot_mass = np.sum(atmass)
    cofm /= tot_mass
    psi4.core.print_out("\nInitial center of mass for molecule ="+str(cofm))

    # original: xyz_array -= cofm
    xyz_array = init_xyz_array - cofm
    # check centered
    psi4.core.print_out("\nMol coordinates (xyz_array) now centered about origin:\n" + str(xyz_array))
    cofm = np.dot(atmass,xyz_array)/tot_mass
    for i in range(3):
        if cofm[i] > 1.e-8:
            psi4.core.print_out("\nERROR: new cofm not at orgin: " + str(cofm))
    
    
    
    psi4.core.print_out("\n\n========= Start forming mass weighted rotational vectors =======")
    
    # form rotational vectors
    rotsv =np.zeros((mat_dim,3),dtype=float,order='F')
    for iat in range(len(atmass)):
        sti = iat*3
        # x rot = (0,z,-y)
        rotsv[sti+1,0] = xyz_array[iat,2]
        rotsv[sti+2,0] = -xyz_array[iat,1]
        # y rot = (-z,0,x)
        rotsv[sti,1] = -xyz_array[iat,2]
        rotsv[sti+2,1] = xyz_array[iat,0]
        # z rot = (y,-x,0)
        rotsv[sti,2] = xyz_array[iat,1]
        rotsv[sti+1,2] = -xyz_array[iat,0]
    #print("formed cartesian rotational vetors\n",rotsv)
    
    # now form mass weighted rotational vectors
    norm = np.zeros(3,dtype=float)
    sqmass = np.repeat(np.sqrt(atmass),3)
    norm_cnt = 0
    for rvec in range(3):
        rotsv[:,rvec] *= sqmass
        norm[rvec] = np.linalg.norm(rotsv[:,rvec])
        if np.abs(norm[norm_cnt]) < 1.e-10:
            # delete zero column from rotsv vectors
            psi4.core.print_out("\nrot vector norm_cnt is zero - deleting from col %d from rotsv" % norm_cnt)
            psi4.core.print_out("\nrotsv[%d] ->" % norm_cnt + str(rotsv[:,norm_cnt]))
            rotsv = np.delete(rotsv,norm_cnt,axis=1)
        else:
            norm_cnt += 1
            
    if norm_cnt < 2:
        psi4.core.print_out("\n\n========== Linear molecule - no rot vectors = %d" % norm_cnt+1)    
    psi4.core.print_out("\nrotv norms = " + str(norm))

    # check that the rotational vectors orthonormal
    momenti_tensor = np.dot(rotsv.T,rotsv)
    
    psi4.core.print_out("\n====moments tensor from rotational disp vectors:\n" + str(momenti_tensor))
    
    # find eigenvals and evecs of rot_orthog_chk

    # TODO: organize any degeneracies m of inertia eigenvalues
    #  using: nerr, str_bendv, head_dir = sbv.get_str_bends(vec_iaja)

    r_eval,r_evec= scipy.linalg.eigh(momenti_tensor)

    num_rot_axis = len(r_eval)
    # check for degeneracies on m of inertia evals
    no_deg_mi = 0
    for iax in range(num_rot_axis-1):
        if np.abs(r_eval[iax] - r_eval[iax+1]) < 1.e-10:
            psi4.core.print_out("\nmoment of inertia %d and %d are degenerate = %12.6f"
                  % (iax,iax+1,r_eval[iax]))
            no_deg_mi += 1

    if no_deg_mi > 0:
        psi4.core.print_out("\nWARNING: need to deal with deg m of inertia in "
              "tran_rot_comp.py")
        psi4.core.print_out("\nNOT SETUP at present jdhd 22-july-2020")

    if num_rot_axis != 3:
        psi4.core.print_out("\nNOTE: number of rotational axis = %d and NOT 3" %
              num_rot_axis)
    
    psi4.core.print_out("\n\n===== eval and evec for momenti_tensor")
    for ivec in range(len(r_eval)):
        psi4.core.print_out("\neval[%d] = %12.6f  evec[:,%d]" % (ivec,r_eval[ivec],ivec) + str(r_evec[:,ivec]))

    #==================================================================

    #print("Rotational metric evals")
    #for iv in range(3):
        #ord_vec = np.argsort(np.abs(r_evec[:,iv]))

        #print("r_eval[%d] = %f abs_sort "
        #      % (iv,r_eval[iv]),ord_vec,"<- max\n evec:", r_evec[:,iv])

    #print("Metric eigenvals: ",r_eval)
    #print("Metric evec:\n",r_evec)
    #print("evec[:,0] = ",r_evec[:,0])
    #print("evec[:,1] = ",r_evec[:,1])
    #print("evec[:,2] = ",r_evec[:,2])
    
    # now calc transformed rot vectors
    
    #trans_rotv = np.matmul(r_evec.T,rotsv)
    trans_rotv = np.dot(rotsv,r_evec)
    
    
    ck_print("transformed rotv:",trans_rotv.shape,"\n",trans_rotv)
    
    trans_rot_orthog_chk =np.dot(trans_rotv.T,trans_rotv)
    
    ck_print("Metric for new trans_rotv:\n",trans_rot_orthog_chk)
    #print("Metric for new trans_rotv:\n",trans_rot_orthog_chk)
    
    #renormalize trans_rotv and place vectors in rotsv
    for i in range(num_rot_axis):
        #norm[i] = 1.0/np.sqrt(trans_rot_orthog_chk[i,i])
        rotsv[:,i] = trans_rotv[:,i]/np.sqrt(trans_rot_orthog_chk[i,i])

        
    ck_print("Confirm final rot vectors orthonormal:\n",np.dot(rotsv.T,rotsv))
    #print("Confirm final rot vectors orthonormal:\n",np.dot(rotsv.T,rotsv))
    
    ck_print("\nOrthonormal mass weighted rotational vectors\n",rotsv)
    
    # form translational modes
    tranv_norm = np.sqrt(tot_mass)
    
    transv = np.zeros((mat_dim,3),dtype=float,order='F')
    for iat in range(len(atmass)):
        sqmass = np.sqrt(atmass[iat])/tranv_norm
        iat3 = iat*3
        for jat in range(3):
            transv[jat+iat3,jat] = sqmass
            
    ck_print("Mass weighted trans vecs =\n",transv)
    
    # check that translational modes orthog to rotations
    #print("translational vec transv:\n",transv)
    trans_rot_orthog = np.dot(transv.T,rotsv)
    psi4.core.print_out("\n\nCheck that trans modes orthog to rotations:\n" + str(trans_rot_orthog))
    
    # combine transv and rotsv into one matrix
    tran_rot_v = np.hstack((transv,rotsv))
    psi4.core.print_out("\nCombined trans and rots vectors shape = " + str(tran_rot_v.shape))
    
    psi4.core.print_out("\n\n\n========================End of gen_mwt_trans_rots============================\n\n")
    
    return tran_rot_v

def vibs_trans_comp(mass_wt_vec,transv,proj= False,inv_hess=False,ckprnt=False):

    """ Compute translation components of mass_wt vib modes"""
    #print("========== Start of finding fraction trans in vib modes=========")
    mat_dim = mass_wt_vec.shape[0]
    # now compute translation in mat_wt_vec
    #for ivec in range(mat_dim):    
    #trans_comp = np.dot(transv.T,mass_wt_vec)
    ck_print("==in vibs_trans_comp transv.shape = ",transv.shape)
    trans_comp = np.dot(transv.T,mass_wt_vec)
    
    ck_print(" trans_comp.shape = ",trans_comp.shape)
    
    if proj:
        psi4.core.print_out("\n\nAfter projection translational components of normal modes")
    else:
        psi4.core.print_out("\n\nTranslational components of normal modes")
    ck_print("Mode       x            y           z",turn_on=ckprnt)
    tot_tnorm = 0.
    xyznorm = np.zeros(3,dtype=float)
    tnorm = np.zeros(mat_dim,dtype=float)
    for imode in range(mat_dim):       
        for jcomp in range(3):
            compsqr = trans_comp[jcomp,imode]**2
            xyznorm[jcomp] += compsqr
            tnorm[imode] += compsqr
            
        tot_tnorm += tnorm[imode]    
        ck_print("%4d %10.6f %10.6f %10.6f  Tot %10.6f" 
              % (imode,trans_comp[0,imode], trans_comp[1,imode], trans_comp[2,imode], tnorm[imode]),turn_on=ckprnt)
    ck_print("===== ================================================",turn_on=ckprnt)    
    psi4.core.print_out("\nTots %10.6f %10.6f %10.6f  Tot %10.6f"
          % (xyznorm[0], xyznorm[1], xyznorm[2],tot_tnorm))

    #return 0,tnorm,transv.T
    return 0,tnorm
    
# In[8]:


def vibs_rots_comp(mass_wt_vec,rotsv,proj=False,inv_hess=False,ckprnt=False):
    """ Compute rotational components of mass_wt vib modes"""
    #print("========== Start of finding fraction rots in vib modes=========")
    mat_dim = mass_wt_vec.shape[0]
    num_rots = rotsv.shape[1]

    ck_print("num_rots = %d" % num_rots)
    
    # now compute rot components in mat_wt_vec
    
    #for ivec in range(mat_dim): 
    ck_print("\n==in vibs_rots_comp rotsv.shape = ",rotsv.shape)
    rots_comp = np.dot(rotsv.T,mass_wt_vec)    
    
    if proj:
        psi4.core.print_out("\n\nAfter projection rotational components in normal modes")
    else:
        psi4.core.print_out("\n\nRotational components in normal modes")
    ck_print("Mode       x            y           z",turn_on=ckprnt)
    tot_rnorm = 0
    xyznorm = np.zeros(3,dtype=float)
    rnorm = np.zeros(mat_dim,dtype=float)
    for imode in range(mat_dim):       
        for jcomp in range(num_rots):
            compsqr = rots_comp[jcomp,imode]**2
            xyznorm[jcomp] += compsqr
            rnorm[imode] += compsqr
        tot_rnorm += rnorm[imode]
        if num_rots == 2:
            ck_print("%4d %10.6f %10.6f   zero      Tot %10.6f"
                     % (imode,rots_comp[0,imode], rots_comp[1,imode], rnorm[imode]),turn_on=ckprnt)
        else:
            ck_print("%4d %10.6f %10.6f %10.6f  Tot %10.6f"
              % (imode,rots_comp[0,imode], rots_comp[1,imode], rots_comp[2,imode], rnorm[imode]),turn_on=ckprnt)
    ck_print("===== ================================================",turn_on=ckprnt) 
    psi4.core.print_out("\nTots %10.6f %10.6f %10.6f  Tot %10.6f"
          % (xyznorm[0], xyznorm[1], xyznorm[2],tot_rnorm))

    return 0,rnorm

# In[10]:


def tot_tr_in_vibs(tcomp,rcomp,freq,proj=False,inv_hess=False,ckprnt=False):      
    tot_trnorm = 0.
    tot_tnorm = 0.
    tot_rnorm = 0.
    if proj:
        psi4.core.print_out("\n\n==Summary of trans and rot components in projected vibrations==")
    else:
        psi4.core.print_out("\n\n====== Summary of trans and rot components in vibrations ======")
    ck_print("Comp    trans     rot              Total          Freq",turn_on=ckprnt)
    for imode in range(len(tcomp)):
        tn = tcomp[imode]
        rn = rcomp[imode]
        trnorm = tn + rn
        tot_tnorm += tn
        tot_rnorm += rn
        tot_trnorm += trnorm
        if trnorm > 1.e-7:
            ck_print("%4d  %10.6f  %10.6f  Tot  %10.6f   %11.3f"
              % (imode,tcomp[imode],rcomp[imode],trnorm,freq[imode]),turn_on=ckprnt)

    ck_print("===== ==========================================",turn_on=ckprnt) 
    psi4.core.print_out("\nTotal %10.6f  %10.6f  Tot  %10.6f" 
          % (tot_tnorm, tot_rnorm, tot_trnorm))
    
    return 0


# In[19]:


# set up overall trans, rotational vibrational comp analysis - NEED TO TEST THIS ?????
def overall_tran_rot_vibcomp(freq,mwt_evec,tran_rot_v,proj=False,inv_hess=False):
    """ analysis of translational and rotational modes in mass weighted mwt_evec
                             
    """
    ret_code,tcomp = vibs_trans_comp(mwt_evec,tran_rot_v[:,:3],proj=False,inv_hess=inv_hess,ckprnt=False)

    ret_code,rcomp = vibs_rots_comp(mwt_evec,tran_rot_v[:,3:],proj=False,inv_hess = inv_hess,ckprnt=False)

    retcode = tot_tr_in_vibs(tcomp,rcomp,freq,proj=False,ckprnt=True)
    
    return retcode ## need to check a few things here
