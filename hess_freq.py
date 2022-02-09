#  routines for calculating vib frequencies from the hessian matrix  14-june-2020

import numpy as np
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

# routine to print out the numpy hess matrix
def print_hess(hess,title="numpy hess matrix",simple_int=False,prnt=False):
    """ print out lower symmetrical part of the hess matrix
        hess is the hessian - numpy darray
        simple_int = False for decimals - set to True if identity matrix
        """
    hess_size = hess.shape[0]
    numat = hess_size//3
    print("%s -- num atoms = %d" % (title,numat))
    if not prnt:
        return
    else:
        if simple_int:
            print(hess)
            return
        row = 0
        for iat in range(numat):
            print("Atom row %d" % iat)
            col = 0
            
            for jat in range(0,iat+1,2):
                rstr = [str("%2d" % (row)),str("%2d"% (row+1)),str("%2d"% (row+2))]
                for jjat in range(2):
                    if jat +jjat > iat:
                        continue
                    else:
                        for add_at in range(3):
                            rstr[add_at] += str(" %2d %10.3e %10.3e %10.3e" %                         (jat+jjat,hess[row+add_at,col],hess[row+add_at,col+1],hess[row+add_at,col+2]))
                        col += 3
                        if jat +jjat == iat:
                            row += 3
                print("%s" % rstr[0])
                print("%s" % rstr[1])
                print("%s" % rstr[2])
                print("-------------------------------------------------------------------------")
                #row += 3
    return

def prnt_vib_evec(evec,freq,mass):
    """ routine to write the vibrational evec

    Parameter
    ---------
    evec: ndarray (3*nat,3*nat)
    freq: ndarray (3*nat)  (in 1/cm)
    mass: list

    Returnu
    ------
    None
    """
    #mm=np.array(mass)
    nmodes=len(freq)
    nat = len(mass)
    if nmodes//3 != nat:
        print("ERROR: nmodes = %d and should = 3*nat, nat = %d" % (nmodes,nat))
        return "Error with nat and nmodes"

    # setup print labels
    print("\n=========== Vibrational normal modes ============\n")
    print("  dvec = orthog mass wt    cvec = cartesian disps")
    #print("type(mass) = ",type(mass))
    #vec = " 1234567 1234567 1234567 1234567"
    Dvec = "    dx      dy      dz      Td  "
    Cvec = "    cx      cy      cz      Tc  "
    for im in range(nmodes):
        avec = evec[:,im].reshape(nat,3)
        Td = np.sum(avec**2,axis=1)
        Totd = np.sum(Td)
        Tc = Td/np.sqrt(mass)
        Totc = np.sum(Tc)
        mTc = mass*Tc
        Tot_mtc = np.sum(mTc)
        print("  imode = %d   freq = %9.4f" % (im,freq[im]))
        print("atom ",Dvec,Cvec,"  m * Tc")
        for iat in range(nat):
           ic = 3 * iat
           D = " %7.3f %7.3f %7.3f %7.3f " % (evec[ic,im],evec[ic+1,im],
                                              evec[ic+2,im],Td[iat])
           sm = 1./np.sqrt(mass[iat])
           C = " %7.3f %7.3f %7.3f %7.3f " % (sm*evec[ic,im],sm*evec[ic+1,im],
                                              sm*evec[ic+2,im],Tc[iat])
           print("%3d %s  %s %7.3f" % (iat,D,C,mTc[iat]))
        print(" Totals:    TotD = %7.3f     TotC = %7.3f    Tot_mtc = %7.3f"
                        % (Totd, Totc, Tot_mtc))

    return

def pseudo_invert_hess(freq, evec, ntran=3, nrot=3, teval= 0., reval = 0., detscl=1., inv_hess = False):
    """ 
    Forming psuedo inverse hess matrix from the freq (evals) and evec of the starting projected mwt hess matrix
    parameter
    ---------
    freq: 1D ndarray
    initial hess eigenvalues in wavenumbers (1/cm) - trans and rot vectors assumed to be listed first in hess evec
    hess: ndarray
    initial hess eigenvecs
    ntran: int
    no of translations modes (3)
    nrot: int
    no of rotational modes (3 or 2)
    teval, reval: float
    values to set the trans and rot modes eigenvalues in inverse
    det_scl:
    mass wt factors for "utmwt" or "atmwt" vib modes -- set to det_scl value for initial matrix
    inv_hess: bool
    True if init hess is an inverse matrix, False if a direct hess

    Returns
    -------
    inverted_mat: ndarray
    shape (3*numat,3*numat)
    
    """

 # unit conversion
    hartree2waveno = 219474.6
    ck_print("hartree2waveno = %f" % hartree2waveno)
    au2amu = 5.4857990907e-04  # CODATA recommend value 2019 = 5.485 799 090 65(16) e-04  ?? corrected 8-may-2020
    #sqrt_au2amu = np.sqrt(au2amu/mass_unit)  # convert amu from g to kg
    sqrt_au2amu = np.sqrt(au2amu/1000.)  # convert amu from g to kg
    #Evib = hbar * omega  = hbar * sqrt(k/m)
    if inv_hess:
        radpsec2waveno = 1./(hartree2waveno*sqrt_au2amu)
    else:     # unit conversion for mass weighted hessian
        radpsec2waveno = hartree2waveno*sqrt_au2amu
    ck_print("au2amu %f inv(au2amu) %f -- radpsec2waveno %f" % (au2amu,1./au2amu,radpsec2waveno))
    
    # set up freq values
    
    print("=== init freq values in 1/cm ->\n",freq)
    
    freq_shft = 100. # freq shift in 1/cm
    if teval > -3*freq_shft or reval > -3.*freq_shft:
        teval = 0.
        reval = 0.
        freq_shft = 0.
        
    print("initial trans %f and rot %f 1/cm freq values -- freq_shft = %f" % (teval,reval,freq_shft))
 
    for itran in range(ntran):
        freq[itran] = teval + freq_shft*itran
        
    for irot in range(nrot):
        freq[ntran+irot] = reval + freq_shft* irot
    
    print("=== shifted freq values in 1/cm ->\n",freq)
    
    
    #set up mass_unit
    
    mass_unit = 1000./detscl
    
    

    # now convert freq in wavenumbers to freq1 in au
    # freq1 corresponds to eigenvals of mwt hessian being inverted
    # freq2 = 1/freq1 and is the eigenvalues of inverted hessian
    # trans/rot frequencies of inverted matrix set to the teval and reval values given function args
    
    scale = np.ones_like(freq)/radpsec2waveno
    scale = scale**2
    scale[freq<0] = -scale[freq<0]
    freq1 = freq**2 * scale / mass_unit # convert freq in wavenumbers to eval of mass wt hessian
    print("mass_unit = %f scale = \n" % mass_unit,scale,"\n freq1 -->\n",freq1)
    #for ivec in range(mat_dim):
    #    print("ivec %d scale[ivec] = %12.5e  freq2[ivec] = %12.5e  1/freq2[ivec] = %12.5e"
    #         % (ivec,scale[ivec],freq2[ivec],1./freq2[ivec]))
    if inv_hess:
        # in freq calc
        #freq = 1./(scale *np.sqrt(freq2))
        # freq2 for inv_hess
        #freq2 = 1./(scale * freq * mass_unit)
        # freq 2 for direct hess = inverted inv hess
        freq2 = freq1.copy()
        # try inverting 
        # 1st effort 
        freq2[ntran+nrot:] = 1./freq1[ntran+nrot:]
        # 2nd effort
        #freq2[ntran+nrot:] = freq1[ntran+nrot:]
        print("Eigenvalues in au for direct mwt hess from inverted mwt hess")
        # invert trans + rot if abs(freq1[0:6]) > 1.e-5
        for imode in range(ntran+nrot):
            if np.abs(freq1[imode]) > 1.e-5:
                freq2[imode] = 1./freq1[imode]
                print("inverting freq1[%d] = %15.7f --> %15.7f" % (imode,freq1[imode],freq2[imode]))
            else:
                print("not inverting freq1[%d] = %15.7f to %15.7f  freq2[^%d] = %15.7f"
                     % (imode,freq1[imode],1./freq1[imode],imode,freq2[imode]))
        
        ############################### here sat
       
    else:   # scale factor of direct mass_wted
        # freq2 for direct hess
        #freq2 = scale * freq * mass_unit
        # freq 2 for inverted mwt hess = inverted direct hess
        freq2 = freq1.copy()
        # trying inverting
        # 1st effort 
        #freq1[ntran+nrot:] = 1./freq2[ntran+nrot:]
        # freq2 is the eval of the inverted matrx
        
        # invert trans + rot if abs(freq1[0:6]) > 1.e-5
        ###for imode in range(ntran+nrot):
            ###if np.abs(freq1[imode]) > 1.e-5:
                ###freq2[imode] = 1./freq1[imode]
                ###print("inverting freq1[%d] = %15.7f --> %15.7f" % (imode,freq1[imode],freq2[imode]))
            ###else:
                ###print("not inverting freq1[%d] = %15.7f to %15.7f  freq2[^%d] = %15.7f"
                     ###% (imode,freq1[imode],1./freq1[imode],imode,freq2[imode]))
        ###freq2[ntran+nrot:] = 1./freq1[ntran+nrot:]
        # 2nd effort
        #freq1[ntran+nrot:] = freq2[ntran+nrot:]
        print("Eigenvalues in au for inverted direct mwt hess")
        # invert trans + rot if abs(freq1[0:6]) > 1.e-5
        for imode in range(ntran+nrot):
            if np.abs(freq1[imode]) > 1.e-5:
                freq2[imode] = 1./freq1[imode]
                print("inverting freq1[%d] = %15.7f --> %15.7f" % (imode,freq1[imode],freq2[imode]))
            else:
                print("not inverting freq1[%d] = %15.7f to %15.7f  freq2[^%d] = %15.7f"
                     % (imode,freq1[imode],1./freq1[imode],imode,freq2[imode]))
        freq2[ntran+nrot:] = 1./freq1[ntran+nrot:]

    inverted_mat = np.zeros_like(evec,dtype=float)
    
    for imode in range(len(freq)):
        print("%3d inv_freq = %16.7f   1/inv_freq = %16.7f  orig freq = %12.3f 1/cm"
              % (imode,freq2[imode],freq1[imode],freq[imode]))
        inverted_mat += freq2[imode] * np.outer(evec[:,imode],evec[:,imode])
        
    if inv_hess:
        print("=============== finished inverting the inverse mwt hessian =================")
    else:
        print("=============== finished inverting the direct mwt hessian =================")
        
    return inverted_mat,not inv_hess

# scale energy hess or invhess with masses before calling freq_calc
def mwt_ehess(mol_name,ehess,mass,mass_type="atmwt",mass_detscl=1.,inv_hess=False):
    """ funtion to set up mwt hess or invhess from energy ehess or einvhess
        parameters
        ----------
        mol_name: string
        ehess: ndarray
        contains hess to be mass weighted
        mass: ndarry
        atom masses - shape (3,3*natom) 
        where jj: (jj=0) mass**1 (jj=1) mass**1/2 (jj=2) mass**(-1/2))
    
        mass_type: str
        Either "atmwt" (traditional) or "utmwt" (unitary)
        
        mass_detscl: float
        determinant scaling factor for masses "atmwt" = 1. "utmwt" = mass_detscl
        inv_hess: bool
        True if starting hess direct, False is init hess inverse - not mass wtd
        
        return
        -------
        mwt_hess: ndarry
        mwt_hess =  ehess[i,j]*(mass[jj,iat]*mass[jj,jat])
    """
    # initially set mwt_hess as a copy of routine input ehess
    mwt_hess = ehess.copy()
    numat = mwt_hess.shape[0]//3
    
    print("\n================ Start of forming mass weighted hessian =============")
    print("=== mass_type = %s   mass_detscl = %10.5f    inv_hess = %s\n" % (mass_type,mass_detscl,inv_hess))
    print("==== mass.shape = ",mass.shape)
    if mass_type == "atmwt":
        print("traditional freq calc on molecule %s with molar mass  %.7f" 
          % (mol_name,np.sum(mass[0])/3.))

    elif mass_type == "utmwt":
        scaled_m = mass.copy()
        print("unitary freq calc on molecule %s unit molecular wt  %.7f and detscl %15.6f"
              % (mol_name,np.sum(mass[0])/3.,mass_detscl))
    else:
        print("ERROR in mwt_ehess - mass_type = %s which is not an allowed option")
        return 1000
        
    if inv_hess:
        # scaled_m = mass ** half
        scaled_m = mass[1]
        print("Forming mass weighted inv_hess")
    else:
        # hess scaled by mass ** -half
        scaled_m = mass[2]
        print("Forming mass weighted hess")

    for i in range(3*numat):
        mwt_hess[i,:] *= scaled_m[:]
        
    for j in range(3*numat):
        mwt_hess[:,j] *= scaled_m[:]
        
    return mwt_hess

def freq_calc(hess,detscl=1.,ref_freq=None,long_freq_out=False,inv_hess=False):
    """ calc vibrational frequencies from the mass weighted hessian matrix hess
       and compare the calc frequencies with ref_freq computed by other hessian calc
       mass_unit gives the scale so that the atomic masses are kg/mol units = 1000. typically
       detscl = geometric mean of atomic masses
       inv_hess=True when hess is a mass_wted inv_hess form in scl_einvhess"""
    print("======== Start of computing vibrational freq from mass weighted hess ========")
    print("============== Trace of hess in freq calc = %16.8e ============="
          % np.trace(hess))
    # unit conversion
    hartree2waveno = 219474.6
    ck_print("hartree2waveno = %f" % hartree2waveno)
    au2amu = 5.4857990907e-04  # CODATA recommend value 2019 = 5.485 799 090 65(16) e-04  ?? corrected 8-may-2020
    #sqrt_au2amu = np.sqrt(au2amu/mass_unit)  # convert amu from g to kg
    sqrt_au2amu = np.sqrt(au2amu/1000.)  # convert amu from g to kg
    #Evib = hbar * omega  = hbar * sqrt(k/m)
    if inv_hess:
        radpsec2waveno = 1./(hartree2waveno*sqrt_au2amu)
    else:     # unit conversion for mass weighted hessian
        radpsec2waveno = hartree2waveno*sqrt_au2amu
    ck_print("au2amu %f inv(au2amu) %f -- radpsec2waveno %f"
                  % (au2amu,1./au2amu,radpsec2waveno))
    #hartree2Hz = 6.579684e3
    #Hz2waveno = hartree2Hz / hartree2waveno
    #print("Hz2waveno = %f" % Hz2waveno)
    #mat_dim = len(nwchem_freq)
    mat_dim = hess.shape[0]
    # symmetrize the hess matrix
        
    # find eigenvalues and mass weighted evec from hess
    freq3,evec= scipy.linalg.eigh(hess)

    # scale the frequency by the mass_unit conversion factor
    freq2 = freq3.copy()
    scale = radpsec2waveno*np.ones_like(freq2)
    scale[freq2<0] = -scale[freq2<0]
    freq2[freq2<0] = -freq2[freq2<0]
    # set up mass_unit
    mass_unit = detscl / 1000.
    print("\n mass_unit = detscl/1000. = %12.6f  detscl = %12.6f" % (mass_unit,detscl))
    #for ivec in range(mat_dim):
    #    print("ivec %d scale[ivec] = %12.5e  freq2[ivec] = %12.5e  1/freq2[ivec] = %12.5e" 
    #         % (ivec,scale[ivec],freq2[ivec],1./freq2[ivec]))
    if inv_hess:
        # comment out mass scaling to see if mass_unit giving a problem
        freq2 *= mass_unit  # need to check this works
        
        freq = 1./(scale *np.sqrt(freq2))
        print("inv_test -- freq = 1/(scale*np.sqrt(freq2)) ->\n,",freq2)
        #junk print("inv_test2 -- freq/mass_unit**2 -->",mass_unit/scale*np.sqrt(freq2))
        print("\n Frequency (1/cm) from inverse hess + mat 1/evals (au)")
        # reverse order of inv_hess eigenvals starting with ivec value when freq2[ivec] > 1.e-5
        for ivec in range(mat_dim):
            if np.abs(freq2[ivec]) < 1.e-5:
                print("abs(freq2[%d]) < 1.e-5 -- freq2 = %9.5e   freq[%d] set to zero"
                      % (ivec,freq2[ivec],ivec))
                freq[ivec]=0.
                
        #print("=== Not doing eval,evec flip flipiv = %d" % flipiv)
        #order freq in increasing order
        fr_ord = np.argsort(freq)
        tmp_fr = freq[fr_ord[:]]
        freq= tmp_fr.copy()
        tmp_fr=freq3[fr_ord[:]]
        freq3 = tmp_fr.copy()
        tmp_vec = evec[:,fr_ord[:]]
        evec = tmp_vec.copy()
        del tmp_fr
        del tmp_vec
        del fr_ord

    else:   # scale factor of direct mass_wted
        # comment out mass scaling to see if mass_unit giving a problem
        freq2 /= mass_unit
        #print("mass_unit = %12.5e freq2[6:10]" % mass_unit,freq2[6:10])
        freq = scale * np.sqrt(freq2)
        ck_print("\n Frequency (1/cm) from dir hess +  mat eigenvals (au) ")
    #print("vib freq from hess:",freq)
    sum_str = ""
    if not long_freq_out:
        print("===== Freq in 1/cm")
    for ivec in range(mat_dim):
        #print("ivec %d  %10.3f -- ev %16.7f  1/ev %16.7f"
        #        % (ivec,freq[ivec],freq2[ivec],1./freq2[ivec]))
        if long_freq_out:
            format("ivec %d  %10.3f 1/cm -- actual ev %16.7f  1/ev %16.7f"
                % (ivec,freq[ivec],freq3[ivec],1./freq3[ivec]))
        else:
            if len(sum_str) > 75:
                print(sum_str)
                sum_str = ""
            sum_str += "%3d %8.1f   " % (ivec,freq[ivec])


    # print out end of sum_str
    if not long_freq_out and len(sum_str) > 0:
        print(sum_str)
    
    #print("ref_freq:\n",ref_freq)
    
    #####################################################################
    #
    # add in reduce mass calc using mass weightet evec from hessian
    print("^^^^^^^^ going to compute reduced mass here ^^^^^^^^")
    #
    #####################################################################
    
    if ref_freq is None:
        # print out just frequencies
        print("========= print out straight freq and their inverse here")
            
    else:   # compare computed freq against ref_freq
        for imode in range(mat_dim):
            #if ref_freq[imode] < 5.0:
            #ratio = 0.5
            #else:
            freq_diff = ref_freq[imode]-freq[imode]
            if np.abs(freq_diff) > 10.:
                freq_diff = ref_freq[imode]/freq[imode]
            #ratio = ref_freq[imode]/freq[imode]
            print("diff ref_freq[%2d] - cmp_freq[%2d] = %9.3f - %9.3f = %10.4f" 
                 % (imode,imode,ref_freq[imode],freq[imode],freq_diff))

    print("============ End of diagonalizing mass weighted Hess ===================")    
    return 0, freq,evec

def test_mwthess_projd(hess,tran_rot_v,detscl=1.,inv_hess=False,
                       test_thres=1.e-10):
    """ routine to check if the trans/rot modes in the mwt_hess have zero frequency
        return
        ------
        tranrot_projd: bool
        True if tran/rot frequencies are zero and no projection needed or False otherwise
    """
    (mat_dim,no_tr_rot_v) = tran_rot_v.shape
    hess_trv = np.dot(hess,tran_rot_v)
    v_hess_v = np.dot(tran_rot_v.T,hess_trv)
    ck_print("tran_rot_v.T*hess*tran_rot_v = ",v_hess_v)
    
    abs_diagsum = 0.
    for ii in range(no_tr_rot_v):
        abs_diagsum += np.abs(v_hess_v[ii,ii])
    tracevhv = np.trace(v_hess_v)
    print("\n Trace of v_hes_v = ",np.trace(v_hess_v))
    print("Abs trace of v_hess_v = ",abs_diagsum)

    # could add here a return True if tracevhv < some threshold
    if abs_diagsum < test_thres:
        print("Test_mwthess_projd trace < test_thres = %10.4e" % test_thres,
              "no need to do Trans/Rots projection")
        #return True
    
    # unit conversion
    hartree2waveno = 219474.6
    #print("hartree2waveno = %f" % hartree2waveno)
    au2amu = 5.485799097e-04
    sqrt_au2amu = np.sqrt(au2amu/1000.)  # convert amu from g to kg
    #Evib = hbar * omega  = hbar * sqrt(k/m)
    radpsec2waveno = hartree2waveno*sqrt_au2amu
    print("au2amu %f inv(au2amu) %f -- radpsec2waveno %f" % (au2amu,1./au2amu,radpsec2waveno))
    
    # diagonalize v_hess_v and check out eigenvectors
    # find eigenvals and evecs of rot_orthog_chk
    
    vhv_eval,vhv_evec= scipy.linalg.eigh(v_hess_v)
    # set up mass_unit
    mass_unit = detscl/1000.   #  corrects atomic masses when unit
    if inv_hess:
        print(" projecting inv_hess - scale vhv_eval by mass_unit = %.9f" % mass_unit)
        vhv_eval *= mass_unit
    else:
        print("projecting hess - divide vhv_eval by mass_unit = %.9f" % mass_unit)
        vhv_eval /= mass_unit
    
    print("\nv_hess_v evals and evecs: sum = %.10e" % np.sum(np.abs(vhv_eval)))
    for iv in range(len(vhv_eval)):
        #tmp_vec = np.abs(vhv_evec[:,iv])
        ord_vec = np.argsort(np.abs(vhv_evec[:,iv]))[::-1]
        if vhv_eval[iv] >= 0.:
            eval_cm = radpsec2waveno*np.sqrt(vhv_eval[iv])
        else:
            eval_cm = -radpsec2waveno*np.sqrt(-vhv_eval[iv])
        print("vhv_eval[%d] = %f freq = %9.3f 1/cm  abs_sort max-> "
              % (iv,vhv_eval[iv],eval_cm),ord_vec)
        ck_print("evec:", vhv_evec[:,iv])
    if abs_diagsum < test_thres:
        return True
    else:
        print("Test_mwthess_projd trace not below %10.5e" % test_thres,
              " - need to project out mwthess Trans/Rots modes")
        return False

def proj_trans_rots_frm_hess(hess,tran_rot_v,detscl=1.,inv_hess=False,ref_freq=None):
    """ routine to project out trans/rotational modes from hess and get new freqs
        hess needs to be symmetrical 
        method uses just 5 or 6 linear combinations of tran_rot_v vectors in projection
        
        Parameters
        ----------
        hess: ndarray
        mwt-hess or inv-hess
        tran_rot_v: ndarray
        detscl: float
        = 1. if 'atmwt' and = X. if 'unit'
        inv_hess: bool
        ref_freq: ndarray
        list of frequency for comparison with freq from proj hessian
        
        Returns
        -------
        0,proj_hess,proj_eval,proj_evec
        """
    print("\n\n===== Projecting trans/rots modes out of mass weighted hess =====")
    ck_print("hess in proj_trans_rots:\n",hess[:,:5])
    ck_print("tran_rot_v.shape = ",tran_rot_v.shape)
    # 
    # get dimension info
    (mat_dim,no_tr_rot_v) = tran_rot_v.shape
    print("Len of normal mode vector = %d    no tran rot vecs = %d" % (mat_dim,no_tr_rot_v))
    
    # checking whether tran/rot eigenvalues are zero before calling proj_trans_rots_frm_hess
    # therefor skip this
    #tran_rot_zero = test_mwthess_projd(hess,tran_rot_v,detscl=detscl,inv_hess=inv_hess)
    # method uses the projector P = 1 - sum_i v[:,i] * v[:,i]
    # where i runs over all the trans + rot vibrational modes
    # then form proj_hess  = P * hess * P
    #print("\n ======= Projecting trans/rots out of mwt hessian matrix -->")
    proj = np.identity(mat_dim,dtype=float)
    for iv in range(no_tr_rot_v):
            
        proj -= np.outer(tran_rot_v[:,iv],tran_rot_v[:,iv])
        
    #print("proj.shape =",proj.shape)
    #print(proj)
        
    proj_hess = np.linalg.multi_dot([proj,hess,proj])
        
    #print("proj_hess.shape = ",proj_hess.shape)
        
    print("\n ===== Finished projecting the trans/rot modes out of mwt hess matrix")
        
        
    
    max_off_diff =0.
    for icol in range(1,mat_dim):
        for jcol in range(icol):
            diff = np.abs(proj_hess[icol,jcol] - proj_hess[jcol,icol]) 
            if diff > max_off_diff:
                max_off_diff = diff
                ii = icol
                jj = jcol
    if max_off_diff > 1.e-10:
        print("***WARNING*** [%2d,%2d] max_off_diff_proj_hess2 = %e" % (ii,jj,diff))
    
    #  freq_calc reminder

    # calc freqs separate to proj fn
    #ret_code,proj_eval,proj_evec = freq_calc(proj_hess,detscl=detscl,ref_freq=ref_freq,
    #                                         freq_out=True,inv_hess=inv_hess)

    #  check that the projected hessian is gives tran_rot_zero = True 
    #  projection seems to be working - so there is not need to do this - keep as check for now
    #
    # add the following 2 lines of code if you want to check if projection working correctly
    
    #tran_rot_zero = test_mwthess_projd(proj_hess,tran_rot_v,detscl=detscl,inv_hess=inv_hess)
    #print("test_mwthess_projd = %s after projecting hessian" % tran_rot_zero)
    
    #return 0,proj_hess,proj_eval,proj_evec
    return 0, proj_hess
