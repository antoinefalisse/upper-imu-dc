import numpy as np
import pandas as pd
from scipy import signal
from scipy.interpolate import interp1d
from sys import path
path.append(r"C:/Users/u0101727/Documents/Software/CasADi/casadi-windows-py37-v3.5.1-64bit")
import casadi as ca
import matplotlib.pyplot as plt  

# Found here: https://github.com/chrisdembia/perimysium/ => thanks Chris
def storage2numpy(storage_file, excess_header_entries=0):
    """Returns the data from a storage file in a numpy format. Skips all lines
    up to and including the line that says 'endheader'.
    Parameters
    ----------
    storage_file : str
        Path to an OpenSim Storage (.sto) file.
    Returns
    -------
    data : np.ndarray (or numpy structure array or something?)
        Contains all columns from the storage file, indexable by column name.
    excess_header_entries : int, optional
        If the header row has more names in it than there are data columns.
        We'll ignore this many header row entries from the end of the header
        row. This argument allows for a hacky fix to an issue that arises from
        Static Optimization '.sto' outputs.
    Examples
    --------
    Columns from the storage file can be obtained as follows:
        >>> data = storage2numpy('<filename>')
        >>> data['ground_force_vy']
    """
    # What's the line number of the line containing 'endheader'?
    f = open(storage_file, 'r')

    header_line = False
    for i, line in enumerate(f):
        if header_line:
            column_names = line.split()
            break
        if line.count('endheader') != 0:
            line_number_of_line_containing_endheader = i + 1
            header_line = True
    f.close()

    # With this information, go get the data.
    if excess_header_entries == 0:
        names = True
        skip_header = line_number_of_line_containing_endheader
    else:
        names = column_names[:-excess_header_entries]
        skip_header = line_number_of_line_containing_endheader + 1
    data = np.genfromtxt(storage_file, names=names,
            skip_header=skip_header)

    return data

def getIK(storage_file, joints, degrees=False):
    # Extract data
    data = storage2numpy(storage_file)
    Qs = pd.DataFrame(data=data['time'], columns=['time'])    
    for count, joint in enumerate(joints):  
        if ((joint == 'pelvis_tx') or (joint == 'pelvis_ty') or 
            (joint == 'pelvis_tz')):
            Qs.insert(count + 1, joint, data[joint])         
        else:
            if degrees == True:
                Qs.insert(count + 1, joint, data[joint])                  
            else:
                Qs.insert(count + 1, joint, data[joint] * np.pi / 180)              
            
    # Filter data    
    fs=1/np.mean(np.diff(Qs['time']))    
    fc = 6  # Cut-off frequency of the filter
    order = 4
    w = fc / (fs / 2) # Normalize the frequency
    b, a = signal.butter(order/2, w, 'low')  
    output = signal.filtfilt(b, a, Qs.loc[:, Qs.columns != 'time'], axis=0, 
                             padtype='odd', padlen=3*(max(len(b),len(a))-1))    
    output = pd.DataFrame(data=output, columns=joints)
    QsFilt = pd.concat([pd.DataFrame(data=data['time'], columns=['time']), 
                        output], axis=1)    
    
    return Qs, QsFilt

def getGRF(storage_file, headers):
    # Extract data
    data = storage2numpy(storage_file)
    GRFs = pd.DataFrame(data=data['time'], columns=['time'])    
    for count, header in enumerate(headers):
        GRFs.insert(count + 1, header, data[header])    
    
    return GRFs

def getID(storage_file, headers):
    # Extract data
    data = storage2numpy(storage_file)
    out = pd.DataFrame(data=data['time'], columns=['time'])    
    for count, header in enumerate(headers):
        if ((header == 'pelvis_tx') or (header == 'pelvis_ty') or 
            (header == 'pelvis_tz')):
            out.insert(count + 1, header, data[header + '_force'])              
        else:
            out.insert(count + 1, header, data[header + '_moment'])    
    
    return out

def getFromStorage(storage_file, headers):
    # Extract data
    data = storage2numpy(storage_file)
    out = pd.DataFrame(data=data['time'], columns=['time'])    
    for count, header in enumerate(headers):
        out.insert(count + 1, header, data[header])    
    
    return out

def getFromTRC(pathTRC, markers, dimensions = ["x", "y", "z"]):    
    import dataman    
    trc = dataman.TRCFile(pathTRC)    
    trc_data = pd.DataFrame(data=trc.time, columns=['time'])
    count = 0
    if trc.units == "mm":
        scaling = 1000
    elif trc.units == "cm":
        scaling = 100
    elif trc.units == "dm":
        scaling = 10
    elif trc.units == "m":
        scaling = 1
    else:
        raise ValueError('TRC data units not recognized.')
    for marker in markers:
        marker_data = trc.marker(marker)
        assert len(dimensions) == marker_data.shape[1]
        for d, dimension in enumerate(dimensions):
            name = marker + "_" + dimension
            trc_data.insert(count + 1, name, marker_data[:,d]/scaling) 
            count += 1
    return trc_data

def getGRM_wrt_groundOrigin(storage_file, fHeaders, pHeaders, mHeaders):
    # Extract data
    data = storage2numpy(storage_file)
    GRFs = pd.DataFrame()    
    for count, fheader in enumerate(fHeaders):
        GRFs.insert(count, fheader, data[fheader])  
    PoAs = pd.DataFrame()    
    for count, pheader in enumerate(pHeaders):
        PoAs.insert(count, pheader, data[pheader]) 
    GRMs = pd.DataFrame()    
    for count, mheader in enumerate(mHeaders):
        GRMs.insert(count, mheader, data[mheader])  
        
    # GRT_x = PoA_y*GRF_z - PoA_z*GRF_y
    # GRT_y = PoA_z*GRF_x - PoA_z*GRF_z + T_y
    # GRT_z = PoA_x*GRF_y - PoA_y*GRF_x
    GRM_wrt_groundOrigin = pd.DataFrame(data=data['time'], columns=['time'])    
    GRM_wrt_groundOrigin.insert(1, mHeaders[0], PoAs[pHeaders[1]] * GRFs[fHeaders[2]]  - PoAs[pHeaders[2]] * GRFs[fHeaders[1]])
    GRM_wrt_groundOrigin.insert(2, mHeaders[1], PoAs[pHeaders[2]] * GRFs[fHeaders[0]]  - PoAs[pHeaders[0]] * GRFs[fHeaders[2]] + GRMs[mHeaders[1]])
    GRM_wrt_groundOrigin.insert(3, mHeaders[2], PoAs[pHeaders[0]] * GRFs[fHeaders[1]]  - PoAs[pHeaders[1]] * GRFs[fHeaders[0]])        
    
    return GRM_wrt_groundOrigin

def getJointIndices(joints, selectedJoints):
    
    jointIndices = []
    for joint in selectedJoints:
        jointIndices.append(joints.index(joint))
            
    return jointIndices

def getMomentArmIndices(muscles, polynomialJoints, polynomialData):
         
    momentArmIndices = {}
    for count, muscle in enumerate(muscles):        
        spanning = polynomialData[muscle]['spanning']
        for i in range(len(spanning)):
            if (spanning[i] == 1):
                momentArmIndices.setdefault(
                        polynomialJoints[i], []).append(count)     
        
    return momentArmIndices

def solve_with_bounds(opti, tolerance):
    # Get guess
    guess = opti.debug.value(opti.x, opti.initial())
    # Sparsity pattern of the constraint Jacobian
    jac = ca.jacobian(opti.g, opti.x)
    sp = (ca.DM(jac.sparsity(), 1)).sparse()
    # Find constraints dependent on one variable
    is_single = np.sum(sp, axis=1)
    is_single_num = np.zeros(is_single.shape[0])
    for i in range(is_single.shape[0]):
        is_single_num[i] = np.equal(is_single[i, 0], 1)
    # Find constraints with linear dependencies or no dependencies
    is_nonlinear = ca.which_depends(opti.g, opti.x, 2, True)
    is_linear = [not i for i in is_nonlinear]
    is_linear_np = np.array(is_linear)
    is_linear_np_num = is_linear_np*1
    # Constraints dependent linearly on one variable should become bounds
    is_simple = is_single_num.astype(int) & is_linear_np_num
    idx_is_simple = [i for i, x in enumerate(is_simple) if x]
    ## Find corresponding variables
    col = np.nonzero(sp[idx_is_simple, :].T)[0]
    # Contraint values
    lbg = opti.lbg
    lbg = opti.value(lbg)
    ubg = opti.ubg
    ubg = opti.value(ubg)
    # Detect  f2(p)x+f1(p)==0
    # This is important should you have scaled variables: x = 10*opti.variable()
    # with a constraint -10 < x < 10. Because in the reformulation we read out the
    # original variable and thus we need to scale the bounds appropriately.
    g = opti.g
    gf = ca.Function('gf', [opti.x, opti.p], [g[idx_is_simple, 0], 
                            ca.jtimes(g[idx_is_simple, 0], opti.x, 
                                      np.ones((opti.nx, 1)))])
    [f1, f2] = gf(0, opti.p)
    f1 = (ca.evalf(f1)).full() # maybe a problem here
    f2 = (ca.evalf(f2)).full()
    lb = (lbg[idx_is_simple] - f1[:, 0]) / np.abs(f2[:, 0])
    ub = (ubg[idx_is_simple] - f1[:, 0]) / np.abs(f2[:, 0])
    # Initialize bound vector
    lbx = -np.inf * np.ones((opti.nx))
    ubx = np.inf * np.ones((opti.nx))
    # Fill bound vector. For unbounded variables, we keep +/- inf.
    for i in range(col.shape[0]):
        lbx[col[i]] = np.maximum(lbx[col[i]], lb[i])
        ubx[col[i]] = np.minimum(ubx[col[i]], ub[i])      
    lbx[col] = (lbg[idx_is_simple] - f1[:, 0]) / np.abs(f2[:, 0])
    ubx[col] = (ubg[idx_is_simple] - f1[:, 0]) / np.abs(f2[:, 0])
    # Updated constraint value vector
    not_idx_is_simple = np.delete(range(0, is_simple.shape[0]), idx_is_simple)
    new_g = g[not_idx_is_simple, 0]
    # Updated bounds
    llb = lbg[not_idx_is_simple]
    uub = ubg[not_idx_is_simple]
    
    prob = {'x': opti.x, 'f': opti.f, 'g': new_g}
    s_opts = {}
    s_opts["expand"] = False
    s_opts["ipopt.hessian_approximation"] = "limited-memory"
    s_opts["ipopt.mu_strategy"] = "adaptive"
    s_opts["ipopt.max_iter"] = 2500
    s_opts["ipopt.tol"] = 10**(-tolerance)
#    s_opts["ipopt.print_frequency_iter"] = 20 
    solver = ca.nlpsol("solver", "ipopt", prob, s_opts)
    # Solve
    arg = {}
    arg["x0"] = guess
    # Bounds on x
    arg["lbx"] = lbx
    arg["ubx"] = ubx
    # Bounds on g
    arg["lbg"] = llb
    arg["ubg"] = uub    
    sol = solver(**arg) 
    # Extract and save results
    w_opt = sol['x'].full()
    stats = solver.stats()
    
    return w_opt, stats

def solve_with_constraints(opti, tolerance):
    s_opts = {"hessian_approximation": "limited-memory",
              "mu_strategy": "adaptive",
              "max_iter": 2,
              "tol": 10**(-tolerance)}
    p_opts = {"expand":False}
    opti.solver("ipopt", p_opts, s_opts)
    sol = opti.solve()  
    
    return sol

# Test
#storage_file = 'IK_average_running_HGC.mot'
#joints = ['pelvis_tx','pelvis_tilt', 'pelvis_rotation']
#
#data = storage2numpy(storage_file)
#Qs, Qsfilt = getIK(storage_file, joints)
    
def numpy2storage(labels, data, storage_file):
    
    assert data.shape[1] == len(labels), "# labels doesn't match columns"
    assert labels[0] == "time"
    
    f = open(storage_file, 'w')
    f.write('name %s\n' %storage_file)
    f.write('datacolumns %d\n' %data.shape[1])
    f.write('datarows %d\n' %data.shape[0])
    f.write('range %f %f\n' %(np.min(data[:, 0]), np.max(data[:, 0])))
    f.write('endheader \n')
    
    for i in range(len(labels)):
        f.write('%s\t' %labels[i])
    f.write('\n')
    
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            f.write('%20.8f\t' %data[i, j])
        f.write('\n')
        
    f.close()  
    
def interpolateDataFrame2Numpy(dataFrame, tIn, tEnd, N):   
    
    tOut = np.linspace(tIn, tEnd, N)
    dataInterp = np.zeros([N, len(dataFrame.columns)])
    for i, col in enumerate(dataFrame.columns):
        set_interp = interp1d(dataFrame['time'], dataFrame[col])
        dataInterp[:,i] = set_interp(tOut)
        
    return dataInterp    

def interpolateDataFrame(dataFrame, tIn, tEnd, N):   
    
    tOut = np.linspace(tIn, tEnd, N)    
    dataInterp = pd.DataFrame() 
    for i, col in enumerate(dataFrame.columns):
        set_interp = interp1d(dataFrame['time'], dataFrame[col])        
        dataInterp.insert(i, col, set_interp(tOut))
        
    return dataInterp 

def scaleDataFrame(dataFrame, scaling, headers):
    dataFrame_scaled = pd.DataFrame(data=dataFrame['time'], columns=['time'])  
    for count, header in enumerate(headers): 
        dataFrame_scaled.insert(count+1, header, dataFrame[header] / scaling.iloc[0][header])
        
    return dataFrame_scaled

def selectFromDataFrame(dataFrame, headers):
    dataFrame_sel = pd.DataFrame(data=dataFrame['time'], columns=['time'])  
    for count, header in enumerate(headers): 
        dataFrame_sel.insert(count+1, header, dataFrame[header])
        
    return dataFrame_sel

def unscaleDataFrame2(dataFrame, scaling, headers):
    dataFrame_scaled = pd.DataFrame(data=dataFrame['time'], columns=['time'])  
    for count, header in enumerate(headers): 
        dataFrame_scaled.insert(count+1, header, dataFrame[header] * scaling.iloc[0][header])
        
    return dataFrame_scaled

def plotVSBounds(y,lb,ub,title=''):    
    ny = np.ceil(np.sqrt(y.shape[0]))   
    fig, axs = plt.subplots(int(ny), int(ny), sharex=True)    
    fig.suptitle(title)
    x = np.linspace(1,y.shape[1],y.shape[1])
    for i, ax in enumerate(axs.flat):
        if i < y.shape[0]:
            ax.plot(x,y[i,:],'k')
            ax.hlines(lb[i,0],x[0],x[-1],'r')
            ax.hlines(ub[i,0],x[0],x[-1],'b')
            
def plotVSvaryingBounds(y,lb,ub,title=''):    
    ny = np.ceil(np.sqrt(y.shape[0]))   
    fig, axs = plt.subplots(int(ny), int(ny), sharex=True)    
    fig.suptitle(title)
    x = np.linspace(1,y.shape[1],y.shape[1])
    for i, ax in enumerate(axs.flat):
        if i < y.shape[0]:
            ax.plot(x,y[i,:],'k')
            ax.plot(x,lb[i,:],'r')
            ax.plot(x,ub[i,:],'b')
    plt.show()
            
def plotParametersVSBounds(y,lb,ub,title='',xticklabels=[]):    
    x = np.linspace(1,y.shape[0],y.shape[0])   
    plt.figure()
    ax = plt.gca()
    ax.scatter(x,lb,c='r',marker='_')
    ax.scatter(x,ub,c='b',marker='_')
    ax.scatter(x,y,c='k')
    ax.set_xticks(x)
    ax.set_xticklabels(xticklabels) 
    ax.set_title(title)
    
def nSubplots(N):
    
    ny_0 = (np.sqrt(N)) 
    ny = np.round(ny_0) 
    ny_a = int(ny)
    ny_b = int(ny)
    if (ny == ny_0) == False:
        if ny_a == 1:
            ny_b = N
        if ny < ny_0:
            ny_b = int(ny+1)
            
    return ny_a, ny_b

def getIdxIC(GRF_opt, threshold):    
    idxIC = np.nan
    N = GRF_opt.shape[1]
    legIC = "undefined"    
    GRF_opt_rl = np.concatenate((GRF_opt[1,:], GRF_opt[3,:]))
    last_noContact = np.argwhere(GRF_opt_rl < threshold)[-1]
    if last_noContact == 2*N - 1:
        first_contact = np.argwhere(GRF_opt_rl > threshold)[0]
    else:
        first_contact = last_noContact + 1
    if first_contact >= N:
        idxIC = first_contact - N
        legIC = "left"
    else:
        idxIC = first_contact
        legIC = "right"
            
    return idxIC, legIC
            
def getRMSE(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

def getRMSENormMinMax(predictions, targets):    
    ROM = np.max(targets) - np.min(targets)    
    return (np.sqrt(((predictions - targets) ** 2).mean()))/ROM

def getRMSENormStd(predictions, targets):    
    std = np.std(targets)
    return (np.sqrt(((predictions - targets) ** 2).mean()))/std

def getR2(predictions, targets):
    return (np.corrcoef(predictions, targets)[0,1])**2 

def getMetrics(predictions, targets):
    r2 = np.zeros((predictions.shape[0]))
    rmse = np.zeros((predictions.shape[0]))
    rmseNormMinMax = np.zeros((predictions.shape[0]))
    rmseNormStd = np.zeros((predictions.shape[0]))
    for i in range(predictions.shape[0]):
        r2[i] = getR2(predictions[i,:], targets[i,:]) 
        rmse[i] = getRMSE(predictions[i,:],targets[i,:])  
        rmseNormMinMax[i] = getRMSENormMinMax(predictions[i,:],targets[i,:])   
        rmseNormStd[i] = getRMSENormStd(predictions[i,:],targets[i,:])        
    return r2, rmse, rmseNormMinMax, rmseNormStd

def eulerIntegration(xk_0, xk_1, uk, delta_t):
    
    return (xk_1 - xk_0) - uk * delta_t


from scipy.spatial.transform import Rotation as R 
def getIMUStorageToNumpy(storage_file, headers, 
                         R_sensor_to_opensim = R.from_euler('xyz', 
                                                            [0,0,0],
                                                            degrees=False), 
                         timeInfo=[np.nan, np.nan],
                         NInterpolate=np.nan, NElements=3):
    """Returns the data from a storage file in a numpy format. The storage file
    contains SimTK::Vec3 or Quaternions, thus multiple entries per cell. This
    function returns the data as numpy array with one entry per cell. Skips all
    lines up to and including the line that says 'endheader'.
    Parameters
    ----------
    storage_file : str
        Path to an OpenSim Storage (.sto) file.
    headers : list of str
        Headers for which data are returned.
    NElements : int
        Number of entries per cell (default is 3 for SimTK::Vec3).
    Returns
    -------
    data : np.array
        Contains data from the storage file corresponding to headers.
    data_sel : np.array
        Contains data from the storage file corresponding to headers, from
        timeInfo[0] to timeInfo[1].
    data_interp : np.array
        Contains data from the storage file corresponding to headers, from
        timeInfo[0] to timeInfo[1], interpolated on NInterpolate points. 
    """
    
    f = open(storage_file, 'r')
    header_line = False
    for i, line in enumerate(f):
        if header_line:
            break
        if line.count('endheader') != 0:
            skip_header = i + 1
            header_line = True
    f.close()
    
    df = pd.read_csv(storage_file, sep='\t', skiprows=skip_header)    
    for i, header in enumerate(headers):
        data_tuple = df[header].map(eval).values                
        data_header = np.zeros((data_tuple.shape[0], NElements))        
        for n in range(data_tuple.shape[0]):    
            if not NElements == 3:
                data_header[n,:] = np.reshape(
                    np.asarray(data_tuple[n]), (-1, 1)).T
            else:
                data_header[n,:] = R_sensor_to_opensim.apply(
                    np.reshape(np.asarray(data_tuple[n]), (-1, 1)).T) 
        if i == 0:
            data = data_header
        else:
            data = np.concatenate((data, data_header), axis=1)
            
    if (not np.isnan(timeInfo[0])) and (not np.isnan(timeInfo[1])):
        timeAll = df['time']
        start, idx_start = find_nearest_idx(timeAll,timeInfo[0])
        end, idx_end = find_nearest_idx(timeAll,timeInfo[1])        
        data_sel = data[idx_start:idx_end+1, :]
        
        if not np.isnan(NInterpolate):
            tOut = np.linspace(start, end, NInterpolate)
            data_interp = np.zeros((NInterpolate, data_sel.shape[1]))            
            for i in range(data_sel.shape[1]):
                set_interp = interp1d(timeAll[idx_start:idx_end+1], 
                                      data_sel[:,i])       
                data_interp[:,i] = set_interp(tOut)
                
            return data, data_sel, data_interp
                
        else:
                
            return data, data_sel
        
    else:
        
        return data   

def getTimeIMUStorageToNumpy(storage_file):
    """Returns the data from a storage file in a numpy format. The storage file
    contains SimTK::Vec3 or Quaternions, thus multiple entries per cell. This
    function returns the data as numpy array with one entry per cell. Skips all
    lines up to and including the line that says 'endheader'.
    Parameters
    ----------
    storage_file : str
        Path to an OpenSim Storage (.sto) file.
    headers : list of str
        Headers for which data are returned.
    NElements : int
        Number of entries per cell (default is 3 for SimTK::Vec3).
    Returns
    -------
    data : np.array
        Contains data from the storage file corresponding to headers.
    data_sel : np.array
        Contains data from the storage file corresponding to headers, from
        timeInfo[0] to timeInfo[1].
    data_interp : np.array
        Contains data from the storage file corresponding to headers, from
        timeInfo[0] to timeInfo[1], interpolated on NInterpolate points. 
    """
    
    f = open(storage_file, 'r')
    header_line = False
    for i, line in enumerate(f):
        if header_line:
            break
        if line.count('endheader') != 0:
            skip_header = i + 1
            header_line = True
    f.close()
    
    df = pd.read_csv(storage_file, sep='\t', skiprows=skip_header)    
    time = df['time'].to_numpy()
        
    return time        

def find_nearest_idx(array,value):
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or np.abs(value - array[idx-1]) < np.abs(value - array[idx])):
        return array[idx-1], idx-1
    else:
        return array[idx], idx    
    
def getHeadingCorrection(path_Orientations, base_imu, heading_axis, ref_axis,
                         rotation_axis,
                         R_sensor_data = R.from_euler('xyz', [0,0,0])):
    if heading_axis[0] == "-":
        direction = -1
    else:
        direction = 1   
        
    if heading_axis[-1] == 'x':
        directionOnIMU = 0
    elif heading_axis[-1] == 'y':
        directionOnIMU = 1
    elif heading_axis[-1] == 'z':
        directionOnIMU = 2
        
    if rotation_axis == 'x':
        rotation_direction = 0
    elif rotation_axis == 'y':
        rotation_direction = 1
    elif rotation_axis == 'z':
        rotation_direction = 2
    
    imu_orientations_quat = getIMUStorageToNumpy(path_Orientations, [base_imu], 
                                                R.from_euler('xyz', [0,0,0]), 
                                                [np.nan, np.nan],
                                                np.nan, 4)  
    # scipy and Simbody have different quaternions representation. 
    # In Simbody, it is [w,x,y,z] and in scipy it is [x,y,z,w].
    imu_orientations_quat_Simbody = imu_orientations_quat.copy()
    idx_to_flip_A = [1,2,3,0]
    imu_orientations_quat_Simbody = imu_orientations_quat[:,idx_to_flip_A]
    idxIMU = list(range(0, 4))
    headingCorrection = np.zeros((imu_orientations_quat_Simbody.shape[0], 1))
    for n in range(imu_orientations_quat_Simbody.shape[0]):    
        R_imu_orientations = R.from_quat(
            [imu_orientations_quat_Simbody[n, idxIMU]])        
        # You can already rotate the data if you like.
        # In OpenSense, they rotate first using the sensor_to_opensim_rotation
        # and then they compute the heading correction.
        R_imu_orientations_rot = R_sensor_data * R_imu_orientations        
        baseHeading = direction * (
            (R_imu_orientations_rot.as_matrix()).T)[directionOnIMU]
        # Compute the value of the angular correction
        headingCorrection[n,0] = np.arccos(
            np.matmul(baseHeading.T, ref_axis))[0,0];
        # Compute the sign of the angular correction
        xproduct = np.cross(ref_axis.T, baseHeading.T)
        if xproduct[0,rotation_direction] > 0:
            headingCorrection[n,0] = headingCorrection[n,0]*-1
            
    headingCorrection_first = headingCorrection[0,0]
    headingCorrection_mean = np.mean(headingCorrection)
    headingCorrection_std = np.std(headingCorrection)
        
    return headingCorrection_mean, headingCorrection_std, headingCorrection_first, headingCorrection   

def getBodyFixedXYZFromDataFrameR(df_R, dimensions = ['x', 'y', 'z']):
    R_data = df_R.to_numpy()  
    
    XYZ_data = np.zeros((R_data.shape[0], 3))
    for count in range(R_data.shape[0]):
        r = R.from_matrix([
            [R_data[count][1], R_data[count][2], R_data[count][3]],
            [R_data[count][4], R_data[count][5], R_data[count][6]],
            [R_data[count][7], R_data[count][8], R_data[count][9]]])
        XYZ_data_count = r.as_euler('XYZ', degrees=False)  
        XYZ_data[count,:] = XYZ_data_count.T
        
    df_XYZ = pd.DataFrame(data=df_R['time'], columns=['time'])    
    for count, dimension in enumerate(dimensions):
        df_XYZ.insert(count + 1, dimension, XYZ_data[:,count])
        
    return df_XYZ
    
    
    