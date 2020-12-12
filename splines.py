import numpy as np
import os
from sys import path

def lexsort_based(data):                 
    sorted_data =  data[np.lexsort(data.T),:]
    row_mask = np.append([True],np.any(np.diff(sorted_data,axis=0),1))
    return sorted_data[row_mask]


def getTrainingLMT(pathMA, pathMuscleAnalysis, joints, muscles, nNodes,
                   OpenSimDict={}):
    
    pathResultsSplines = os.path.join(pathMA, "MA_Splines")
    if not os.path.exists(pathResultsSplines):
        os.makedirs(pathResultsSplines)       
    
    nCoords = len(joints)
    nMuscles = len(muscles)

    from variousFunctions import getIK
    pathDummyMotion = os.path.join(pathMA, 'dummy_motion.mot')
    jointCoordinates = (getIK(pathDummyMotion, joints)[0]).to_numpy()[:,1::] 
    
    # Get moment arms
    from variousFunctions import getFromStorage
    momentArms = np.zeros((jointCoordinates.shape[0], nMuscles, nCoords))
    for i, joint in enumerate(joints):
        pathMomentArm = pathMuscleAnalysis + 'MomentArm_' + joint + '.sto'
        # getFromStorage outputs time vector as well so [:,1::]
        momentArms[:, :, i] = getFromStorage(
            pathMomentArm, muscles).to_numpy()[:,1::]
    spanningInfo = np.sum(momentArms, axis=0)    
    spanningInfo = np.where(np.logical_and(spanningInfo<=0.0001, 
                                            spanningInfo>=-0.0001), 0, 1)   
    
    spanningInfo_unique = lexsort_based(spanningInfo) 
    
    maxima = np.max(jointCoordinates, axis=0)*180/np.pi
    minima = np.min(jointCoordinates, axis=0)*180/np.pi
    
    intervals = np.zeros((nNodes, nCoords))
    for nCoord in range(nCoords):
        intervals[:, nCoord] = np.linspace(minima[nCoord], 
                                            maxima[nCoord], nNodes)  
        
    from variousFunctions import numpy2storage
    lmt = {}
    lmt_all = {}
    for sp in range(spanningInfo_unique.shape[0]-2):
        lmt[str(sp)] = {}
        lmt_all[str(sp)] = {}
        
        
        spanningInfo_sp = spanningInfo_unique[sp,:]
        nCoords_sp = np.sum(spanningInfo_sp)
        idx_sp = np.where(spanningInfo_sp == 1)
        
        lmt_all[str(sp)]["a"] = np.empty((nCoords_sp,))
        lmt_all[str(sp)]["b"] = np.empty((nCoords_sp,))
        lmt_all[str(sp)]["n"] = np.empty((nCoords_sp,), dtype=int)
        
        nSamples = nNodes**nCoords_sp
        angles = np.zeros((nSamples, nCoords_sp))
        labels = ['time']
        for nCoord in range(nCoords_sp):
            lmt_all[str(sp)]["a"][nCoord] = minima[idx_sp[0][nCoord]]*np.pi/180
            lmt_all[str(sp)]["b"][nCoord] = maxima[idx_sp[0][nCoord]]*np.pi/180
            lmt_all[str(sp)]["n"][nCoord] = nNodes
            
            anglest = np.empty([intervals.shape[0],0])
            for count in range(intervals.shape[0]) :
                anglest = np.append(anglest, 
                                    np.tile(intervals[count,idx_sp[0][nCoord]], 
                                            nNodes**(nCoords_sp-(nCoords_sp-nCoord))))
            angles[:, nCoord] = np.tile(anglest, nNodes**(nCoords_sp-(1+nCoord)))
            labels += [joints[idx_sp[0][nCoord]]]
            
        # Write motion files for Muscle Analysis. Since `angles` might become huge
        # when the number of nodes increase, we split `angles` into shunks of
        # max 10,000 rows. That way, we can then run the Muscle Analysis in batch.
        nRowPerChunk = 10000    
        nChunks, nRem = np.divmod(nSamples, nRowPerChunk)                  
         
        nChunk = np.NaN
        chunkData = {}
        for nChunk in range(nChunks):
            chunkData[str(nChunk)] = {}            
            tgrid = np.array([np.linspace(nChunk*nRowPerChunk+1, 
                                          nChunk*nRowPerChunk+nRowPerChunk, 
                                          nRowPerChunk)])/100  
            chunkData[str(nChunk)]["time"] = [tgrid[0][0], tgrid[0][-1]]
            data = np.concatenate((tgrid.T, 
                                    angles[nChunk*nRowPerChunk:
                                          nChunk*nRowPerChunk+nRowPerChunk,:]),
                                  axis=1)
            filename = 'angle_in_n' + str(nNodes) + '_dof' + str(nCoords_sp) + '_' + str(nChunk) + '.mot'    
            pathFilename = os.path.join(pathResultsSplines, filename)
            chunkData[str(nChunk)]["pathFilename"] = pathFilename
            chunkData[str(nChunk)]["filename"] = filename
            numpy2storage(labels, data, pathFilename)
        if np.isnan(nChunk):
            nChunk = 0
        else:
            nChunk += 1
        chunkData[str(nChunk)] = {} 
        tgrid = np.array([np.linspace(nChunk*nRowPerChunk+1, 
                                          nChunk*nRowPerChunk+nRem, 
                                          nRem)])/100    
        chunkData[str(nChunk)]["time"] = [tgrid[0][0], tgrid[0][-1]]
        data = np.concatenate((tgrid.T, 
                                angles[nChunk*nRowPerChunk:
                                      nChunk*nRowPerChunk+nRem,:]),
                              axis=1)
        filename = 'angle_in_n' + str(nNodes) + '_dof' + str(nCoords_sp) + '_' + str(nChunk) + '.mot'    
        pathFilename = os.path.join(pathResultsSplines, filename)
        chunkData[str(nChunk)]["pathFilename"] = pathFilename
        chunkData[str(nChunk)]["filename"] = filename
        numpy2storage(labels, data, pathFilename)    
        
        # Generate setup files for muscle analysis
        if OpenSimDict:
            path.insert(0, OpenSimDict["pathOS"])
            import opensim
            pathOpenSimModel = OpenSimDict["pathOpenSimModel"];
            opensim.LoadOpenSimLibrary('ScapulothoracicJointPlugin40.dll')
            genericSetupFile = os.path.join(pathMA, 'SetupMA_lmt.xml')
            ATool = opensim.AnalyzeTool(genericSetupFile);
            ATool.setModelFilename(pathOpenSimModel)     
            lmt[str(sp)]
            for i in chunkData:
                ATool.setStartTime(chunkData[i]["time"][0])
                ATool.setFinalTime(chunkData[i]["time"][-1])                
                pathResultsSplinesCase = os.path.join(
                    pathResultsSplines, chunkData[i]["filename"][:-4])
                if not os.path.exists(pathResultsSplinesCase):
                    os.makedirs(pathResultsSplinesCase)  
                ATool.setResultsDir(pathResultsSplinesCase)
                ATool.setCoordinatesFileName(chunkData[i]["pathFilename"])
                ATool.printToXML(chunkData[i]["pathFilename"][:-4] + ".xml")   
                # TODO run tool - make things work with joint
                ATool.run()
                
         
        # Get MT-length
        for i in chunkData:
            # Get MT-length
            # Only select the relevant muscles, ie the muscles of that
            # spanning set.             
            pathResultsSplinesCase = os.path.join(
                    pathResultsSplines, chunkData[i]["filename"][:-4])
            pathLMT = os.path.join(pathResultsSplinesCase, 
                                   'subject01_MuscleAnalysis_Length.sto')                 
            idx_muscles_sp = np.where(
                (spanningInfo == spanningInfo_sp).all(axis=1))
            muscles_sp = []
            for m in range(idx_muscles_sp[0].shape[0]):
                muscles_sp += [muscles[idx_muscles_sp[0][m]]]                
            lmt[str(sp)][i] = getFromStorage(
                pathLMT, muscles_sp).to_numpy()[:,1::]
            
        # Reconstruct full matrix (multiple chunks)
        lmt_all[str(sp)]["muscles"] = muscles_sp
        lmt_all[str(sp)]["data"] = lmt[str(sp)]["0"]
        if nChunks > 0:
            for i in range(1,nChunks+1):
                lmt_all[str(sp)]["data"] = np.concatenate(
                    (lmt_all[str(sp)]["data"], lmt[str(sp)][str(i)]),
                    axis=0)
                    
    return lmt_all

def getCoeffs(pathLib, trainingLMT):
    
    nCoords = len(trainingLMT["a"])
    if nCoords == 2:
        C = getCoeffs_2(pathLib, trainingLMT["data"], trainingLMT["a"],
                        trainingLMT["b"], trainingLMT["n"])
    elif nCoords == 3:
        C = getCoeffs_3(pathLib, trainingLMT["data"], trainingLMT["a"],
                        trainingLMT["b"], trainingLMT["n"])
    elif nCoords == 4:
        C = getCoeffs_4(pathLib, trainingLMT["data"], trainingLMT["a"],
                        trainingLMT["b"], trainingLMT["n"])
    elif nCoords == 5:
        C = getCoeffs_5(pathLib, trainingLMT["data"], trainingLMT["a"],
                        trainingLMT["b"], trainingLMT["n"])
    elif nCoords == 6:
        C = getCoeffs_6(pathLib, trainingLMT["data"], trainingLMT["a"],
                        trainingLMT["b"], trainingLMT["n"])
    elif nCoords == 7:
        C = getCoeffs_7(pathLib, trainingLMT["data"], trainingLMT["a"],
                        trainingLMT["b"], trainingLMT["n"])
    elif nCoords == 8:
        C = getCoeffs_8(pathLib, trainingLMT["data"], trainingLMT["a"],
                        trainingLMT["b"], trainingLMT["n"])
    elif nCoords == 9:
        C = getCoeffs_9(pathLib, trainingLMT["data"], trainingLMT["a"],
                        trainingLMT["b"], trainingLMT["n"])
        
    return C

from ctypes import CDLL, c_int
def getCoeffs_2(pathLib, trainingData, a, b, n):
    """Returns the spline coefficients in a numpy format.
    Parameters
    ----------
    pathLib : str
        Path to createSpline.dll.
    trainingData : numpy array of float64
        Muscle-tendon lengths used for training (nSamples x nMuscles).
    a : numpy array of float64
        Minima of range of motion used to generate muscle-tendon lengths 
        (nCoordinates,).
    b : numpy array of float64
        Maxima of range of motion used to generate muscle-tendon lengths 
        (nCoordinates,).
    n : int
        Number of intervals (nNodes - 1)
    Returns
    -------
    coeffs : numpy array of float64
        Spline coefficients (nCoefficients x nMuscles)
    
    """    
    mylib = CDLL(pathLib)

    ND_POINTER_1 = np.ctypeslib.ndpointer(dtype=np.float64, ndim=1)
    ND_POINTER_2 = np.ctypeslib.ndpointer(dtype=np.int32, ndim=1)
    
    mylib.createSpline_2.argtypes = [ND_POINTER_1, ND_POINTER_1, ND_POINTER_1,
                                     ND_POINTER_2, c_int, c_int, ND_POINTER_1]
    
    if len(trainingData.shape) == 1:
        nMuscles = 1
    else:       
        nMuscles = trainingData.shape[1]
    nSamples = trainingData.shape[0]
    
    numberOfPreCoeffs = 1;
    nDOFs = a.shape[0]
    for i in range(nDOFs):
        numberOfPreCoeffs *= ((n[i])+3)
    
    coeff_flat = np.zeros((numberOfPreCoeffs*nMuscles))
    mylib.createSpline_2(trainingData.flatten(), a, b, n, nMuscles, nSamples,
                         coeff_flat)
    coeffs = np.reshape(coeff_flat, (numberOfPreCoeffs, nMuscles))
    
    C = {}
    C = dict(coefficients=coeffs, minima_ROM=a, maxima_ROM=b,
             number_intervals= n)
    
    return C

def getCoeffs_3(pathLib, trainingData, a, b, n):
    """Returns the spline coefficients in a numpy format.
    Parameters
    ----------
    pathLib : str
        Path to createSpline.dll.
    trainingData : numpy array of float64
        Muscle-tendon lengths used for training (nSamples x nMuscles).
    a : numpy array of float64
        Minima of range of motion used to generate muscle-tendon lengths 
        (nCoordinates,).
    b : numpy array of float64
        Maxima of range of motion used to generate muscle-tendon lengths 
        (nCoordinates,).
    n : int
        Number of intervals (nNodes - 1)
    Returns
    -------
    coeffs : numpy array of float64
        Spline coefficients (nCoefficients x nMuscles)
    
    """    
    mylib = CDLL(pathLib)

    ND_POINTER_1 = np.ctypeslib.ndpointer(dtype=np.float64, ndim=1)
    ND_POINTER_2 = np.ctypeslib.ndpointer(dtype=np.int32, ndim=1)
    
    mylib.createSpline_3.argtypes = [ND_POINTER_1, ND_POINTER_1, ND_POINTER_1,
                                     ND_POINTER_2, c_int, c_int, ND_POINTER_1]
    
    if len(trainingData.shape) == 1:
        nMuscles = 1
    else:       
        nMuscles = trainingData.shape[1]
    nSamples = trainingData.shape[0]
    
    numberOfPreCoeffs = 1;
    nDOFs = a.shape[0]
    for i in range(nDOFs):
        numberOfPreCoeffs *= ((n[i])+3)
    
    coeff_flat = np.zeros((numberOfPreCoeffs*nMuscles))
    mylib.createSpline_3(trainingData.flatten(), a, b, n, nMuscles, nSamples,
                         coeff_flat)
    coeffs = np.reshape(coeff_flat, (numberOfPreCoeffs, nMuscles))
    
    C = {}
    C = dict(coefficients=coeffs, minima_ROM=a, maxima_ROM=b,
             number_intervals= n)
    
    return C

def getCoeffs_4(pathLib, trainingData, a, b, n):
    """Returns the spline coefficients in a numpy format.
    Parameters
    ----------
    pathLib : str
        Path to createSpline.dll.
    trainingData : numpy array of float64
        Muscle-tendon lengths used for training (nSamples x nMuscles).
    a : numpy array of float64
        Minima of range of motion used to generate muscle-tendon lengths 
        (nCoordinates,).
    b : numpy array of float64
        Maxima of range of motion used to generate muscle-tendon lengths 
        (nCoordinates,).
    n : int
        Number of intervals (nNodes - 1)
    Returns
    -------
    coeffs : numpy array of float64
        Spline coefficients (nCoefficients x nMuscles)
    
    """    
    mylib = CDLL(pathLib)

    ND_POINTER_1 = np.ctypeslib.ndpointer(dtype=np.float64, ndim=1)
    ND_POINTER_2 = np.ctypeslib.ndpointer(dtype=np.int32, ndim=1)
    
    mylib.createSpline_4.argtypes = [ND_POINTER_1, ND_POINTER_1, ND_POINTER_1,
                                     ND_POINTER_2, c_int, c_int, ND_POINTER_1]
    
    if len(trainingData.shape) == 1:
        nMuscles = 1
    else:       
        nMuscles = trainingData.shape[1]
    nSamples = trainingData.shape[0]
    
    numberOfPreCoeffs = 1;
    nDOFs = a.shape[0]
    for i in range(nDOFs):
        numberOfPreCoeffs *= ((n[i])+3)
    
    coeff_flat = np.zeros((numberOfPreCoeffs*nMuscles))
    mylib.createSpline_4(trainingData.flatten(), a, b, n, nMuscles, nSamples,
                         coeff_flat)
    coeffs = np.reshape(coeff_flat, (numberOfPreCoeffs, nMuscles))
    
    C = {}
    C = dict(coefficients=coeffs, minima_ROM=a, maxima_ROM=b,
             number_intervals= n)
    
    return C

def getCoeffs_5(pathLib, trainingData, a, b, n):
    """Returns the spline coefficients in a numpy format.
    Parameters
    ----------
    pathLib : str
        Path to createSpline.dll.
    trainingData : numpy array of float64
        Muscle-tendon lengths used for training (nSamples x nMuscles).
    a : numpy array of float64
        Minima of range of motion used to generate muscle-tendon lengths 
        (nCoordinates,).
    b : numpy array of float64
        Maxima of range of motion used to generate muscle-tendon lengths 
        (nCoordinates,).
    n : int
        Number of intervals (nNodes - 1)
    Returns
    -------
    coeffs : numpy array of float64
        Spline coefficients (nCoefficients x nMuscles)
    
    """    
    mylib = CDLL(pathLib)

    ND_POINTER_1 = np.ctypeslib.ndpointer(dtype=np.float64, ndim=1)
    ND_POINTER_2 = np.ctypeslib.ndpointer(dtype=np.int32, ndim=1)
    
    mylib.createSpline_5.argtypes = [ND_POINTER_1, ND_POINTER_1, ND_POINTER_1,
                                     ND_POINTER_2, c_int, c_int, ND_POINTER_1]
    
    if len(trainingData.shape) == 1:
        nMuscles = 1
    else:       
        nMuscles = trainingData.shape[1]
    nSamples = trainingData.shape[0]
    
    numberOfPreCoeffs = 1;
    nDOFs = a.shape[0]
    for i in range(nDOFs):
        numberOfPreCoeffs *= ((n[i])+3)
    
    coeff_flat = np.zeros((numberOfPreCoeffs*nMuscles))
    mylib.createSpline_5(trainingData.flatten(), a, b, n, nMuscles, nSamples,
                         coeff_flat)
    coeffs = np.reshape(coeff_flat, (numberOfPreCoeffs, nMuscles))
    
    C = {}
    C = dict(coefficients=coeffs, minima_ROM=a, maxima_ROM=b,
             number_intervals= n)
    
    return C

def getCoeffs_6(pathLib, trainingData, a, b, n):
    """Returns the spline coefficients in a numpy format.
    Parameters
    ----------
    pathLib : str
        Path to createSpline.dll.
    trainingData : numpy array of float64
        Muscle-tendon lengths used for training (nSamples x nMuscles).
    a : numpy array of float64
        Minima of range of motion used to generate muscle-tendon lengths 
        (nCoordinates,).
    b : numpy array of float64
        Maxima of range of motion used to generate muscle-tendon lengths 
        (nCoordinates,).
    n : int
        Number of intervals (nNodes - 1)
    Returns
    -------
    coeffs : numpy array of float64
        Spline coefficients (nCoefficients x nMuscles)
    
    """    
    mylib = CDLL(pathLib)

    ND_POINTER_1 = np.ctypeslib.ndpointer(dtype=np.float64, ndim=1)
    ND_POINTER_2 = np.ctypeslib.ndpointer(dtype=np.int32, ndim=1)
    
    mylib.createSpline_6.argtypes = [ND_POINTER_1, ND_POINTER_1, ND_POINTER_1,
                                     ND_POINTER_2, c_int, c_int, ND_POINTER_1]
    
    if len(trainingData.shape) == 1:
        nMuscles = 1
    else:       
        nMuscles = trainingData.shape[1]
    nSamples = trainingData.shape[0]
    
    numberOfPreCoeffs = 1;
    nDOFs = a.shape[0]
    for i in range(nDOFs):
        numberOfPreCoeffs *= ((n[i])+3)
    
    coeff_flat = np.zeros((numberOfPreCoeffs*nMuscles))
    mylib.createSpline_6(trainingData.flatten(), a, b, n, nMuscles, nSamples,
                         coeff_flat)
    coeffs = np.reshape(coeff_flat, (numberOfPreCoeffs, nMuscles))
    
    C = {}
    C = dict(coefficients=coeffs, minima_ROM=a, maxima_ROM=b,
             number_intervals= n)
    
    return C

def getCoeffs_7(pathLib, trainingData, a, b, n):
    """Returns the spline coefficients in a numpy format.
    Parameters
    ----------
    pathLib : str
        Path to createSpline.dll.
    trainingData : numpy array of float64
        Muscle-tendon lengths used for training (nSamples x nMuscles).
    a : numpy array of float64
        Minima of range of motion used to generate muscle-tendon lengths 
        (nCoordinates,).
    b : numpy array of float64
        Maxima of range of motion used to generate muscle-tendon lengths 
        (nCoordinates,).
    n : int
        Number of intervals (nNodes - 1)
    Returns
    -------
    coeffs : numpy array of float64
        Spline coefficients (nCoefficients x nMuscles)
    
    """    
    mylib = CDLL(pathLib)

    ND_POINTER_1 = np.ctypeslib.ndpointer(dtype=np.float64, ndim=1)
    ND_POINTER_2 = np.ctypeslib.ndpointer(dtype=np.int32, ndim=1)
    
    mylib.createSpline_7.argtypes = [ND_POINTER_1, ND_POINTER_1, ND_POINTER_1,
                                     ND_POINTER_2, c_int, c_int, ND_POINTER_1]
    
    if len(trainingData.shape) == 1:
        nMuscles = 1
    else:       
        nMuscles = trainingData.shape[1]
    nSamples = trainingData.shape[0]
    
    numberOfPreCoeffs = 1;
    nDOFs = a.shape[0]
    for i in range(nDOFs):
        numberOfPreCoeffs *= ((n[i])+3)
    
    coeff_flat = np.zeros((numberOfPreCoeffs*nMuscles))
    mylib.createSpline_7(trainingData.flatten(), a, b, n, nMuscles, nSamples,
                         coeff_flat)
    coeffs = np.reshape(coeff_flat, (numberOfPreCoeffs, nMuscles))
    
    C = {}
    C = dict(coefficients=coeffs, minima_ROM=a, maxima_ROM=b,
             number_intervals= n)
    
    return C

def getCoeffs_8(pathLib, trainingData, a, b, n):
    """Returns the spline coefficients in a numpy format.
    Parameters
    ----------
    pathLib : str
        Path to createSpline.dll.
    trainingData : numpy array of float64
        Muscle-tendon lengths used for training (nSamples x nMuscles).
    a : numpy array of float64
        Minima of range of motion used to generate muscle-tendon lengths 
        (nCoordinates,).
    b : numpy array of float64
        Maxima of range of motion used to generate muscle-tendon lengths 
        (nCoordinates,).
    n : int
        Number of intervals (nNodes - 1)
    Returns
    -------
    coeffs : numpy array of float64
        Spline coefficients (nCoefficients x nMuscles)
    
    """    
    mylib = CDLL(pathLib)

    ND_POINTER_1 = np.ctypeslib.ndpointer(dtype=np.float64, ndim=1)
    ND_POINTER_2 = np.ctypeslib.ndpointer(dtype=np.int32, ndim=1)
    
    mylib.createSpline_8.argtypes = [ND_POINTER_1, ND_POINTER_1, ND_POINTER_1,
                                     ND_POINTER_2, c_int, c_int, ND_POINTER_1]
    
    if len(trainingData.shape) == 1:
        nMuscles = 1
    else:       
        nMuscles = trainingData.shape[1]
    nSamples = trainingData.shape[0]
    
    numberOfPreCoeffs = 1;
    nDOFs = a.shape[0]
    for i in range(nDOFs):
        numberOfPreCoeffs *= ((n[i])+3)
    
    coeff_flat = np.zeros((numberOfPreCoeffs*nMuscles))
    mylib.createSpline_8(trainingData.flatten(), a, b, n, nMuscles, nSamples,
                         coeff_flat)
    coeffs = np.reshape(coeff_flat, (numberOfPreCoeffs, nMuscles))
    
    C = {}
    C = dict(coefficients=coeffs, minima_ROM=a, maxima_ROM=b,
             number_intervals= n)
    
    return C

def getCoeffs_9(pathLib, trainingData, a, b, n):
    """Returns the spline coefficients in a numpy format.
    Parameters
    ----------
    pathLib : str
        Path to createSpline.dll.
    trainingData : numpy array of float64
        Muscle-tendon lengths used for training (nSamples x nMuscles).
    a : numpy array of float64
        Minima of range of motion used to generate muscle-tendon lengths 
        (nCoordinates,).
    b : numpy array of float64
        Maxima of range of motion used to generate muscle-tendon lengths 
        (nCoordinates,).
    n : int
        Number of intervals (nNodes - 1)
    Returns
    -------
    coeffs : numpy array of float64
        Spline coefficients (nCoefficients x nMuscles)
    
    """    
    mylib = CDLL(pathLib)

    ND_POINTER_1 = np.ctypeslib.ndpointer(dtype=np.float64, ndim=1)
    ND_POINTER_2 = np.ctypeslib.ndpointer(dtype=np.int32, ndim=1)
    
    mylib.createSpline_9.argtypes = [ND_POINTER_1, ND_POINTER_1, ND_POINTER_1,
                                     ND_POINTER_2, c_int, c_int, ND_POINTER_1]
    
    if len(trainingData.shape) == 1:
        nMuscles = 1
    else:       
        nMuscles = trainingData.shape[1]
    nSamples = trainingData.shape[0]
    
    numberOfPreCoeffs = 1;
    nDOFs = a.shape[0]
    for i in range(nDOFs):
        numberOfPreCoeffs *= ((n[i])+3)
    
    coeff_flat = np.zeros((numberOfPreCoeffs*nMuscles))
    mylib.createSpline_9(trainingData.flatten(), a, b, n, nMuscles, nSamples,
                         coeff_flat)
    coeffs = np.reshape(coeff_flat, (numberOfPreCoeffs, nMuscles))
    
    C = {}
    C = dict(coefficients=coeffs, minima_ROM=a, maxima_ROM=b,
             number_intervals= n)
    
    return C
                    