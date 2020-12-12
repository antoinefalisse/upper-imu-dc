import numpy as np
import os
from sys import path

def lexsort_based(data):                 
    sorted_data =  data[np.lexsort(data.T),:]
    row_mask = np.append([True],np.any(np.diff(sorted_data,axis=0),1))
    return sorted_data[row_mask]


def generateAnglesInFiles(pathMA, pathMuscleAnalysis, joints, muscles, nNodes,
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
    for sp in range(spanningInfo_unique.shape[0]):
        lmt[str(sp)] = {}
        lmt_all[str(sp)] = {}
        
        spanningInfo_sp = spanningInfo_unique[sp,:]
        nCoords_sp = np.sum(spanningInfo_sp)
        idx_sp = np.where(spanningInfo_sp == 1)
        
        nSamples = nNodes**nCoords_sp
        angles = np.zeros((nSamples, nCoords_sp))
        labels = ['time']
        for nCoord in range(nCoords_sp):
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
        
        # # Generate setup files for muscle analysis
        if OpenSimDict:
            path.insert(0, OpenSimDict["pathOS"])
            import opensim
            pathOpenSimModel = OpenSimDict["pathOpenSimModel"];
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
                
                # Get MT-length
                # Only select the relevant muscles, ie the muscles of that
                # spanning set.                
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
                    