import numpy as np
import os
from sys import path
import glob

def lexsort_based(data):                 
    sorted_data =  data[np.lexsort(data.T),:]
    row_mask = np.append([True],np.any(np.diff(sorted_data,axis=0),1))
    return sorted_data[row_mask]


def getInputsMA(pathMuscleAnalysis, ubounds, lbounds, joints, nNodes, 
                OpenSimDict):
    
    NJoints = len(joints)
    pathResultsMA = os.path.join(pathMuscleAnalysis, "ResultsMA", "grid_" +
                                 str(nNodes) + "nodes_" + str(NJoints) + "dim")
    if not os.path.exists(pathResultsMA):
        os.makedirs(pathResultsMA)         

    # from variousFunctions import getIK
    # pathDummyMotion = os.path.join(pathMA, 'dummy_motion.mot')
    # jointCoordinates = (getIK(pathDummyMotion, joints)[0]).to_numpy()[:,1::] 
    
    # # Get moment arms
    # from variousFunctions import getFromStorage
    # momentArms = np.zeros((jointCoordinates.shape[0], NMuscles, NJoints))
    # for i, joint in enumerate(joints):
    #     pathMomentArm = pathMuscleAnalysis + 'MomentArm_' + joint + '.sto'
    #     # getFromStorage outputs time vector as well so [:,1::]
    #     momentArms[:, :, i] = getFromStorage(
    #         pathMomentArm, muscles).to_numpy()[:,1::]
    # spanningInfo = np.sum(momentArms, axis=0)    
    # spanningInfo = np.where(np.logical_and(spanningInfo<=0.0001, 
    #                                         spanningInfo>=-0.0001), 0, 1)   
    
    # spanningInfo_unique = lexsort_based(spanningInfo) 
    
    maxima = np.zeros((NJoints,))
    minima = np.zeros((NJoints,))
    for count, joint in enumerate(joints):
        maxima[count] = ubounds[joint]*180/np.pi
        minima[count] = lbounds[joint]*180/np.pi    
    
    intervals = np.zeros((nNodes, NJoints))
    for nJoint in range(NJoints):
        intervals[:, nJoint] = np.linspace(minima[nJoint], 
                                           maxima[nJoint], nNodes)         
        
    from variousFunctions import numpy2storage
    # lmt = {}
    # lmt_all = {}
        # lmt[str(sp)] = {}
        # lmt_all[str(sp)] = {}
        
        
    # spanningInfo_sp = spanningInfo_unique[sp,:]
    # nCoords_sp = np.sum(spanningInfo_sp)
    # idx_sp = np.where(spanningInfo_sp == 1)
    
    # lmt_all[str(sp)]["a"] = np.empty((nCoords_sp,))
    # lmt_all[str(sp)]["b"] = np.empty((nCoords_sp,))
    # lmt_all[str(sp)]["n"] = np.empty((nCoords_sp,), dtype=int)
    
    nSamples = nNodes**NJoints
    angles = np.zeros((nSamples, NJoints))
    labels = ['time']
    for nJoint in range(NJoints):
        # lmt_all[str(sp)]["a"][nJoint] = minima[idx_sp[0][nJoint]]*np.pi/180
        # lmt_all[str(sp)]["b"][nJoint] = maxima[idx_sp[0][nJoint]]*np.pi/180
        # lmt_all[str(sp)]["n"][nJoint] = nNodes
        
        anglest = np.empty([intervals.shape[0],0])
        for count in range(intervals.shape[0]) :
            anglest = np.append(anglest, 
                                np.tile(intervals[count,nJoint], 
                                        nNodes**(NJoints-(NJoints-nJoint))))
        angles[:, nJoint] = np.tile(anglest, nNodes**(NJoints-(1+nJoint)))
        labels += [joints[nJoint]] 
        
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
        filename = 'angles_' + str(nChunk) + '.mot'    
        pathFilename = os.path.join(pathResultsMA, filename)
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
    filename = 'angles_' + str(nChunk) + '.mot'    
    pathFilename = os.path.join(pathResultsMA, filename)
    chunkData[str(nChunk)]["pathFilename"] = pathFilename
    chunkData[str(nChunk)]["filename"] = filename
    numpy2storage(labels, data, pathFilename)    
    
    # Generate dict to run MA in parallel.
    inputs_MA = {}        
    pathGenericSetupFile = os.path.join(pathMuscleAnalysis, 
                                           'SetupMA_generic.xml')
    for i in chunkData:
        inputs_MA[i] = {}
        inputs_MA[i]["pathOS"] = OpenSimDict["pathOS"]
        inputs_MA[i]["pathOpenSimModel"] = OpenSimDict["pathOpenSimModel"]
        inputs_MA[i]["pathGenericSetupFile"] = pathGenericSetupFile
        inputs_MA[i]["setStartTime"] = chunkData[i]["time"][0]
        inputs_MA[i]["setFinalTime"] = chunkData[i]["time"][-1]
        inputs_MA[i]["pathResultsSplinesCase"] = os.path.join(
            pathResultsMA, chunkData[i]["filename"][:-4])
        inputs_MA[i]["setCoordinatesFileName"] = chunkData[i]["pathFilename"]
        inputs_MA[i]["pathSetup"] = chunkData[i]["pathFilename"][:-4] + ".xml"
        inputs_MA[i]["clean1"] = (inputs_MA[i]["pathResultsSplinesCase"] + 
                                  '\subject01_MuscleAnalysis_Length.')
        inputs_MA[i]["clean2"] = (inputs_MA[i]["pathResultsSplinesCase"] + 
                                  '\subject01_MuscleAnalysis_MomentArm')
        inputs_MA[i]["clean3"] = (inputs_MA[i]["pathResultsSplinesCase"] + 
                                  '\subject01_Kinematics_q')
        inputs_MA[i]["pathResultsMA"] = pathResultsMA
        
    return inputs_MA    
    
    # # Generate setup files for muscle analysis
    # if OpenSimDict:
    #     path.insert(0, OpenSimDict["pathOS"])
    #     import opensim
    #     pathOpenSimModel = OpenSimDict["pathOpenSimModel"]
    #     # opensim.LoadOpenSimLibrary('ScapulothoracicJointPlugin40.dll')
    #     pathGenericSetupFile = os.path.join(pathMuscleAnalysis, 
    #                                         'SetupMA_generic.xml')
    #     ATool = opensim.AnalyzeTool(pathGenericSetupFile);
    #     ATool.setModelFilename(pathOpenSimModel)     
    #     # lmt[str(sp)]
    #     for i in chunkData:
    #         ATool.setStartTime(chunkData[i]["time"][0])
    #         ATool.setFinalTime(chunkData[i]["time"][-1])                
    #         pathResultsSplinesCase = os.path.join(
    #             pathResultsMA, chunkData[i]["filename"][:-4])
    #         if not os.path.exists(pathResultsSplinesCase):
    #             os.makedirs(pathResultsSplinesCase)  
    #         ATool.setResultsDir(pathResultsSplinesCase)
    #         ATool.setCoordinatesFileName(chunkData[i]["pathFilename"])
    #         pathSetup = chunkData[i]["pathFilename"][:-4] + ".xml"
    #         ATool.printToXML(pathSetup)   
    #         # ATool.run()
    #         command = 'opensim-cmd' + ' run-tool ' + pathSetup
    #         # os.system(command)
    #         # Delete unused files (all but lengths and moment arms)
    #         for CleanUp in glob.glob(pathResultsSplinesCase + '/*.*'):
    #             if ((not CleanUp.startswith(pathResultsSplinesCase + '\subject01_MuscleAnalysis_Length.')) and 
    #                 (not CleanUp.startswith(pathResultsSplinesCase + '\subject01_MuscleAnalysis_MomentArm'))):    
    #                 os.remove(CleanUp)
                    
def MA_parallel(inputs_MA):  
    path.insert(0, inputs_MA["pathOS"])
    import opensim
    ATool = opensim.AnalyzeTool(inputs_MA["pathGenericSetupFile"]);
    ATool.setModelFilename(inputs_MA["pathOpenSimModel"])     
    ATool.setStartTime(inputs_MA["setStartTime"])
    ATool.setFinalTime(inputs_MA["setFinalTime"])   
    if not os.path.exists(inputs_MA["pathResultsSplinesCase"]):
        os.makedirs(inputs_MA["pathResultsSplinesCase"])  
    ATool.setResultsDir(inputs_MA["pathResultsSplinesCase"])
    ATool.setCoordinatesFileName(inputs_MA["setCoordinatesFileName"])
    ATool.printToXML(inputs_MA["pathSetup"])   
    # # ATool.run()
    command = 'opensim-cmd' + ' run-tool ' + inputs_MA["pathSetup"]
    os.system(command)
    # Delete unused files (all but lengths and moment arms)
    for CleanUp in glob.glob(inputs_MA["pathResultsSplinesCase"] + '/*.*'):
        if ((not CleanUp.startswith(inputs_MA["clean1"])) and 
            (not CleanUp.startswith(inputs_MA["clean2"])) and
            (not CleanUp.startswith(inputs_MA["clean3"]))):    
            os.remove(CleanUp)
     
    # # Get MT-length
    # for i in chunkData:
    #     # Get MT-length
    #     # Only select the relevant muscles, ie the muscles of that
    #     # spanning set.             
    #     pathResultsSplinesCase = os.path.join(
    #             pathResultsMA, chunkData[i]["filename"][:-4])
    #     pathLMT = os.path.join(pathResultsSplinesCase, 
    #                            'subject01_MuscleAnalysis_Length.sto')                 
    #     idx_muscles_sp = np.where(
    #         (spanningInfo == spanningInfo_sp).all(axis=1))
    #     muscles_sp = []
    #     for m in range(idx_muscles_sp[0].shape[0]):
    #         muscles_sp += [muscles[idx_muscles_sp[0][m]]]                
    #     lmt[str(sp)][i] = getFromStorage(
    #         pathLMT, muscles_sp).to_numpy()[:,1::]
        
    # # Reconstruct full matrix (multiple chunks)
    # lmt_all[str(sp)]["muscles"] = muscles_sp
    # lmt_all[str(sp)]["data"] = lmt[str(sp)]["0"]
    # if nChunks > 0:
    #     for i in range(1,nChunks+1):
    #         lmt_all[str(sp)]["data"] = np.concatenate(
    #             (lmt_all[str(sp)]["data"], lmt[str(sp)][str(i)]),
    #             axis=0)
               
                
    # return lmt_all
    # '''
    
def getTrainingData(inputs_MA, muscles):
    from variousFunctions import getFromStorage
    lmt = {}
    for key in inputs_MA:
        # Get MT-length        
        pathLMT = os.path.join(inputs_MA[key]["pathResultsSplinesCase"], 
                               'subject01_MuscleAnalysis_Length.sto')
        lmt[key] = getFromStorage(pathLMT, muscles).to_numpy()             
        
        
    # Reconstruct full matrix (multiple chunks)
    lmt_all = {}
    lmt_all["labels"] = ["time"] + muscles
    lmt_all["data"] = lmt["0"]
    if len(inputs_MA) > 1:
        for key in inputs_MA:
            if not key == "0":
                lmt_all["data"] = np.concatenate(
                    (lmt_all["data"], lmt[key]), axis=0)
    from variousFunctions import numpy2storage    

    pathResultsFolder = os.path.join(inputs_MA["0"]["pathResultsMA"], "all")
    pathResultsLMTAll = os.path.join(pathResultsFolder, 
                                     "MuscleAnalysis_Length.sto")
    if not os.path.exists(pathResultsFolder):
        os.makedirs(pathResultsFolder) 
    
    numpy2storage(lmt_all["labels"], lmt_all["data"], pathResultsLMTAll)
                    