from sys import path
import os
import numpy as np

def getMTParameters(pathOS, pathModel, muscles, loadMTParameters,
                    pathMTParameters=0, modelName=''):
    
    if loadMTParameters:        
        mtParameters = np.load(os.path.join(pathMTParameters, 
                                            modelName + '_mtParameters.npy'), 
                               allow_pickle=True)     
        
    else:   
        path.insert(0, pathOS)
        import opensim
        model = opensim.Model(pathModel)
        mtParameters = np.zeros([5,len(muscles)])
        model_muscles = model.getMuscles()
        for i in range(len(muscles)):
           muscle = model_muscles.get(muscles[i])
           mtParameters[0,i] = muscle.getMaxIsometricForce()
           mtParameters[1,i] = muscle.getOptimalFiberLength()
           mtParameters[2,i] = muscle.getTendonSlackLength()
           mtParameters[3,i] = muscle.getPennationAngleAtOptimalFiberLength()
           mtParameters[4,i] = muscle.getMaxContractionVelocity()*muscle.getOptimalFiberLength()
        if pathMTParameters != 0:
            np.save(os.path.join(pathMTParameters, modelName + '_mtParameters.npy'),
                   mtParameters)
       
    return mtParameters  

def getPolynomialData(loadPolynomialData, pathPolynomialData, modelName='', 
                      pathCoordinates='', pathMuscleAnalysis='', joints=[],
                      muscles=[]):
    
    if loadPolynomialData:
        polynomialData = np.load(os.path.join(pathPolynomialData, 
                                              modelName + '_polynomialData.npy'), 
                                 allow_pickle=True) 
        
    else:       
        from polynomials import getPolynomialCoefficients
        polynomialData = getPolynomialCoefficients(pathCoordinates, pathMuscleAnalysis, joints, muscles)
        if pathPolynomialData != 0:
            np.save(os.path.join(pathPolynomialData, modelName + 
                                 '_polynomialData.npy'), polynomialData)
           
    return polynomialData

def getSpanningInfo(pathCoordinates, pathMuscleAnalysis, joints, muscles):
    
    # Get joint coordinates.
    from variousFunctions import getIK
    jointCoordinates = (getIK(pathCoordinates, joints)[0]).to_numpy()[:,1::]
    
    # Get muscle-tendon lengths
    from variousFunctions import getFromStorage
    
    # Get moment arms
    momentArms = np.zeros((jointCoordinates.shape[0], len(muscles), len(joints)))
    for i, joint in enumerate(joints):
        pathMomentArm = pathMuscleAnalysis + 'MomentArm_' + joint + '.sto'
        # getFromStorage outputs time vector as well so [:,1::]
        momentArms[:, :, i] = getFromStorage(pathMomentArm, muscles).to_numpy()[:,1::] 
    # Detect which muscles actuate which joints (moment arm different than [-0.0001:0.0001]) 
    spanningInfo = np.sum(momentArms, axis=0)    
    spanningInfo = np.where(np.logical_and(spanningInfo<=0.0001, spanningInfo>=-0.0001), 0, 1)
    
    idxSpanningJoints = {}
    for c, joint in enumerate(joints):
        idxSpanningJoints[joint] = np.where(spanningInfo[:,c] == 1)[0]
    
    return idxSpanningJoints

def getTendonCompliance(NSideMuscles):
    tendonCompliance = np.full((1, NSideMuscles), 35)
    
    return tendonCompliance

def getTendonShift(NSideMuscles):
    tendonShift = np.full((1, NSideMuscles), 0)
    
    return tendonShift

def getSpecificTension(NSideMuscles):
    specificTension = np.full((1, NSideMuscles), 0.75)
    
    return specificTension
