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

def getTendonCompliance(NSideMuscles):
    tendonCompliance = np.full((1, NSideMuscles), 35)
    
    return tendonCompliance

def getTendonShift(NSideMuscles):
    tendonShift = np.full((1, NSideMuscles), 0)
    
    return tendonShift

def getSpecificTension(NSideMuscles):
    specificTension = np.full((1, NSideMuscles), 0.75)
    
    return specificTension