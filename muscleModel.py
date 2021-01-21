import numpy as np

class muscleModel:
    
    def __init__(self, mtParameters, activation, mtLength, mtVelocity,
                 normTendonForce, normTendonForceDT, tendonCompliance,
                 tendonShift, specificTension):
        self.mtParameters = mtParameters
        
        self.maximalIsometricForce = mtParameters[0]
        self.optimalFiberLength = mtParameters[1]          
        self.tendonSlackLength = mtParameters[2]
        self.optimalPennationAngle = mtParameters[3]        
        self.maximalFiberVelocity = mtParameters[4]
        
        self.activation = activation
        self.mtLength = mtLength
        self.mtVelocity = mtVelocity
        self.normTendonForce = normTendonForce
        self.normTendonForceDT = normTendonForceDT
        self.tendonCompliance = tendonCompliance
        self.tendonShift = tendonShift
        self.specificTension = specificTension
        self.paramFLa = np.array([0.814483478343008, 1.05503342897057,
                                  0.162384573599574, 0.0633034484654646,
                                  0.433004984392647, 0.716775413397760, 
                                  -0.0299471169706956, 0.200356847296188])
        self.paramFLp = np.array([-0.995172050006169, 53.5981500331442])
        self.paramFV = np.array([-0.318323436899127, -8.14915604347525,
                                 -0.374121508647863, 0.885644059915004])
    
    def getMuscleVolume(self):
        self.muscleVolume = np.multiply(self.maximalIsometricForce, 
                                        self.optimalFiberLength)
        return self.muscleVolume
        
        
    def getMuscleMass(self):                
        muscleMass = np.divide(np.multiply(self.muscleVolume, 1059.7), 
                               np.multiply(self.specificTension, 1e6))
        
        return muscleMass        
        
    def getTendonForce(self):          
        tendonForce = np.multiply(self.normTendonForce, 
                                  self.maximalIsometricForce)  
        
        return tendonForce
            
    def getTendonLength(self):          
        # Tendon force-length relationship
        self.normTendonLength = np.divide(
                np.log(5*(self.normTendonForce + 0.25 - self.tendonShift)), 
                self.tendonCompliance) + 0.995                                     
        self.tendonLength = np.multiply(self.tendonSlackLength, 
                                        self.normTendonLength)
        
        return self.tendonLength, self.normTendonLength
                
    def getFiberLength(self):
        # Hill-type muscle model: geometric relationships    
        self.getTendonLength()
        w = np.multiply(self.optimalFiberLength, 
                        np.sin(self.optimalPennationAngle))        
        self.fiberLength = np.sqrt(
                (self.mtLength - self.tendonLength)**2 + w**2)
        self.normFiberLength = np.divide(self.fiberLength, 
                                         self.optimalFiberLength)   

        return self.fiberLength, self.normFiberLength         
    
    def getFiberVelocity(self):            
        # Hill-type muscle model: geometric relationships 
        self.getFiberLength()
        tendonVelocity = np.divide(np.multiply(self.tendonSlackLength, 
                                               self.normTendonForceDT), 
            0.2 * self.tendonCompliance * np.exp(self.tendonCompliance * 
                                                 (self.normTendonLength - 
                                                  0.995)))        
        self.cosPennationAngle = np.divide((self.mtLength - self.tendonLength), 
                                           self.fiberLength)        
        self.fiberVelocity = np.multiply((self.mtVelocity - tendonVelocity), 
                                         self.cosPennationAngle)        
        self.normFiberVelocity = np.divide(self.fiberVelocity, 
                                           self.maximalFiberVelocity)  
        
        return self.fiberVelocity, self.normFiberVelocity 
    
    def getActiveFiberLengthForce(self):  
        self.getFiberLength()        
        # Active muscle force-length relationship
        b11 = self.paramFLa[0]
        b21 = self.paramFLa[1]
        b31 = self.paramFLa[2]
        b41 = self.paramFLa[3]
        b12 = self.paramFLa[4]
        b22 = self.paramFLa[5]
        b32 = self.paramFLa[6]
        b42 = self.paramFLa[7]
        b13 = 0.1
        b23 = 1
        b33 = 0.5 * np.sqrt(0.5)
        b43 = 0
        num3 = self.normFiberLength - b23
        den3 = b33 + b43 * self.normFiberLength
        FMtilde3 = b13 * np.exp(-0.5 * (np.divide(num3**2, den3**2)))
        num1 = self.normFiberLength - b21
        den1 = b31 + b41 * self.normFiberLength        
        FMtilde1 = b11 * np.exp(-0.5 * (np.divide(num1**2, den1**2)))
        num2 = self.normFiberLength - b22
        den2 = b32 + b42 * self.normFiberLength
        FMtilde2 = b12 * np.exp(-0.5 * (np.divide(num2**2, den2**2)))
        self.normActiveFiberLengthForce = FMtilde1 + FMtilde2 + FMtilde3
        
        return self.normActiveFiberLengthForce
        
    def getActiveFiberVelocityForce(self):   
        self.getFiberVelocity()        
        # Active muscle force-velocity relationship
        e1 = self.paramFV[0]
        e2 = self.paramFV[1]
        e3 = self.paramFV[2]
        e4 = self.paramFV[3]
        
        self.normActiveFiberVelocityForce = e1 * np.log(
                (e2 * self.normFiberVelocity + e3) 
                + np.sqrt((e2 * self.normFiberVelocity + e3)**2 + 1)) + e4
        
    def getActiveFiberForce(self):
        d = 0.01
        self.getActiveFiberLengthForce()
        self.getActiveFiberVelocityForce()
        
        self.normActiveFiberForce = ((self.activation * 
                                      self.normActiveFiberLengthForce * 
                                      self.normActiveFiberVelocityForce) + 
            d * self.normFiberVelocity)
            
        activeFiberForce = (self.normActiveFiberForce * 
                            self.maximalIsometricForce)
            
        return activeFiberForce, self.normActiveFiberForce
        
    def getPassiveFiberForce(self):
        paramFLp = self.paramFLp
        self.getFiberLength()
        
        # Passive muscle force-length relationship
        e0 = 0.6
        kpe = 4        
        t5 = np.exp(kpe * (self.normFiberLength - 1) / e0)
        self.normPassiveFiberForce = np.divide(((t5 - 1) - paramFLp[0]), 
                                               paramFLp[1])
        
        passiveFiberForce = (self.normPassiveFiberForce * 
                             self.maximalIsometricForce)
        
        return passiveFiberForce, self.normPassiveFiberForce
        
    def deriveHillEquilibrium(self):        
        self.getActiveFiberForce()
        self.getPassiveFiberForce()
        
        hillEquilibrium = ((np.multiply(self.normActiveFiberForce + 
                                        self.normPassiveFiberForce, 
                                        self.cosPennationAngle)) - 
                                        self.normTendonForce)
        
        return hillEquilibrium
    
    def deriveHillEquilibriumNoPassive(self):        
        self.getActiveFiberForce()
        
        hillEquilibrium = ((np.multiply(self.normActiveFiberForce, 
                                        self.cosPennationAngle)) - 
                                        self.normTendonForce)
        
        return hillEquilibrium
    
class muscleModel_rigidTendon:
    
    def __init__(self, mtParameters, activation, mtLength, mtVelocity,
                 specificTension):
        self.mtParameters = mtParameters        
        self.maximalIsometricForce = mtParameters[0]
        self.optimalFiberLength = mtParameters[1]          
        self.tendonSlackLength = mtParameters[2]
        self.optimalPennationAngle = mtParameters[3]        
        self.maximalFiberVelocity = mtParameters[4]        
        self.activation = activation
        self.mtLength = mtLength
        self.mtVelocity = mtVelocity
        self.specificTension = specificTension
        self.paramFLa = np.array([0.814483478343008, 1.05503342897057,
                                  0.162384573599574, 0.0633034484654646,
                                  0.433004984392647, 0.716775413397760, 
                                  -0.0299471169706956, 0.200356847296188])
        self.paramFLp = np.array([-0.995172050006169, 53.5981500331442])
        self.paramFV = np.array([-0.318323436899127, -8.14915604347525,
                                 -0.374121508647863, 0.885644059915004])
    
    def getMuscleVolume(self):
        self.muscleVolume = np.multiply(self.maximalIsometricForce, 
                                        self.optimalFiberLength)
        return self.muscleVolume
        
    def getMuscleMass(self):                
        muscleMass = np.divide(np.multiply(self.muscleVolume, 1059.7), 
                               np.multiply(self.specificTension, 1e6))
        
        return muscleMass 
            
    def getTendonLength(self):          
        # Tendon force-length relationship
        normTendonLength = 1                                    
        tendonLength = self.tendonSlackLength
        
        return tendonLength, normTendonLength
                
    def getFiberLength(self):
        # Hill-type muscle model: geometric relationships    
        self.getTendonLength()
        w = np.multiply(self.optimalFiberLength, 
                        np.sin(self.optimalPennationAngle))  
        # Tendon length is tendon slack length with rigid tendon.
        self.fiberLength = np.sqrt(
                (self.mtLength - self.tendonSlackLength)**2 + w**2)
        self.normFiberLength = np.divide(self.fiberLength, 
                                         self.optimalFiberLength)   

        return self.fiberLength, self.normFiberLength         
    
    def getFiberVelocity(self):            
        # Hill-type muscle model: geometric relationships 
        self.getFiberLength()     
        # Tendon length is tendon slack length with rigid tendon.
        self.cosPennationAngle = np.divide(
            (self.mtLength - self.tendonSlackLength), 
            self.fiberLength)       
        self.normFiberVelocity = np.multiply(
            np.divide(self.mtVelocity, self.maximalFiberVelocity),
            self.cosPennationAngle)                                             
        
        return self.normFiberVelocity 
    
    def getActiveFiberLengthForce(self):  
        self.getFiberLength()        
        # Active muscle force-length relationship
        b11 = self.paramFLa[0]
        b21 = self.paramFLa[1]
        b31 = self.paramFLa[2]
        b41 = self.paramFLa[3]
        b12 = self.paramFLa[4]
        b22 = self.paramFLa[5]
        b32 = self.paramFLa[6]
        b42 = self.paramFLa[7]
        b13 = 0.1
        b23 = 1
        b33 = 0.5 * np.sqrt(0.5)
        b43 = 0
        num3 = self.normFiberLength - b23
        den3 = b33 + b43 * self.normFiberLength
        FMtilde3 = b13 * np.exp(-0.5 * (np.divide(num3**2, den3**2)))
        num1 = self.normFiberLength - b21
        den1 = b31 + b41 * self.normFiberLength        
        FMtilde1 = b11 * np.exp(-0.5 * (np.divide(num1**2, den1**2)))
        num2 = self.normFiberLength - b22
        den2 = b32 + b42 * self.normFiberLength
        FMtilde2 = b12 * np.exp(-0.5 * (np.divide(num2**2, den2**2)))
        self.normActiveFiberLengthForce = FMtilde1 + FMtilde2 + FMtilde3
        
        return self.normActiveFiberLengthForce
        
    def getActiveFiberVelocityForce(self):   
        self.getFiberVelocity()        
        # Active muscle force-velocity relationship
        e1 = self.paramFV[0]
        e2 = self.paramFV[1]
        e3 = self.paramFV[2]
        e4 = self.paramFV[3]
        
        self.normActiveFiberVelocityForce = e1 * np.log(
                (e2 * self.normFiberVelocity + e3) 
                + np.sqrt((e2 * self.normFiberVelocity + e3)**2 + 1)) + e4
        
    def getActiveFiberForce(self):
        d = 0.01
        self.getActiveFiberLengthForce()
        self.getActiveFiberVelocityForce()
        
        self.normActiveFiberForce = ((self.activation * 
                                      self.normActiveFiberLengthForce * 
                                      self.normActiveFiberVelocityForce) + 
            d * self.normFiberVelocity)
            
        self.activeFiberForce = (self.normActiveFiberForce * 
                                 self.maximalIsometricForce)
            
        return self.activeFiberForce, self.normActiveFiberForce
        
    def getPassiveFiberForce(self):
        paramFLp = self.paramFLp
        self.getFiberLength()
        
        # Passive muscle force-length relationship
        e0 = 0.6
        kpe = 4        
        t5 = np.exp(kpe * (self.normFiberLength - 1) / e0)
        self.normPassiveFiberForce = np.divide(((t5 - 1) - paramFLp[0]), 
                                                paramFLp[1])
        
        self.passiveFiberForce = (self.normPassiveFiberForce * 
                                  self.maximalIsometricForce)
        
        return self.passiveFiberForce, self.normPassiveFiberForce
    
    def getTotalFiberForce(self):
        self.getActiveFiberForce()
        self.getPassiveFiberForce()
        
        totalNormFiberForce = (self.normActiveFiberForce + 
                               self.normPassiveFiberForce)
        
        totalFiberForce = (self.activeFiberForce + 
                           self.passiveFiberForce)
        
        return totalFiberForce, totalNormFiberForce
        
    # def deriveHillEquilibrium(self):        
    #     self.getActiveFiberForce()
    #     self.getPassiveFiberForce()
        
    #     hillEquilibrium = ((np.multiply(self.normActiveFiberForce + 
    #                                     self.normPassiveFiberForce, 
    #                                     self.cosPennationAngle)) - 
    #                                     self.normTendonForce)
        
    #     return hillEquilibrium
    
    # def deriveHillEquilibriumNoPassive(self):        
    #     self.getActiveFiberForce()
        
    #     hillEquilibrium = ((np.multiply(self.normActiveFiberForce, 
    #                                     self.cosPennationAngle)) - 
    #                                     self.normTendonForce)
        
    #     return hillEquilibrium
    
#def main():
#    mtParameters = np.array([[2.5],[4.5],[6.5],[8.5],[10.5]])
#    activation = 0.8
#    mtLength = 1.4
#    mtVelocity = 0.8
#    normTendonForce = 1.5
#    normTendonForceDT = 15
#    tendonCompliance = 35
#    tendonShift = 0
#    specificTension = 25
#    
#    muscle = muscleModel(mtParameters, activation, mtLength, mtVelocity,
#                         normTendonForce, normTendonForceDT, tendonCompliance,
#                         tendonShift, specificTension)
#    
#    fiberLength, normFiberLength = muscle.getFiberLength()
#    print(fiberLength)
#    print(normFiberLength)
#    
#    fiberVelocity, normFiberVelocity = muscle.getFiberVelocity()
#    print(fiberVelocity)
#    print(normFiberVelocity)
#    
#    normActiveFiberLengthForce = muscle.getActiveFiberLengthForce()
#    print(normActiveFiberLengthForce)
#    
#    normActiveFiberForce = muscle.getActiveFiberForce()
#    print(normActiveFiberForce)
#    
#    normPassiveFiberForce = muscle.getPassiveFiberForce()
#    print(normPassiveFiberForce)
#    
#    hillEquilibrium = muscle.deriveHillEquilibrium()
#    print(hillEquilibrium)
#    
#    tendonForce = muscle.getTendonForce()
#    print(tendonForce)
#    
#main()