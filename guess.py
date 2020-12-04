import pandas as pd
import numpy as np
import scipy.interpolate as interpolate
from scipy.interpolate import interp1d
   
# %% Quasi-random initial guess    
class quasiRandomGuess:    
    def __init__(self, N, d, joints, muscles, time, Qs):      
        
        self.N = N
        self.d = d
        self.joints = joints
        self.guessFinalTime = time
        self.muscles = muscles
        self.targetSpeed = 1.2
        self.Qs = Qs
    
    # Mesh points
    def getGuessPosition(self, scaling):
        g = [0] * (self.N + 1)
        
        pelvis_tx_first = self.Qs["pelvis_tx"][0]
        g_pelvis_tx = np.linspace(
            pelvis_tx_first, 
            pelvis_tx_first + self.guessFinalTime * self.targetSpeed, 
            self.N)
        g_pelvis_tx = np.append(g_pelvis_tx, g_pelvis_tx[-1] + 
                                (g_pelvis_tx[-1] - g_pelvis_tx[-2]))
        g_pelvis_ty =  [0.9385] * (self.N + 1)
        self.guessPosition = pd.DataFrame()  
        for count, joint in enumerate(self.joints): 
            if joint == 'pelvis_tx':
                self.guessPosition.insert(count, joint, 
                                          g_pelvis_tx / scaling.iloc[0][joint])
            elif joint == 'pelvis_ty':
                self.guessPosition.insert(count, joint, 
                                          g_pelvis_ty / scaling.iloc[0][joint])                    
            else:
                self.guessPosition.insert(count, joint, 
                                          g / scaling.iloc[0][joint])
        
        return self.guessPosition
    
    def getGuessVelocity(self, scaling):
        g = [0] * (self.N + 1)
        g_pelvis_tx =  [self.targetSpeed] * (self.N + 1)
        self.guessVelocity = pd.DataFrame()  
        for count, joint in enumerate(self.joints): 
            if joint == 'pelvis_tx':
                self.guessVelocity.insert(count, joint,
                                          g_pelvis_tx / scaling.iloc[0][joint])             
            else:
                self.guessVelocity.insert(count, joint, 
                                          g / scaling.iloc[0][joint])
        
        return self.guessVelocity
    
    # TODO: zeroAcceleration to match data-driven - not great
    def getGuessAcceleration(self, scaling, zeroAcceleration=True):
        if zeroAcceleration:
            g = [0] * self.N
        else:
            raise ValueError('Guess acceleration - zero')
        self.guessAcceleration = pd.DataFrame()  
        for count, joint in enumerate(self.joints):
            self.guessAcceleration.insert(count, joint, 
                                          g / scaling.iloc[0][joint])
            
        return self.guessAcceleration
    
    def getGuessActivation(self, scaling):
        g = [0.1] * (self.N + 1)
        self.guessActivation = pd.DataFrame()  
        for count, muscle in enumerate(self.muscles):
            self.guessActivation.insert(count, muscle, 
                                        g / scaling.iloc[0][muscle])
            
        return self.guessActivation
    
    def getGuessActivationDerivative(self, scaling):
        g = [0.01] * self.N
        guessActivationDerivative = pd.DataFrame()  
        for count, muscle in enumerate(self.muscles):
            guessActivationDerivative.insert(count, muscle, 
                                             g / scaling.iloc[0][muscle])
            
        return guessActivationDerivative
    
    def getGuessForce(self, scaling):
        g = [0.1] * (self.N + 1)
        self.guessForce = pd.DataFrame()  
        for count, muscle in enumerate(self.muscles):
            self.guessForce.insert(count, muscle, g / scaling.iloc[0][muscle])
            
        return self.guessForce
    
    def getGuessForceDerivative(self, scaling):
        g = [0.01] * self.N
        self.guessForceDerivative = pd.DataFrame()  
        for count, muscle in enumerate(self.muscles):
            self.guessForceDerivative.insert(count, muscle, 
                                        g / scaling.iloc[0][muscle])
            
        return self.guessForceDerivative
    
    def getGuessTMActivation(self, joints):
        g = [0.1] * (self.N + 1)
        self.guessTMActivation = pd.DataFrame()  
        for count, TMJoint in enumerate(joints):
            self.guessTMActivation.insert(
                    count, TMJoint, g)
            
        return self.guessTMActivation
    
    def getGuessTMExcitation(self, joints):
        g = [0.1] * (self.N)
        guessTMExcitation = pd.DataFrame()  
        for count, TMJoint in enumerate(joints):
            guessTMExcitation.insert(count, TMJoint, g)
            
        return guessTMExcitation
    
    # Collocation points   
    def getGuessActivationCol(self):            
        guessActivationCol = pd.DataFrame(columns=self.muscles)          
        for k in range(self.N):
            for c in range(self.d):          
                guessActivationCol = guessActivationCol.append(
                        self.guessActivation.iloc[k], ignore_index=True)
            
        return guessActivationCol
    
    def getGuessForceCol(self):
        guessForceCol = pd.DataFrame(columns=self.muscles)          
        for k in range(self.N):
            for c in range(self.d):          
                guessForceCol = guessForceCol.append(
                        self.guessForce.iloc[k], ignore_index=True)
            
        return guessForceCol
    
    def getGuessForceDerivativeCol(self):
        guessForceDerivativeCol = pd.DataFrame(columns=self.muscles)          
        for k in range(self.N):
            for c in range(self.d):          
                guessForceDerivativeCol = guessForceDerivativeCol.append(
                        self.guessForceDerivative.iloc[k], ignore_index=True)
            
        return guessForceDerivativeCol
    
    def getGuessTMActivationCol(self, joints):
        guessTMActivationCol = (
                pd.DataFrame(columns=joints))         
        for k in range(self.N):
            for c in range(self.d):          
                guessTMActivationCol = (
                        guessTMActivationCol.append(
                        self.guessTMActivation.iloc[k], 
                        ignore_index=True))
            
        return guessTMActivationCol        
    
    def getGuessPositionCol(self):
        guessPositionCol = pd.DataFrame(columns=self.joints)          
        for k in range(self.N):
            for c in range(self.d):          
                guessPositionCol = guessPositionCol.append(
                        self.guessPosition.iloc[k], ignore_index=True)
        
        return guessPositionCol
    
    def getGuessVelocityCol(self):
        guessVelocityCol = pd.DataFrame(columns=self.joints)       
        for k in range(self.N):
            for c in range(self.d):          
                guessVelocityCol = guessVelocityCol.append(
                        self.guessVelocity.iloc[k], ignore_index=True)
        
        return guessVelocityCol
    
    def getGuessAccelerationCol(self):
        guessAccelerationCol = pd.DataFrame(columns=self.joints)  
        for k in range(self.N):
            for c in range(self.d):          
                guessAccelerationCol = guessAccelerationCol.append(
                        self.guessAcceleration.iloc[k], ignore_index=True)
                
        return guessAccelerationCol
    
    def getGuessMarker(self, markers, marker_data, scaling, 
                       dimensions = ["x", "y", "z"]):
        guessMarker = pd.DataFrame() 
        count = 0
        for marker in markers:  
            for dimension in dimensions:
                guessMarker.insert(count, marker + "_" + dimension, 
                                   marker_data[marker + "_" + dimension] / 
                                   scaling.iloc[0][marker + "_" + dimension]) 
                count += 1
        
        return guessMarker
    
    def getGuessOffset(self, scaling):
        
        guessOffset = 0 / scaling
        
        return guessOffset
    
# %% Data-driven initial guess    
class dataDrivenGuess:    
    def __init__(self, Qs, N, d, joints, holConstraints, muscles=[]):        
        
        self.Qs = Qs
        self.N = N
        self.d = d
        self.joints = joints
        self.muscles = muscles
        self.holConstraints = holConstraints
        
    def splineQs(self):
        
        self.Qs_spline = self.Qs.copy()
        self.Qdots_spline = self.Qs.copy()
        self.Qdotdots_spline = self.Qs.copy()

        for joint in self.joints:
            spline = interpolate.InterpolatedUnivariateSpline(self.Qs['time'], 
                                                              self.Qs[joint],
                                                              k=3) 
            self.Qs_spline[joint] = spline(self.Qs['time'])
            splineD1 = spline.derivative(n=1)
            self.Qdots_spline[joint] = splineD1(self.Qs['time'])
            splineD2 = spline.derivative(n=2)
            self.Qdotdots_spline[joint] = splineD2(self.Qs['time'])
            
    def interpQs(self):
        self.splineQs()            
        tOut = np.linspace(self.Qs['time'].iloc[0], 
                           self.Qs['time'].iloc[-1], 
                           self.N + 1)    
        
        self.Qs_spline_interp = pd.DataFrame()  
        self.Qdots_spline_interp = pd.DataFrame()  
        self.Qdotdots_spline_interp = pd.DataFrame()  
        for count, joint in enumerate(self.joints):  
            set_interp = interp1d(self.Qs['time'], self.Qs_spline[joint])
            self.Qs_spline_interp.insert(count, joint, set_interp(tOut))
            
            set_interp = interp1d(self.Qs['time'], self.Qdots_spline[joint])
            self.Qdots_spline_interp.insert(count, joint, set_interp(tOut))
            
            set_interp = interp1d(self.Qs['time'], self.Qdotdots_spline[joint])
            self.Qdotdots_spline_interp.insert(count, joint, set_interp(tOut))
        
    
    # Mesh points
    def getGuessPosition(self, scaling, zeroMTP=False):
        self.interpQs()
        self.guessPosition = pd.DataFrame()  
        g = [0] * (self.N + 1)
        for count, joint in enumerate(self.joints):  
            if joint == 'mtp_angle_l' or joint == 'mtp_angle_r':
                self.guessPosition.insert(count, joint, g) 
            
            else:
                self.guessPosition.insert(count, joint, 
                                          self.Qs_spline_interp[joint] / 
                                          scaling.iloc[0][joint]) 
        
        return self.guessPosition
    
    def getGuessVelocity(self, scaling, zeroVelocity=False):
        self.splineQs()
        self.guessVelocity = pd.DataFrame()  
        g = [0] * (self.N + 1)
        for count, joint in enumerate(self.joints): 
            if zeroVelocity:
                self.guessVelocity.insert(count, joint, g)             
            else:
                self.guessVelocity.insert(count, joint, 
                                          self.Qdots_spline_interp[joint] / 
                                          scaling.iloc[0][joint])       
        return self.guessVelocity
    
    def getGuessAcceleration(self, scaling, zeroAcceleration=False):
        self.splineQs()
        self.guessAcceleration = pd.DataFrame()  
        g = [0] * self.N
        for count, joint in enumerate(self.joints):   
            if zeroAcceleration:
                self.guessAcceleration.insert(
                    count, joint, g / scaling.iloc[0][joint]) 
            else:               
                self.guessAcceleration.insert(
                    count, joint, self.Qdotdots_spline_interp[joint] /
                    scaling.iloc[0][joint])                               
                    
        return self.guessAcceleration
    
    def getGuessActivation(self, scaling):
        g = [0.1] * (self.N + 1)
        self.guessActivation = pd.DataFrame()  
        for count, muscle in enumerate(self.muscles):
            self.guessActivation.insert(count, muscle, 
                                        g / scaling.iloc[0][muscle])
            
        return self.guessActivation
    
    def getGuessActivationDerivative(self, scaling):
        g = [0.01] * self.N
        guessActivationDerivative = pd.DataFrame()  
        for count, muscle in enumerate(self.muscles):
            guessActivationDerivative.insert(count, muscle, 
                                             g / scaling.iloc[0][muscle])
            
        return guessActivationDerivative
    
    def getGuessForce(self, scaling):
        g = [0.1] * (self.N + 1)
        self.guessForce = pd.DataFrame()  
        for count, muscle in enumerate(self.muscles):
            self.guessForce.insert(count, muscle, g / scaling.iloc[0][muscle])
            
        return self.guessForce
    
    def getGuessForceDerivative(self, scaling):
        g = [0.01] * self.N
        self.guessForceDerivative = pd.DataFrame()  
        for count, muscle in enumerate(self.muscles):
            self.guessForceDerivative.insert(count, muscle, 
                                        g / scaling.iloc[0][muscle])
            
        return self.guessForceDerivative
    
    def getGuessTMActivation(self, actJoints):
        g = [0.1] * (self.N + 1)
        self.actJoints = actJoints
        self.guessTMActivation = pd.DataFrame()  
        for count, TMJoint in enumerate(self.actJoints):
            self.guessTMActivation.insert(
                    count, TMJoint, g)
            
        return self.guessTMActivation
    
    def getGuessTMExcitation(self, joints):
        g = [0.1] * (self.N)
        guessTMExcitation = pd.DataFrame()  
        for count, TMJoint in enumerate(joints):
            guessTMExcitation.insert(count, TMJoint, g)
            
        return guessTMExcitation
    
    def getGuessMultipliers(self):
        g = [0.] * (self.N + 1)

        self.guessMultipliers = pd.DataFrame()  
        for count, holConstraints_title in enumerate(self.holConstraints):
            self.guessMultipliers.insert(
                    count, holConstraints_title, g)
            
        return self.guessMultipliers
    
    def getGuessVelCorrs(self):
        g = [0.] * (self.N + 1)

        self.guessVelCorrs = pd.DataFrame()  
        for count, holConstraints_title in enumerate(self.holConstraints):
            self.guessVelCorrs.insert(
                    count, holConstraints_title, g)
            
        return self.guessVelCorrs 
    
    # Collocation points   
    def getGuessActivationCol(self):            
        guessActivationCol = pd.DataFrame(columns=self.muscles)          
        for k in range(self.N):
            for c in range(self.d):          
                guessActivationCol = guessActivationCol.append(
                        self.guessActivation.iloc[k], ignore_index=True)
            
        return guessActivationCol
    
    def getGuessForceCol(self):
        guessForceCol = pd.DataFrame(columns=self.muscles)          
        for k in range(self.N):
            for c in range(self.d):          
                guessForceCol = guessForceCol.append(
                        self.guessForce.iloc[k], ignore_index=True)
            
        return guessForceCol
    
    def getGuessForceDerivativeCol(self):
        guessForceDerivativeCol = pd.DataFrame(columns=self.muscles)          
        for k in range(self.N):
            for c in range(self.d):          
                guessForceDerivativeCol = guessForceDerivativeCol.append(
                        self.guessForceDerivative.iloc[k], ignore_index=True)
            
        return guessForceDerivativeCol
    
    def getGuessTMActivationCol(self):
        guessTMActivationCol = (
                pd.DataFrame(columns=self.actJoints))         
        for k in range(self.N):
            for c in range(self.d):          
                guessTMActivationCol = (
                        guessTMActivationCol.append(
                        self.guessTMActivation.iloc[k], 
                        ignore_index=True))
            
        return guessTMActivationCol        
    
    def getGuessPositionCol(self):
        guessPositionCol = pd.DataFrame(columns=self.joints)          
        for k in range(self.N):
            for c in range(self.d):          
                guessPositionCol = guessPositionCol.append(
                        self.guessPosition.iloc[k], ignore_index=True)
        
        return guessPositionCol
    
    def getGuessVelocityCol(self):
        guessVelocityCol = pd.DataFrame(columns=self.joints)       
        for k in range(self.N):
            for c in range(self.d):          
                guessVelocityCol = guessVelocityCol.append(
                        self.guessVelocity.iloc[k], ignore_index=True)
        
        return guessVelocityCol
    
    def getGuessAccelerationCol(self):
        guessAccelerationCol = pd.DataFrame(columns=self.joints)  
        for k in range(self.N):
            for c in range(self.d):          
                guessAccelerationCol = guessAccelerationCol.append(
                        self.guessAcceleration.iloc[k], ignore_index=True)
                
        return guessAccelerationCol
    
    def getGuessMultipliersCol(self):
        guessMultipliersCol = pd.DataFrame(columns=self.holConstraints)  
        for k in range(self.N):
            for c in range(self.d):          
                guessMultipliersCol = guessMultipliersCol.append(
                        self.guessMultipliers.iloc[k], ignore_index=True)
                
        return guessMultipliersCol
    
    def getGuessVelCorrsCol(self):
        guessVelCorrsCol = pd.DataFrame(columns=self.holConstraints)  
        for k in range(self.N):
            for c in range(self.d):          
                guessVelCorrsCol = guessVelCorrsCol.append(
                        self.guessVelCorrs.iloc[k], ignore_index=True)
                
        return guessVelCorrsCol
    
    def getGuessMarker(self, markers, marker_data, scaling, 
                       dimensions = ["x", "y", "z"]):
        guessMarker = pd.DataFrame() 
        count = 0
        for marker in markers:  
            for dimension in dimensions:
                guessMarker.insert(count, marker + "_" + dimension, 
                                   marker_data[marker + "_" + dimension] / 
                                   scaling.iloc[0][marker + "_" + dimension]) 
                count += 1
        
        return guessMarker
    
    def getGuessOffset(self, scaling):
        
        guessOffset = 0 / scaling
        
        return guessOffset
    
    def getGuessIMU(self, imus, imu_data, scaling):
        guessIMU = pd.DataFrame() 
        for count, imu in enumerate(imus):  
            guessIMU.insert(count, imu, imu_data[imu] / scaling.iloc[0][imu]) 
        
        return guessIMU
    