import scipy.interpolate as interpolate
import pandas as pd
import numpy as np

class bounds:  
    def __init__(self, joints, rotationalJoints, translationalJoints=[], 
                 muscles=[]):
        self.joints = joints
        self.rotationalJoints = rotationalJoints
        self.translationalJoints = translationalJoints
        self.muscles = muscles       
            
    def getBoundsPosition(self):
        upperBoundsPosition_all = pd.DataFrame()   
        lowerBoundsPosition_all = pd.DataFrame() 
        
        upperBoundsPosition_all['ground_thorax_rot_x'] = [20 * np.pi / 180]
        upperBoundsPosition_all['ground_thorax_rot_y'] = [20 * np.pi / 180]
        upperBoundsPosition_all['ground_thorax_rot_z'] = [10 * np.pi / 180]
        upperBoundsPosition_all['ground_thorax_tx'] = [1]
        upperBoundsPosition_all['ground_thorax_ty'] = [1]
        upperBoundsPosition_all['ground_thorax_tz'] = [1]        
        upperBoundsPosition_all['clav_prot'] = [0 * np.pi / 180]
        upperBoundsPosition_all['clav_elev'] = [40 * np.pi / 180]
        upperBoundsPosition_all['scapula_abduction'] = [30 * np.pi / 180]
        upperBoundsPosition_all['scapula_elevation'] = [30 * np.pi / 180]
        upperBoundsPosition_all['scapula_upward_rot'] = [100 * np.pi / 180]
        upperBoundsPosition_all['scapula_winging'] = [30 * np.pi / 180]        
        upperBoundsPosition_all['plane_elv'] = [90 * np.pi / 180]
        upperBoundsPosition_all['shoulder_elv'] = [110 * np.pi / 180]
        upperBoundsPosition_all['axial_rot'] = [90 * np.pi / 180]
        upperBoundsPosition_all['elbow_flexion'] = [20 * np.pi / 180]
        upperBoundsPosition_all['pro_sup'] = [50 * np.pi / 180]
        
        lowerBoundsPosition_all['ground_thorax_rot_x'] = [-20 * np.pi / 180]
        lowerBoundsPosition_all['ground_thorax_rot_y'] = [-30 * np.pi / 180]
        lowerBoundsPosition_all['ground_thorax_rot_z'] = [-30 * np.pi / 180]
        lowerBoundsPosition_all['ground_thorax_tx'] = [-1]
        lowerBoundsPosition_all['ground_thorax_ty'] = [-1]
        lowerBoundsPosition_all['ground_thorax_tz'] = [-1]        
        lowerBoundsPosition_all['clav_prot'] = [-60 * np.pi / 180]
        lowerBoundsPosition_all['clav_elev'] = [-30 * np.pi / 180]
        lowerBoundsPosition_all['scapula_abduction'] = [-50 * np.pi / 180]
        lowerBoundsPosition_all['scapula_elevation'] = [-30 * np.pi / 180]
        lowerBoundsPosition_all['scapula_upward_rot'] = [-10 * np.pi / 180]
        lowerBoundsPosition_all['scapula_winging'] = [-30 * np.pi / 180]        
        lowerBoundsPosition_all['plane_elv'] = [-60 * np.pi / 180]
        lowerBoundsPosition_all['shoulder_elv'] = [-20 * np.pi / 180]
        lowerBoundsPosition_all['axial_rot'] = [-50 * np.pi / 180]
        lowerBoundsPosition_all['elbow_flexion'] = [-20 * np.pi / 180]
        lowerBoundsPosition_all['pro_sup'] = [10 * np.pi / 180]
        
        # Rotational joints
        # The goal is to use the same scaling for each joint
        max_rotationalJoints = np.zeros(len(self.rotationalJoints))
        for count, rotationalJoint in enumerate(self.rotationalJoints): 
            max_rotationalJoints[count] = pd.concat(
                    [abs(upperBoundsPosition_all[rotationalJoint]),
                     abs(lowerBoundsPosition_all[rotationalJoint])]).max(level=0)[0]
        maxAll_rotationalJoints = np.max(max_rotationalJoints)      
            
        # Translational joints
        # The goal is to use the same scaling for each joint
        if self.translationalJoints:
            max_translationalJoints = np.zeros(len(self.translationalJoints))
            for count, translationalJoint in enumerate(self.translationalJoints): 
                max_translationalJoints[count] = pd.concat(
                        [abs(upperBoundsPosition_all[translationalJoint]),
                         abs(lowerBoundsPosition_all[translationalJoint])]).max(level=0)[0]
            maxAll_translationalJoints = np.max(max_translationalJoints)   
        
        upperBoundsPosition = pd.DataFrame()   
        lowerBoundsPosition = pd.DataFrame()
        scalingPosition = pd.DataFrame() 
        
        for count, joint in enumerate(self.joints):         
            # Scaling            
            if joint in self.rotationalJoints:
                scalingPosition.insert(count, joint, [maxAll_rotationalJoints])
            elif joint in self.translationalJoints:
                scalingPosition.insert(count, joint, [maxAll_translationalJoints])
            else:
                 raise ValueError('Unknown joint')
            upperBoundsPosition.insert(
                 count, joint, [upperBoundsPosition_all.iloc[0][joint] / 
                                scalingPosition.iloc[0][joint]])
            lowerBoundsPosition.insert(
                 count, joint, [lowerBoundsPosition_all.iloc[0][joint] / 
                                scalingPosition.iloc[0][joint]])  
                
        return upperBoundsPosition, lowerBoundsPosition, scalingPosition 
    
    def getBoundsPositionConservative(self):
        upperBoundsPosition_all = pd.DataFrame()   
        lowerBoundsPosition_all = pd.DataFrame() 
        
        upperBoundsPosition_all['ground_thorax_rot_x'] = [20 * np.pi / 180]
        upperBoundsPosition_all['ground_thorax_rot_y'] = [20 * np.pi / 180]
        upperBoundsPosition_all['ground_thorax_rot_z'] = [10 * np.pi / 180]
        upperBoundsPosition_all['ground_thorax_tx'] = [1]
        upperBoundsPosition_all['ground_thorax_ty'] = [1]
        upperBoundsPosition_all['ground_thorax_tz'] = [1]        
        upperBoundsPosition_all['clav_prot'] = [0 * np.pi / 180]
        upperBoundsPosition_all['clav_elev'] = [40 * np.pi / 180]
        upperBoundsPosition_all['scapula_abduction'] = [30 * np.pi / 180]
        upperBoundsPosition_all['scapula_elevation'] = [30 * np.pi / 180]
        upperBoundsPosition_all['scapula_upward_rot'] = [100 * np.pi / 180]
        upperBoundsPosition_all['scapula_winging'] = [30 * np.pi / 180]        
        upperBoundsPosition_all['plane_elv'] = [90 * np.pi / 180]
        upperBoundsPosition_all['shoulder_elv'] = [110 * np.pi / 180]
        upperBoundsPosition_all['axial_rot'] = [90 * np.pi / 180]
        upperBoundsPosition_all['elbow_flexion'] = [20 * np.pi / 180]
        upperBoundsPosition_all['pro_sup'] = [50 * np.pi / 180]
        
        lowerBoundsPosition_all['ground_thorax_rot_x'] = [-20 * np.pi / 180]
        lowerBoundsPosition_all['ground_thorax_rot_y'] = [-30 * np.pi / 180]
        lowerBoundsPosition_all['ground_thorax_rot_z'] = [-30 * np.pi / 180]
        lowerBoundsPosition_all['ground_thorax_tx'] = [-1]
        lowerBoundsPosition_all['ground_thorax_ty'] = [-1]
        lowerBoundsPosition_all['ground_thorax_tz'] = [-1]        
        lowerBoundsPosition_all['clav_prot'] = [-60 * np.pi / 180]
        lowerBoundsPosition_all['clav_elev'] = [-30 * np.pi / 180]
        lowerBoundsPosition_all['scapula_abduction'] = [-50 * np.pi / 180]
        lowerBoundsPosition_all['scapula_elevation'] = [-30 * np.pi / 180]
        lowerBoundsPosition_all['scapula_upward_rot'] = [-10 * np.pi / 180]
        lowerBoundsPosition_all['scapula_winging'] = [-30 * np.pi / 180]        
        lowerBoundsPosition_all['plane_elv'] = [-60 * np.pi / 180]
        lowerBoundsPosition_all['shoulder_elv'] = [-20 * np.pi / 180]
        lowerBoundsPosition_all['axial_rot'] = [-50 * np.pi / 180]
        lowerBoundsPosition_all['elbow_flexion'] = [-20 * np.pi / 180]
        lowerBoundsPosition_all['pro_sup'] = [10 * np.pi / 180]
        
        # Rotational joints
        # The goal is to use the same scaling for each joint
        max_rotationalJoints = np.zeros(len(self.rotationalJoints))
        for count, rotationalJoint in enumerate(self.rotationalJoints): 
            max_rotationalJoints[count] = pd.concat(
                    [abs(upperBoundsPosition_all[rotationalJoint]),
                     abs(lowerBoundsPosition_all[rotationalJoint])]).max(level=0)[0]
        maxAll_rotationalJoints = np.max(max_rotationalJoints)      
            
        # Translational joints
        # The goal is to use the same scaling for each joint
        if self.translationalJoints:
            max_translationalJoints = np.zeros(len(self.translationalJoints))
            for count, translationalJoint in enumerate(self.translationalJoints): 
                max_translationalJoints[count] = pd.concat(
                        [abs(upperBoundsPosition_all[translationalJoint]),
                         abs(lowerBoundsPosition_all[translationalJoint])]).max(level=0)[0]
            maxAll_translationalJoints = np.max(max_translationalJoints)   
        
        upperBoundsPosition = pd.DataFrame()   
        lowerBoundsPosition = pd.DataFrame()
        scalingPosition = pd.DataFrame() 
        
        for count, joint in enumerate(self.joints):         
            # Scaling            
            if joint in self.rotationalJoints:
                scalingPosition.insert(count, joint, [maxAll_rotationalJoints])
            elif joint in self.translationalJoints:
                scalingPosition.insert(count, joint, [maxAll_translationalJoints])
            else:
                 raise ValueError('Unknown joint')
            upperBoundsPosition.insert(
                 count, joint, [upperBoundsPosition_all.iloc[0][joint] / 
                                scalingPosition.iloc[0][joint]])
            lowerBoundsPosition.insert(
                 count, joint, [lowerBoundsPosition_all.iloc[0][joint] / 
                                scalingPosition.iloc[0][joint]])
                
        return upperBoundsPosition, lowerBoundsPosition, scalingPosition
    
    def getBoundsVelocity(self):
        upperBoundsVelocity_all = pd.DataFrame()   
        lowerBoundsVelocity_all = pd.DataFrame()        
        
        upperBoundsVelocity_all['ground_thorax_rot_x'] = [300 * np.pi / 180]
        upperBoundsVelocity_all['ground_thorax_rot_y'] = [300 * np.pi / 180]
        upperBoundsVelocity_all['ground_thorax_rot_z'] = [300 * np.pi / 180]
        upperBoundsVelocity_all['ground_thorax_tx'] = [5]
        upperBoundsVelocity_all['ground_thorax_ty'] = [5]
        upperBoundsVelocity_all['ground_thorax_tz'] = [5]        
        upperBoundsVelocity_all['clav_prot'] = [300 * np.pi / 180]
        upperBoundsVelocity_all['clav_elev'] = [300 * np.pi / 180]
        upperBoundsVelocity_all['scapula_abduction'] = [300 * np.pi / 180]
        upperBoundsVelocity_all['scapula_elevation'] = [300 * np.pi / 180]
        upperBoundsVelocity_all['scapula_upward_rot'] = [300 * np.pi / 180]
        upperBoundsVelocity_all['scapula_winging'] = [300 * np.pi / 180]        
        upperBoundsVelocity_all['plane_elv'] = [300 * np.pi / 180]
        upperBoundsVelocity_all['shoulder_elv'] = [300 * np.pi / 180]
        upperBoundsVelocity_all['axial_rot'] = [300 * np.pi / 180]
        upperBoundsVelocity_all['elbow_flexion'] = [300 * np.pi / 180]
        upperBoundsVelocity_all['pro_sup'] = [300 * np.pi / 180]
        
        lowerBoundsVelocity_all['ground_thorax_rot_x'] = [-300 * np.pi / 180]
        lowerBoundsVelocity_all['ground_thorax_rot_y'] = [-300 * np.pi / 180]
        lowerBoundsVelocity_all['ground_thorax_rot_z'] = [-300 * np.pi / 180]
        lowerBoundsVelocity_all['ground_thorax_tx'] = [-5]
        lowerBoundsVelocity_all['ground_thorax_ty'] = [-5]
        lowerBoundsVelocity_all['ground_thorax_tz'] = [-5]        
        lowerBoundsVelocity_all['clav_prot'] = [-300 * np.pi / 180]
        lowerBoundsVelocity_all['clav_elev'] = [-300 * np.pi / 180]
        lowerBoundsVelocity_all['scapula_abduction'] = [-300 * np.pi / 180]
        lowerBoundsVelocity_all['scapula_elevation'] = [-300 * np.pi / 180]
        lowerBoundsVelocity_all['scapula_upward_rot'] = [-300 * np.pi / 180]
        lowerBoundsVelocity_all['scapula_winging'] = [-300 * np.pi / 180]        
        lowerBoundsVelocity_all['plane_elv'] = [-300 * np.pi / 180]
        lowerBoundsVelocity_all['shoulder_elv'] = [-300 * np.pi / 180]
        lowerBoundsVelocity_all['axial_rot'] = [-300 * np.pi / 180]
        lowerBoundsVelocity_all['elbow_flexion'] = [-300 * np.pi / 180]
        lowerBoundsVelocity_all['pro_sup'] = [-300 * np.pi / 180]
        
        # Rotational joints
        # The goal is to use the same scaling for each joint
        max_rotationalJoints = np.zeros(len(self.rotationalJoints))
        for count, rotationalJoint in enumerate(self.rotationalJoints): 
            max_rotationalJoints[count] = pd.concat(
                    [abs(upperBoundsVelocity_all[rotationalJoint]),
                     abs(lowerBoundsVelocity_all[rotationalJoint])]).max(level=0)[0]
        maxAll_rotationalJoints = np.max(max_rotationalJoints)       
            
        # Translational joints
        # The goal is to use the same scaling for each joint
        if self.translationalJoints:
            max_translationalJoints = np.zeros(len(self.translationalJoints))
            for count, translationalJoint in enumerate(self.translationalJoints): 
                max_translationalJoints[count] = pd.concat(
                        [abs(upperBoundsVelocity_all[translationalJoint]),
                         abs(lowerBoundsVelocity_all[translationalJoint])]).max(level=0)[0]
            maxAll_translationalJoints = np.max(max_translationalJoints) 
        
        upperBoundsVelocity = pd.DataFrame()   
        lowerBoundsVelocity = pd.DataFrame() 
        scalingVelocity = pd.DataFrame()
        
        for count, joint in enumerate(self.joints):         
            # Scaling            
            if joint in self.rotationalJoints:
                scalingVelocity.insert(count, joint, [maxAll_rotationalJoints])
            elif joint in self.translationalJoints:
                scalingVelocity.insert(count, joint, [maxAll_translationalJoints])
            else:
                 raise ValueError('Unknown joint')   
            upperBoundsVelocity.insert(
                 count, joint, [upperBoundsVelocity_all.iloc[0][joint] / 
                                scalingVelocity.iloc[0][joint]])
            lowerBoundsVelocity.insert(
                 count, joint, [lowerBoundsVelocity_all.iloc[0][joint] / 
                                scalingVelocity.iloc[0][joint]]) 
            
        return upperBoundsVelocity, lowerBoundsVelocity, scalingVelocity
    
    def getBoundsAcceleration(self):
        upperBoundsAcceleration_all = pd.DataFrame()
        lowerBoundsAcceleration_all = pd.DataFrame()
        
        upperBoundsAcceleration_all['ground_thorax_rot_x'] = [150]
        upperBoundsAcceleration_all['ground_thorax_rot_y'] = [150]
        upperBoundsAcceleration_all['ground_thorax_rot_z'] = [150]
        upperBoundsAcceleration_all['ground_thorax_tx'] = [30]
        upperBoundsAcceleration_all['ground_thorax_ty'] = [30]
        upperBoundsAcceleration_all['ground_thorax_tz'] = [30]
        upperBoundsAcceleration_all['clav_prot'] = [150]
        upperBoundsAcceleration_all['clav_elev'] = [150] 
        upperBoundsAcceleration_all['scapula_abduction'] = [150] 
        upperBoundsAcceleration_all['scapula_elevation'] = [150]
        upperBoundsAcceleration_all['scapula_upward_rot'] = [150] 
        upperBoundsAcceleration_all['scapula_winging'] = [150] 
        upperBoundsAcceleration_all['plane_elv'] = [150]
        upperBoundsAcceleration_all['shoulder_elv'] = [150]
        upperBoundsAcceleration_all['axial_rot'] = [150]
        upperBoundsAcceleration_all['elbow_flexion'] = [150]
        upperBoundsAcceleration_all['pro_sup'] = [150]
        
        lowerBoundsAcceleration_all['ground_thorax_rot_x'] = [-150]
        lowerBoundsAcceleration_all['ground_thorax_rot_y'] = [-150]
        lowerBoundsAcceleration_all['ground_thorax_rot_z'] = [-150]
        lowerBoundsAcceleration_all['ground_thorax_tx'] = [-30]
        lowerBoundsAcceleration_all['ground_thorax_ty'] = [-30]
        lowerBoundsAcceleration_all['ground_thorax_tz'] = [-30]
        lowerBoundsAcceleration_all['clav_prot'] = [-150]
        lowerBoundsAcceleration_all['clav_elev'] = [-150] 
        lowerBoundsAcceleration_all['scapula_abduction'] = [-150] 
        lowerBoundsAcceleration_all['scapula_elevation'] = [-150]
        lowerBoundsAcceleration_all['scapula_upward_rot'] = [-150] 
        lowerBoundsAcceleration_all['scapula_winging'] =  [-150]
        lowerBoundsAcceleration_all['plane_elv'] = [-150]
        lowerBoundsAcceleration_all['shoulder_elv'] = [-150]
        lowerBoundsAcceleration_all['axial_rot'] = [-150]
        lowerBoundsAcceleration_all['elbow_flexion'] = [-150]
        lowerBoundsAcceleration_all['pro_sup'] = [-150]
        
        # Rotational joints
        # The goal is to use the same scaling for each joint
        max_rotationalJoints = np.zeros(len(self.rotationalJoints))
        for count, rotationalJoint in enumerate(self.rotationalJoints): 
            max_rotationalJoints[count] = pd.concat(
                    [abs(upperBoundsAcceleration_all[rotationalJoint]),
                     abs(lowerBoundsAcceleration_all[rotationalJoint])]).max(level=0)[0]
        maxAll_rotationalJoints = np.max(max_rotationalJoints)          
            
        # Translational joints
        # The goal is to use the same scaling for each joint
        if self.translationalJoints:
            max_translationalJoints = np.zeros(len(self.translationalJoints))
            for count, translationalJoint in enumerate(self.translationalJoints): 
                max_translationalJoints[count] = pd.concat(
                        [abs(upperBoundsAcceleration_all[translationalJoint]),
                         abs(lowerBoundsAcceleration_all[translationalJoint])]).max(level=0)[0]
            maxAll_translationalJoints = np.max(max_translationalJoints)        
        
        upperBoundsAcceleration = pd.DataFrame()   
        lowerBoundsAcceleration = pd.DataFrame() 
        scalingAcceleration = pd.DataFrame() 
        
        for count, joint in enumerate(self.joints):         
            # Scaling            
            if joint in self.rotationalJoints:
                scalingAcceleration.insert(count, joint, [maxAll_rotationalJoints])
            elif joint in self.translationalJoints:
                scalingAcceleration.insert(count, joint, [maxAll_translationalJoints])
            else:
                 raise ValueError('Unknown joint')   
            upperBoundsAcceleration.insert(
                 count, joint, [upperBoundsAcceleration_all.iloc[0][joint] / 
                                scalingAcceleration.iloc[0][joint]])
            lowerBoundsAcceleration.insert(
                 count, joint, [lowerBoundsAcceleration_all.iloc[0][joint] / 
                                scalingAcceleration.iloc[0][joint]]) 

        return (upperBoundsAcceleration, lowerBoundsAcceleration, 
                scalingAcceleration)
    
    def getBoundsActivation(self):
        lb = [0.05] 
        lb_vec = lb * len(self.muscles)
        ub = [1]
        ub_vec = ub * len(self.muscles)
        s = [1]
        s_vec = s * len(self.muscles)
        upperBoundsActivation = pd.DataFrame([ub_vec], columns=self.muscles)   
        lowerBoundsActivation = pd.DataFrame([lb_vec], columns=self.muscles) 
        scalingActivation = pd.DataFrame([s_vec], columns=self.muscles)
        upperBoundsActivation = upperBoundsActivation.div(scalingActivation)
        lowerBoundsActivation = lowerBoundsActivation.div(scalingActivation)
        for count, muscle in enumerate(self.muscles):
            upperBoundsActivation.insert(count + len(self.muscles), 
                                         muscle[:-1] + 'l', ub)
            lowerBoundsActivation.insert(count + len(self.muscles), 
                                         muscle[:-1] + 'l', lb)  

            # Scaling                       
            scalingActivation.insert(count + len(self.muscles), 
                                     muscle[:-1] + 'l', s)  
            upperBoundsActivation[
                    muscle[:-1] + 'l'] /= scalingActivation[muscle[:-1] + 'l']
            lowerBoundsActivation[
                    muscle[:-1] + 'l'] /= scalingActivation[muscle[:-1] + 'l']
        
        return upperBoundsActivation, lowerBoundsActivation, scalingActivation
    
    def getBoundsForce(self):
        lb = [0] 
        lb_vec = lb * len(self.muscles)
        ub = [5]
        ub_vec = ub * len(self.muscles)
        s = max([abs(lbi) for lbi in lb], [abs(ubi) for ubi in ub])
        s_vec = s * len(self.muscles)
        upperBoundsForce = pd.DataFrame([ub_vec], columns=self.muscles)   
        lowerBoundsForce = pd.DataFrame([lb_vec], columns=self.muscles) 
        scalingForce = pd.DataFrame([s_vec], columns=self.muscles)
        upperBoundsForce = upperBoundsForce.div(scalingForce)
        lowerBoundsForce = lowerBoundsForce.div(scalingForce)
        for count, muscle in enumerate(self.muscles):
            upperBoundsForce.insert(count + len(self.muscles), 
                                    muscle[:-1] + 'l', ub)
            lowerBoundsForce.insert(count + len(self.muscles), 
                                    muscle[:-1] + 'l', lb)  

            # Scaling                       
            scalingForce.insert(count + len(self.muscles), 
                                         muscle[:-1] + 'l', s)   
            upperBoundsForce[
                    muscle[:-1] + 'l'] /= scalingForce[muscle[:-1] + 'l']
            lowerBoundsForce[
                    muscle[:-1] + 'l'] /= scalingForce[muscle[:-1] + 'l']
        
        return upperBoundsForce, lowerBoundsForce, scalingForce
    
    def getBoundsActivationDerivative(self):
        activationTimeConstant = 0.015
        deactivationTimeConstant = 0.06
        lb = [-1 / deactivationTimeConstant] 
        lb_vec = lb * len(self.muscles)
        ub = [1 / activationTimeConstant]
        ub_vec = ub * len(self.muscles)
        s = [100]
        s_vec = s * len(self.muscles)
        upperBoundsActivationDerivative = pd.DataFrame([ub_vec], 
                                                       columns=self.muscles)   
        lowerBoundsActivationDerivative = pd.DataFrame([lb_vec], 
                                                       columns=self.muscles) 
        scalingActivationDerivative = pd.DataFrame([s_vec], 
                                                   columns=self.muscles)
        upperBoundsActivationDerivative = upperBoundsActivationDerivative.div(
                scalingActivationDerivative)
        lowerBoundsActivationDerivative = lowerBoundsActivationDerivative.div(
                scalingActivationDerivative)
        for count, muscle in enumerate(self.muscles):
            upperBoundsActivationDerivative.insert(count + len(self.muscles), 
                                                   muscle[:-1] + 'l', ub)
            lowerBoundsActivationDerivative.insert(count + len(self.muscles), 
                                                   muscle[:-1] + 'l', lb) 

            # Scaling                       
            scalingActivationDerivative.insert(count + len(self.muscles), 
                                               muscle[:-1] + 'l', s)  
            upperBoundsActivationDerivative[muscle[:-1] + 'l'] /= (
                    scalingActivationDerivative[muscle[:-1] + 'l'])
            lowerBoundsActivationDerivative[muscle[:-1] + 'l'] /= (
                    scalingActivationDerivative[muscle[:-1] + 'l'])             
        
        return (upperBoundsActivationDerivative, 
                lowerBoundsActivationDerivative, scalingActivationDerivative)
    
    def getBoundsForceDerivative(self):
        lb = [-100] 
        lb_vec = lb * len(self.muscles)
        ub = [100]
        ub_vec = ub * len(self.muscles)
        s = [100]
        s_vec = s * len(self.muscles)
        upperBoundsForceDerivative = pd.DataFrame([ub_vec], 
                                                  columns=self.muscles)   
        lowerBoundsForceDerivative = pd.DataFrame([lb_vec], 
                                                  columns=self.muscles) 
        scalingForceDerivative = pd.DataFrame([s_vec], 
                                                   columns=self.muscles)
        upperBoundsForceDerivative = upperBoundsForceDerivative.div(
                scalingForceDerivative)
        lowerBoundsForceDerivative = lowerBoundsForceDerivative.div(
                scalingForceDerivative)
        for count, muscle in enumerate(self.muscles):
            upperBoundsForceDerivative.insert(count + len(self.muscles), 
                                              muscle[:-1] + 'l', ub)
            lowerBoundsForceDerivative.insert(count + len(self.muscles), 
                                              muscle[:-1] + 'l', lb)   
            
            # Scaling                       
            scalingForceDerivative.insert(count + len(self.muscles), 
                                               muscle[:-1] + 'l', s)  
            upperBoundsForceDerivative[muscle[:-1] + 'l'] /= (
                    scalingForceDerivative[muscle[:-1] + 'l'])
            lowerBoundsForceDerivative[muscle[:-1] + 'l'] /= (
                    scalingForceDerivative[muscle[:-1] + 'l']) 
        
        return (upperBoundsForceDerivative, lowerBoundsForceDerivative, 
                scalingForceDerivative)
    
    def getBoundsTMExcitation(self, joints):
        lb = [-1] 
        lb_vec = lb * len(joints)
        ub = [1]
        ub_vec = ub * len(joints)
        s = [30]
        s_vec = s * len(joints)
        upperBoundsArmExcitation = pd.DataFrame([ub_vec], 
                                                columns=joints)   
        lowerBoundsArmExcitation = pd.DataFrame([lb_vec], 
                                                columns=joints)            
        scalingArmExcitation = pd.DataFrame([s_vec], columns=joints)
        
        return (upperBoundsArmExcitation, lowerBoundsArmExcitation,
                scalingArmExcitation)
    
    def getBoundsTMActivation(self, joints):
        lb = [-1] 
        lb_vec = lb * len(joints)
        ub = [1]
        ub_vec = ub * len(joints)
        s = [30]
        s_vec = s * len(joints)
        upperBoundsArmActivation = pd.DataFrame([ub_vec], 
                                                columns=joints)   
        lowerBoundsArmActivation = pd.DataFrame([lb_vec], 
                                                columns=joints) 
        scalingArmActivation = pd.DataFrame([s_vec], columns=joints)                  
        
        return (upperBoundsArmActivation, lowerBoundsArmActivation, 
                scalingArmActivation)
    
    def getBoundsFinalTime(self):
        upperBoundsFinalTime = pd.DataFrame([1], columns=['time'])   
        lowerBoundsFinalTime = pd.DataFrame([0.1], columns=['time'])  
        
        return upperBoundsFinalTime, lowerBoundsFinalTime
    
    def getUniformBoundsMarker(self, markers, markers_scaling,
                               dimensions = ["x", "y", "z"]):
        lb = [-1] 
        lb_vec = lb * len(markers) * len(dimensions)
        ub = [1]
        ub_vec = ub * len(markers) * len(dimensions)
        s = [markers_scaling]
        s_vec = s * len(markers) * len(dimensions)
        
        marker_titles = []
        for marker in markers:
            for dimension in dimensions:
                marker_titles.append(marker + '_' + dimension)   
        
        upperBoundsMarker = pd.DataFrame([ub_vec], columns=marker_titles)   
        lowerBoundsMarker = pd.DataFrame([lb_vec], columns=marker_titles) 
        scalingMarker = pd.DataFrame([s_vec], columns=marker_titles)                 
        
        return upperBoundsMarker, lowerBoundsMarker, scalingMarker 
    
    def getTreadmillSpecificBoundsMarker(self, markers, markers_scaling,
                                         dimensions = ["x", "y", "z"]):
        
        upperBoundsMarker_all = pd.DataFrame()   
        lowerBoundsMarker_all = pd.DataFrame()        
        
        s = [markers_scaling]
        
        upperBoundsMarker_all['Neck_x'] = [1.056]
        upperBoundsMarker_all['Neck_y'] = [1.431]
        upperBoundsMarker_all['Neck_z'] = [-0.369] 
        upperBoundsMarker_all['RShoulder_x'] = [1.072]
        upperBoundsMarker_all['RShoulder_y'] = [1.442]
        upperBoundsMarker_all['RShoulder_z'] = [-0.222]
        upperBoundsMarker_all['LShoulder_x'] = [1.072]
        upperBoundsMarker_all['LShoulder_y'] = [1.442]
        upperBoundsMarker_all['LShoulder_z'] = [-0.516]
        upperBoundsMarker_all['MidHip_x'] = [1.035]
        upperBoundsMarker_all['MidHip_y'] = [0.963]
        upperBoundsMarker_all['MidHip_z'] = [-0.411]
        upperBoundsMarker_all['RHip_x'] = [1.037]
        upperBoundsMarker_all['RHip_y'] = [0.969]
        upperBoundsMarker_all['RHip_z'] = [-0.315]
        upperBoundsMarker_all['LHip_x'] = [1.037]
        upperBoundsMarker_all['LHip_y'] = [0.969]
        upperBoundsMarker_all['LHip_z'] = [-0.508]
        upperBoundsMarker_all['RKnee_x'] = [1.278]
        upperBoundsMarker_all['RKnee_y'] = [0.602]
        upperBoundsMarker_all['RKnee_z'] = [-0.272]
        upperBoundsMarker_all['LKnee_x'] = [1.278]
        upperBoundsMarker_all['LKnee_y'] = [0.602]
        upperBoundsMarker_all['LKnee_z'] = [-0.542]
        upperBoundsMarker_all['RAnkle_x'] = [1.463]
        upperBoundsMarker_all['RAnkle_y'] = [0.369]
        upperBoundsMarker_all['RAnkle_z'] = [-0.321]
        upperBoundsMarker_all['LAnkle_x'] = [1.463]
        upperBoundsMarker_all['LAnkle_y'] = [0.369]
        upperBoundsMarker_all['LAnkle_z'] = [-0.532]
        upperBoundsMarker_all['RHeel_x'] = [1.444]
        upperBoundsMarker_all['RHeel_y'] = [0.416]
        upperBoundsMarker_all['RHeel_z'] = [-0.31]
        upperBoundsMarker_all['LHeel_x'] = [1.444]
        upperBoundsMarker_all['LHeel_y'] = [0.416]
        upperBoundsMarker_all['LHeel_z'] = [-0.517]
        upperBoundsMarker_all['RBigToe_x'] = [1.664]
        upperBoundsMarker_all['RBigToe_y'] = [0.215]
        upperBoundsMarker_all['RBigToe_z'] = [-0.326]
        upperBoundsMarker_all['LBigToe_x'] = [1.664]
        upperBoundsMarker_all['LBigToe_y'] = [0.215]
        upperBoundsMarker_all['LBigToe_z'] = [-0.528]
        upperBoundsMarker_all['RSmallToe_x'] = [1.62]
        upperBoundsMarker_all['RSmallToe_y'] = [0.225]
        upperBoundsMarker_all['RSmallToe_z'] = [-0.259]
        upperBoundsMarker_all['LSmallToe_x'] = [1.62]
        upperBoundsMarker_all['LSmallToe_y'] = [0.225]
        upperBoundsMarker_all['LSmallToe_z'] = [-0.603]
        
        lowerBoundsMarker_all['Neck_x'] = [0.735]
        lowerBoundsMarker_all['Neck_y'] = [1.348]
        lowerBoundsMarker_all['Neck_z'] = [-0.588] 
        lowerBoundsMarker_all['RShoulder_x'] = [0.699]
        lowerBoundsMarker_all['RShoulder_y'] = [1.335]
        lowerBoundsMarker_all['RShoulder_z'] = [-0.444]
        lowerBoundsMarker_all['LShoulder_x'] = [0.699]
        lowerBoundsMarker_all['LShoulder_y'] = [1.335]
        lowerBoundsMarker_all['LShoulder_z'] = [-0.732]
        lowerBoundsMarker_all['MidHip_x'] = [0.708]
        lowerBoundsMarker_all['MidHip_y'] = [0.883]
        lowerBoundsMarker_all['MidHip_z'] = [-0.589]
        lowerBoundsMarker_all['RHip_x'] = [0.7]
        lowerBoundsMarker_all['RHip_y'] = [0.88]
        lowerBoundsMarker_all['RHip_z'] = [-0.493]
        lowerBoundsMarker_all['LHip_x'] = [0.7]
        lowerBoundsMarker_all['LHip_y'] = [0.88]
        lowerBoundsMarker_all['LHip_z'] = [-0.685]
        lowerBoundsMarker_all['RKnee_x'] = [0.501]
        lowerBoundsMarker_all['RKnee_y'] = [0.456]
        lowerBoundsMarker_all['RKnee_z'] = [-0.448]
        lowerBoundsMarker_all['LKnee_x'] = [0.501]
        lowerBoundsMarker_all['LKnee_y'] = [0.456]
        lowerBoundsMarker_all['LKnee_z'] = [-0.669]
        lowerBoundsMarker_all['RAnkle_x'] = [0.182]
        lowerBoundsMarker_all['RAnkle_y'] = [0.078]
        lowerBoundsMarker_all['RAnkle_z'] = [-0.454]
        lowerBoundsMarker_all['LAnkle_x'] = [0.182]
        lowerBoundsMarker_all['LAnkle_y'] = [0.078]
        lowerBoundsMarker_all['LAnkle_z'] = [-0.651]
        lowerBoundsMarker_all['RHeel_x'] = [0.083]
        lowerBoundsMarker_all['RHeel_y'] = [0.003]
        lowerBoundsMarker_all['RHeel_z'] = [-0.466]
        lowerBoundsMarker_all['LHeel_x'] = [0.083]
        lowerBoundsMarker_all['LHeel_y'] = [0.003]
        lowerBoundsMarker_all['LHeel_z'] = [-0.664]
        lowerBoundsMarker_all['RBigToe_x'] = [0.179]
        lowerBoundsMarker_all['RBigToe_y'] = [0.018]
        lowerBoundsMarker_all['RBigToe_z'] = [-0.454]
        lowerBoundsMarker_all['LBigToe_x'] = [0.179]
        lowerBoundsMarker_all['LBigToe_y'] = [0.018]
        lowerBoundsMarker_all['LBigToe_z'] = [-0.673]
        lowerBoundsMarker_all['RSmallToe_x'] = [0.141]
        lowerBoundsMarker_all['RSmallToe_y'] = [0.013]
        lowerBoundsMarker_all['RSmallToe_z'] = [-0.384]
        lowerBoundsMarker_all['LSmallToe_x'] = [0.141]
        lowerBoundsMarker_all['LSmallToe_y'] = [0.013]
        lowerBoundsMarker_all['LSmallToe_z'] = [-0.738]    
        
        s_vec = s * len(markers) * len(dimensions)
        
        marker_titles = []
        for marker in markers:
            for dimension in dimensions:
                marker_titles.append(marker + '_' + dimension) 
                
        upperBoundsMarker = pd.DataFrame()   
        lowerBoundsMarker = pd.DataFrame() 
        
        for count, marker in enumerate(marker_titles):
             upperBoundsMarker.insert(
                 count, marker, [upperBoundsMarker_all.iloc[0][marker] / s[0]])
             lowerBoundsMarker.insert(
                 count, marker, [lowerBoundsMarker_all.iloc[0][marker] / s[0]])
             
        scalingMarker = pd.DataFrame([s_vec], columns=marker_titles)  

        return upperBoundsMarker, lowerBoundsMarker, scalingMarker
    
    def getBoundsOffset(self, scaling):
        upperBoundsOffset = pd.DataFrame([0.05] / scaling, columns=['offset_y']) 
        lowerBoundsOffset = pd.DataFrame([-0.05] / scaling, columns=['offset_y'])
        
        return upperBoundsOffset, lowerBoundsOffset
    
    def getBoundsMultipliers(self, NHolConstraints):
        
        lb = [-1] 
        lb_vec = lb * NHolConstraints
        ub = [1]
        ub_vec = ub * NHolConstraints
        s = [1000]
        s_vec = s * NHolConstraints
        
        holConstraints_titles = []
        for count in range(NHolConstraints):
            holConstraints_titles.append('hol_constraint_' + str(count)) 
            
        upperBoundsHolConstraints = pd.DataFrame([ub_vec],
                                                 columns=holConstraints_titles)   
        lowerBoundsHolConstraints = pd.DataFrame([lb_vec],
                                                 columns=holConstraints_titles) 
        scalingHolConstraints = pd.DataFrame([s_vec],
                                             columns=holConstraints_titles) 
        
        return (upperBoundsHolConstraints, lowerBoundsHolConstraints,
                scalingHolConstraints)
    
    def getBoundsVelCorrs(self, NHolConstraints):
        
        lb = [-1] 
        lb_vec = lb * NHolConstraints
        ub = [1]
        ub_vec = ub * NHolConstraints
        s = [0.1]
        s_vec = s * NHolConstraints
        
        velCorrs_titles = []
        for count in range(NHolConstraints):
            velCorrs_titles.append('hol_constraint_' + str(count)) 
            
        upperBoundsVelCorrs = pd.DataFrame([ub_vec],
                                                 columns=velCorrs_titles)   
        lowerBoundsVelCorrs = pd.DataFrame([lb_vec],
                                                 columns=velCorrs_titles) 
        scalingVelCorrs = pd.DataFrame([s_vec],
                                             columns=velCorrs_titles) 
        
        return upperBoundsVelCorrs, lowerBoundsVelCorrs, scalingVelCorrs
    
    def getBoundsAngVel(self, imus):
        lb = [-1] 
        lb_vec = lb * len(imus)
        ub = [1]
        ub_vec = ub * len(imus)
        s = [10]
        s_vec = s * len(imus)
        upperBoundsAngVel = pd.DataFrame([ub_vec], columns=imus)   
        lowerBoundsAngVel = pd.DataFrame([lb_vec], columns=imus)   
        scalingAngVel = pd.DataFrame([s_vec], columns=imus)               
        
        return (upperBoundsAngVel, lowerBoundsAngVel, scalingAngVel)
    
    def getBoundsLinAcc(self, imus):
        lb = [-1] 
        lb_vec = lb * len(imus)
        ub = [1]
        ub_vec = ub * len(imus)
        s = [30]
        s_vec = s * len(imus)
        upperBoundsLinAcc = pd.DataFrame([ub_vec], columns=imus)   
        lowerBoundsLinAcc = pd.DataFrame([lb_vec], columns=imus)  
        scalingLinAcc = pd.DataFrame([s_vec], columns=imus)               
        
        return (upperBoundsLinAcc, lowerBoundsLinAcc, scalingLinAcc)
    
    def getBoundsXYZ(self, imus):
        lb = [-1] 
        lb_vec = lb * len(imus)
        ub = [1]
        ub_vec = ub * len(imus)
        s = [np.pi]
        s_vec = s * len(imus)
        upperBoundsXYZ = pd.DataFrame([ub_vec], columns=imus)   
        lowerBoundsXYZ = pd.DataFrame([lb_vec], columns=imus)  
        scalingXYZ = pd.DataFrame([s_vec], columns=imus)               
        
        return (upperBoundsXYZ, lowerBoundsXYZ, scalingXYZ) 
    