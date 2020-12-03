from sys import path
import os
if os.environ['COMPUTERNAME'] == 'GBW-L-W2003':
    path.append(r"C:/Users/u0101727/Documents/Software/CasADi/casadi-windows-py37-v3.5.1-64bit")
    pathOS = "C:/Users/u0101727/Documents/MyRepositories/opensim-fork/install/sdk/Python"
elif os.environ['COMPUTERNAME'] == 'GBW-D-W0529':
    path.append(r"D:/u0101727/MySoftware/casadi-windows-py37-v3.5.1-64bit")
    pathOS = "C:/OpenSim_4.1/sdk/Python"
elif os.environ['COMPUTERNAME'] == 'GBW-D-W2711':
    path.append(r"C:/Users/Public/Documents/Software/casadi-windows-py37-v3.5.1-64bit")
    pathOS = "C:/OpenSim_4.1/sdk/Python"
import casadi as ca
import numpy as np
import copy

# User settings
# run_options = [True, True, True, True, True, True, True, True, False, False]
run_options = [False, False, True, True, True, True, True, True, True, False]

solveProblem = run_options[0]
saveResults = run_options[1]
analyzeResults = run_options[2]
loadResults = run_options[3]
writeMotionFile = run_options[4]
writeGRF = run_options[5]
visualizeTracking = run_options[6]
decomposeCost = run_options[7]
visualizeSimulationResults = run_options[8]
visualizeConstraintErrors = run_options[9]

cases = ["6"]

# loadMTParameters = True 
# loadPolynomialData = True
plotPolynomials = False
plotGuessVsBounds = False
visualizeResultsAgainstBounds = True
plotMarkerTrackingAtInitialGuess = False
writeIMUFile = True

# Numerical Settings
tol = 4
d = 3
NThreads = 20
parallelMode = "thread"

from settings import getSettings     
settings = getSettings() 
# from settings import getSubjectData     
# subjectData = getSubjectData() 

for case in cases:
    # Weights in cost function
    weights = {
        'activationTerm': settings[case]['w_activationTerm'],
        'jointAccelerationTerm': settings[case]['w_jointAccelerationTerm'],
        'actJExcitationTerm': settings[case]['w_actJExcitationTerm'], 
        'gtJExcitationTerm': settings[case]['w_gtJExcitationTerm'], 
        'lambdaTerm': settings[case]['w_lambdaTerm'],
        'gammaTerm': settings[case]['w_gammaTerm'],
        'trackingTerm': settings[case]['w_trackingTerm']}   
    
    # Other settings
    subjectID = settings[case]['subjectID']
    subject = "subject" + subjectID
    model = subject + "_" + settings[case]['model']
    trial = settings[case]['trial']
    timeInterval = settings[case]['timeInterval']
    filter_coordinates_toTrack = settings[case]['filter_coordinates_toTrack']    
    tracking_data = settings[case]['tracking_data']
    N = settings[case]['N']    
    guessType = settings[case]['guessType']  
    timeElapsed = timeInterval[1] - timeInterval[0]
    guess_zeroVelocity = settings[case]['guess_zeroVelocity']
    guess_zeroAcceleration = settings[case]['guess_zeroAcceleration']
    velocity_correction = settings[case]['velocity_correction']
    constraint_pos = settings[case]['constraint_pos']
    constraint_vel = settings[case]['constraint_vel']
    constraint_acc = settings[case]['constraint_acc']
    
    norm_std = False
    if tracking_data == "markers":
        markers_toTrack = settings[case]['markers_toTrack']
        norm_std = settings[case]['norm_std']
        # boundsMarker = settings[case]['boundsMarker']
        # markers_as_controls = settings[case]['markers_as_controls']
        # markers_scaling = settings[case]['markers_scaling']
    
    if tracking_data == "coordinates":
        coordinates_toTrack = settings[case]['coordinates_toTrack']  
        if coordinates_toTrack['translational']:
            TrCoordinates_toTrack_Bool = True
            weights['trackingTerm_tr'] = settings[case]['w_trackingTerm_tr']
        else:
            TrCoordinates_toTrack_Bool = False     
        
    tgrid = np.linspace(timeInterval[0], timeInterval[1], N+1)
    tgridf = np.zeros((1, N+1))
    tgridf[:,:] = tgrid.T

    # Paths
    pathMain = os.getcwd()
    pathSubject = os.path.join(pathMain, 'OpenSim', subject)
    pathModels = os.path.join(pathSubject, 'Models')
    pathOpenSimModel = os.path.join(pathModels, model + ".osim")    
    # pathCoordinates = os.path.join(pathSubject, 'MA', 'dummy_motion' 
    #                                + KA_suffix + '.mot')
    # pathMuscleAnalysis = os.path.join(pathSubject, 'MA', 'ResultsMA', 
    #                                   subject + "_scaled" + KA_suffix,
    #                                   'subject01_MuscleAnalysis_')
    pathTRC = os.path.join(pathSubject, 'TRC', trial + ".trc")
    pathExternalFunctions = os.path.join(pathMain, 'ExternalFunctions')
    pathIKFolder = os.path.join(pathSubject, 'IK', model)
    # pathGRFFolder = os.path.join(pathSubject, 'GRF')
    
    filename = os.path.basename(__file__)
    pathCase = 'Case_' + case
    pathResults = os.path.join(pathMain, 'Results', filename[:-3], pathCase)  
    if not os.path.exists(pathResults):
        os.makedirs(pathResults)     
    
    # %% Muscle part
    '''
    muscles = ['glut_med1_r', 'glut_med2_r', 'glut_med3_r', 'glut_min1_r', 
               'glut_min2_r', 'glut_min3_r', 'semimem_r', 'semiten_r',
               'bifemlh_r', 'bifemsh_r', 'sar_r', 'add_long_r', 'add_brev_r',
               'add_mag1_r', 'add_mag2_r', 'add_mag3_r', 'tfl_r', 'pect_r',
               'grac_r', 'glut_max1_r', 'glut_max2_r', 'glut_max3_r',
               'iliacus_r', 'psoas_r', 'quad_fem_r', 'gem_r', 'peri_r',
               'rect_fem_r', 'vas_med_r', 'vas_int_r', 'vas_lat_r',
               'med_gas_r', 'lat_gas_r', 'soleus_r', 'tib_post_r',
               'flex_dig_r', 'flex_hal_r', 'tib_ant_r', 'per_brev_r',
               'per_long_r', 'per_tert_r', 'ext_dig_r', 'ext_hal_r',
               'ercspn_r', 'intobl_r', 'extobl_r', 'ercspn_l', 'intobl_l',
               'extobl_l']
    rightSideMuscles = muscles[:-3]
    leftSideMuscles = [muscle[:-1] + 'l' for muscle in rightSideMuscles]
    bothSidesMuscles = leftSideMuscles + rightSideMuscles
    NMuscles = len(bothSidesMuscles)
    NSideMuscles = len(rightSideMuscles)
    
    from muscleData import getMTParameters
    sideMtParameters = getMTParameters(pathOS, pathOpenSimModel,
                                       rightSideMuscles,
                                       loadMTParameters, pathModels, model)
    mtParameters = np.concatenate((sideMtParameters, sideMtParameters), axis=1)
    
    from muscleData import tendonCompliance
    sideTendonCompliance = tendonCompliance(NSideMuscles)
    tendonCompliance = np.concatenate((sideTendonCompliance, 
                                       sideTendonCompliance), axis=1)
    
    from muscleData import tendonShift
    sideTendonShift = tendonShift(NSideMuscles)
    tendonShift = np.concatenate((sideTendonShift, sideTendonShift), axis=1)
    
    from muscleData import specificTension_3D
    sideSpecificTension = specificTension_3D(rightSideMuscles)
    specificTension = np.concatenate((sideSpecificTension, 
                                      sideSpecificTension), axis=1)
    
    from functionCasADi import hillEquilibrium
    f_hillEquilibrium = hillEquilibrium(mtParameters, tendonCompliance, 
                                        tendonShift, specificTension)
    # Time constants
    activationTimeConstant = 0.015
    deactivationTimeConstant = 0.06
    # Symmetry
    idxSymmetricMuscles = (list(range(NSideMuscles, NMuscles)) + 
                           list(range(0, NSideMuscles)))
    '''
    
    # %% Joints
    from variousFunctions import getJointIndices
    joints = ['ground_thorax_rot_x', 'ground_thorax_rot_y',
              'ground_thorax_rot_z', 'clav_prot', 'clav_elev',
              'scapula_abduction', 'scapula_elevation', 'scapula_upward_rot', 
              'scapula_winging', 'plane_elv', 'shoulder_elv', 'axial_rot',
              'elbow_flexion', 'pro_sup']
    NJoints = len(joints)
    # Rotational degrees of freedom
    rotationalJoints = ['ground_thorax_rot_x', 'ground_thorax_rot_y',
                        'ground_thorax_rot_z', 'clav_prot', 'clav_elev',
                        'scapula_abduction', 'scapula_elevation',
                        'scapula_upward_rot', 'scapula_winging', 'plane_elv',
                        'shoulder_elv', 'axial_rot', 'elbow_flexion',
                        'pro_sup']      
    idxRotationalJoints = getJointIndices(joints, rotationalJoints)
    # # Translational degrees of freedom
    # translationalJoints = ['ground_thorax_rot_tx', 'ground_thorax_rot_ty',
    #                        'ground_thorax_rot_tz'] 
    # Ground thorax joints
    groundThoraxJoints = ['ground_thorax_rot_x', 'ground_thorax_rot_y',
                          'ground_thorax_rot_z']
    idxGroundThoraxJoints = getJointIndices(joints, groundThoraxJoints)
    NGroundThoraxJoints = len(groundThoraxJoints)
    # Actuated joints (ideal torque motors or muscles)
    actJoints = copy.deepcopy(joints)
    for groundThoraxJoint in groundThoraxJoints:
        actJoints.remove(groundThoraxJoint)
    idxActJoints = getJointIndices(joints, actJoints)
    NActJoints = len(actJoints)
        
    # %% Ideal torque motor dynamics
    from functionCasADi import torqueMotorDynamics
    f_actJointsDynamics = torqueMotorDynamics(NActJoints)
    f_groundThoraxJointsDynamics = torqueMotorDynamics(NGroundThoraxJoints)
    
    # %% Polynomials
    '''
    from functionCasADi import polynomialApproximation
    leftPolynomialJoints = ['hip_flexion_l', 'hip_adduction_l', 
                            'hip_rotation_l', 'knee_angle_l',
                            'knee_adduction_l', 'ankle_angle_l',
                            'subtalar_angle_l', 'mtp_angle_l', 
                            'lumbar_extension', 'lumbar_bending',
                            'lumbar_rotation'] 
    rightPolynomialJoints = ['hip_flexion_r', 'hip_adduction_r', 
                             'hip_rotation_r', 'knee_angle_r',
                             'knee_adduction_r', 'ankle_angle_r',
                             'subtalar_angle_r', 'mtp_angle_r',
                             'lumbar_extension', 'lumbar_bending',
                             'lumbar_rotation'] 
    if not knee_adduction:
        leftPolynomialJoints.remove('knee_adduction_l')
        rightPolynomialJoints.remove('knee_adduction_r') 
    
    from muscleData import getPolynomialData      
    polynomialData = getPolynomialData(loadPolynomialData, pathModels, model,
                                       pathCoordinates, pathMuscleAnalysis,
                                       rightPolynomialJoints, muscles)        
    if loadPolynomialData:
        polynomialData = polynomialData.item()
    
    NPolynomials = len(leftPolynomialJoints)
    f_polynomial = polynomialApproximation(muscles, polynomialData,
                                           NPolynomials)
    leftPolynomialJointIndices = getJointIndices(joints, leftPolynomialJoints)
    rightPolynomialJointIndices = getJointIndices(joints,rightPolynomialJoints)    
    leftPolynomialMuscleIndices = list(range(43)) +  list(range(46, 49))
    rightPolynomialMuscleIndices = list(range(46))
    from variousFunctions import getMomentArmIndices
    momentArmIndices = getMomentArmIndices(rightSideMuscles,
                                           leftPolynomialJoints,
                                           rightPolynomialJoints, 
                                           polynomialData)
    trunkMomentArmPolynomialIndices = list(range(46, 49)) + list(range(43, 46))
    
    from functionCasADi import sumProd
    f_NHipSumProd = sumProd(len(momentArmIndices['hip_flexion_r']))
    f_NKneeSumProd = sumProd(len(momentArmIndices['knee_angle_r']))
    f_NAnkleSumProd = sumProd(len(momentArmIndices['ankle_angle_r']))
    f_NSubtalarSumProd = sumProd(len(momentArmIndices['subtalar_angle_r']))
    f_NTrunkSumProd = sumProd(len(momentArmIndices['lumbar_extension']))
    
    # Test polynomials
    if plotPolynomials:
        from polynomials import testPolynomials
        momentArms = testPolynomials(pathCoordinates, pathMuscleAnalysis, 
                                      rightPolynomialJoints, muscles, 
                                      f_polynomial, polynomialData, 
                                      momentArmIndices,
                                      trunkMomentArmPolynomialIndices)
    
#     # %% Metabolic energy model    
#     """
#     maximalIsometricForce = mtParameters[0, :]
#     optimalFiberLength = mtParameters[1, :]
#     muscleVolume = np.multiply(maximalIsometricForce, optimalFiberLength)
#     muscleMass = np.divide(np.multiply(muscleVolume, 1059.7), 
#                            np.multiply(specificTension[0, :].T, 1e6))
#     from muscleData import slowTwitchRatio_3D
#     sideSlowTwitchRatio = slowTwitchRatio_3D(rightSideMuscles)
#     slowTwitchRatio = (np.concatenate((sideSlowTwitchRatio, 
#                                       sideSlowTwitchRatio), axis=1))[0, :].T
#     smoothingConstant = 10
#     from functionCasADi import metabolicsBhargava
#     f_metabolicsBhargava = metabolicsBhargava(slowTwitchRatio, 
#                                               maximalIsometricForce,
#                                               muscleMass, 
#                                               smoothingConstant)
#     """
    
    # %% Passive joint torques
    from functionCasADi import passiveJointTorque
    from muscleData import passiveJointTorqueData_3D
    if genericMTPTorqueLimits:
        lbMTPTorqueLimits = passiveJointTorqueData_3D('mtp_angle_r')[1][0]
        ubMTPTorqueLimits = passiveJointTorqueData_3D('mtp_angle_r')[1][1]
    else:
        lbMTPTorqueLimits = -1.134464013796314 
        ubMTPTorqueLimits = passiveJointTorqueData_3D('mtp_angle_r')[1][1]
    rangeMTPTorqueLimits = [lbMTPTorqueLimits, ubMTPTorqueLimits]
    if genericSubtalarTorqueLimits:
        lbSubtalarTorqueLimits = (
            passiveJointTorqueData_3D('subtalar_angle_r')[1][0])
        ubSubtalarTorqueLimits = (
            passiveJointTorqueData_3D('subtalar_angle_r')[1][1])
    else:
        lbSubtalarTorqueLimits = -1
        ubSubtalarTorqueLimits = 1
    rangeSubtalarTorqueLimits = [lbSubtalarTorqueLimits, 
                                 ubSubtalarTorqueLimits]        
    
    damping = 0.1
    f_passiveJointTorque_hip_flexion = passiveJointTorque(
            passiveJointTorqueData_3D('hip_flexion_r')[0],
            passiveJointTorqueData_3D('hip_flexion_r')[1], damping)
    f_passiveJointTorque_hip_adduction = passiveJointTorque(
            passiveJointTorqueData_3D('hip_adduction_r')[0],
            passiveJointTorqueData_3D('hip_adduction_r')[1], damping)
    f_passiveJointTorque_hip_rotation = passiveJointTorque(
            passiveJointTorqueData_3D('hip_rotation_r')[0],
            passiveJointTorqueData_3D('hip_rotation_r')[1], damping)
    f_passiveJointTorque_knee_angle = passiveJointTorque(
            passiveJointTorqueData_3D('knee_angle_r')[0],
            passiveJointTorqueData_3D('knee_angle_r')[1], damping)
    f_passiveJointTorque_ankle_angle = passiveJointTorque(
            passiveJointTorqueData_3D('ankle_angle_r')[0],
            passiveJointTorqueData_3D('ankle_angle_r')[1], damping)
    f_passiveJointTorque_subtalar_angle = passiveJointTorque(
            passiveJointTorqueData_3D('subtalar_angle_r')[0],
            rangeSubtalarTorqueLimits, damping)
    f_passiveJointTorque_mtp_angle = passiveJointTorque(
            passiveJointTorqueData_3D('mtp_angle_r')[0],
            rangeMTPTorqueLimits, damping)
    f_passiveJointTorque_lumbar_extension = passiveJointTorque(
            passiveJointTorqueData_3D('lumbar_extension')[0],
            passiveJointTorqueData_3D('lumbar_extension')[1], damping)
    f_passiveJointTorque_lumbar_bending = passiveJointTorque(
            passiveJointTorqueData_3D('lumbar_bending')[0],
            passiveJointTorqueData_3D('lumbar_bending')[1], damping)
    f_passiveJointTorque_lumbar_rotation = passiveJointTorque(
            passiveJointTorqueData_3D('lumbar_rotation')[0],
            passiveJointTorqueData_3D('lumbar_rotation')[1], damping)
    
    from functionCasADi import passiveTorqueActuatedJointTorque
    stiffnessMtp = 0.25
    dampingMtp = 0.4
    f_linearPassiveMtpTorque = passiveTorqueActuatedJointTorque(stiffnessMtp,
                                                                dampingMtp)
    '''
    
    # %% Marker data
    NVec3 = 3
    dimensions = ['x', 'y', 'z']    
    from variousFunctions import interpolateDataFrame
    if tracking_data == "markers":
        from variousFunctions import getFromTRC
        NMarkers = len(markers_toTrack)
        marker_data = getFromTRC(pathTRC, markers_toTrack)        
        marker_data_interp = interpolateDataFrame(marker_data, 
                                                  timeInterval[0], 
                                                  timeInterval[1], N)        
        NEl_toTrack = NVec3 * NMarkers
        marker_titles = []
        for marker in markers_toTrack:
            for dimension in dimensions:
                marker_titles.append(marker + '_' + dimension)
        
    # %% Kinematic data
    pathIK = os.path.join(pathIKFolder, 'IK_' + trial + '.mot')
    from variousFunctions import getIK
    # Extract joint positions from walking trial.
    Qs_fromIK, Qs_fromIK_filt = getIK(pathIK, joints) 
    if filter_coordinates_toTrack:
        Qs_fromIK_interp = interpolateDataFrame(
            Qs_fromIK_filt, timeInterval[0], timeInterval[1], N+1)
    else:
        Qs_fromIK_interp = interpolateDataFrame(
            Qs_fromIK, timeInterval[0], timeInterval[1], N+1)
    if tracking_data == "coordinates":        
        # Rotational DOFs
        NEl_toTrack = len(coordinates_toTrack['rotational'])
        idxRotCoordinates_toTrack = getJointIndices(
            joints, coordinates_toTrack['rotational'])
        # Translational DOFs
        if coordinates_toTrack['translational']:
            NEl_toTrack_tr = len(coordinates_toTrack['translational'])
            idxTrCoordinates_toTrack = getJointIndices(
                joints, coordinates_toTrack['translational'])
    
    # %% Load external functions
    NHolConstraints = 3
    holConstraints_titles = []
    for count in range(NHolConstraints):
        holConstraints_titles.append('hol_constraint_' + str(count))     
    NVelCorrs = 6 # clavicle and scapula mobilities
    os.chdir(pathExternalFunctions)
    if tracking_data == "markers":
        print("Not supported") 
    elif tracking_data == "coordinates":
        if velocity_correction:
            if constraint_pos and constraint_vel and constraint_acc:
                F = ca.external('F','Shoulder_' + subject + '_test0.dll')  
                NKinConstraints = 3*NHolConstraints
                NOutput_F = NJoints + NKinConstraints + NVelCorrs                
            elif constraint_pos and constraint_vel and not constraint_acc:
                F = ca.external('F','Shoulder_' + subject + '_test1.dll')  
                NKinConstraints = 2*NHolConstraints
                NOutput_F = NJoints + NKinConstraints + NVelCorrs                 
            elif constraint_pos and not constraint_vel and not constraint_acc:
                F = ca.external('F','Shoulder_' + subject + '_test2.dll')  
                NKinConstraints = 1*NHolConstraints
                NOutput_F = NJoints + NKinConstraints + NVelCorrs              
        else:
            if constraint_pos and constraint_vel and constraint_acc:
                F = ca.external('F','Shoulder_' + subject + '_test3.dll')  
                NKinConstraints = 3*NHolConstraints
                NOutput_F = NJoints + NKinConstraints                
            elif constraint_pos and constraint_vel and not constraint_acc:
                F = ca.external('F','Shoulder_' + subject + '_test4.dll')  
                NKinConstraints = 2*NHolConstraints
                NOutput_F = NJoints + NKinConstraints                 
            elif constraint_pos and not constraint_vel and not constraint_acc:
                F = ca.external('F','Shoulder_' + subject + '_test5.dll')  
                NKinConstraints = 1*NHolConstraints
                NOutput_F = NJoints + NKinConstraints   
        if analyzeResults:
            if velocity_correction:
                F1 = ca.external('F','Shoulder_' + subject + '_pp.dll')  
                NOutput_F1 = NJoints + 3*NHolConstraints + NVelCorrs + 4*3 
                
    os.chdir(pathMain)
    '''
    vec_in_1 = -np.ones((14*2, 1))
    vec_in_2 = -np.ones((14, 1))
    vec_in_3 = -np.ones((3, 1))
    vec_in_4 = -np.ones((3, 1))
    vec_in = np.concatenate((vec_in_1,vec_in_2,vec_in_3,vec_in_4))
    vec_out = (F(vec_in)).full()  
    '''    
    # Helper indices.    
    idxKinConstraints = {}
    idxKinConstraints["Position"] = list(
            range(NJoints, NJoints + NHolConstraints))
    idxKinConstraints["Velocity"] = list(
            range(1 + idxKinConstraints["Position"][-1], 
                  1 + idxKinConstraints["Position"][-1] + NHolConstraints))
    idxKinConstraints["Acceleration"] = list(
            range(1 + idxKinConstraints["Velocity"][-1], 
                  1 + idxKinConstraints["Velocity"][-1] + NHolConstraints)) 
    
    if constraint_pos:        
        idxKinConstraints["applied"] = idxKinConstraints["Position"] 
    if constraint_vel:        
        idxKinConstraints["applied"] = (idxKinConstraints["Position"]  + 
                                        idxKinConstraints["Velocity"]) 
    if constraint_acc:        
        idxKinConstraints["applied"] = (idxKinConstraints["Position"] + 
                                        idxKinConstraints["Velocity"] + 
                                        idxKinConstraints["Acceleration"])  
    idxKinConstraints["all"] = (idxKinConstraints["Position"] + 
                                idxKinConstraints["Velocity"] + 
                                idxKinConstraints["Acceleration"])         
    
    idxVelCorrs = {}
    if velocity_correction:
        idxVelCorrs["applied"] = list(
            range(1 + idxKinConstraints["applied"][-1], 
                  1 + idxKinConstraints["applied"][-1] + NVelCorrs)) 
        # Joints for which the velocity correction should be applied.    
        jointVelCorrs = ['clav_prot', 'clav_elev', 'scapula_abduction',
                         'scapula_elevation', 'scapula_upward_rot', 
                         'scapula_winging']
        jointNoVelCorrs = copy.deepcopy(joints)
        for jointVelCorr in jointVelCorrs:    
            jointNoVelCorrs.remove(jointVelCorr)    
        idxJointVelCorr = getJointIndices(joints, jointVelCorrs)
        idxNoJointVelCorr = getJointIndices(joints, jointNoVelCorrs)
    idxVelCorrs["all"] = list(
            range(1 + idxKinConstraints["all"][-1], 
                  1 + idxKinConstraints["all"][-1] + NVelCorrs)) 
    
    idxStations = {}
    idxStations["clavicle"] = list(
            range(1 + idxVelCorrs["all"][-1], 
                  1 + idxVelCorrs["all"][-1] + 3))
    idxStations["scapula"] = list(
            range(1 + idxStations["clavicle"][-1], 
                  1 + idxStations["clavicle"][-1] + 3))
    idxStations["all"] = idxStations["clavicle"] + idxStations["scapula"]
    
    idxIMUs = {}
    idxIMUs["radius"] = {}
    idxIMUs["radius"]["angVel"] = list(
            range(1 + idxStations["all"][-1], 
                  1 + idxStations["all"][-1] + 3))
    idxIMUs["radius"]["linAcc"] = list(
            range(1 + idxIMUs["radius"]["angVel"][-1], 
                  1 + idxIMUs["radius"]["angVel"][-1] + 3))
    idxIMUs["radius"]["all"] = (idxIMUs["radius"]["angVel"] + 
                                idxIMUs["radius"]["linAcc"])    
   
    # %% CasADi helper functions
    from functionCasADi import normSumPow
    from functionCasADi import diffTorques
    from functionCasADi import mySum
    from functionCasADi import normSqrtDiff
    # f_NMusclesSum2 = normSumPow(NMuscles, 2)
    f_NGroundThoraxJointsSum2 = normSumPow(NGroundThoraxJoints, 2)
    f_NActJointsSum2 = normSumPow(NActJoints, 2)
    f_NJointsSum2 = normSumPow(NJoints, 2)
    f_NHolConstraintsSum2 = normSumPow(NHolConstraints, 2)
    f_diffTorques = diffTorques()
    f_mySum = mySum(N)        
    if norm_std and tracking_data == "markers":
        from functionCasADi import normSqrtDiffStd
        f_normSqrtDiffStd_tracking = normSqrtDiffStd(NEl_toTrack)
    else:
        f_normSqrtDiff_tracking = normSqrtDiff(NEl_toTrack)
    if TrCoordinates_toTrack_Bool:
        f_normSqrtDiff_tracking_b = normSqrtDiff(NEl_toTrack_tr)        
        
    # %% Functions for tracking terms 
    if norm_std and tracking_data == "markers":
        dataToTrackk = ca.MX.sym('dataToTrackk', NEl_toTrack) 
        dataToTrack_expk = ca.MX.sym('dataToTrack_expk', NEl_toTrack)   
        dataToTrack_exp_stdk = ca.MX.sym('dataToTrack_exp_stdk', NEl_toTrack)  
        dataTrackingTerm = f_normSqrtDiffStd_tracking(dataToTrackk,
                                                      dataToTrack_expk,
                                                      dataToTrack_exp_stdk) 
        f_track_k = ca.Function('f_track_k', [dataToTrackk, dataToTrack_expk,
                                              dataToTrack_exp_stdk],
                                [dataTrackingTerm])
        f_track_k_map = f_track_k.map(N, parallelMode, NThreads)        
    else:
        dataToTrackk = ca.MX.sym('dataToTrackk', NEl_toTrack) 
        dataToTrack_expk = ca.MX.sym('dataToTrack_expk', NEl_toTrack)    
        dataTrackingTerm = f_normSqrtDiff_tracking(dataToTrackk,
                                                   dataToTrack_expk) 
        f_track_k = ca.Function('f_track_k', [dataToTrackk, dataToTrack_expk],
                                [dataTrackingTerm])
        if tracking_data == "markers":
            f_track_k_map = f_track_k.map(N, parallelMode, NThreads)
        elif tracking_data == "coordinates":
            f_mySumTrack = mySum(N+1)
            f_track_k_map = f_track_k.map(N+1, parallelMode, NThreads)
    if TrCoordinates_toTrack_Bool:
        dataToTrack_trk = ca.MX.sym('dataToTrack_trk', NEl_toTrack_tr) 
        dataToTrack_tr_expk = ca.MX.sym('dataToTrack_tr_expk', NEl_toTrack_tr)    
        dataTrackingTerm_tr = f_normSqrtDiff_tracking_b(dataToTrack_trk,
                                                        dataToTrack_tr_expk) 
        f_track_tr_k = ca.Function('f_track_tr_k',
                                  [dataToTrack_trk, dataToTrack_tr_expk],
                                  [dataTrackingTerm_tr])
        f_track_tr_k_map = f_track_tr_k.map(N+1, parallelMode, NThreads)        
        
    # %% Bounds
    from bounds import bounds
    bounds = bounds(joints, rotationalJoints)    
    # States
    # uBoundsA, lBoundsA, scalingA = bounds.getBoundsActivation()
    # uBoundsAk = ca.vec(uBoundsA.to_numpy().T * np.ones((1, N+1))).full()
    # lBoundsAk = ca.vec(lBoundsA.to_numpy().T * np.ones((1, N+1))).full()
    # uBoundsAj = ca.vec(uBoundsA.to_numpy().T * np.ones((1, d*N))).full()
    # lBoundsAj = ca.vec(lBoundsA.to_numpy().T * np.ones((1, d*N))).full()
    
    # uBoundsF, lBoundsF, scalingF = bounds.getBoundsForce()
    # uBoundsFk = ca.vec(uBoundsF.to_numpy().T * np.ones((1, N+1))).full()
    # lBoundsFk = ca.vec(lBoundsF.to_numpy().T * np.ones((1, N+1))).full()
    # uBoundsFj = ca.vec(uBoundsF.to_numpy().T * np.ones((1, d*N))).full()
    # lBoundsFj = ca.vec(lBoundsF.to_numpy().T * np.ones((1, d*N))).full()
    
    # if conservative_bounds:
    #     uBoundsQs, lBoundsQs, scalingQs, _, _ = (
    #         bounds.getBoundsPositionConservative()) 
    # else:
        
    uBoundsQs, lBoundsQs, scalingQs = bounds.getBoundsPosition()    
    uBoundsQsk = ca.vec(uBoundsQs.to_numpy().T * np.ones((1, N+1))).full()
    lBoundsQsk = ca.vec(lBoundsQs.to_numpy().T * np.ones((1, N+1))).full()
    uBoundsQsj = ca.vec(uBoundsQs.to_numpy().T * np.ones((1, d*N))).full()
    lBoundsQsj = ca.vec(lBoundsQs.to_numpy().T * np.ones((1, d*N))).full()
    
    uBoundsQdots, lBoundsQdots, scalingQdots = bounds.getBoundsVelocity()
    uBoundsQdotsk = ca.vec(uBoundsQdots.to_numpy().T*np.ones((1, N+1))).full()
    lBoundsQdotsk = ca.vec(lBoundsQdots.to_numpy().T*np.ones((1, N+1))).full()
    uBoundsQdotsj = ca.vec(uBoundsQdots.to_numpy().T*np.ones((1, d*N))).full()
    lBoundsQdotsj = ca.vec(lBoundsQdots.to_numpy().T*np.ones((1, d*N))).full()
        
    uBoundsActJA, lBoundsActJA, scalingActJA = bounds.getBoundsTMActivation(actJoints)
    uBoundsActJAk = ca.vec(uBoundsActJA.to_numpy().T * np.ones((1, N+1))).full()
    lBoundsActJAk = ca.vec(lBoundsActJA.to_numpy().T * np.ones((1, N+1))).full()
    uBoundsActJAj = ca.vec(uBoundsActJA.to_numpy().T * np.ones((1, d*N))).full()
    lBoundsActJAj = ca.vec(lBoundsActJA.to_numpy().T * np.ones((1, d*N))).full()
    
    uBoundsGTJA, lBoundsGTJA, scalingGTJA = bounds.getBoundsTMActivation(groundThoraxJoints)
    uBoundsGTJAk = ca.vec(uBoundsGTJA.to_numpy().T * np.ones((1, N+1))).full()
    lBoundsGTJAk = ca.vec(lBoundsGTJA.to_numpy().T * np.ones((1, N+1))).full()
    uBoundsGTJAj = ca.vec(uBoundsGTJA.to_numpy().T * np.ones((1, d*N))).full()
    lBoundsGTJAj = ca.vec(lBoundsGTJA.to_numpy().T * np.ones((1, d*N))).full()
    
    # Controls
    # uBoundsADt, lBoundsADt, scalingADt = bounds.getBoundsActivationDerivative()
    # uBoundsADtk = ca.vec(uBoundsADt.to_numpy().T * np.ones((1, N))).full()
    # lBoundsADtk = ca.vec(lBoundsADt.to_numpy().T * np.ones((1, N))).full()
    
    uBoundsActJE, lBoundsActJE, scalingActJE = bounds.getBoundsTMExcitation(actJoints)
    uBoundsActJEk = ca.vec(uBoundsActJE.to_numpy().T * np.ones((1, N))).full()
    lBoundsActJEk = ca.vec(lBoundsActJE.to_numpy().T * np.ones((1, N))).full()
    
    uBoundsGTJE, lBoundsGTJE, scalingGTJE = bounds.getBoundsTMExcitation(groundThoraxJoints)
    uBoundsGTJEk = ca.vec(uBoundsGTJE.to_numpy().T * np.ones((1, N))).full()
    lBoundsGTJEk = ca.vec(lBoundsGTJE.to_numpy().T * np.ones((1, N))).full()
    
    # Slack controls
    uBoundsQdotdots, lBoundsQdotdots, scalingQdotdots = (
            bounds.getBoundsAcceleration())
    uBoundsQdotdotsj = ca.vec(uBoundsQdotdots.to_numpy().T * 
                              np.ones((1, d*N))).full()
    lBoundsQdotdotsj = ca.vec(lBoundsQdotdots.to_numpy().T *
                              np.ones((1, d*N))).full()
    
    uBoundsLambda, lBoundsLambda, scalingLambda = (
            bounds.getBoundsMultipliers(NHolConstraints))
    uBoundsLambdaj = ca.vec(uBoundsLambda.to_numpy().T * 
                              np.ones((1, d*N))).full()
    lBoundsLambdaj = ca.vec(lBoundsLambda.to_numpy().T *
                              np.ones((1, d*N))).full()
    
    if velocity_correction:
        uBoundsGamma, lBoundsGamma, scalingGamma = (
                bounds.getBoundsMultipliers(NHolConstraints))
        uBoundsGammaj = ca.vec(uBoundsGamma.to_numpy().T * 
                                  np.ones((1, d*N))).full()
        lBoundsGammaj = ca.vec(lBoundsGamma.to_numpy().T *
                                  np.ones((1, d*N))).full()    
    
    # uBoundsFDt, lBoundsFDt, scalingFDt = bounds.getBoundsForceDerivative()
    # uBoundsFDtj = ca.vec(uBoundsFDt.to_numpy().T * np.ones((1, d*N))).full()
    # lBoundsFDtj = ca.vec(lBoundsFDt.to_numpy().T * np.ones((1, d*N))).full()
    
#     if tracking_data == "markers":
#         # Additional controls
#         if boundsMarker == "uniformBoundsMarker":
#             uBoundsMarker, lBoundsMarker, scalingMarker = (
#                 bounds.getUniformBoundsMarker(markers_toTrack, 
#                                               markers_scaling))
#         elif boundsMarker == "treadmillSpecificBoundsMarker":
#             uBoundsMarker, lBoundsMarker, scalingMarker = (
#                 bounds.getTreadmillSpecificBoundsMarker(markers_toTrack, 
#                                                         markers_scaling))
                
#         uBoundsMarkerk = ca.vec(
#             uBoundsMarker.to_numpy().T * np.ones((1, N))).full()
#         lBoundsMarkerk = ca.vec(
#             lBoundsMarker.to_numpy().T * np.ones((1, N))).full()
#         if offset_ty:
#             # Static parameters  
#             scalingOffset = scalingMarker.iloc[0][scalingMarker.columns[0]]
#             uBoundsOffset, lboundsOffset = bounds.getBoundsOffset(
#                 scalingOffset)
#             uBoundsOffsetk = uBoundsOffset.to_numpy()
#             lBoundsOffsetk = lboundsOffset.to_numpy()        
#     elif tracking_data == "coordinates" and offset_ty:
#         # Static parameters 
#         scalingOffset = scalingQs.iloc[0]["pelvis_ty"]
#         uBoundsOffset, lboundsOffset = bounds.getBoundsOffset(scalingOffset)
#         uBoundsOffsetk = uBoundsOffset.to_numpy()
#         lBoundsOffsetk = lboundsOffset.to_numpy()
    
    # %% Guesses and scaling   
    Qs_fromIK_filt_interp = interpolateDataFrame(
        Qs_fromIK_filt, timeInterval[0], timeInterval[1], N)
    if guessType == "dataDriven":         
        from guess import dataDrivenGuess
        guess = dataDrivenGuess(Qs_fromIK_filt_interp, N, d, joints, 
                                holConstraints_titles)    
    # elif guessType == "quasiRandom": 
    #     from guesses import quasiRandomGuess
    #     guess = quasiRandomGuess(N, d, joints, bothSidesMuscles, timeElapsed,
    #                               Qs_fromIK_filt_interp)
    # if offset_ty:
    #     # Static parameters
    #     guessOffset = guess.getGuessOffset(scalingOffset)
    # States
    # guessA = guess.getGuessActivation(scalingA)
    # guessACol = guess.getGuessActivationCol()
    # guessF = guess.getGuessForce(scalingF)
    # guessFCol = guess.getGuessForceCol()
    guessQs = guess.getGuessPosition(scalingQs)
    guessQsCol = guess.getGuessPositionCol()
    guessQdots = guess.getGuessVelocity(scalingQdots, guess_zeroVelocity)
    guessQdotsCol = guess.getGuessVelocityCol()    
    guessActJA = guess.getGuessTMActivation(actJoints)
    guessActJACol = guess.getGuessTMActivationCol()
    guessGTJA = guess.getGuessTMActivation(groundThoraxJoints)
    guessGTJACol = guess.getGuessTMActivationCol()
    # Controls
    # guessADt = guess.getGuessActivationDerivative(scalingADt)
    guessActJE = guess.getGuessTMExcitation(actJoints)
    guessGTJE = guess.getGuessTMExcitation(groundThoraxJoints)
    # Slack controls
    guessQdotdots = guess.getGuessAcceleration(scalingQdotdots, 
                                               guess_zeroAcceleration)
    guessQdotdotsCol = guess.getGuessAccelerationCol()    
    guessLambda = guess.getGuessMultipliers()
    guessLambdaCol = guess.getGuessMultipliersCol()
    if velocity_correction:
        guessGamma = guess.getGuessVelCorrs()
        guessGammaCol = guess.getGuessVelCorrsCol()       
    # guessFDt = guess.getGuessForceDerivative(scalingFDt)
    # guessFDtCol = guess.getGuessForceDerivativeCol()  
    
    # if tracking_data == "markers":
    #     guessMarker = guess.getGuessMarker(
    #         markers_toTrack, marker_data_interp, scalingMarker)    
    #     # Scale marker data for tracking term
    #     dataToTrack_sc = (marker_data_interp.to_numpy()[:,1::].T / 
    #                       scalingMarker.to_numpy().T)
    #     dataToTrack_nsc = (marker_data_interp.to_numpy()[:,1::].T)
    #     if norm_std:
    #         dataToTrack_std_sc = np.reshape(np.std(dataToTrack_sc, axis=1),
    #                                         (-1, 1)) * np.ones((1, N))        
    if tracking_data == "coordinates":
        from variousFunctions import scaleDataFrame        
        dataToTrack_sc = scaleDataFrame(
            Qs_fromIK_interp, scalingQs, 
            coordinates_toTrack['rotational']).to_numpy()[:,1::].T
        from variousFunctions import selectFromDataFarme 
        dataToTrack_nsc = selectFromDataFarme(
            Qs_fromIK_interp, 
            coordinates_toTrack['rotational']).to_numpy()[:,1::].T     
        dataToTrack_nsc_rot_deg = copy.deepcopy(dataToTrack_nsc)        
        dataToTrack_nsc_rot_deg = dataToTrack_nsc_rot_deg * 180 / np.pi
        if coordinates_toTrack['translational']:            
            dataToTrack_tr_sc = scaleDataFrame(
                Qs_fromIK_interp, scalingQs, 
                coordinates_toTrack['translational']).to_numpy()[:,1::].T
            dataToTrack_tr_nsc = selectFromDataFarme(
                Qs_fromIK_interp, 
                coordinates_toTrack['translational']).to_numpy()[:,1::].T
        # if constraintPelvis_ty["end_cond"] or constraintPelvis_ty["env_cond"]:
        #     pelvis_ty_sc = scaleDataFrame(Qs_fromIK_interp, scalingQs, 
        #                                   ["pelvis_ty"]).to_numpy()[:,1::].T
            
#     # %% Compare simulated marker trajectories to trajectories to track at IG    
#     Qsk_IG_sc = guessQs.to_numpy().T[:,1::] 
#     Qsk_IG_nsc = Qsk_IG_sc * (scalingQs.to_numpy().T * np.ones((1, N)))
#     QsQdotsk_IG_nsc = np.zeros((NJoints*2, N))
#     QsQdotsk_IG_nsc[::2, :] = Qsk_IG_nsc
#     Qdotdotsk_IG_nsc = np.zeros((NJoints, N))
#     F1_out_IG = np.zeros((NOutput_F1 , N))    
#     F_out_IG = np.zeros((NOutput_F , N)) 
#     for k in range(N):        
#         F1_out_IG[:, k] = F1(np.concatenate((QsQdotsk_IG_nsc[:,k], 
#                                              Qdotdotsk_IG_nsc[:,k]), 
#                                             axis=0)).full().T     
#         F_out_IG[:, k] = F(np.concatenate((QsQdotsk_IG_nsc[:,k], 
#                                            Qdotdotsk_IG_nsc[:,k]), 
#                                           axis=0)).full().T  
#     assert np.alltrue(F_out_IG ==  F1_out_IG[:NOutput_F,:]), "F vs F1"
    
#     if plotMarkerTrackingAtInitialGuess:
#         markers_IG = F1_out_IG[idxMarker["toTrack"], :]
#         import matplotlib.pyplot as plt 
#         if tracking_data == "markers":    
#             fig, axs = plt.subplots(NMarkers, NVec3, sharex=True)    
#             fig.suptitle('Tracking of marker trajectories')
#             for i, ax in enumerate(axs.flat):
#                 ax.plot(tgridf[0,1:].T, 
#                         dataToTrack_nsc[i:i+1,:].T, 
#                         'tab:blue', label='experimental')
#                 ax.plot(tgridf[0,1:].T, 
#                         markers_IG[i:i+1, :].T, 
#                         'tab:orange', label='simulated')
#                 ax.set_title(marker_titles[i])
#             plt.setp(axs[-1, :], xlabel='Time (s)')
#             plt.setp(axs[:, 0], ylabel='(m)')
#             fig.align_ylabels()
#             handles, labels = ax.get_legend_handles_labels()
#             plt.legend(handles, labels, loc='upper right')    
        
    # %% Formulate ocp
    # Time step    
    h = timeElapsed / N
    
    ###########################################################################
    # Collocation matrices
    tau = ca.collocation_points(d,'radau');
    [C,D] = ca.collocation_interpolators(tau);
    # Missing matrix B, add manually
    B = [-8.88178419700125e-16, 0.376403062700467, 0.512485826188421, 
          0.111111111111111]
    
    if solveProblem: 
        #######################################################################
        # Initialize opti instance
        opti = ca.Opti()        
        #######################################################################
        # if offset_ty:
        #     # Static parameters
        #     offset = opti.variable(1)
        #     opti.subject_to(
        #         opti.bounded(lBoundsOffsetk, offset, uBoundsOffsetk))
        #     opti.set_initial(offset, guessOffset)
        
        #######################################################################
        # States
        '''
        # Musle activation at mesh points
        a = opti.variable(NMuscles, N+1)
        opti.subject_to(opti.bounded(lBoundsAk, ca.vec(a), uBoundsAk))
        opti.set_initial(a, guessA.to_numpy().T)
        assert np.alltrue(lBoundsAk <= ca.vec(guessA.to_numpy().T).full()), "lb Musle activation"
        assert np.alltrue(uBoundsAk >= ca.vec(guessA.to_numpy().T).full()), "ub Musle activation"
        # Musle activation at collocation points
        a_col = opti.variable(NMuscles, d*N)
        opti.subject_to(opti.bounded(lBoundsAj, ca.vec(a_col), uBoundsAj))
        opti.set_initial(a_col, guessACol.to_numpy().T)
        assert np.alltrue(lBoundsAj <= ca.vec(guessACol.to_numpy().T).full()), "lb Musle activation col"
        assert np.alltrue(uBoundsAj >= ca.vec(guessACol.to_numpy().T).full()), "ub Musle activation col"
        # Musle force at mesh points
        normF = opti.variable(NMuscles, N+1)
        opti.subject_to(opti.bounded(lBoundsFk, ca.vec(normF), uBoundsFk))
        opti.set_initial(normF, guessF.to_numpy().T)
        assert np.alltrue(lBoundsFk <= ca.vec(guessF.to_numpy().T).full()), "lb Musle force"
        assert np.alltrue(uBoundsFk >= ca.vec(guessF.to_numpy().T).full()), "ub Musle force"
        # Musle force at collocation points
        normF_col = opti.variable(NMuscles, d*N)
        opti.subject_to(opti.bounded(lBoundsFj, ca.vec(normF_col), uBoundsFj))
        opti.set_initial(normF_col, guessFCol.to_numpy().T)
        assert np.alltrue(lBoundsFj <= ca.vec(guessFCol.to_numpy().T).full()), "lb Musle force col"
        assert np.alltrue(uBoundsFj >= ca.vec(guessFCol.to_numpy().T).full()), "ub Musle force col"
        '''
        # Joint position at mesh points
        Qs = opti.variable(NJoints, N+1)
        opti.subject_to(opti.bounded(lBoundsQsk, ca.vec(Qs), uBoundsQsk))
        opti.set_initial(Qs, guessQs.to_numpy().T)
        assert np.alltrue(lBoundsQsk - ca.vec(guessQs.to_numpy().T).full() < 1e-12), "lb Joint position"
        assert np.alltrue(uBoundsQsk - ca.vec(guessQs.to_numpy().T).full() > -1e-12), "ub Joint position"        
        # testA = lBoundsQs.to_numpy().T * np.ones((1, N+1))
        # testB = guessQs.to_numpy().T
        # testC = testA - testB <-1e-12        
        # Joint position at collocation points
        Qs_col = opti.variable(NJoints, d*N)
        opti.subject_to(opti.bounded(lBoundsQsj, ca.vec(Qs_col), uBoundsQsj))
        opti.set_initial(Qs_col, guessQsCol.to_numpy().T)
        assert np.alltrue(lBoundsQsj <= ca.vec(guessQsCol.to_numpy().T).full()), "lb Joint position col"
        assert np.alltrue(uBoundsQsj >= ca.vec(guessQsCol.to_numpy().T).full()), "ub Joint position col"
        # Joint velocity at mesh points
        Qdots = opti.variable(NJoints, N+1)
        opti.subject_to(opti.bounded(lBoundsQdotsk, ca.vec(Qdots), uBoundsQdotsk))
        opti.set_initial(Qdots, guessQdots.to_numpy().T)
        assert np.alltrue(lBoundsQdotsk <= ca.vec(guessQdots.to_numpy().T).full()), "lb Joint velocity"
        assert np.alltrue(uBoundsQdotsk >= ca.vec(guessQdots.to_numpy().T).full()), "ub Joint velocity"        
        # Joint velocity at collocation points
        Qdots_col = opti.variable(NJoints, d*N)
        opti.subject_to(opti.bounded(lBoundsQdotsj, ca.vec(Qdots_col), uBoundsQdotsj))
        opti.set_initial(Qdots_col, guessQdotsCol.to_numpy().T)
        assert np.alltrue(lBoundsQdotsj <= ca.vec(guessQdotsCol.to_numpy().T).full()), "lb Joint velocity col"
        assert np.alltrue(uBoundsQdotsj >= ca.vec(guessQdotsCol.to_numpy().T).full()), "ub Joint velocity col"
        # Actuated joints activation at mesh points
        aActJ = opti.variable(NActJoints, N+1)
        opti.subject_to(opti.bounded(lBoundsActJAk, ca.vec(aActJ), uBoundsActJAk))
        opti.set_initial(aActJ, guessActJA.to_numpy().T)
        assert np.alltrue(lBoundsActJAk <= ca.vec(guessActJA.to_numpy().T).full()), "lb ActJ activation"
        assert np.alltrue(uBoundsActJAk >= ca.vec(guessActJA.to_numpy().T).full()), "ub ActJ activation"
        # Actuated joints activation at collocation points
        aActJ_col = opti.variable(NActJoints, d*N)
        opti.subject_to(opti.bounded(lBoundsActJAj, ca.vec(aActJ_col), uBoundsActJAj))
        opti.set_initial(aActJ_col, guessActJACol.to_numpy().T)
        assert np.alltrue(lBoundsActJAj <= ca.vec(guessActJACol.to_numpy().T).full()), "lb ActJ activation col"
        assert np.alltrue(uBoundsActJAj >= ca.vec(guessActJACol.to_numpy().T).full()), "ub ActJ activation col"
        # Ground thorax joints activation at mesh points
        aGTJ = opti.variable(NGroundThoraxJoints, N+1)
        opti.subject_to(opti.bounded(lBoundsGTJAk, ca.vec(aGTJ), uBoundsGTJAk))
        opti.set_initial(aGTJ, guessGTJA.to_numpy().T)
        assert np.alltrue(lBoundsGTJAk <= ca.vec(guessGTJA.to_numpy().T).full()), "lb GTJ activation"
        assert np.alltrue(uBoundsGTJAk >= ca.vec(guessGTJA.to_numpy().T).full()), "ub GTJ activation"
        # Ground thorax joints activation at collocation points
        aGTJ_col = opti.variable(NGroundThoraxJoints, d*N)
        opti.subject_to(opti.bounded(lBoundsGTJAj, ca.vec(aGTJ_col), uBoundsGTJAj))
        opti.set_initial(aGTJ_col, guessGTJACol.to_numpy().T)
        assert np.alltrue(lBoundsGTJAj <= ca.vec(guessGTJACol.to_numpy().T).full()), "lb GTJ activation col"
        assert np.alltrue(uBoundsGTJAj >= ca.vec(guessGTJACol.to_numpy().T).full()), "ub GTJ activation col"
        
        #######################################################################
        # Controls
        '''
        # Muscle activation derivative at mesh points
        aDt = opti.variable(NMuscles, N)
        opti.subject_to(opti.bounded(lBoundsADtk, ca.vec(aDt), uBoundsADtk))
        opti.set_initial(aDt, guessADt.to_numpy().T)
        assert np.alltrue(lBoundsADtk <= ca.vec(guessADt.to_numpy().T).full()), "lb Muscle activation derivative"
        assert np.alltrue(uBoundsADtk >= ca.vec(guessADt.to_numpy().T).full()), "ub Muscle activation derivative"
        '''
        # Actuated joints excitation at mesh points
        eActJ = opti.variable(NActJoints, N)
        opti.subject_to(opti.bounded(lBoundsActJEk, ca.vec(eActJ), uBoundsActJEk))
        opti.set_initial(eActJ, guessActJE.to_numpy().T)
        assert np.alltrue(lBoundsActJEk <= ca.vec(guessActJE.to_numpy().T).full()), "lb ActJ excitation"
        assert np.alltrue(uBoundsActJEk >= ca.vec(guessActJE.to_numpy().T).full()), "ub ActJ excitation"
        # Ground thorax joints excitation at mesh points
        eGTJ = opti.variable(NGroundThoraxJoints, N)
        opti.subject_to(opti.bounded(lBoundsGTJEk, ca.vec(eGTJ), uBoundsGTJEk))
        opti.set_initial(eGTJ, guessGTJE.to_numpy().T)
        assert np.alltrue(lBoundsGTJEk <= ca.vec(guessGTJE.to_numpy().T).full()), "lb GTJ excitation"
        assert np.alltrue(uBoundsGTJEk >= ca.vec(guessGTJE.to_numpy().T).full()), "ub GTJ excitation"
        
        #######################################################################
        # Slack controls
        '''
        # Muscle force derivative at collocation points
        normFDt_col = opti.variable(NMuscles, d*N)
        opti.subject_to(opti.bounded(lBoundsFDtj, ca.vec(normFDt_col), uBoundsFDtj))
        opti.set_initial(normFDt_col, guessFDtCol.to_numpy().T)
        assert np.alltrue(lBoundsFDtj <= ca.vec(guessFDtCol.to_numpy().T).full()), "lb Muscle force derivative"
        assert np.alltrue(uBoundsFDtj >= ca.vec(guessFDtCol.to_numpy().T).full()), "ub Muscle force derivative"
        '''
        # Joint velocity derivative (acceleration) at collocation points
        Qdotdots_col = opti.variable(NJoints, d*N)
        opti.subject_to(opti.bounded(lBoundsQdotdotsj, ca.vec(Qdotdots_col),
                                      uBoundsQdotdotsj))
        opti.set_initial(Qdotdots_col, guessQdotdotsCol.to_numpy().T)
        assert np.alltrue(lBoundsQdotdotsj <= ca.vec(guessQdotdotsCol.to_numpy().T).full()), "lb Joint velocity derivative"
        assert np.alltrue(uBoundsQdotdotsj >= ca.vec(guessQdotdotsCol.to_numpy().T).full()), "ub Joint velocity derivative"
        # Lagrange multipliers
        lambda_col = opti.variable(NHolConstraints, d*N)
        opti.subject_to(opti.bounded(lBoundsLambdaj, ca.vec(lambda_col),
                                     uBoundsLambdaj))
        opti.set_initial(lambda_col, guessLambdaCol.to_numpy().T)
        assert np.alltrue(lBoundsLambdaj <= ca.vec(guessLambdaCol.to_numpy().T).full()), "lb Lagrange Multipliers"
        assert np.alltrue(uBoundsLambdaj >= ca.vec(guessLambdaCol.to_numpy().T).full()), "ub Lagrange Multipliers"   
        # Velocity correctors
        if velocity_correction:
            gamma_col = opti.variable(NHolConstraints, d*N)
            opti.subject_to(opti.bounded(lBoundsGammaj, ca.vec(gamma_col),
                                         uBoundsGammaj))
            opti.set_initial(gamma_col, guessGammaCol.to_numpy().T)
            assert np.alltrue(lBoundsGammaj <= ca.vec(guessGammaCol.to_numpy().T).full()), "lb Velocity Correctors"
            assert np.alltrue(uBoundsGammaj >= ca.vec(guessGammaCol.to_numpy().T).full()), "ub Velocity Correctors"
        
            
#         #######################################################################
#         # Additional controls
#         if tracking_data == "markers" and markers_as_controls:
#             # Marker trajectories
#             marker_u = opti.variable(NEl_toTrack, N)
#             opti.subject_to(opti.bounded(lBoundsMarkerk, ca.vec(marker_u),
#                                          uBoundsMarkerk))
#             opti.set_initial(marker_u, guessMarker.to_numpy().T)
#             # Unscale for equality constraints with model markers
#             marker_u_nsc = marker_u * (
#                 scalingMarker.to_numpy().T *np.ones((1, N)))
            
        #######################################################################
        if plotGuessVsBounds:   
            from variousFunctions import plotVSBounds
            # States
            '''
            # Muscle activation at mesh points            
            lb = lBoundsA.to_numpy().T
            ub = uBoundsA.to_numpy().T
            y = guessA.to_numpy().T
            title='Muscle activation at mesh points'            
            plotVSBounds(y,lb,ub,title)  
            # Muscle activation at collocation points
            lb = lBoundsA.to_numpy().T
            ub = uBoundsA.to_numpy().T
            y = guessACol.to_numpy().T
            title='Muscle activation at collocation points' 
            plotVSBounds(y,lb,ub,title)  
            # Muscle force at mesh points
            lb = lBoundsF.to_numpy().T
            ub = uBoundsF.to_numpy().T
            y = guessF.to_numpy().T
            title='Muscle force at mesh points' 
            plotVSBounds(y,lb,ub,title)  
            # Muscle force at collocation points
            lb = lBoundsF.to_numpy().T
            ub = uBoundsF.to_numpy().T
            y = guessFCol.to_numpy().T
            title='Muscle force at collocation points' 
            plotVSBounds(y,lb,ub,title)
            '''
            # Joint position at mesh points
            lb = lBoundsQs.to_numpy().T
            ub = uBoundsQs.to_numpy().T
            y = guessQs.to_numpy().T
            title='Joint position at mesh points' 
            plotVSBounds(y,lb,ub,title)             
            # Joint position at collocation points
            lb = lBoundsQs.to_numpy().T
            ub = uBoundsQs.to_numpy().T
            y = guessQsCol.to_numpy().T
            title='Joint position at collocation points' 
            plotVSBounds(y,lb,ub,title) 
            # Joint velocity at mesh points
            lb = lBoundsQdots.to_numpy().T
            ub = uBoundsQdots.to_numpy().T
            y = guessQdots.to_numpy().T
            title='Joint velocity at mesh points' 
            plotVSBounds(y,lb,ub,title) 
            # Joint velocity at collocation points
            lb = lBoundsQdots.to_numpy().T
            ub = uBoundsQdots.to_numpy().T
            y = guessQdotsCol.to_numpy().T
            title='Joint velocity at collocation points' 
            plotVSBounds(y,lb,ub,title) 
            # Actuated joints activation at mesh points
            lb = lBoundsActJA.to_numpy().T
            ub = uBoundsActJA.to_numpy().T
            y = guessActJA.to_numpy().T
            title='ActJ activation at mesh points' 
            plotVSBounds(y,lb,ub,title) 
            # Actuated joints activation at collocation points
            lb = lBoundsActJA.to_numpy().T
            ub = uBoundsActJA.to_numpy().T
            y = guessActJACol.to_numpy().T
            title='ActJ activation at collocation points' 
            plotVSBounds(y,lb,ub,title) 
            # Ground thorax joints activation at mesh points
            lb = lBoundsGTJA.to_numpy().T
            ub = uBoundsGTJA.to_numpy().T
            y = guessGTJA.to_numpy().T
            title='GTJ activation at mesh points' 
            plotVSBounds(y,lb,ub,title) 
            # Ground thorax joints activation at collocation points
            lb = lBoundsGTJA.to_numpy().T
            ub = uBoundsGTJA.to_numpy().T
            y = guessGTJACol.to_numpy().T
            title='GTJ activation at collocation points' 
            plotVSBounds(y,lb,ub,title) 
            ###################################################################
            # Controls
            '''
            # Muscle activation derivative at mesh points
            lb = lBoundsADt.to_numpy().T
            ub = uBoundsADt.to_numpy().T
            y = guessADt.to_numpy().T
            title='Muscle activation derivative at mesh points' 
            plotVSBounds(y,lb,ub,title) 
            '''
            # Actuated joints excitation at mesh points
            lb = lBoundsActJE.to_numpy().T
            ub = uBoundsActJE.to_numpy().T
            y = guessActJE.to_numpy().T
            title='ActJ excitation at mesh points' 
            plotVSBounds(y,lb,ub,title) 
            # Ground thorax joints excitation at mesh points
            lb = lBoundsGTJE.to_numpy().T
            ub = uBoundsGTJE.to_numpy().T
            y = guessGTJE.to_numpy().T
            title='GTJ excitation at mesh points' 
            plotVSBounds(y,lb,ub,title)               
            ###################################################################
            # Slack controls
            '''
            # Muscle force derivative at collocation points
            lb = lBoundsFDt.to_numpy().T
            ub = uBoundsFDt.to_numpy().T
            y = guessFDtCol.to_numpy().T
            title='Muscle force derivative at collocation points' 
            plotVSBounds(y,lb,ub,title)
            '''
            # Joint velocity derivative (acceleration) at collocation points
            lb = lBoundsQdotdots.to_numpy().T
            ub = uBoundsQdotdots.to_numpy().T
            y = guessQdotdotsCol.to_numpy().T
            title='Joint velocity derivative (acceleration) at collocation points' 
            plotVSBounds(y,lb,ub,title)             
            # Lagrange multipliers at collocation points
            lb = lBoundsLambda.to_numpy().T
            ub = uBoundsLambda.to_numpy().T
            y = guessLambdaCol.to_numpy().T
            title='Lagrange multipliers at collocation points' 
            plotVSBounds(y,lb,ub,title)            
            # Velocity correctors at collocation points
            if velocity_correction:
                lb = lBoundsGamma.to_numpy().T
                ub = uBoundsGamma.to_numpy().T
                y = guessGammaCol.to_numpy().T
                title='Velocity correctors at collocation points' 
                plotVSBounds(y,lb,ub,title)        
            
            # if tracking_data == "markers" and markers_as_controls:
            #     # Marker trajectories
            #     lb = lBoundsMarker.to_numpy().T
            #     ub = uBoundsMarker.to_numpy().T
            #     y = guessMarker.to_numpy().T
            #     title='Marker trajectories at mesh points' 
            #     plotVSBounds(y,lb,ub,title) 
        
        #######################################################################
        # Parallel formulation
        # States
        '''
        ak = ca.MX.sym('ak', NMuscles)
        aj = ca.MX.sym('aj', NMuscles, d)    
        akj = ca.horzcat(ak, aj)    
        normFk = ca.MX.sym('normFk', NMuscles)
        normFj = ca.MX.sym('normFj', NMuscles, d)
        normFkj = ca.horzcat(normFk, normFj)   
        '''
        Qsk = ca.MX.sym('Qsk', NJoints)
        Qsj = ca.MX.sym('Qsj', NJoints, d)
        Qskj = ca.horzcat(Qsk, Qsj)    
        Qdotsk = ca.MX.sym('Qdotsk', NJoints)
        Qdotsj = ca.MX.sym('Qdotsj', NJoints, d)
        Qdotskj = ca.horzcat(Qdotsk, Qdotsj)    
        aActJk = ca.MX.sym('aActJk', NActJoints)
        aActJj = ca.MX.sym('aActJj', NActJoints, d)
        aActJkj = ca.horzcat(aActJk, aActJj)  
        aGTJk = ca.MX.sym('aGTJk', NGroundThoraxJoints)
        aGTJj = ca.MX.sym('aGTJj', NGroundThoraxJoints, d)
        aGTJkj = ca.horzcat(aGTJk, aGTJj) 
        # Controls
        '''
        aDtk = ca.MX.sym('aDtk', NMuscles)    
        '''
        eActJk = ca.MX.sym('eActJk', NActJoints)
        eGTJk = ca.MX.sym('eGTJk', NGroundThoraxJoints)
        # Slack controls
        '''
        normFDtj = ca.MX.sym('normFDtj', NMuscles, d);
        '''
        Qdotdotsj = ca.MX.sym('Qdotdotsj', NJoints, d)     
        lambdaj = ca.MX.sym('lambdaj', NHolConstraints, d)   
        if velocity_correction:
            gammaj = ca.MX.sym('gammaj', NHolConstraints, d)   
               
        #######################################################################
        # Initialize cost function and constraint vectors
        J = 0
        eq_constr = []
        # ineq_constr1 = []
        # ineq_constr2 = []
            
        #######################################################################
        # Loop over collocation points
        for j in range(d):
            ###################################################################
            # Unscale variables
            # States
            # normFkj_nsc = normFkj * (scalingF.to_numpy().T * np.ones((1, d+1)))
            Qskj_nsc = Qskj * (scalingQs.to_numpy().T * np.ones((1, d+1)))
            Qdotskj_nsc = Qdotskj * (scalingQdots.to_numpy().T * np.ones((1, d+1)))
            # Controls
            # aDtk_nsc = aDtk * (scalingADt.to_numpy().T)
            # Slack controls
            # normFDtj_nsc = normFDtj * (scalingFDt.to_numpy().T * 
            #                             np.ones((1, d)))
            Qdotdotsj_nsc = Qdotdotsj * (scalingQdotdots.to_numpy().T * np.ones((1, d))) 
            lambdaj_nsc = lambdaj * (scalingLambda.to_numpy().T * np.ones((1, d)))
            if velocity_correction:
                gammaj_nsc = gammaj * (scalingGamma.to_numpy().T * np.ones((1, d)))            
            
            # Qs and Qdots are intertwined in external function
            QsQdotskj_nsc = ca.MX(NJoints*2, d+1)
            QsQdotskj_nsc[::2, :] = Qskj_nsc
            QsQdotskj_nsc[1::2, :] = Qdotskj_nsc   
            
#             ###################################################################
#             # Polynomial approximations
#             # Left leg
#             Qsinj_l = Qskj_nsc[leftPolynomialJointIndices, j+1]
#             Qdotsinj_l = Qdotskj_nsc[leftPolynomialJointIndices, j+1]
#             [lMTj_l, vMTj_l, dMj_l] = f_polynomial(Qsinj_l, Qdotsinj_l)        
#             dMj_hip_flexion_l = dMj_l[momentArmIndices['hip_flexion_l'],
#                                       leftPolynomialJoints.index(
#                                           'hip_flexion_l')]
#             dMj_hip_adduction_l = dMj_l[momentArmIndices['hip_adduction_l'],
#                                         leftPolynomialJoints.index(
#                                           'hip_adduction_l')]
#             dMj_hip_rotation_l = dMj_l[momentArmIndices['hip_rotation_l'],
#                                        leftPolynomialJoints.index(
#                                           'hip_rotation_l')]
#             dMj_knee_angle_l = dMj_l[momentArmIndices['knee_angle_l'],
#                                        leftPolynomialJoints.index(
#                                           'knee_angle_l')]
#             dMj_ankle_angle_l = dMj_l[momentArmIndices['ankle_angle_l'],
#                                       leftPolynomialJoints.index(
#                                           'ankle_angle_l')]
#             dMj_subtalar_angle_l = dMj_l[momentArmIndices['subtalar_angle_l'],
#                                          leftPolynomialJoints.index(
#                                           'subtalar_angle_l')]        
#             # Right leg
#             Qsinj_r = Qskj_nsc[rightPolynomialJointIndices, j+1]
#             Qdotsinj_r = Qdotskj_nsc[rightPolynomialJointIndices, j+1]
#             [lMTj_r, vMTj_r, dMj_r] = f_polynomial(Qsinj_r, Qdotsinj_r)
#             dMj_hip_flexion_r = dMj_r[momentArmIndices['hip_flexion_l'],
#                                       leftPolynomialJoints.index(
#                                           'hip_flexion_l')]
#             dMj_hip_adduction_r = dMj_r[momentArmIndices['hip_adduction_l'],
#                                         leftPolynomialJoints.index(
#                                           'hip_adduction_l')]
#             dMj_hip_rotation_r = dMj_r[momentArmIndices['hip_rotation_l'],
#                                        leftPolynomialJoints.index(
#                                           'hip_rotation_l')]
#             dMj_knee_angle_r = dMj_r[momentArmIndices['knee_angle_l'],
#                                        leftPolynomialJoints.index(
#                                           'knee_angle_l')]
#             dMj_ankle_angle_r = dMj_r[momentArmIndices['ankle_angle_l'],
#                                       leftPolynomialJoints.index(
#                                           'ankle_angle_l')]
#             dMj_subtalar_angle_r =dMj_r[momentArmIndices['subtalar_angle_l'],
#                                          leftPolynomialJoints.index(
#                                           'subtalar_angle_l')]
#             # Trunk
#             dMj_lumbar_extension = dMj_l[trunkMomentArmPolynomialIndices,
#                                          leftPolynomialJoints.index(
#                                           'lumbar_extension')]
#             dMj_lumbar_bending = dMj_l[trunkMomentArmPolynomialIndices,
#                                          leftPolynomialJoints.index(
#                                           'lumbar_bending')]
#             dMj_lumbar_rotation = dMj_l[trunkMomentArmPolynomialIndices,
#                                          leftPolynomialJoints.index(
#                                           'lumbar_rotation')]        
#             # Both legs        
#             lMTj_lr = ca.vertcat(lMTj_l[leftPolynomialMuscleIndices], 
#                                   lMTj_r[rightPolynomialMuscleIndices])
#             vMTj_lr = ca.vertcat(vMTj_l[leftPolynomialMuscleIndices], 
#                                   vMTj_r[rightPolynomialMuscleIndices])
            
#             ###################################################################
#             # Derive Hill-equilibrium        
#             [hillEquilibriumj, Fj, activeFiberForcej, passiveFiberForcej,
#               normActiveFiberLengthForcej, normFiberLengthj, fiberVelocityj]=(
#               f_hillEquilibrium(akj[:, j+1], lMTj_lr, vMTj_lr, 
#                                 normFkj_nsc[:, j+1], normFDtj_nsc[:, j]))  
                  
#             ###################################################################
#             # Get passive joint torques
#             if enableLimitTorques:                
#                 passiveJointTorque_hip_flexion_rj = (
#                     f_passiveJointTorque_hip_flexion(
#                         Qskj_nsc[joints.index('hip_flexion_r'), j+1], 
#                         Qdotskj_nsc[joints.index('hip_flexion_r'), j+1]))
#                 passiveJointTorque_hip_flexion_lj = (
#                     f_passiveJointTorque_hip_flexion(
#                         Qskj_nsc[joints.index('hip_flexion_l'), j+1], 
#                         Qdotskj_nsc[joints.index('hip_flexion_l'), j+1]))        
#                 passiveJointTorque_hip_adduction_rj = (
#                     f_passiveJointTorque_hip_adduction(
#                         Qskj_nsc[joints.index('hip_adduction_r'), j+1], 
#                         Qdotskj_nsc[joints.index('hip_adduction_r'), j+1]))
#                 passiveJointTorque_hip_adduction_lj = (
#                     f_passiveJointTorque_hip_adduction(
#                         Qskj_nsc[joints.index('hip_adduction_l'), j+1], 
#                         Qdotskj_nsc[joints.index('hip_adduction_l'), j+1]))        
#                 passiveJointTorque_hip_rotation_rj = (
#                     f_passiveJointTorque_hip_rotation(
#                         Qskj_nsc[joints.index('hip_rotation_r'), j+1], 
#                         Qdotskj_nsc[joints.index('hip_rotation_r'), j+1]))
#                 passiveJointTorque_hip_rotation_lj = (
#                     f_passiveJointTorque_hip_rotation(
#                         Qskj_nsc[joints.index('hip_rotation_l'), j+1], 
#                         Qdotskj_nsc[joints.index('hip_rotation_l'), j+1]))        
#                 passiveJointTorque_knee_angle_rj = (
#                     f_passiveJointTorque_knee_angle(
#                         Qskj_nsc[joints.index('knee_angle_r'), j+1], 
#                         Qdotskj_nsc[joints.index('knee_angle_r'), j+1]))
#                 passiveJointTorque_knee_angle_lj = (
#                     f_passiveJointTorque_knee_angle(
#                         Qskj_nsc[joints.index('knee_angle_l'), j+1], 
#                         Qdotskj_nsc[joints.index('knee_angle_l'), j+1]))        
#                 passiveJointTorque_ankle_angle_rj = (
#                     f_passiveJointTorque_ankle_angle(
#                         Qskj_nsc[joints.index('ankle_angle_r'), j+1], 
#                         Qdotskj_nsc[joints.index('ankle_angle_r'), j+1]))
#                 passiveJointTorque_ankle_angle_lj = (
#                     f_passiveJointTorque_ankle_angle(
#                         Qskj_nsc[joints.index('ankle_angle_l'), j+1], 
#                         Qdotskj_nsc[joints.index('ankle_angle_l'), j+1]))        
#                 passiveJointTorque_subtalar_angle_rj = (
#                     f_passiveJointTorque_subtalar_angle(
#                         Qskj_nsc[joints.index('subtalar_angle_r'), j+1], 
#                         Qdotskj_nsc[joints.index('subtalar_angle_r'), j+1]))
#                 passiveJointTorque_subtalar_angle_lj = (
#                     f_passiveJointTorque_subtalar_angle(
#                         Qskj_nsc[joints.index('subtalar_angle_l'), j+1], 
#                         Qdotskj_nsc[joints.index('subtalar_angle_l'), j+1]))    
#                 passiveJointTorque_mtp_angle_rj = (
#                     f_passiveJointTorque_mtp_angle(
#                         Qskj_nsc[joints.index('mtp_angle_r'), j+1], 
#                         Qdotskj_nsc[joints.index('mtp_angle_r'), j+1]))
#                 passiveJointTorque_mtp_angle_lj = (
#                     f_passiveJointTorque_mtp_angle(
#                         Qskj_nsc[joints.index('mtp_angle_l'), j+1], 
#                         Qdotskj_nsc[joints.index('mtp_angle_l'), j+1]))    
#                 passiveJointTorque_lumbar_extensionj = (
#                     f_passiveJointTorque_lumbar_extension(
#                         Qskj_nsc[joints.index('lumbar_extension'), j+1], 
#                         Qdotskj_nsc[joints.index('lumbar_extension'), j+1]))        
#                 passiveJointTorque_lumbar_bendingj = (
#                     f_passiveJointTorque_lumbar_bending(
#                         Qskj_nsc[joints.index('lumbar_bending'), j+1], 
#                         Qdotskj_nsc[joints.index('lumbar_bending'), j+1]))        
#                 passiveJointTorque_lumbar_rotationj = (
#                     f_passiveJointTorque_lumbar_rotation(
#                         Qskj_nsc[joints.index('lumbar_rotation'), j+1], 
#                         Qdotskj_nsc[joints.index('lumbar_rotation'), j+1]))   
#             else:
#                 passiveJointTorque_hip_flexion_rj = 0
#                 passiveJointTorque_hip_flexion_lj = 0       
#                 passiveJointTorque_hip_adduction_rj = 0
#                 passiveJointTorque_hip_adduction_lj = 0
#                 passiveJointTorque_hip_rotation_rj = 0
#                 passiveJointTorque_hip_rotation_lj = 0     
#                 passiveJointTorque_knee_angle_rj = 0
#                 passiveJointTorque_knee_angle_lj = 0       
#                 passiveJointTorque_ankle_angle_rj = 0
#                 passiveJointTorque_ankle_angle_lj = 0      
#                 passiveJointTorque_subtalar_angle_rj = 0
#                 passiveJointTorque_subtalar_angle_lj = 0  
#                 passiveJointTorque_mtp_angle_rj = 0
#                 passiveJointTorque_mtp_angle_lj = 0   
#                 passiveJointTorque_lumbar_extensionj = 0        
#                 passiveJointTorque_lumbar_bendingj = 0     
#                 passiveJointTorque_lumbar_rotationj = 0
            
#             linearPassiveJointTorque_mtp_angle_lj = f_linearPassiveMtpTorque(
#                     Qskj_nsc[joints.index('mtp_angle_l'), j+1],
#                     Qdotskj_nsc[joints.index('mtp_angle_l'), j+1])
#             linearPassiveJointTorque_mtp_angle_rj = f_linearPassiveMtpTorque(
#                     Qskj_nsc[joints.index('mtp_angle_r'), j+1],
#                     Qdotskj_nsc[joints.index('mtp_angle_r'), j+1])     
            
#             passiveJointTorquesj = ca.vertcat(
#                     passiveJointTorque_hip_flexion_rj,
#                     passiveJointTorque_hip_flexion_lj,
#                     passiveJointTorque_hip_adduction_rj,
#                     passiveJointTorque_hip_adduction_lj,
#                     passiveJointTorque_hip_rotation_rj,
#                     passiveJointTorque_hip_rotation_lj,
#                     passiveJointTorque_knee_angle_rj,
#                     passiveJointTorque_knee_angle_lj,
#                     passiveJointTorque_ankle_angle_rj,
#                     passiveJointTorque_ankle_angle_lj,
#                     passiveJointTorque_subtalar_angle_rj,
#                     passiveJointTorque_subtalar_angle_lj,
#                     passiveJointTorque_mtp_angle_rj,
#                     passiveJointTorque_mtp_angle_lj,
#                     passiveJointTorque_lumbar_extensionj,
#                     passiveJointTorque_lumbar_bendingj,
#                     passiveJointTorque_lumbar_rotationj)
                    
            ###################################################################
            # Cost function
            # activationTerm = f_NMusclesSum2(akj[:, j+1])     
            actJExcitationTerm = f_NActJointsSum2(eActJk) 
            gtJExcitationTerm = f_NGroundThoraxJointsSum2(eGTJk) 
            jointAccelerationTerm = f_NJointsSum2(Qdotdotsj[:, j])   
            lambdaTerm = f_NHolConstraintsSum2(lambdaj[:, j])
            if velocity_correction:
                gammaTerm = f_NHolConstraintsSum2(gammaj[:, j]) 
            
            
            # passiveJointTorqueTerm = (
            #         f_NPassiveTorqueJointsSum2(passiveJointTorquesj))       
            # activationDtTerm = f_NMusclesSum2(aDtk)
            # forceDtTerm = f_NMusclesSum2(normFDtj[:, j])
                    
            # J += ((weights['activationTerm'] * activationTerm + 
            #         weights['actJExcitationTerm'] * actJExcitationTerm + 
            #         weights['jointAccelerationTerm'] * jointAccelerationTerm +                
            #         weights['passiveJointTorqueTerm'] * passiveJointTorqueTerm + 
            #         weights['controls'] * (forceDtTerm + activationDtTerm)
            #         ) * h * B[j + 1])
            
            if velocity_correction:
                J += ((weights['actJExcitationTerm'] * actJExcitationTerm + 
                       weights['gtJExcitationTerm'] * gtJExcitationTerm + 
                       weights['jointAccelerationTerm'] * jointAccelerationTerm +                
                       weights['lambdaTerm'] * lambdaTerm + 
                       weights['gammaTerm'] * gammaTerm) * h * B[j + 1])
            else:
                J += ((weights['actJExcitationTerm'] * actJExcitationTerm + 
                       weights['gtJExcitationTerm'] * gtJExcitationTerm +
                       weights['jointAccelerationTerm'] * jointAccelerationTerm +                
                       weights['lambdaTerm'] * lambdaTerm) * h * B[j + 1])
            
            
            # Call external function (run inverse dynamics among other)
            if velocity_correction:
                Tj = F(ca.vertcat(QsQdotskj_nsc[:, j+1], Qdotdotsj_nsc[:, j], 
                                  lambdaj_nsc[:, j], gammaj_nsc[:, j]))
                # Extract the velocity coorectors and reconstruct vector.
                qdotCorrj = Tj[idxVelCorrs["applied"]]
                qdotCorr_allj = ca.MX(NJoints, 1)
                qdotCorr_allj[idxNoJointVelCorr,:] = 0
                qdotCorr_allj[idxJointVelCorr,:] = qdotCorrj   
            else:
                Tj = F(ca.vertcat(QsQdotskj_nsc[:, j+1], Qdotdotsj_nsc[:, j], 
                                  lambdaj_nsc[:, j]))
            
            ###################################################################
            # Expression for the state derivatives at the collocation points
            # ap = ca.mtimes(akj, C[j+1])        
            # normFp_nsc = ca.mtimes(normFkj_nsc, C[j+1])
            Qsp_nsc = ca.mtimes(Qskj_nsc, C[j+1])
            Qdotsp_nsc = ca.mtimes(Qdotskj_nsc, C[j+1])      
            aActJp = ca.mtimes(aActJkj, C[j+1])
            aGTJp = ca.mtimes(aGTJkj, C[j+1])
            # Append collocation equations
            # Muscle activation dynamics (implicit formulation)
            # eq_constr.append((h*aDtk_nsc - ap))
            # Muscle contraction dynamics (implicit formulation)  
            # eq_constr.append((h*normFDtj_nsc[:, j] - normFp_nsc) / 
            #                 scalingF.to_numpy().T)
            # Skeleton dynamics (explicit formulation) 
            # Position derivative
            if velocity_correction:
                eq_constr.append((h*(Qdotskj_nsc[:, j+1] + qdotCorr_allj) - 
                                  Qsp_nsc) / scalingQs.to_numpy().T)
            else:
                eq_constr.append((h*(Qdotskj_nsc[:, j+1]) - 
                                  Qsp_nsc) / scalingQs.to_numpy().T)
            # Velocity derivative
            eq_constr.append((h*Qdotdotsj_nsc[:, j] - Qdotsp_nsc) / 
                             scalingQdots.to_numpy().T)
            # Actuated joints activation dynamics (explicit formulation) 
            aActJDtj = f_actJointsDynamics(eActJk, aActJkj[:, j+1])
            eq_constr.append(h*aActJDtj - aActJp)
            # ground thorax joints activation dynamics (explicit formulation) 
            aGTJDtj = f_groundThoraxJointsDynamics(eGTJk, aGTJkj[:, j+1])
            eq_constr.append(h*aGTJDtj - aGTJp)
            
            ###################################################################
            # Path constraints        
#             if tracking_data == "markers":
#                 # Extract marker trajectories for tracking terms;
#                 # markerj is overwritten in the loop over j but we are only
#                 # interested in the last collocation point, since it
#                 # corresponds to the mesh point. So fine... but not great.
#                 markerj = Tj[idxMarker["toTrack"]]        
                
            # # Enforce dynamic consistency (null thorax residuals)
            # eq_constr.append(Tj[idxGroundThoraxJoints])
            
            # Actuate joints with ideal motor torques
            for count, joint in enumerate(joints[joints.index("clav_prot"):]):
                diffTj = f_diffTorques(
                    Tj[joints.index(joint)] / scalingActJE.iloc[0][joint],
                    aActJkj[count, j+1], 0)
                eq_constr.append(diffTj)
                
            # Actuate ground thorax joints with ideal motor torques
            for count, joint in enumerate(groundThoraxJoints):
                diffTj = f_diffTorques(
                    Tj[joints.index(joint)] / scalingGTJE.iloc[0][joint],
                    aGTJkj[count, j+1], 0)
                eq_constr.append(diffTj)
                
            # Enforce kinematics constraints 
            eq_constr.append(Tj[idxKinConstraints["applied"]])           
            
#             ###################################################################
#             # Muscle-driven joint torques
#             # Hip flexion: left
#             Fj_hip_flexion_l = Fj[momentArmIndices['hip_flexion_l']] 
#             mTj_hip_flexion_l = f_NHipSumProd(dMj_hip_flexion_l,
#                                               Fj_hip_flexion_l)        
#             diffTj_hip_flexion_l = f_diffTorques(
#                     Tj[joints.index('hip_flexion_l')], mTj_hip_flexion_l, 
#                     passiveJointTorque_hip_flexion_lj)
#             eq_constr.append(diffTj_hip_flexion_l)
#             # Hip flexion: right
#             Fj_hip_flexion_r = Fj[momentArmIndices['hip_flexion_r']]
#             mTj_hip_flexion_r = f_NHipSumProd(dMj_hip_flexion_r,
#                                               Fj_hip_flexion_r)
#             diffTj_hip_flexion_r = f_diffTorques(
#                     Tj[joints.index('hip_flexion_r')], mTj_hip_flexion_r, 
#                     passiveJointTorque_hip_flexion_rj)
#             eq_constr.append(diffTj_hip_flexion_r)
#             # Hip adduction: left
#             Fj_hip_adduction_l = Fj[momentArmIndices['hip_adduction_l']] 
#             mTj_hip_adduction_l = f_NHipSumProd(dMj_hip_adduction_l, 
#                                                 Fj_hip_adduction_l)
#             diffTj_hip_adduction_l = f_diffTorques(
#                     Tj[joints.index('hip_adduction_l')], mTj_hip_adduction_l, 
#                     passiveJointTorque_hip_adduction_lj)
#             eq_constr.append(diffTj_hip_adduction_l)
#             # Hip adduction: right
#             Fj_hip_adduction_r = Fj[momentArmIndices['hip_adduction_r']]
#             mTj_hip_adduction_r = f_NHipSumProd(dMj_hip_adduction_r, 
#                                                 Fj_hip_adduction_r)
#             diffTj_hip_adduction_r = f_diffTorques(
#                     Tj[joints.index('hip_adduction_r')], mTj_hip_adduction_r, 
#                     passiveJointTorque_hip_adduction_rj)
#             eq_constr.append(diffTj_hip_adduction_r)
#             # Hip rotation: left
#             Fj_hip_rotation_l = Fj[momentArmIndices['hip_rotation_l']] 
#             mTj_hip_rotation_l = f_NHipSumProd(dMj_hip_rotation_l, 
#                                                 Fj_hip_rotation_l)
#             diffTj_hip_rotation_l = f_diffTorques(
#                     Tj[joints.index('hip_rotation_l')], mTj_hip_rotation_l, 
#                     passiveJointTorque_hip_rotation_lj)
#             eq_constr.append(diffTj_hip_rotation_l)
#             # Hip rotation: right
#             Fj_hip_rotation_r = Fj[momentArmIndices['hip_rotation_r']]
#             mTj_hip_rotation_r = f_NHipSumProd(dMj_hip_rotation_r, 
#                                                 Fj_hip_rotation_r)
#             diffTj_hip_rotation_r = f_diffTorques(
#                     Tj[joints.index('hip_rotation_r')], mTj_hip_rotation_r, 
#                     passiveJointTorque_hip_rotation_rj)
#             eq_constr.append(diffTj_hip_rotation_r)
#             # Knee angle: left
#             Fj_knee_angle_l = Fj[momentArmIndices['knee_angle_l']] 
#             mTj_knee_angle_l = f_NKneeSumProd(dMj_knee_angle_l, Fj_knee_angle_l)
#             diffTj_knee_angle_l = f_diffTorques(
#                     Tj[joints.index('knee_angle_l')], mTj_knee_angle_l, 
#                     passiveJointTorque_knee_angle_lj)
#             eq_constr.append(diffTj_knee_angle_l)
#             # Knee angle: right
#             Fj_knee_angle_r = Fj[momentArmIndices['knee_angle_r']]
#             mTj_knee_angle_r = f_NKneeSumProd(dMj_knee_angle_r, Fj_knee_angle_r)
#             diffTj_knee_angle_r = f_diffTorques(
#                     Tj[joints.index('knee_angle_r')], mTj_knee_angle_r, 
#                     passiveJointTorque_knee_angle_rj)
#             eq_constr.append(diffTj_knee_angle_r)
#             # Ankle angle: left
#             Fj_ankle_angle_l = Fj[momentArmIndices['ankle_angle_l']] 
#             mTj_ankle_angle_l = f_NAnkleSumProd(dMj_ankle_angle_l, 
#                                                 Fj_ankle_angle_l)
#             diffTj_ankle_angle_l = f_diffTorques(
#                     Tj[joints.index('ankle_angle_l')], mTj_ankle_angle_l, 
#                     passiveJointTorque_ankle_angle_lj)
#             eq_constr.append(diffTj_ankle_angle_l)
#             # Ankle angle: right
#             Fj_ankle_angle_r = Fj[momentArmIndices['ankle_angle_r']]
#             mTj_ankle_angle_r = f_NAnkleSumProd(dMj_ankle_angle_r, 
#                                                 Fj_ankle_angle_r)
#             diffTj_ankle_angle_r = f_diffTorques(
#                     Tj[joints.index('ankle_angle_r')], mTj_ankle_angle_r, 
#                     passiveJointTorque_ankle_angle_rj)
#             eq_constr.append(diffTj_ankle_angle_r)
#             # Subtalar angle: left
#             Fj_subtalar_angle_l = Fj[momentArmIndices['subtalar_angle_l']] 
#             mTj_subtalar_angle_l = f_NSubtalarSumProd(dMj_subtalar_angle_l, 
#                                                       Fj_subtalar_angle_l)
#             diffTj_subtalar_angle_l = f_diffTorques(
#                     Tj[joints.index('subtalar_angle_l')], mTj_subtalar_angle_l, 
#                     passiveJointTorque_subtalar_angle_lj)
#             eq_constr.append(diffTj_subtalar_angle_l)
#             # Subtalar angle: right
#             Fj_subtalar_angle_r = Fj[momentArmIndices['subtalar_angle_r']]
#             mTj_subtalar_angle_r = f_NSubtalarSumProd(dMj_subtalar_angle_r, 
#                                                       Fj_subtalar_angle_r)
#             diffTj_subtalar_angle_r = f_diffTorques(
#                     Tj[joints.index('subtalar_angle_r')], mTj_subtalar_angle_r, 
#                     passiveJointTorque_subtalar_angle_rj)
#             eq_constr.append(diffTj_subtalar_angle_r)
#             # Trunk extension
#             Fj_lumbar_extension = Fj[momentArmIndices['lumbar_extension']]      
#             mTj_lumbar_extension = f_NTrunkSumProd(dMj_lumbar_extension, 
#                                                     Fj_lumbar_extension)
#             diffTj_lumbar_extension = f_diffTorques(
#                     Tj[joints.index('lumbar_extension')], mTj_lumbar_extension, 
#                     passiveJointTorque_lumbar_extensionj)
#             eq_constr.append(diffTj_lumbar_extension)
#             # Trunk bending
#             Fj_lumbar_bending = Fj[momentArmIndices['lumbar_bending']] 
#             mTj_lumbar_bending = f_NTrunkSumProd(dMj_lumbar_bending, 
#                                                   Fj_lumbar_bending)
#             diffTj_lumbar_bending = f_diffTorques(
#                     Tj[joints.index('lumbar_bending')], mTj_lumbar_bending, 
#                     passiveJointTorque_lumbar_bendingj)
#             eq_constr.append(diffTj_lumbar_bending)
#             # Trunk rotation
#             Fj_lumbar_rotation = Fj[momentArmIndices['lumbar_rotation']]  
#             mTj_lumbar_rotation = f_NTrunkSumProd(dMj_lumbar_rotation, 
#                                                   Fj_lumbar_rotation)
#             diffTj_lumbar_rotation = f_diffTorques(
#                     Tj[joints.index('lumbar_rotation')], mTj_lumbar_rotation, 
#                     passiveJointTorque_lumbar_rotationj)
#             eq_constr.append(diffTj_lumbar_rotation)
                
#             ###################################################################
#             # Torque-driven joint torques (mtp joints)     
#             diffTj_mtp_angle_l = f_diffTorques(
#                     Tj[joints.index('mtp_angle_l')] / 
#                     scalingMtpE.iloc[0]['mtp_angle_l'],
#                     aTMkj[0, j+1],
#                     (passiveJointTorque_mtp_angle_lj +
#                       linearPassiveJointTorque_mtp_angle_lj) /
#                     scalingMtpE.iloc[0]['mtp_angle_l'])
#             eq_constr.append(diffTj_mtp_angle_l)
#             diffTj_mtp_angle_r = f_diffTorques(
#                     Tj[joints.index('mtp_angle_r')] / 
#                     scalingMtpE.iloc[0]['mtp_angle_r'], 
#                     aTMkj[1, j+1], 
#                     (passiveJointTorque_mtp_angle_rj +
#                       linearPassiveJointTorque_mtp_angle_rj) /
#                     scalingMtpE.iloc[0]['mtp_angle_r'])
#             eq_constr.append(diffTj_mtp_angle_r)
            
#             ###################################################################
#             # Activation dynamics (implicit formulation)
#             act1 = aDtk_nsc + akj[:, j+1] / deactivationTimeConstant
#             act2 = aDtk_nsc + akj[:, j+1] / activationTimeConstant
#             ineq_constr1.append(act1)
#             ineq_constr2.append(act2)
            
#             ###################################################################
#             # Contraction dynamics (implicit formulation)
#             eq_constr.append(hillEquilibriumj)
        # End loop over collocation points
        
        #######################################################################
        # Flatten constraint vectors
        eq_constr = ca.vertcat(*eq_constr)
        # ineq_constr1 = ca.vertcat(*ineq_constr1)
        # ineq_constr2 = ca.vertcat(*ineq_constr2)
        # Create function for map construct (parallel computing)    
        '''
        if tracking_data == "markers":
            f_coll = ca.Function('f_coll', [ak, aj, normFk, normFj, Qsk, 
                                            Qsj, Qdotsk, Qdotsj, 
                                            aTMk, aTMj, aDtk, eTMk,
                                            normFDtj, Qdotdotsj],
                [eq_constr, ineq_constr1, ineq_constr2, J, markerj])     
        if tracking_data == "coordinates":
            f_coll = ca.Function('f_coll', [ak, aj, normFk, normFj, Qsk, 
                                            Qsj, Qdotsk, Qdotsj, 
                                            aTMk, aTMj, aDtk, eTMk,
                                            normFDtj, Qdotdotsj],
                [eq_constr, ineq_constr1, ineq_constr2, J])                 
        # Create map construct
        f_coll_map = f_coll.map(N, parallelMode, NThreads)   
        # Call function with opti variables and set constraints
        if tracking_data == "markers":
            (coll_eq_constr, coll_ineq_constr1, coll_ineq_constr2, JPred,
              marker_sim) = (
                      f_coll_map(a[:, :-1], a_col, normF[:, :-1], normF_col, 
                                Qs[:, :-1], Qs_col, Qdots[:, :-1], Qdots_col, 
                                aTM[:, :-1], aTM_col,
                                aDt, eTM, normFDt_col, Qdotdots_col))    
        elif tracking_data == "coordinates":
            (coll_eq_constr, coll_ineq_constr1, coll_ineq_constr2, JPred) = (
                f_coll_map(a[:, :-1], a_col, normF[:, :-1], normF_col, 
                            Qs[:, :-1], Qs_col, Qdots[:, :-1], Qdots_col, 
                            aTM[:, :-1], aTM_col, aDt, eTM,
                            normFDt_col, Qdotdots_col))  
        opti.subject_to(ca.vec(coll_eq_constr) == 0)
        opti.subject_to(ca.vec(coll_ineq_constr1) >= 0)
        opti.subject_to(ca.vec(coll_ineq_constr2) <= 1/activationTimeConstant)  
        '''
        if tracking_data == "coordinates":
            if velocity_correction:
                f_coll = ca.Function('f_coll', 
                                     [Qsk, Qsj, Qdotsk, Qdotsj, aActJk, aActJj,
                                      aGTJk, aGTJj, eActJk, eGTJk, Qdotdotsj,
                                      lambdaj, gammaj], [eq_constr, J]) 
            else:
                f_coll = ca.Function('f_coll', 
                                     [Qsk, Qsj, Qdotsk, Qdotsj, aActJk, aActJj,
                                      aGTJk, aGTJj, eActJk, eGTJk, Qdotdotsj,
                                      lambdaj], [eq_constr, J])         
        # Create map construct
        f_coll_map = f_coll.map(N, parallelMode, NThreads)   
        # Call function with opti variables and set constraints
        if tracking_data == "coordinates":
            if velocity_correction:
                (coll_eq_constr, JPred) = (
                    f_coll_map(Qs[:, :-1], Qs_col, Qdots[:, :-1], Qdots_col, 
                               aActJ[:, :-1], aActJ_col, aGTJ[:, :-1],
                               aGTJ_col, eActJ, eGTJ, Qdotdots_col, lambda_col,
                               gamma_col))  
            else:
                (coll_eq_constr, JPred) = (
                    f_coll_map(Qs[:, :-1], Qs_col, Qdots[:, :-1], Qdots_col, 
                               aActJ[:, :-1], aActJ_col, aGTJ[:, :-1],
                               aGTJ_col, eActJ, eGTJ, Qdotdots_col,
                               lambda_col))
                
        opti.subject_to(ca.vec(coll_eq_constr) == 0)
                
        #######################################################################
        # Equality / continuity constraints
        # Loop over mesh points
        for k in range(N):
            # akj2 = (ca.horzcat(a[:, k], a_col[:, k*d:(k+1)*d]))
            # normFkj2 = (ca.horzcat(normF[:, k], normF_col[:, k*d:(k+1)*d]))
            Qskj2 = (ca.horzcat(Qs[:, k], Qs_col[:, k*d:(k+1)*d]))
            Qdotskj2 = (ca.horzcat(Qdots[:, k], Qdots_col[:, k*d:(k+1)*d]))  
            aActJkj2 = (ca.horzcat(aActJ[:, k], aActJ_col[:, k*d:(k+1)*d]))
            aGTJkj2 = (ca.horzcat(aGTJ[:, k], aGTJ_col[:, k*d:(k+1)*d]))
            
            # opti.subject_to(a[:, k+1] == ca.mtimes(akj2, D))
            # opti.subject_to(normF[:, k+1] == ca.mtimes(normFkj2, D))    
            opti.subject_to(Qs[:, k+1] == ca.mtimes(Qskj2, D))
            opti.subject_to(Qdots[:, k+1] == ca.mtimes(Qdotskj2, D))    
            opti.subject_to(aActJ[:, k+1] == ca.mtimes(aActJkj2, D))
            opti.subject_to(aGTJ[:, k+1] == ca.mtimes(aGTJkj2, D))
            
        #######################################################################
        # Add tracking terms, only at the mesh points.  
        # Adjust the y location of the markers using the offset 
        # if tracking_data == "markers": 
        #     dataToTrack_sc_offset = ca.MX(dataToTrack_sc.shape[0],
        #                                   dataToTrack_sc.shape[1])
        #     dataToTrack_sc_offset[0::3, :] = dataToTrack_sc[0::3, :]
        #     dataToTrack_sc_offset[2::3, :] = dataToTrack_sc[2::3, :]
        #     if offset_ty:  
        #         dataToTrack_sc_offset[1::3, :] = (
        #             dataToTrack_sc[1::3, :] + offset)
        #     else:
        #         dataToTrack_sc_offset[1::3, :] = dataToTrack_sc[1::3, :]
        #     if markers_as_controls:
        #         if norm_std:
        #             JTrack = f_track_k_map(marker_u, dataToTrack_sc_offset,
        #                                     dataToTrack_std_sc)
        #         else:
        #             JTrack = f_track_k_map(marker_u, dataToTrack_sc_offset)
        #         # Since we have additional controls, we need to add constraints
        #         # imposing those controls to match the simulated data.
        #         opti.subject_to(marker_u_nsc - marker_sim == 0)   
        #     else:
        #         # Scale the simulated marker data (modeling choice)
        #         marker_sim_sc = marker_sim / (
        #             scalingMarker.to_numpy().T *np.ones((1, N)))
        #         if norm_std:
        #             JTrack = f_track_k_map(marker_sim_sc, 
        #                                     dataToTrack_sc_offset,
        #                                     dataToTrack_std_sc)
        #         else:
        #             JTrack = f_track_k_map(marker_sim_sc, 
        #                                     dataToTrack_sc_offset)
        #     JTrack_sc = (
        #         weights['trackingTerm'] * f_mySum(JTrack) * h / timeElapsed)              
        if tracking_data == "coordinates":  
            # Rotational DOFs
            JTrack_rot = f_track_k_map(Qs[idxRotCoordinates_toTrack,:],
                                       dataToTrack_sc)            
            JTrack_rot_sc = (weights['trackingTerm'] * 
                             f_mySumTrack(JTrack_rot) * h / timeElapsed)
            # # Translational DOFs            
            # if coordinates_toTrack['translational']:
            #     if offset_ty:                    
            #         dataToTrack_tr_sc_offset = ca.MX(
            #             dataToTrack_tr_sc.shape[0], dataToTrack_tr_sc.shape[1])                    
            #         for j, joint in enumerate(
            #                 coordinates_toTrack['translational']):                        
            #             if joint == "pelvis_ty":                        
            #                 dataToTrack_tr_sc_offset[j, :] = (
            #                     dataToTrack_tr_sc[j, :] + offset)
            #             else:
            #                 dataToTrack_tr_sc_offset[j, :] = (
            #                     dataToTrack_tr_sc[j, :])
            #         JTrack_tr = f_track_tr_k_map(
            #             Qs[idxTrCoordinates_toTrack,:],
            #             dataToTrack_tr_sc_offset)                    
            #     else:
            #         JTrack_tr = f_track_tr_k_map(
            #             Qs[idxTrCoordinates_toTrack,:],
            #             dataToTrack_tr_sc)
            #     JTrack_tr_sc = (weights['trackingTerm_tr'] * 
            #                     f_mySumTrack(JTrack_tr) * h / timeElapsed)
            #     JTrack_sc = JTrack_rot_sc + JTrack_tr_sc
            # else:
            JTrack_sc = JTrack_rot_sc
                
#         #######################################################################
#         if enforceSpeed:
#             # Average speed constraint
#             Qs_nsc = Qs * (scalingQs.to_numpy().T * np.ones((1, N+1)))
#             distTraveled =  (Qs_nsc[joints.index('pelvis_tx'), -1] - 
#                              Qs_nsc[joints.index('pelvis_tx'), 0])
#             simSpeed = distTraveled / timeElapsed
#             opti.subject_to(simSpeed - targetSpeed == 0)
            
#         #######################################################################
#         if tracking_data == "coordinates": 
#             if offset_ty:
#                 pelvis_ty_offset_sc = pelvis_ty_sc + offset
#             else:
#                 pelvis_ty_offset_sc = pelvis_ty_sc
#             if constraintPelvis_ty["end_cond"]:
#                 # Endpoint constraints on pelvis_ty
#                 opti.subject_to(opti.bounded(
#                     -constraintPelvis_ty["end_bound"] / 
#                     scalingQs.iloc[0]["pelvis_ty"], 
#                     Qs[joints.index("pelvis_ty"), 0] - 
#                     pelvis_ty_offset_sc[0, 0], 
#                     constraintPelvis_ty["end_bound"] / 
#                     scalingQs.iloc[0]["pelvis_ty"]))
#                 opti.subject_to(opti.bounded(
#                     -constraintPelvis_ty["end_bound"] / 
#                     scalingQs.iloc[0]["pelvis_ty"], 
#                     Qs[joints.index("pelvis_ty"), -1] - 
#                     pelvis_ty_offset_sc[0, -1], 
#                     constraintPelvis_ty["end_bound"] / 
#                     scalingQs.iloc[0]["pelvis_ty"]))   
#             # Constraint imposing first and last pelvis_ty value to be both
#             # either larger or small than experimental data; the first value
#             # being larger (smaller) and the last value being larger (smaller)
#             # is thus prevented. This is because the model is cheating by
#             # starting the movement into the ground
#             if constraintPelvis_ty["side"]:
#                 first_diff_pelvis_ty = (Qs[joints.index("pelvis_ty"), 0] - 
#                                         pelvis_ty_offset_sc[0, 0])
#                 last_diff_pelvis_ty = (Qs[joints.index("pelvis_ty"), -1] - 
#                                         pelvis_ty_offset_sc[0, -1])
#                 opti.subject_to(first_diff_pelvis_ty*last_diff_pelvis_ty >= 0)  
                
#             if constraintPelvis_ty["env_cond"]:
#                 # Constraints on pelvis_ty over entire trajectory
#                 opti.subject_to(opti.bounded(
#                     -constraintPelvis_ty["env_bound"] / 
#                     scalingQs.iloc[0]["pelvis_ty"], 
#                     Qs[joints.index("pelvis_ty"), :] - 
#                     pelvis_ty_offset_sc[0, :], 
#                     constraintPelvis_ty["env_bound"] / 
#                     scalingQs.iloc[0]["pelvis_ty"]))
                
#         if tracking_data == "markers":
#             if constraintMidHip_y["env_cond"]:         
#                 midHip_y_sc_offset = dataToTrack_sc_offset[
#                     marker_titles.index('MidHip_y'), :]
#                 # Constraints on MidHip_y over entire trajectory
#                 opti.subject_to(opti.bounded(
#                     -constraintMidHip_y["env_bound"] / 
#                     scalingMarker.iloc[0]["MidHip_y"],           
#                     marker_u[marker_titles.index('MidHip_y'), :] - 
#                     midHip_y_sc_offset[0, :], 
#                     constraintMidHip_y["env_bound"] / 
#                     scalingMarker.iloc[0]["MidHip_y"]))
            
        
        #######################################################################
        # Scale cost function with distance traveled
        JPred_sc = f_mySum(JPred) / timeElapsed  
        
        #######################################################################
        # Create NLP solver        
        opti.minimize(JPred_sc + JTrack_sc)
        # Solve problem
        from variousFunctions import solve_with_bounds
        w_opt, stats = solve_with_bounds(opti, tol)
        if saveResults:               
            np.save(os.path.join(pathResults, 'w_opt.npy'), w_opt)
            np.save(os.path.join(pathResults, 'stats.npy'), stats)
        
    # %% Analyze results
    if analyzeResults:
        if loadResults:
            w_opt = np.load(os.path.join(pathResults, 'w_opt.npy'))
            stats = np.load(os.path.join(pathResults, 'stats.npy'), 
                            allow_pickle=True).item()  
        if not stats['success'] == True:
            print("WARNING: PROBLEM DID NOT CONVERGE - " 
                  + stats['return_status'])  
        # if offset_ty:
        #     NParameters = 1    
        #     offset_opt = w_opt[:NParameters]
        # else:
        NParameters = 0
        starti = NParameters    
        '''
        a_opt = (np.reshape(w_opt[starti:starti+NMuscles*(N+1)],
                                  (N+1, NMuscles))).T
        starti = starti + NMuscles*(N+1)
        a_col_opt = (np.reshape(w_opt[starti:starti+NMuscles*(d*N)],
                                      (d*N, NMuscles))).T    
        starti = starti + NMuscles*(d*N)
        normF_opt = (np.reshape(w_opt[starti:starti+NMuscles*(N+1)],
                                      (N+1, NMuscles))  ).T  
        starti = starti + NMuscles*(N+1)
        normF_col_opt = (np.reshape(w_opt[starti:starti+NMuscles*(d*N)],
                                          (d*N, NMuscles))).T
        starti = starti + NMuscles*(d*N)
        '''
        Qs_opt = (np.reshape(w_opt[starti:starti+NJoints*(N+1)],
                                    (N+1, NJoints))  ).T  
        starti = starti + NJoints*(N+1)    
        Qs_col_opt = (np.reshape(w_opt[starti:starti+NJoints*(d*N)],
                                        (d*N, NJoints))).T
        starti = starti + NJoints*(d*N)
        Qdots_opt = (np.reshape(w_opt[starti:starti+NJoints*(N+1)],
                                      (N+1, NJoints)) ).T   
        starti = starti + NJoints*(N+1)    
        Qdots_col_opt = (np.reshape(w_opt[starti:starti+NJoints*(d*N)],
                                          (d*N, NJoints))).T
        starti = starti + NJoints*(d*N)      
        aActJ_opt = (np.reshape(w_opt[starti:starti+NActJoints*(N+1)],
                                      (N+1, NActJoints))).T
        starti = starti + NActJoints*(N+1)    
        aActJ_col_opt = (np.reshape(w_opt[starti:starti+NActJoints*(d*N)],
                                          (d*N, NActJoints))).T
        starti = starti + NActJoints*(d*N)
        aGTJ_opt = (np.reshape(w_opt[starti:starti+NGroundThoraxJoints*(N+1)],
                                      (N+1, NGroundThoraxJoints))).T
        starti = starti + NGroundThoraxJoints*(N+1)    
        aGTJ_col_opt = (np.reshape(w_opt[starti:starti+NGroundThoraxJoints*(d*N)],
                                          (d*N, NGroundThoraxJoints))).T
        starti = starti + NGroundThoraxJoints*(d*N)
        '''
        aDt_opt = (np.reshape(w_opt[starti:starti+NMuscles*N],
                              (N, NMuscles))).T
        starti = starti + NMuscles*N 
        '''
        eActJ_opt = (np.reshape(w_opt[starti:starti+NActJoints*N],
                                (N, NActJoints))).T
        starti = starti + NActJoints*N
        eGTJ_opt = (np.reshape(w_opt[starti:starti+NGroundThoraxJoints*N],
                                (N, NGroundThoraxJoints))).T
        starti = starti + NGroundThoraxJoints*N
        '''
        normFDt_col_opt = (np.reshape(w_opt[starti:starti+NMuscles*(d*N)],
                                            (d*N, NMuscles))).T
        starti = starti + NMuscles*(d*N)
        '''
        Qdotdots_col_opt = (np.reshape(w_opt[starti:starti+NJoints*(d*N)],
                                              (d*N, NJoints))).T
        starti = starti + NJoints*(d*N)
        
        lambda_col_opt = (np.reshape(w_opt[starti:starti+NHolConstraints*(d*N)],
                                              (d*N, NHolConstraints))).T
        starti = starti + NHolConstraints*(d*N)
        if velocity_correction:
            gamma_col_opt = (np.reshape(w_opt[starti:starti+NHolConstraints*(d*N)],
                                                  (d*N, NHolConstraints))).T
            starti = starti + NHolConstraints*(d*N)
        
        # if tracking_data == "markers" and markers_as_controls:
        #     marker_u_opt = (np.reshape(w_opt[starti:starti+NEl_toTrack*(N)],
        #                                 (N, NEl_toTrack))).T
        #     starti = starti + NEl_toTrack*(N)
        assert (starti == w_opt.shape[0]), "error when extracting results"
            
        # %% Unscale results
        # normF_opt_nsc = normF_opt * (scalingF.to_numpy().T * np.ones((1, N+1)))
        # normF_col_opt_nsc = normF_col_opt * (scalingF.to_numpy().T * 
        #                                       np.ones((1, d*N)))    
        Qs_opt_nsc = Qs_opt * (scalingQs.to_numpy().T * np.ones((1, N+1)))
        Qs_col_opt_nsc = Qs_col_opt * (scalingQs.to_numpy().T * 
                                        np.ones((1, d*N)))
        Qdots_opt_nsc = Qdots_opt * (scalingQdots.to_numpy().T * 
                                      np.ones((1, N+1)))
        Qdots_col_opt_nsc = Qdots_col_opt * (scalingQdots.to_numpy().T * 
                                              np.ones((1, d*N)))
        # aDt_opt_nsc = aDt_opt * (scalingADt.to_numpy().T * np.ones((1, N)))
        Qdotdots_col_opt_nsc = Qdotdots_col_opt * (
            scalingQdotdots.to_numpy().T * np.ones((1, d*N)))
        lambda_col_opt_nsc = lambda_col_opt * (
            scalingLambda.to_numpy().T * np.ones((1, d*N)))
        if velocity_correction:
            gamma_col_opt_nsc = gamma_col_opt * (
                scalingGamma.to_numpy().T * np.ones((1, d*N)))
        
        
        # normFDt_col_opt_nsc = normFDt_col_opt * (scalingFDt.to_numpy().T * 
        #                                           np.ones((1, d*N)))
#         if tracking_data == "markers" and markers_as_controls:
#             marker_u_opt_nsc = marker_u_opt * (scalingMarker.to_numpy().T * 
#                                                np.ones((1, N)))
#         if offset_ty:
#             offset_opt_nsc = offset_opt * scalingOffset
            
#         if enforceSpeed:
#             # Assert speed
#             distTraveled_opt = (Qs_opt_nsc[joints.index('pelvis_tx'), -1] - 
#                                 Qs_opt_nsc[joints.index('pelvis_tx'), 0])
#             simSpeed_opt = distTraveled_opt / timeElapsed
#             if stats['success']:
#                 assert (np.abs(simSpeed_opt-targetSpeed) < 10**(-5)), "error speed"
        
#         # %% Extract passive joint torques
#         linearPassiveJointTorque_mtp_angle_l_opt = np.zeros((1, N+1))
#         linearPassiveJointTorque_mtp_angle_r_opt = np.zeros((1, N+1))  
#         passiveJointTorque_mtp_angle_l_opt = np.zeros((1, N+1))
#         passiveJointTorque_mtp_angle_r_opt = np.zeros((1, N+1))
#         for k in range(N+1):
#             linearPassiveJointTorque_mtp_angle_l_opt[0, k] = (
#                 f_linearPassiveMtpTorque(
#                     Qs_opt_nsc[joints.index('mtp_angle_l'), k],
#                     Qdots_opt_nsc[joints.index('mtp_angle_l'), k]))
#             linearPassiveJointTorque_mtp_angle_r_opt[0, k] = (
#                 f_linearPassiveMtpTorque(
#                     Qs_opt_nsc[joints.index('mtp_angle_r'), k],
#                     Qdots_opt_nsc[joints.index('mtp_angle_r'), k]))             
#             if enableLimitTorques:
#                 passiveJointTorque_mtp_angle_l_opt[0, k] = (
#                     f_passiveJointTorque_mtp_angle(
#                         Qs_opt_nsc[joints.index('mtp_angle_l'), k],
#                         Qdots_opt_nsc[joints.index('mtp_angle_l'), k]))
#                 passiveJointTorque_mtp_angle_r_opt[0, k] = (
#                     f_passiveJointTorque_mtp_angle(
#                         Qs_opt_nsc[joints.index('mtp_angle_r'), k],
#                         Qdots_opt_nsc[joints.index('mtp_angle_r'), k]))
#             else:
#                 passiveJointTorque_mtp_angle_l_opt[0, k] = 0
#                 passiveJointTorque_mtp_angle_r_opt[0, k] = 0                
            
        # %% Extract joint torques and ground reaction forces
        QsQdots_opt_nsc = np.zeros((NJoints*2, N+1))
        QsQdots_opt_nsc[::2, :] = Qs_opt_nsc
        QsQdots_opt_nsc[1::2, :] = Qdots_opt_nsc
        Qdotdots_opt = Qdotdots_col_opt_nsc[:,d-1::d] 
        lambda_opt = lambda_col_opt_nsc[:,d-1::d] 
        if velocity_correction:
            gamma_opt = gamma_col_opt_nsc[:,d-1::d] 
        F1_out = np.zeros((NOutput_F1 , N))
        for k in range(N):    
            if velocity_correction:
                Tj = F1(ca.vertcat(QsQdots_opt_nsc[:, k+1], Qdotdots_opt[:, k],
                                   lambda_opt[:, k], gamma_opt[:, k]))
            else:
                Tj = F(ca.vertcat(QsQdots_opt_nsc[:, k+1], Qdotdots_opt[:, k],
                                  lambda_opt[:, k]))
            F1_out[:, k] = Tj.full().T                    
#         if tracking_data == "markers":
#             marker_sim_opt = F1_out[idxMarker["toTrack"], :] 
#             if stats['success'] and markers_as_controls:
#                 assert np.alltrue(
#                     np.abs(marker_sim_opt - marker_u_opt_nsc) 
#                     < 10**(-5)), "error slack markers"                
#             marker_sim_opt_sc = marker_sim_opt / (
#                     scalingMarker.to_numpy().T *np.ones((1, N)))
        
        torques_opt = F1_out[getJointIndices(joints, joints), :] 
        kinCon_opt = F1_out[idxKinConstraints["all"], :] 
        if velocity_correction:
            qdotCorr_opt = F1_out[idxVelCorrs["all"], :] 
        stations_opt = F1_out[idxStations["all"], :]
        assert np.alltrue(stations_opt[:3,:] - 
                          stations_opt[3:,:] < 10**(-5)), "error stations"   
        angVel_opt = F1_out[idxIMUs["radius"]["angVel"], :]
        linAcc_opt = F1_out[idxIMUs["radius"]["linAcc"], :]         
        
#         # %% Data to track - Adjust for offset  
#         if tracking_data == "markers": 
#             if offset_ty:
#                 dataToTrack_sc_offset_opt = np.zeros((dataToTrack_sc.shape[0],
#                                                       dataToTrack_sc.shape[1]))
#                 dataToTrack_sc_offset_opt[0::3, :] = dataToTrack_sc[0::3, :]
#                 dataToTrack_sc_offset_opt[2::3, :] = dataToTrack_sc[2::3, :]
#                 dataToTrack_sc_offset_opt[1::3, :] = (
#                     dataToTrack_sc[1::3, :] + offset_opt)
#             else:
#                 dataToTrack_sc_offset_opt = dataToTrack_sc
#             dataToTrack_nsc_offset_opt = dataToTrack_sc_offset_opt * (
#                 scalingMarker.to_numpy().T * np.ones((1, N)))
                    
#         if TrCoordinates_toTrack_Bool:            
#             dataToTrack_tr_sc_offset_opt = np.zeros(
#                 (dataToTrack_tr_sc.shape[0], dataToTrack_tr_sc.shape[1]))                    
#             for j, joint in enumerate(coordinates_toTrack['translational']):                        
#                 if joint == "pelvis_ty":                        
#                     dataToTrack_tr_sc_offset_opt[j, :] = (
#                         dataToTrack_tr_sc[j, :] + offset_opt)
#                 else:
#                     dataToTrack_tr_sc_offset_opt[j, :] = (
#                         dataToTrack_tr_sc[j, :])
        
        # %% Re-organize data for plotting and GUI   
        import copy
        Qs_opt_nsc_deg = copy.deepcopy(Qs_opt_nsc)
        Qs_opt_nsc_deg[idxRotationalJoints, :] = (
            Qs_opt_nsc_deg[idxRotationalJoints, :] * 180 / np.pi)             
        
        # %% Write motion file for visualization in OpenSim GUI
        if writeMotionFile:    
            # muscleLabels = ([bothSidesMuscle + '/activation' 
            #                   for bothSidesMuscle in bothSidesMuscles])        
            labels = ['time'] + joints   
            # labels_w_muscles = labels + muscleLabels
            labels_w_muscles = labels
            # data = np.concatenate((tgridf.T, Qs_opt_nsc_deg.T, a_opt.T),axis=1)    
            data = np.concatenate((tgridf.T, Qs_opt_nsc_deg.T),axis=1)
            from variousFunctions import numpy2storage
            numpy2storage(labels_w_muscles, data, os.path.join(
                pathResults, 'kinematics.mot'))
            
        # %% Write IMU files with synthetic data
        imu_labels = []
        linAcc_labels = []
        for dimension in dimensions:
            imu_labels = imu_labels + ["radius_imu_" + dimension]
        if writeIMUFile:
            imu_labels_all = ['time'] + imu_labels  
            angVel_data = np.concatenate((tgridf.T[1::], angVel_opt.T),axis=1)
            linAcc_data = np.concatenate((tgridf.T[1::], linAcc_opt.T),axis=1)            
            from variousFunctions import numpy2storage
            numpy2storage(imu_labels_all, angVel_data, os.path.join(
                pathResults, 'angularVelocities.mot'))
            numpy2storage(imu_labels_all, linAcc_data, os.path.join(
                pathResults, 'linearAccelerations.mot'))
            

        # %% Visualize tracking results
        if visualizeTracking:
            import matplotlib.pyplot as plt 
#             if tracking_data == "markers":                 
#                 marker_titles = []
#                 for marker in markers_toTrack:
#                     for dimension in dimensions:
#                         marker_titles.append(marker + '_' + dimension)    
#                 fig, axs = plt.subplots(NMarkers, NVec3, sharex=True)    
#                 fig.suptitle('Tracking of marker trajectories')
#                 for i, ax in enumerate(axs.flat):
#                     ax.plot(tgridf[0,1:].T, 
#                             dataToTrack_nsc_offset_opt[i:i+1,:].T, 
#                             'tab:blue', label='experimental')
#                     ax.plot(tgridf[0,1:].T, 
#                             marker_sim_opt[i:i+1, :].T, 
#                             'tab:orange', label='simulated')
#                     ax.set_title(marker_titles[i])
#                     ymin, ymax = ax.get_ylim()
#                     ax.set_yticks(np.round(np.linspace(ymin, ymax, 2), 2))
#                 plt.setp(axs[-1, :], xlabel='Time (s)')
#                 plt.setp(axs[:, 0], ylabel='(m)')
#                 fig.align_ylabels()
#                 handles, labels = ax.get_legend_handles_labels()
#                 plt.legend(handles, labels, loc='upper right')
            refData_nsc = Qs_fromIK_interp.to_numpy()[:,1::].T
            refData_offset_nsc = copy.deepcopy(refData_nsc)
            # if offset_ty and tracking_data == "coordinates":                    
            #     refData_offset_nsc[joints.index("pelvis_ty")] = (
            #         refData_nsc[joints.index("pelvis_ty")] + 
            #         offset_opt_nsc)    
            if tracking_data == "coordinates":                
                ny = np.ceil(np.sqrt(NJoints))   
                fig, axs = plt.subplots(int(ny), int(ny), sharex=True)    
                fig.suptitle('Tracking of joint coordinates')                  
                for i, ax in enumerate(axs.flat):
                    if i < NJoints:
                        if joints[i] in rotationalJoints:
                            scale_angles = 180 / np.pi
                        else:
                            scale_angles = 1
                        # reference data
                        ax.plot(tgridf[0,:].T, 
                                refData_offset_nsc[i:i+1,:].T * scale_angles, 
                                c='black', label='experimental')
                        # simulated data
                        if (joints[i] in coordinates_toTrack["rotational"] or 
                            joints[i] in coordinates_toTrack["translational"]):
                            col_sim = 'orange'
                        else:
                            col_sim = 'blue'
                        
                        ax.plot(tgridf[0,:].T, 
                                Qs_opt_nsc[i:i+1,:].T * scale_angles, 
                                c=col_sim, label='simulated')
                        ax.set_title(joints[i])
                plt.setp(axs[-1, :], xlabel='Time (s)')
                plt.setp(axs[:, 0], ylabel='(deg or m)')
                fig.align_ylabels()
                handles, labels = ax.get_legend_handles_labels()
                plt.legend(handles, labels, loc='upper right')
                          
        # %% Contribution to the cost function
        if decomposeCost:     
            # activationTerm_opt_all = 0
            actJExcitationTerm_opt_all = 0
            gtJExcitationTerm_opt_all = 0
            jointAccelerationTerm_opt_all = 0
            # activationDtTerm_opt_all = 0
            # forceDtTerm_opt_all = 0
            lambdaTerm_opt_all = 0
            if velocity_correction:
                gammaTerm_opt_all = 0
            for k in range(N):
                # States 
                # akj_opt = (ca.horzcat(a_opt[:, k], a_col_opt[:, k*d:(k+1)*d]))
                # normFkj_opt = (ca.horzcat(normF_opt[:, k], normF_col_opt[:, k*d:(k+1)*d]))
                # normFkj_opt_nsc = normFkj_opt * (scalingF.to_numpy().T * np.ones((1, d+1)))   
                Qskj_opt = (ca.horzcat(Qs_opt[:, k], Qs_col_opt[:, k*d:(k+1)*d]))
                Qskj_opt_nsc = Qskj_opt * (scalingQs.to_numpy().T * np.ones((1, d+1)))
                Qdotskj_opt = (ca.horzcat(Qdots_opt[:, k], Qdots_col_opt[:, k*d:(k+1)*d]))
                Qdotskj_opt_nsc = Qdotskj_opt * (scalingQdots.to_numpy().T * np.ones((1, d+1)))
                # Controls
                # aDtk_opt = aDt_opt[:, k]
                # aDtk_opt_nsc = aDt_opt_nsc[:, k]
                eActJk_opt = eActJ_opt[:, k]
                eGTJk_opt = eGTJ_opt[:, k]
                # Slack controls
                Qdotdotsj_opt = Qdotdots_col_opt[:, k*d:(k+1)*d]
                Qdotdotsj_opt_nsc = Qdotdotsj_opt * (scalingQdotdots.to_numpy().T * np.ones((1, d)))
                # normFDtj_opt = normFDt_col_opt[:, k*d:(k+1)*d] 
                # normFDtj_opt_nsc = normFDtj_opt * (scalingFDt.to_numpy().T * np.ones((1, d)))
                lambdaj_opt = lambda_col_opt[:, k*d:(k+1)*d]
                lambdaj_opt_nsc = lambdaj_opt * (scalingLambda.to_numpy().T * np.ones((1, d)))
                if velocity_correction:
                    gammaj_opt = gamma_col_opt[:, k*d:(k+1)*d]
                    gammaj_opt_nsc = gammaj_opt * (scalingGamma.to_numpy().T * np.ones((1, d)))                
                
                QsQdotskj_opt_nsc = ca.DM(NJoints*2, d+1)
                QsQdotskj_opt_nsc[::2, :] = Qskj_opt_nsc
                QsQdotskj_opt_nsc[1::2, :] = Qdotskj_opt_nsc
                
                for j in range(d):                    
                    # Motor control terms.
                    # activationTerm_opt = f_NMusclesSum2(akj_opt[:, j+1])     
                    actJExcitationTerm_opt = f_NActJointsSum2(eActJk_opt) 
                    gtJExcitationTerm_opt = f_NGroundThoraxJointsSum2(eGTJk_opt) 
                    jointAccelerationTerm_opt = f_NJointsSum2(Qdotdotsj_opt[:, j])       
                    # activationDtTerm_opt = f_NMusclesSum2(aDtk_opt)
                    # forceDtTerm_opt = f_NMusclesSum2(normFDtj_opt[:, j])
                    lambdaTerm_opt = f_NHolConstraintsSum2(lambdaj_opt[:, j])  
                    if velocity_correction:
                        gammaTerm_opt = f_NHolConstraintsSum2(gammaj_opt[:, j])  
                        gammaTerm_opt_all += weights['gammaTerm'] * gammaTerm_opt * h * B[j + 1] / timeElapsed
                        
                    # activationTerm_opt_all += weights['activationTerm'] * activationTerm_opt * h * B[j + 1] / timeElapsed 
                    actJExcitationTerm_opt_all += weights['actJExcitationTerm'] * actJExcitationTerm_opt * h * B[j + 1] / timeElapsed
                    gtJExcitationTerm_opt_all += weights['gtJExcitationTerm'] * gtJExcitationTerm_opt * h * B[j + 1] / timeElapsed 
                    jointAccelerationTerm_opt_all += weights['jointAccelerationTerm'] * jointAccelerationTerm_opt * h * B[j + 1] / timeElapsed 
                    # activationDtTerm_opt_all += weights['controls'] * activationDtTerm_opt * h * B[j + 1] / timeElapsed 
                    # forceDtTerm_opt_all += weights['controls'] * forceDtTerm_opt * h * B[j + 1] / timeElapsed          
                    lambdaTerm_opt_all += weights['lambdaTerm'] * lambdaTerm_opt * h * B[j + 1] / timeElapsed 
                    
            
            # Tracking terms
            # if tracking_data == "markers":
            #     if markers_as_controls:
            #         if norm_std:
            #             JTrack_opt = f_track_k_map(
            #                 marker_u_opt, dataToTrack_sc_offset_opt,
            #                 dataToTrack_std_sc)  
            #         else:
            #             JTrack_opt = f_track_k_map(
            #                 marker_u_opt, dataToTrack_sc_offset_opt) 
            #     else:
            #         if norm_std:
            #             JTrack_opt = f_track_k_map(
            #                 marker_sim_opt_sc, dataToTrack_sc_offset_opt,
            #                 dataToTrack_std_sc)  
            #         else:
            #             JTrack_opt = f_track_k_map(
            #                 marker_sim_opt_sc, dataToTrack_sc_offset_opt)                    
            #     JTrack_opt_sc = (weights['trackingTerm'] * f_mySum(JTrack_opt)
            #                       *  h / timeElapsed).full()
            if tracking_data == "coordinates":
                # Rotational
                JTrack_rot_opt = f_track_k_map(
                    Qs_opt[idxRotCoordinates_toTrack,:], dataToTrack_sc)
                JTrack_rot_opt_sc = (
                    weights['trackingTerm'] * f_mySumTrack(JTrack_rot_opt) 
                    * h / timeElapsed).full()  
                # # Translational
                # if coordinates_toTrack['translational']:
                #     if offset_ty:
                #         JTrack_tr_opt = f_track_tr_k_map(
                #             Qs_opt[idxTrCoordinates_toTrack,:],
                #             dataToTrack_tr_sc_offset_opt) 
                #     else:
                #         JTrack_tr_opt = f_track_tr_k_map(
                #             Qs_opt[idxTrCoordinates_toTrack,:],
                #             dataToTrack_tr_sc)
                #     JTrack_tr_opt_sc = (
                #         weights['trackingTerm_tr'] * 
                #         f_mySumTrack(JTrack_tr_opt) * h / timeElapsed).full()
                #     JTrack_opt_sc = JTrack_rot_opt_sc + JTrack_tr_opt_sc
                # else:
                JTrack_opt_sc = JTrack_rot_opt_sc
                    
            # Motor control term
            if velocity_correction:
                JMotor_opt = (actJExcitationTerm_opt_all.full() +
                              gtJExcitationTerm_opt_all.full() +
                              jointAccelerationTerm_opt_all.full() + 
                              lambdaTerm_opt_all.full() + 
                              gammaTerm_opt_all.full())      
            else:
                JMotor_opt = (actJExcitationTerm_opt_all.full() + 
                              gtJExcitationTerm_opt_all.full() +
                              jointAccelerationTerm_opt_all.full() + 
                              lambdaTerm_opt_all.full())                    
            # Combined term
            JAll_opt = JTrack_opt_sc + JMotor_opt
            assert np.alltrue(
                np.abs(JAll_opt[0][0] - stats['iterations']['obj'][-1]) 
                <= 1e-5), "decomposition cost"
            
            JTerms = {}
            # JTerms["activationTerm"] = activationTerm_opt_all.full()[0][0]
            JTerms["actJExcitationTerm"] = actJExcitationTerm_opt_all.full()[0][0]
            JTerms["gtJExcitationTerm"] = gtJExcitationTerm_opt_all.full()[0][0]
            JTerms["jointAccelerationTerm"] = jointAccelerationTerm_opt_all.full()[0][0]
            JTerms["lambdaTerm"] = lambdaTerm_opt_all.full()[0][0]
            if velocity_correction:
                JTerms["gammaTerm"] = gammaTerm_opt_all.full()[0][0]
            JTerms["trackingTerm"] = JTrack_opt_sc[0][0]
            # JTerms["activationTerm_sc"] = JTerms["activationTerm"] / JAll_opt[0][0]
            JTerms["actJExcitationTerm_sc"] = JTerms["actJExcitationTerm"] / JAll_opt[0][0]
            JTerms["gtJExcitationTerm_sc"] = JTerms["gtJExcitationTerm"] / JAll_opt[0][0]
            JTerms["jointAccelerationTerm_sc"] = JTerms["jointAccelerationTerm"] / JAll_opt[0][0]
            # JTerms["activationDtTerm_sc"] = JTerms["activationDtTerm"] / JAll_opt[0][0]
            # JTerms["forceDtTerm_sc"] = JTerms["forceDtTerm"] / JAll_opt[0][0]
            JTerms["lambdaTerm_sc"] = JTerms["lambdaTerm"] / JAll_opt[0][0]
            if velocity_correction:
                JTerms["gammaTerm_sc"] = JTerms["gammaTerm"] / JAll_opt[0][0]
            JTerms["trackingTerm_sc"] = JTerms["trackingTerm"] / JAll_opt[0][0]
            
            # print("Activations: " + str(np.round(JTerms["activationTerm_sc"] * 100, 2)) + "%")
            print("ActJ Excitations: " + str(np.round(JTerms["actJExcitationTerm_sc"] * 100, 2)) + "%")
            print("GTJ Excitations: " + str(np.round(JTerms["gtJExcitationTerm_sc"] * 100, 2)) + "%")
            print("Joint Accelerations: " + str(np.round(JTerms["jointAccelerationTerm_sc"] * 100, 2)) + "%")
            print("Lambda: " + str(np.round(JTerms["lambdaTerm_sc"] * 100, 2)) + "%")
            if velocity_correction:
                print("Gamma: " + str(np.round(JTerms["gammaTerm_sc"] * 100, 2)) + "%")
            # print("Activations dt: " + str(np.round(JTerms["activationDtTerm_sc"] * 100, 2)) + "%")
            # print("Forces dt: " + str(np.round(JTerms["forceDtTerm_sc"] * 100, 2)) + "%")
            print("Tracking: " + str(np.round(JTerms["trackingTerm_sc"] * 100, 2)) + "%")
            print("# Iterations: " + str(stats["iter_count"]))
            
        # %% Visualize results against bounds
        if visualizeResultsAgainstBounds:
            from variousFunctions import plotVSBounds
            # States
            '''
            # Muscle activation at mesh points            
            lb = lBoundsA.to_numpy().T
            ub = uBoundsA.to_numpy().T
            y = a_opt
            title='Muscle activation at mesh points'            
            plotVSBounds(y,lb,ub,title)  
            # Muscle activation at collocation points
            lb = lBoundsA.to_numpy().T
            ub = uBoundsA.to_numpy().T
            y = a_col_opt
            title='Muscle activation at collocation points' 
            plotVSBounds(y,lb,ub,title)  
            # Muscle force at mesh points
            lb = lBoundsF.to_numpy().T
            ub = uBoundsF.to_numpy().T
            y = normF_opt
            title='Muscle force at mesh points' 
            plotVSBounds(y,lb,ub,title)  
            # Muscle force at collocation points
            lb = lBoundsF.to_numpy().T
            ub = uBoundsF.to_numpy().T
            y = normF_col_opt
            title='Muscle force at collocation points' 
            plotVSBounds(y,lb,ub,title)
            '''
            # Joint position at mesh points
            lb = lBoundsQs.to_numpy().T
            ub = uBoundsQs.to_numpy().T
            y = Qs_opt
            title='Joint position at mesh points' 
            plotVSBounds(y,lb,ub,title)             
            # Joint position at collocation points
            lb = lBoundsQs.to_numpy().T
            ub = uBoundsQs.to_numpy().T
            y = Qs_col_opt
            title='Joint position at collocation points' 
            plotVSBounds(y,lb,ub,title) 
            # Joint velocity at mesh points
            lb = lBoundsQdots.to_numpy().T
            ub = uBoundsQdots.to_numpy().T
            y = Qdots_opt
            title='Joint velocity at mesh points' 
            plotVSBounds(y,lb,ub,title) 
            # Joint velocity at collocation points
            lb = lBoundsQdots.to_numpy().T
            ub = uBoundsQdots.to_numpy().T
            y = Qdots_col_opt
            title='Joint velocity at collocation points' 
            plotVSBounds(y,lb,ub,title) 
            # Actuated joints activation at mesh points
            lb = lBoundsActJA.to_numpy().T
            ub = uBoundsActJA.to_numpy().T
            y = aActJ_opt
            title='ActJ activation at mesh points' 
            plotVSBounds(y,lb,ub,title) 
            # Actuated joints activation at collocation points
            lb = lBoundsActJA.to_numpy().T
            ub = uBoundsActJA.to_numpy().T
            y = aActJ_col_opt
            title='ActJ activation at collocation points' 
            plotVSBounds(y,lb,ub,title)
            # Ground thorax joints activation at mesh points
            lb = lBoundsGTJA.to_numpy().T
            ub = uBoundsGTJA.to_numpy().T
            y = aGTJ_opt
            title='GTJ activation at mesh points' 
            plotVSBounds(y,lb,ub,title) 
            # Ground thorax joints activation at collocation points
            lb = lBoundsGTJA.to_numpy().T
            ub = uBoundsGTJA.to_numpy().T
            y = aGTJ_col_opt
            title='GTJ activation at collocation points' 
            plotVSBounds(y,lb,ub,title) 
            #######################################################################
            # Controls
            '''
            # Muscle activation derivative at mesh points
            lb = lBoundsADt.to_numpy().T
            ub = uBoundsADt.to_numpy().T
            y = aDt_opt
            title='Muscle activation derivative at mesh points' 
            plotVSBounds(y,lb,ub,title) 
            '''
            # Actuated joints excitation at mesh points
            lb = lBoundsActJE.to_numpy().T
            ub = uBoundsActJE.to_numpy().T
            y = eActJ_opt
            title='ActJ excitation at mesh points' 
            plotVSBounds(y,lb,ub,title) 
            # Ground thorax joints excitation at mesh points
            lb = lBoundsGTJE.to_numpy().T
            ub = uBoundsGTJE.to_numpy().T
            y = eGTJ_opt
            title='GTJ excitation at mesh points' 
            plotVSBounds(y,lb,ub,title)                 
            #######################################################################
            # Slack controls
            '''
            # Muscle force derivative at collocation points
            lb = lBoundsFDt.to_numpy().T
            ub = uBoundsFDt.to_numpy().T
            y = normFDt_col_opt
            title='Muscle force derivative at collocation points' 
            plotVSBounds(y,lb,ub,title)
            '''
            # Joint velocity derivative (acceleration) at collocation points
            lb = lBoundsQdotdots.to_numpy().T
            ub = uBoundsQdotdots.to_numpy().T
            y = Qdotdots_col_opt
            title='Joint velocity derivative (acceleration) at collocation points' 
            plotVSBounds(y,lb,ub,title)   
            # Lagrange multipliers at collocation points
            lb = lBoundsLambda.to_numpy().T
            ub = uBoundsLambda.to_numpy().T
            y = lambda_col_opt
            title='Lagrange multipliers at collocation points' 
            plotVSBounds(y,lb,ub,title)         
            if velocity_correction:
                # Velocity correctors at collocation points
                lb = lBoundsGamma.to_numpy().T
                ub = uBoundsGamma.to_numpy().T
                y = gamma_col_opt
                title='Velocity correctors at collocation points' 
                plotVSBounds(y,lb,ub,title)  
            # # Marker trajectories
            # if markers_as_controls:
            #     lb = lBoundsMarker.to_numpy().T
            #     ub = uBoundsMarker.to_numpy().T
            #     y = marker_u_opt
            #     title='Marker trajectories at mesh points' 
            #     plotVSBounds(y,lb,ub,title)    
            
        if visualizeSimulationResults:
            ncol = 6 
            nrow = np.ceil(NJoints/ncol)           
            fig, axs = plt.subplots(int(nrow), ncol, sharex=True)  
            fig.suptitle('Joint torques')     
            for i, ax in enumerate(axs.flat):
                if i < NJoints:
                    # reference data: full marker set
                    # ax.plot(tgridf[0,1::].T, 
                    #         ID_exp_interp[i:i+1,1::].T, 
                    #         c='black', label='experimental - full marker set')
                    # ax.plot(tgridf[0,1::].T, 
                    #         ID_exp_OpenPose_interp[i:i+1,1::].T, 
                    #         c='blue', label='experimental - OpenPose')
                    # reference data: OpenPose marker set                  
                    ax.plot(tgridf[0,1::].T, 
                            torques_opt[i:i+1,:].T, 
                            c='orange', label='simulated')
                    ax.set_title(joints[i])
                    handles, labels = ax.get_legend_handles_labels()
            plt.setp(axs[-1, :], xlabel='Time (s)')
            plt.setp(axs[:, 0], ylabel='(Nm or N)')
            fig.align_ylabels()            
            plt.legend(handles, labels, loc='upper right')
            
        if visualizeConstraintErrors:
            # Contraint errors       
            constraint_levels = ["positions", "velocity", "acceleration"]
            constraint_labels = []
            for constraint_level in constraint_levels:
                for count in range(NHolConstraints):
                    constraint_labels.append(constraint_level + '_' + str(count))
            
            import matplotlib.pyplot as plt 
            fig, axs = plt.subplots(3, 3, sharex=True) 
            fig.suptitle('Constraint errors')     
            for i, ax in enumerate(axs.flat):
                    ax.plot(tgridf[0,1::].T, 
                            kinCon_opt[i:i+1,:].T, 
                            c='orange', label='simulated')                
                    ax.set_title(constraint_labels[i])
            plt.setp(axs[-1, :], xlabel='Time (s)')
            plt.setp(axs[:, 0], ylabel='(todo)')
            fig.align_ylabels()
            
#         if visualizeSimulationResults:
#             #TODO: path references
#             # Reference from full marker set
#             pathIKRefFolder = os.path.join(pathSubject, 'IK', 
#                                            "subject1_scaled_fullMarkerSet_KA")            
#             pathIKRef = os.path.join(pathIKRefFolder, 
#                                      'IK_eval_pre1_fullMarkerSet.mot')
#             from variousFunctions import getIK
#             _, QsRef_fromIK_filt = getIK(pathIKRef, joints) 
#             QsRef_fromIK_filt_interp = interpolateDataFrame(
#                 QsRef_fromIK_filt, 
#                 timeInterval[0], timeInterval[1], N+1).to_numpy()[:,1::].T    
#             ncol = 6
#             nrow = np.ceil(NJoints/ncol)           
#             fig, axs = plt.subplots(int(nrow), ncol, sharex=True)    
#             fig.suptitle('Joint coordinates')
#             for i, ax in enumerate(axs.flat):
#                 if i < NJoints:
#                     if joints[i] in rotationalJoints:
#                         scale_angles = 180 / np.pi
#                     else:
#                         scale_angles = 1
#                     # reference data: full marker set
#                     ax.plot(tgridf[0,:].T, 
#                             QsRef_fromIK_filt_interp[i:i+1,:].T * scale_angles, 
#                             c='black', label='experimental - full marker set')
#                     # reference data: OpenPose marker set
#                     ax.plot(tgridf[0,:].T, 
#                             refData_offset_nsc[i:i+1,:].T * scale_angles, 
#                             c='blue', label='experimental - OpenPose')
#                     # simulated data                    
#                     ax.plot(tgridf[0,:].T, 
#                             Qs_opt_nsc[i:i+1,:].T * scale_angles, 
#                             c='orange', label='simulated')
#                     ax.set_title(joints[i])
#                     handles, labels = ax.get_legend_handles_labels()
#             plt.setp(axs[-1, :], xlabel='Time (s)')
#             plt.setp(axs[:, 0], ylabel='(deg or m)')
#             fig.align_ylabels()            
#             plt.legend(handles, labels, loc='upper right')
            
#             # Joint torques
#             # Reference from full marker set
#             pathIDRefFolder = os.path.join(pathSubject, 'ID', 
#                                            "subject1_scaled_fullMarkerSet_KA")            
#             pathIDRef = os.path.join(pathIDRefFolder, 
#                                      'ID_eval_pre1_fullMarkerSet.sto')
#             from variousFunctions import getID
#             ID_exp = getID(pathIDRef, joints)
#             from variousFunctions import interpolateDataFrame2Numpy
#             ID_exp_interp = interpolateDataFrame2Numpy(
#                 ID_exp, timeInterval[0], timeInterval[1], N+1)[:,1::].T 
            
#             # Reference from OpenPose marker set   
#             pathIDRefFolderOpenPose = os.path.join(pathSubject, 'ID', model)            
#             pathIDRefOpenPose = os.path.join(pathIDRefFolderOpenPose, 
#                                              'ID_' + trial + '.sto')
#             ID_exp_OpenPose = getID(pathIDRefOpenPose, joints)
#             ID_exp_OpenPose_interp = interpolateDataFrame2Numpy(
#                 ID_exp_OpenPose, 
#                 timeInterval[0], timeInterval[1], N+1)[:,1::].T 
#             ncol = 6 
#             nrow = np.ceil(NJoints/ncol)           
#             fig, axs = plt.subplots(int(nrow), ncol, sharex=True)    
#             fig.suptitle('Joint torques')     
#             for i, ax in enumerate(axs.flat):
#                 if i < NJoints:
#                     # reference data: full marker set
#                     ax.plot(tgridf[0,1::].T, 
#                             ID_exp_interp[i:i+1,1::].T, 
#                             c='black', label='experimental - full marker set')
#                     ax.plot(tgridf[0,1::].T, 
#                             ID_exp_OpenPose_interp[i:i+1,1::].T, 
#                             c='blue', label='experimental - OpenPose')
#                     # reference data: OpenPose marker set                  
#                     ax.plot(tgridf[0,1::].T, 
#                             torques_opt[i:i+1,:].T, 
#                             c='orange', label='simulated')
#                     ax.set_title(joints[i])
#                     handles, labels = ax.get_legend_handles_labels()
#             plt.setp(axs[-1, :], xlabel='Time (s)')
#             plt.setp(axs[:, 0], ylabel='(Nm or N)')
#             fig.align_ylabels()            
#             plt.legend(handles, labels, loc='upper right')
            
#             # GRF
#             # Get experimental data
#             from variousFunctions import getGRF                     
#             pathGRF = os.path.join(pathGRFFolder, 'GRF_eval_pre1.mot')
#             GRF_headers = ["ground_force_vx", "ground_force_vy",
#                            "ground_force_vz", "1_ground_force_vx",
#                            "1_ground_force_vy", "1_ground_force_vz"]            
#             GRF_exp = getGRF(pathGRF, GRF_headers)                        
#             GRF_exp_interp = interpolateDataFrame2Numpy(
#                 GRF_exp, timeInterval[0], timeInterval[1], N+1)[:,1::].T
            
#             fig, axs = plt.subplots(2, 3, sharex=True) 
#             fig.suptitle('Ground reaction forces')     
#             for i, ax in enumerate(axs.flat):
#                 # reference data
#                 ax.plot(tgridf[0,1::].T, 
#                         GRF_exp_interp[i:i+1,1::].T, 
#                         c='black', label='experimental')
#                 # simulated data  
#                 if i < GRF_all_r_opt.shape[0]:                 
#                     ax.plot(tgridf[0,1::].T, 
#                             GRF_all_r_opt[i:i+1,:].T, 
#                             c='orange', label='simulated')
#                     ax.set_title(GR_labels["GRF"]["all"]["r"][i])
#                 else:
#                     ax.plot(tgridf[0,1::].T, 
#                             GRF_all_l_opt[i-3:i+1-3,:].T, 
#                             c='orange', label='simulated')
                    
#                     ax.set_title(GR_labels["GRF"]["all"]["l"][i-3])
#             plt.setp(axs[-1, :], xlabel='Time (s)')
#             plt.setp(axs[:, 0], ylabel='(N)')
#             fig.align_ylabels()
#             handles, labels = ax.get_legend_handles_labels()
#             plt.legend(handles, labels, loc='upper right')            
 
# # %% Code that might be usefull when debugging
# # Test limit torques
# # lb_theta = np.array([[0,0,0,0,0,0,-0.6981, -0.5, -0.92,
# #                       -0.6981, -0.5, -0.92,-2.4,-2.4,-0.74,-0.74,
# #                       -1,-1,-1.134464013796314,-1.134464013796314,
# #                       -0.5235987755982988,
# #                       -0.3490658503988659,-0.3490658503988659]]).T
# # lb_theta_sc = lb_theta / scalingQs.to_numpy().T
# # ub_theta = np.array([[0,0,0,0,0,0,1.81, 0.5, 0.92,
# #                       1.81, 0.5, 0.92,0.13,0.13,0.52,0.52,
# #                       1,1,1.134464013796314,1.134464013796314,
# #                       0.17,
# #                       0.3490658503988659,0.3490658503988659]]).T
# # ub_theta_sc = ub_theta / scalingQs.to_numpy().T
# # from variousFunctions import plotVSBounds
# # lb = lb_theta_sc
# # ub = ub_theta_sc
# # y = guessQs.to_numpy().T
# # title='Joint position at mesh points' 
# # plotVSBounds(y,lb,ub,title) 

# # #######################################################################
# # # Get metabolic energy rate
# # metabolicEnergyRatej = f_metabolicsBhargava(akj[:, j+1], akj[:, j+1], 
# #                             normFiberLengthj, fiberVelocityj, 
# #                             activeFiberForcej, passiveFiberForcej, 
# #                             normActiveFiberLengthForcej)[5]
# #            metabolicEnergyTerm = (f_NMusclesSum2(metabolicEnergyRatej) / 
# #                                       modelMass)
# #            ##################################################################
# #            # Prevent inter-penetrations of body parts
# #            diffCalcOrs = f_sumSqr(Tj[idxCalcOr_r] - Tj[idxCalcOr_l])
# #            ineq_constr3.append(diffCalcOrs)
# #            diffFemurHandOrs_r = f_sumSqr(Tj[idxFemurOr_r] - Tj[idxHandOr_r])
# #            ineq_constr4.append(diffFemurHandOrs_r)
# #            diffFemurHandOrs_l = f_sumSqr(Tj[idxFemurOr_l] - Tj[idxHandOr_l])
# #            ineq_constr4.append(diffFemurHandOrs_l)
# #            diffTibiaOrs = f_sumSqr(Tj[idxTibiaOr_r] - Tj[idxTibiaOr_l])
# #            ineq_constr5.append(diffTibiaOrs)
# #            diffToesOrs = f_sumSqr(Tj[idxToesOr_r] - Tj[idxToesOr_l])
# #            ineq_constr6.append(diffToesOrs)