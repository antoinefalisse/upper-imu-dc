import os
import numpy as np
import matplotlib.pyplot as plt  

# %% Settings 
# cases = ['76', '78', '81'] # flexion
# cases = ['70','79', '82', '85'] # abduction
cases = ['71','80', '83', '86'] # shrugging
mainName = "main_tracking"
subject = "subject1"
reference_case_IMU = cases[0]

# %% Fixed settings
pathMain = os.getcwd()
# Load results
pathTrajectories = os.path.join(pathMain, 'Results', mainName)
optimaltrajectories = np.load(os.path.join(pathTrajectories, 
                                           'optimalTrajectories.npy'),
                              allow_pickle=True).item()
    
# %% Visualize results
plt.close('all')
case_colors = {}

# %% Joint coordinates
# kinematic_ylim_ub = [20, 1, 1, 50, 50, 20, 20, 30, 30, 60, 60, 20]
# kinematic_ylim_lb = [-20, -1, 0.8, -30, -30, -80, -80, -30, -30, -20, -20, -20]
joints = optimaltrajectories[cases[0]]['joints']
jointsToPlot = joints
from variousFunctions import getJointIndices
idxJointsToPlot = getJointIndices(joints, jointsToPlot)
NJointsToPlot = len(jointsToPlot)    
ny = np.ceil(np.sqrt(NJointsToPlot))   
fig, axs = plt.subplots(int(ny), int(ny), sharex=True)  
fig.suptitle('Joint coordinates')
for i, ax in enumerate(axs.flat):
    if i < NJointsToPlot:
        color=iter(plt.cm.rainbow(np.linspace(0,1,len(cases))))   
        for case in cases:
            if case == cases[0]:
                ax.plot(optimaltrajectories[case]['time'].T,
                        optimaltrajectories[case]['ref_coordinate_values'][idxJointsToPlot[i]:idxJointsToPlot[i]+1, :].T, c='black', label='Experimental')            
            col = next(color)
            ax.plot(optimaltrajectories[case]['time'].T,
                    optimaltrajectories[case]['sim_coordinate_values'][idxJointsToPlot[i]:idxJointsToPlot[i]+1, :].T, c=col, label='case_' + case)         
            case_colors[case] = col
        ax.set_title(joints[idxJointsToPlot[i]])
        # ax.set_ylim((kinematic_ylim_lb[i],kinematic_ylim_ub[i]))
        handles, labels = ax.get_legend_handles_labels()
        plt.legend(handles, labels, loc='upper right')
plt.setp(axs[-1, :], xlabel='Time (s)')
plt.setp(axs[:, 0], ylabel='(Angle (deg))')
fig.align_ylabels()

# %% Joint torques
# kinematic_ylim_ub = [20, 1, 1, 50, 50, 20, 20, 30, 30, 60, 60, 20]
# kinematic_ylim_lb = [-20, -1, 0.8, -30, -30, -80, -80, -30, -30, -20, -20, -20]
joints = optimaltrajectories[cases[0]]['joints']
jointsToPlot = joints
from variousFunctions import getJointIndices
idxJointsToPlot = getJointIndices(joints, jointsToPlot)
NJointsToPlot = len(jointsToPlot)    
ny = np.ceil(np.sqrt(NJointsToPlot))   
fig, axs = plt.subplots(int(ny), int(ny), sharex=True)  
fig.suptitle('Joint torques')
for i, ax in enumerate(axs.flat):
    if i < NJointsToPlot:
        color=iter(plt.cm.rainbow(np.linspace(0,1,len(cases))))   
        for case in cases:          
            ax.plot(optimaltrajectories[case]['time'][0,1::].T,
                    optimaltrajectories[case]['sim_coordinate_torques'][idxJointsToPlot[i]:idxJointsToPlot[i]+1, :].T, c=next(color), label='case_' + case)         
        ax.set_title(joints[idxJointsToPlot[i]])
        # ax.set_ylim((kinematic_ylim_lb[i],kinematic_ylim_ub[i]))
        handles, labels = ax.get_legend_handles_labels()
        plt.legend(handles, labels, loc='upper right')
plt.setp(axs[-1, :], xlabel='Time (s)')
plt.setp(axs[:, 0], ylabel='(Torque (Nm))')
fig.align_ylabels()

# %% IMU tracking
joints = optimaltrajectories[cases[0]]['joints']
jointsToPlot = joints
from variousFunctions import getJointIndices
idxJointsToPlot = getJointIndices(joints, jointsToPlot)
NJointsToPlot = len(jointsToPlot)    

plotIMUTracking = False
for case in cases:
    if 'ref_imu_data' in optimaltrajectories[case]:
        plotIMUTracking = True
        plotIMUTrackingR = False
        nrows = 2
        if 'ref_imu_data_R' in optimaltrajectories[case]:
            plotIMUTrackingR = True
            nrows = 3
imu_titles = ["Angular velocity x", "Angular velocity y",
              "Angular velocity z", "Linear Acceleration x", 
              "Linear Acceleration y", "Linear Acceleration z",
              "Euler angle x", "Euler angle y", "Euler angle z"] 
if plotIMUTracking:
    fig, axs = plt.subplots(nrows, 3, sharex=True)  
    fig.suptitle('IMU tracking')
    for i, ax in enumerate(axs.flat):
            # color=iter(plt.cm.rainbow(np.linspace(0,1,len(cases))))
            referenceIsPlotted = False
            for case in cases:                
                if 'ref_imu_data' in optimaltrajectories[case]:
                    if i < 6:                                                
                        if not referenceIsPlotted:
                            ax.plot(optimaltrajectories[case]['time'][0,1::].T,
                                    optimaltrajectories[case]['ref_imu_data'][i:i+1, :].T, c=case_colors[reference_case_IMU], label='Synthetic')            
                            referenceIsPlotted = True
                        ax.plot(optimaltrajectories[case]['time'][0,1::].T,
                                optimaltrajectories[case]['sim_imu_data'][i:i+1, :].T, c=case_colors[case], label='case_' + case)         
                    if plotIMUTrackingR and i > 5:
                        if not referenceIsPlotted:
                            ax.plot(optimaltrajectories[case]['time'][0,1::].T,
                                    optimaltrajectories[case]['ref_imu_data_R'][i-6:i-6+1, :].T, c=case_colors[reference_case_IMU], label='Synthetic')            
                            referenceIsPlotted = True
                        ax.plot(optimaltrajectories[case]['time'][0,1::].T,
                                optimaltrajectories[case]['sim_imu_data_R'][i-6:i-6+1, :].T, c=case_colors[case], label='case_' + case)
            ax.set_title(imu_titles[i])
            # ax.set_ylim((kinematic_ylim_lb[i],kinematic_ylim_ub[i]))
            handles, labels = ax.get_legend_handles_labels()
            plt.legend(handles, labels, loc='upper right')
    plt.setp(axs[-1, :], xlabel='Time (s)')
    plt.setp(axs[:, 0], ylabel='(todo)')
    fig.align_ylabels()


# # %% Muscle activations
# muscles = optimaltrajectories[cases[0]]['muscles']
# musclesToPlot = ['glut_med1_r', 'glut_med2_r', 'glut_med3_r', 'glut_min1_r', 
#                  'glut_min2_r', 'glut_min3_r', 'semimem_r', 'semiten_r', 
#                  'bifemlh_r', 'bifemsh_r', 'sar_r', 'add_long_r', 'add_brev_r',
#                  'add_mag1_r', 'add_mag2_r', 'add_mag3_r', 'tfl_r', 'pect_r',
#                  'grac_r', 'glut_max1_r', 'glut_max2_r', 'glut_max3_r',
#                  'iliacus_r', 'psoas_r', 'quad_fem_r', 'gem_r', 'peri_r',
#                  'rect_fem_r', 'vas_med_r', 'vas_int_r', 'vas_lat_r',
#                  'med_gas_r', 'lat_gas_r', 'soleus_r', 'tib_post_r',
#                  'flex_dig_r', 'flex_hal_r', 'tib_ant_r', 'per_brev_r',
#                  'per_long_r', 'per_tert_r', 'ext_dig_r', 'ext_hal_r',
#                  'ercspn_r', 'intobl_r', 'extobl_r']
# mappingEMG = {'glut_med1_r': 'GluMed_r', 
#               'glut_med2_r': 'GluMed_r', 
#               'glut_med3_r': 'GluMed_r',
#               'semimem_r': 'HamM_r',
#               'semiten_r': 'HamM_r',
#               'bifemlh_r': 'HamL_r',
#               'bifemsh_r': 'HamL_r',
#               'add_long_r': 'AddL_r',
#               'tfl_r': 'TFL_r',
#               'rect_fem_r': 'RF_r',
#               'vas_med_r': 'VM_r',
#               'vas_int_r': 'VL_r',
#               'vas_lat_r': 'VL_r',
#               'med_gas_r': 'GM_r',
#               'lat_gas_r': 'GL_r',
#               'soleus_r': 'Sol_r',
#               'tib_ant_r': 'TA_r',
#               'per_brev_r': 'PerB_l',
#               'per_long_r': 'PerL_l',
#               'glut_med1_l': 'GluMed_l', 
#               'glut_med2_l': 'GluMed_l', 
#               'glut_med3_l': 'GluMed_l',
#               'semimem_l': 'HamM_l',
#               'semiten_l': 'HamM_l',
#               'bifemlh_l': 'HamL_l',
#               'bifemsh_l': 'HamL_l',
#               'add_long_l': 'AddL_l',
#               'tfl_l': 'TFL_l',
#               'rect_fem_l': 'RF_l',
#               'vas_med_l': 'VM_l',
#               'vas_int_l': 'VL_l',
#               'vas_lat_l': 'VL_l',
#               'med_gas_l': 'GM_l',
#               'lat_gas_l': 'GL_l',
#               'soleus_l': 'Sol_l',
#               'tib_ant_l': 'TA_l',
#               'per_brev_l': 'PerB_l',
#               'per_long_l': 'PerL_l'}
# NMusclesToPlot = len(musclesToPlot)
# idxMusclesToPlot = getJointIndices(muscles, musclesToPlot)
# fig, axs = plt.subplots(8, 6, sharex=True)    
# fig.suptitle('Muscle activations')
# for i, ax in enumerate(axs.flat):
#     color=iter(plt.cm.rainbow(np.linspace(0,1,len(cases))))   
#     if i < NMusclesToPlot:
#         for case in cases:
#             ax.plot(optimaltrajectories[case]['GC_percent'],
#                     optimaltrajectories[case]['muscle_activations'][idxMusclesToPlot[i]:idxMusclesToPlot[i]+1, :].T, c=next(color), label='case_' + case)            
#             if musclesToPlot[i] in mappingEMG:                
#                 # Normalize EMG such that peak mean EMG = peak activation             
#                 exp_mean = experimentalData[subject]["EMG"]["mean"][mappingEMG[musclesToPlot[i]]]
#                 exp_mean_peak = np.max(exp_mean)
#                 sim = optimaltrajectories[case]['muscle_activations'][idxMusclesToPlot[i], :].T
#                 sim_peak = np.max(sim)
#                 scaling_emg = sim_peak / exp_mean_peak
#                 ax.fill_between(experimentalData[subject]["EMG"]["GC_percent"],
#                         experimentalData[subject]["EMG"]["mean"][mappingEMG[musclesToPlot[i]]] * scaling_emg + 2*experimentalData[subject]["EMG"]["std"][mappingEMG[musclesToPlot[i]]] * scaling_emg,
#                         experimentalData[subject]["EMG"]["mean"][mappingEMG[musclesToPlot[i]]] * scaling_emg - 2*experimentalData[subject]["EMG"]["std"][mappingEMG[musclesToPlot[i]]] * scaling_emg)
#         ax.set_title(muscles[idxMusclesToPlot[i]])
#         ax.set_ylim((0,1))
#         handles, labels = ax.get_legend_handles_labels()
#         plt.legend(handles, labels, loc='upper right')
# plt.setp(axs[-1, :], xlabel='Gait cycle (%)')
# plt.setp(axs[:, 0], ylabel='(-)')
# fig.align_ylabels()

# # %% Contact forces 
# # contact_ylim_ub = [300, 1500, 300, 1200]
# # contact_ylim_lb = [-300, 0, -300, 0]
# GRF_labels = optimaltrajectories[cases[0]]['GRF_labels']
# GRFToPlot = ['GRF_x_r', 'GRF_y_r', 'GRF_z_r', 'GRF_x_l','GRF_y_l', 'GRF_z_l']
# NGRFToPlot = len(GRFToPlot)
# idxGRFToPlot = getJointIndices(GRF_labels, GRFToPlot)
# fig, axs = plt.subplots(2, 3, sharex=True)    
# fig.suptitle('Ground reaction forces')
# #color=iter(plt.cm.rainbow(np.linspace(0,1,len(trials))))
# for i, ax in enumerate(axs.flat):
#     color=iter(plt.cm.rainbow(np.linspace(0,1,len(cases))))   
#     for case in cases:
#         ax.plot(optimaltrajectories[case]['GC_percent'],
#                 optimaltrajectories[case]['GRF'][idxGRFToPlot[i]:idxGRFToPlot[i]+1, :].T, c=next(color), label='case_' + case) 
#         ax.fill_between(experimentalData[subject]["GRF"]["GC_percent"],
#                         experimentalData[subject]["GRF"]["mean"][GRFToPlot[i]] + 2*experimentalData[subject]["GRF"]["std"][GRFToPlot[i]],
#                         experimentalData[subject]["GRF"]["mean"][GRFToPlot[i]] - 2*experimentalData[subject]["GRF"]["std"][GRFToPlot[i]])
#     ax.set_title(GRF_labels[idxGRFToPlot[i]])
#     # ax.set_ylim((contact_ylim_lb[i],contact_ylim_ub[i]))
#     handles, labels = ax.get_legend_handles_labels()
#     plt.legend(handles, labels, loc='upper right')
# plt.setp(axs[-1, :], xlabel='Gait cycle (%)')
# plt.setp(axs[:, 0], ylabel='(N)')
# fig.align_ylabels()

# # %% MTP actuators
# mtpJoints = optimaltrajectories[cases[0]]['mtp_joints']
# mtpJointsToPlot = ['mtp_angle_r']
# idxMTPJointsToPlot = getJointIndices(mtpJoints, mtpJointsToPlot)
# fig, ax = plt.subplots()     
# color=iter(plt.cm.rainbow(np.linspace(0,1,len(cases))))   
# for case in cases:
#     ax.plot(optimaltrajectories[case]['GC_percent'],
#             optimaltrajectories[case]['mtp_activations'][idxMTPJointsToPlot[0]:idxMTPJointsToPlot[0]+1, :].T, c=next(color), label='case_' + case)  
# ax.set(xlabel='Gait cycle (%)', ylabel='(Nm)',
#        title='MTP activations')
# # ax.set_ylim((kinetic_ylim_lb[i],kinetic_ylim_ub[i]))
# handles, labels = ax.get_legend_handles_labels()
# plt.legend(handles, labels, loc='upper right')

# # %% Metabolic cost and cost function value
# fig, (ax1, ax2) = plt.subplots(1, 2)
# color=iter(plt.cm.rainbow(np.linspace(0,1,len(cases))))   
# for count, case in enumerate(cases):
#     print(optimaltrajectories[case]["COT"])
#     ax1.scatter(count, optimaltrajectories[case]["COT"], s=80)
#     ax2.scatter(count, optimaltrajectories[case]["objective"], s=80)
# ax1.set_title("Cost of Transport")
# ax1.set_ylabel("(J/Kg/m)")    
# ax2.set_title("Optimal cost value")
# ax2.set_ylabel("()")
# x_locations = np.linspace(0, len(cases)-1, len(cases))
# ax1.set_xticks(x_locations)
# xticklabels = ["Case_" + case for case in cases]
# ax1.set_xticklabels(xticklabels)
# ax2.set_xticks(x_locations)
# ax2.set_xticklabels(xticklabels)
