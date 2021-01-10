clear all
clc
close all

%% User settings
folderNameTrainingData = 'grid_9nodes_3dim';
% This is usually fixed.
prefix = 'subject01_MuscleAnalysis_';

%% Coordinates and muscles
coordinates = {'clav_prot', 'clav_elev', 'scapula_abduction', ...
               'scapula_elevation', 'scapula_upward_rot', ...
               'scapula_winging', 'plane_elv', 'shoulder_elv', ...
               'axial_rot'};

muscles = {'TrapeziusScapula_M', 'TrapeziusScapula_S', ...
           'TrapeziusScapula_I', 'TrapeziusClavicle_S', ...
           'SerratusAnterior_I', 'SerratusAnterior_M', ...
           'SerratusAnterior_S', 'Rhomboideus_S', 'Rhomboideus_I', ....
           'LevatorScapulae', 'Coracobrachialis', 'DeltoideusClavicle_A', ...
           'DeltoideusScapula_P', 'DeltoideusScapula_M', ...
           'LatissimusDorsi_S', 'LatissimusDorsi_M', 'LatissimusDorsi_I', ...
           'PectoralisMajorClavicle_S', 'PectoralisMajorThorax_I', ...
           'PectoralisMajorThorax_M', 'TeresMajor', 'Infraspinatus_I', ...
           'Infraspinatus_S', 'PectoralisMinor', 'TeresMinor', ...
           'Subscapularis_S', 'Subscapularis_M', 'Subscapularis_I', ...
           'Supraspinatus_P', 'Supraspinatus_A', 'TRIlong', 'BIC_long', ... 
           'BIC_brevis'};

%% Load training data
% Muscle-tendon lengths
pathLengths = [folderNameTrainingData, '/all/', prefix, 'Length.sto'];
lMT = importdata(pathLengths);
nSamples = size(lMT.data,1);
% Moment arms
for i = 1:length(coordinates)
    MA.([coordinates{i}]) = importdata([folderNameTrainingData, '/all/', prefix,'MomentArm_',coordinates{i},'.sto']);
end
% Joint coordinate values
pathMotion = [folderNameTrainingData, '/all/training_q.mot'];
coordinate_values = importdata(pathMotion);

for m = 1:length(muscles)
    MuscleData.lMT(:,m) = lMT.data(:,strcmp(lMT.colheaders,muscles{m}));
    for i = 1:length(coordinates)
        MuscleData.dM(:,m,i) = MA.([coordinates{i}]).data(:,strcmp(lMT.colheaders,muscles{m}));
    end
end

%% Format data as required by multi-dim-poly
% For some reasons, there are NaNs sometimes...
MuscleData.dM(isnan(MuscleData.dM)) = 0;
% Weird things sometimes, like a moment arm that is almost always 0 and
% then suddenly isn't 0 for a certain configuration. In such cases, we want
% 0. We can check it there is a majority of 0 for the moment arms of the
% muscle, and assign 0 for the moment arm in such case.
test = reshape(sum((MuscleData.dM == 0),1),length(muscles),length(coordinates));
test2=false(length(muscles),length(coordinates));
test2(test>0.5*nSamples)=true;
for m = 1:length(muscles)
    for c = 1:length(coordinates)
        if test2(m,c)
           MuscleData.dM(:,m,c)=0;
        end
    end
end
% if contribution < 1mm -> out
MuscleData.dM(abs(MuscleData.dM)<0.001) = 0;

% Get spanning matrix
spanning = squeeze(sum(MuscleData.dM, 1));
spanning(spanning<=0.0001 & spanning>=-0.0001) = 0;
spanning(spanning~=0) = 1;

for m = 1:length(muscles)
    metaData = struct('sMuscle',[]);
    metaData.sMuscle = muscles{m};
    idxSpanning = find(spanning(m,:) == 1);
    metaData_temp = struct('nDOF',[]);
    for c = 1:sum(spanning(m,:))
       metaData.sDOFlist{c} = coordinates{idxSpanning(c)};        
       metaData_temp.nDOF(:,c) = coordinate_values.data(:,strcmp(coordinate_values.colheaders, coordinates{idxSpanning(c)}))*pi/180;
       metaData_temp.nMomArm(:,c) = MuscleData.dM(:,m,idxSpanning(c));       
    end
    % Numerical error
    metaData_temp.nDOF(abs(metaData_temp.nDOF)<1e-8)=0;
    % We round to 3 decimals to account for tiny differences in angles 
    % thatshould not matter.
    [metaData.nDOF, IA, ~] = unique(round(metaData_temp.nDOF,3), 'rows', 'stable');
    metaData.nMomArm = metaData_temp.nMomArm(IA, :);
    metaData.nLength = MuscleData.lMT(IA,m); 
    save([folderNameTrainingData, '/all/', muscles{m},'MomentArm'], 'metaData')
end

%% Get info to fill out .csv files
nRangeMin = zeros(length(coordinates),1);
nRangeMax = zeros(length(coordinates),1);
for c = 1:length(coordinates)
    nRangeMin(c) =  min(coordinate_values.data(:,strcmp(coordinate_values.colheaders, coordinates{c}))*pi/180);
    nRangeMax(c) =  max(coordinate_values.data(:,strcmp(coordinate_values.colheaders, coordinates{c}))*pi/180);
end

idxSpanning_str = cell(length(muscles),1);
for m = 1:length(muscles)
   idxSpanning_str{m} = find(spanning(m,:) == 1);
end
       