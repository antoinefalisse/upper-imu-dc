clear all
clc
close all

%% User settings
nNodes = 3;
nDim = 9;
% This is usually fixed.
folderNameTrainingData = ['grid_',num2str(nNodes),'nodes_',num2str(nDim),'dim'];
prefix = 'subject01_MuscleAnalysis_';
nSamples = nNodes^nDim;
nRowPerChunk = 10000;
nChunks = ceil(nSamples/nRowPerChunk);
remDiv = rem(nSamples,nRowPerChunk);

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
lMT_all = zeros(nSamples,length(muscles)+1);
for nChunk=1:nChunks
    if nChunk < nChunks
        idxRows = (nChunk-1)*nRowPerChunk+1:(nChunk-1)*nRowPerChunk+nRowPerChunk;
    else
        idxRows = (nChunk-1)*nRowPerChunk+1:(nChunk-1)*nRowPerChunk+remDiv;
    end
    
    pathLengths = [folderNameTrainingData, '/angles_',num2str(nChunk-1),'/', prefix, 'Length.sto'];
    lMT = importdata(pathLengths);
    lMT_all(idxRows,:) = lMT.data;
end

% % Moment arms
% for i = 1:length(coordinates)
%     MA.([coordinates{i}]) = importdata([folderNameTrainingData, '/all/', prefix,'MomentArm_',coordinates{i},'.sto']);
% end
% % Joint coordinate values
% pathMotion = [folderNameTrainingData, '/all/training_q.mot'];
% coordinate_values = importdata(pathMotion);
% 
% for m = 1:length(muscles)
%     MuscleData.lMT(:,m)     = lMT.data(:,strcmp(lMT.colheaders,muscles{m}));
%     for i = 1:length(coordinates)
%         MuscleData.dM(:,m,i) = MA.([coordinates{i}]).data(:,strcmp(lMT.colheaders,muscles{m}));
%     end
% end
% 
% %% Format data as required by multi-dim-poly
% % Get spanning matrix
% spanning = squeeze(sum(MuscleData.dM, 1));
% spanning(spanning<=0.0001 & spanning>=-0.0001) = 0;
% spanning(spanning~=0) = 1;
% 
% for m = 1:length(muscles)
%    metaData = struct('nLength',[]);
%    metaData.nLength = lMT.data(:,strcmp(lMT.colheaders, muscles{m}));     
%    metaData.sMuscle = muscles{m};
%    idxSpanning = find(spanning(m,:) == 1);
%    for c = 1:sum(spanning(m,:))
%        metaData.sDOFlist{c} = coordinates{idxSpanning(c)};        
%        metaData.nDOF(:,c) = coordinate_values.data(:,strcmp(coordinate_values.colheaders, coordinates{idxSpanning(c)}))*pi/180;
%        metaData.nMomArm(:,c) = MA.([coordinates{idxSpanning(c)}]).data(:,strcmp(MA.([coordinates{idxSpanning(c)}]).colheaders, muscles{m}));
%    end
%    save([folderNameTrainingData, '/all/', muscles{m},'MomentArm'], 'metaData')
% end
% 
% %% Get info to fill out .csv files
% nRangeMin = zeros(length(coordinates),1);
% nRangeMax = zeros(length(coordinates),1);
% for c = 1:length(coordinates)
%     nRangeMin(c) =  min(coordinate_values.data(:,strcmp(coordinate_values.colheaders, coordinates{c}))*pi/180);
%     nRangeMax(c) =  max(coordinate_values.data(:,strcmp(coordinate_values.colheaders, coordinates{c}))*pi/180);
% end
% 
% idxSpanning_str = cell(length(muscles),1);
% for m = 1:length(muscles)
%    idxSpanning_str{m} = find(spanning(m,:) == 1);
% end
%        