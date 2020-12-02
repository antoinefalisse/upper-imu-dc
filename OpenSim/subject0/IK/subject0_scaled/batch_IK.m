import org.opensim.modeling.*;
% taskNames={'abd01';'abd02';'abd03';'flx01';'flx02';'flx03';'shrug01';'shrug02';'shrug03';...
%     'abd21';'abd22';'abd23';'flx21';'flx22';'flx23';'shrug21';'shrug22';'shrug23'};
taskNames={'flx01'};

Model.LoadOpenSimLibrary('C:\Users\u0101727\Documents\Visual Studio 2017\Projects\ScapulothoracicJointPlugin\install\plugins\ScapulothoracicJointPlugin40.dll');

for i=1:length(taskNames)
    ikTool = InverseKinematicsTool('IKsetup.xml');
    trcFile = ['..\TRC\', taskNames{i},'.trc'];
    ikTool.setMarkerDataFileName(trcFile);
    output = ['IK_', taskNames{i}, '.mot'];
    ikTool.setOutputMotionFileName(output);
    ikTool.setStartTime(0);
    ikTool.setEndTime(15);
    path_setup = ['.\Setup_IK_', taskNames{i}, '.xml'];
    ikTool.print(path_setup);
    Command = ['opensim-cmd' ' run-tool ' path_setup];
    system(Command);
end   
