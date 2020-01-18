clc
clear all
close all

% C3D features for videos
C3D_CNN_Path = 'Testing_C3D_Feature';
% Path of mp4 videos
Testing_VideoPath = 'Testing_Videos';
% Path of Temporal Annotations (Here is .mat format. You can use .txt format if you want.)
AllAnn_Path = 'Temporal_Anomaly_Annotation_For_Testing_Videos/Matlab_formate';
% AllAnn_Path = 'Temporal_Anomaly_Annotation_For_Testing_Videos/Txt_formate';
% Path of Pretrained Model score on Testing videos (32 numbers for 32 temporal segments)
Model_Score_Folder = 'Model_Res';
% Path to save results
Paper_Results = 'Eval_Res';

frm_counter = 1;
All_Detect = zeros(1, 1000000);
All_GT = zeros(1, 1000000);

All_Videos_scores = dir(Model_Score_Folder);
All_Videos_scores=All_Videos_scores(3:end);
nVideos = length(All_Videos_scores);
for ivideo = 1:nVideos
    Ann_Path = [AllAnn_Path, '/', All_Videos_scores(ivideo).name(1:end-4), '.mat'];
    load(Ann_Path)
    
    VideoPath = [Testing_VideoPath, '/', All_Videos_scores(ivideo).name(1:end-4), '.mp4'];
    ScorePath = [Model_Score_Folder, '/', All_Videos_scores(ivideo).name];
    
    % load videos
    try
        xyloObj = VideoReader(VideoPath);
    catch
        error('load videos failed')
    end
    
    Predic_scores = load(ScorePath);
    Actual_frames = round(xyloObj.Duration * xyloObj.FrameRate);
    
    Folder_Path = [C3D_CNN_Path, '/', All_Videos_scores(ivideo).name(1:end-4)];
    AllFiles = dir([Folder_Path, '/*.fc6-1']);
    % As the features were computed for every 16 frames
    nFrame = length(AllFiles) * 16;
    
    % 32 shots
    Detection_score_32shots = zeros(1, nFrame);
    Thirty2_shots = round(linspace(1, length(AllFiles), 33));
    
    for ishots = 1:length(Thirty2_shots) - 1
        ss = Thirty2_shots(ishots);
        ee = Thirty2_shots(ishots + 1) - 1;
        if ee < ss
            Detection_score_32shots((ss - 1) * 16 + 1:ss * 16) = Predic_scores.prediction(ishots);
        else
            Detection_score_32shots((ss - 1) * 16 + 1:ee * 16) = Predic_scores.prediction(ishots);            
        end        
    end
    
    Final_score = [Detection_score_32shots, repmat(Detection_score_32shots(end), [1, Actual_frames - nFrame])];
    
    % For Normal Videos
    if strcmp(Annotation_file.EventName, 'Normal')
        GT = zeros(1, Actual_frames);
    else
        st_fr = max(Annotation_file.Ann(1), 1);
        end_fr = min(Testing_Videos1.Ann(2), Actual_frames);
        GT(st_fr:end_fr) = 1;
    end

    All_Detect(frm_counter:frm_counter + length(Final_score) - 1) = Final_score;
    All_GT(frm_counter:frm_counter + length(Final_score) - 1) = GT;
    frm_counter = frm_counter + length(Final_score);
    
end

% decrease size of All_Detect and All_GT
All_Detect = (All_Detect(1 : frm_counter - 1));
All_GT = All_GT(1 : frm_counter - 1);

% draw ROC and calculate AUC
[B, I] = sort(All_Detect, 'descend');
tp = B > 0; tp = cumsum(tp);
fp = B == 0; fp = cumsum(fp);
tpr = tp / sum(All_GT); fpr = fp / sum(All_GT == 0);
AUC = trapz(fpr, tpr);
precison = tp ./ (fp + tp);
% [X, Y, T, AUC] = perfcurve(All_GT, All_Detect, 1);


