% This code save already computed C3D features into 32 (video features) segments.

clc
clear all
close all

C3D_Path = './Testing_C3D_Feature';
C3D_Path_Seg = './Testing_Video_Feature';

if ~exist(C3D_Path_Seg, 'dir')
    mkdir(C3D_Path_Seg)
end

All_Folder = dir(C3D_Path);
All_Folder = All_Folder(3:end);
subcript = '.txt';

for ifolder = 1:length(All_Folder)
    % Folder_Path is path of a folder which contains C3D features (for every 16 frames) for a paricular video.
    Folder_Path = [C3D_Path, '/', All_Folder(ifolder).name];
    AllFiles = dir([Folder_Path, '/*.fc6-1']);
    Feature_vect = zeros(length(AllFiles), 4096);
    for ifile = 1:length(AllFiles)
        FilePath = [Folder_Path, '/', AllFiles(ifile).name];
        [s, data] = read_binary_blob(FilePath);
        Feature_vect(ifile,:) = data;
        clear data
    end
    
    % Assume each element of Feature_vect is not zero
    if sum(Feature_vect(:)) == 0
        error('??')
    end
    if ~isempty(find(sum(Feature_vect, 2) == 0, 1))
        error('??')
    end
    % Assume each element of Feature_vect must be numerical and not Inf
    if ~isempty(find(isnan(Feature_vect(:)), 1))
        error('??')
    end
    if ~isempty(find(Feature_vect(:) == Inf, 1))
        error('??')
    end
    
    % Write video features into text file
    % In Training_AnomalyDetector_public.py, You can directly use .mat format if you want.
    fid1 = fopen([C3D_Path_Seg, '/', All_Folder(ifolder).name, subcript], 'w');
    
    % C3D features (for every 16 frames) of a video -> 32 Segments_Feature
    Segments_Feature = zeros(32, 4096);
    thirty2_shots = round(linspace(1, length(AllFiles), 33));
    
    for ishots = 1: length(thirty2_shots) - 1
        ss = thirty2_shots(ishots);
        ee = thirty2_shots(ishots + 1) - 1;
        
        if ss >= ee
            temp_vect = Feature_vect(ss, :);        
        else
            temp_vect = mean(Feature_vect(ss : ee, :));
        end
        
        % normalization
        if norm(temp_vect) == 0
           error('??')
        end
        temp_vect = temp_vect / norm(temp_vect);
        
        Segments_Feature(ishot, :) = temp_vect;
    end
    
    if ~isempty(find(isnan(Segments_Feature(:)), 1))
        error('??')
    end
    if ~isempty(find(sum(Segments_Feature, 2) == 0, 1))
        error('??')
    end
    if ~isempty(find(Segments_Feature(:) == Inf, 1))
        error('??')
    end
    
    for ii = 1:size(Segments_Feature, 1)
       feat_text = Segments_Feature(ii, :);
       fprintf(fid1, '%f ', feat_text);
       fprintf(fid1, '\n');
    end
          
    fclose(fid1);
end
