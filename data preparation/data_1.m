% Data Preparation for Train and Test Set 1

% Data Taken from: http://medicalsegmentation.com/covid19/
% Data Download link :Training images as .nii.gz (151.8 Mb)  – 100 slices
%                     Training  masks as .nii.gz (1.4 Mb)    – 100 masks
% lung masks by Johannes Hofmanninger: Lung masks as .nii.gz - 100 masks
% (0.3Mb). 
% https://github.com/JoHof/lungmask
% 


% Phaneendra Yalavarthy
% Created on: March 31, 2020; Updated by Naveen Paluru on: April 18, 2020
% 


% Do some clean up
clear all;
close all
clc; 
% 
%
%
%% Load the data

% CT slice data
im = niftiread('tr_im.nii');

% Labeled Data
mim = niftiread('tr_mask.nii'); %and

% Lung Mask data
mlim = niftiread('tr_lungmasks.nii');

% Number of Slices
noslices = size(im, 3);
%% Now Run it for available no of slices available
for i = 1:100
    
    % Take one frame
    frame = squeeze(im(:,:, i));
    
    % Lung mask (make sure it is a binary mask)
    mlim1 = squeeze(mlim(:,:, i));
    mlim1(mlim1(:)>1) = 1;
    
    % Make background 0
    frame = frame.*double(mlim1);
    % It is tilted, so transpose
    image_scan_original = frame';
    [m, n] = size(frame');
    
    % Convert to unit8 (DO NOT USE unit8 command, it does not work!)
    image_scan = uint8(255 * mat2gray(image_scan_original));
    
        
    % Background
    ml = squeeze(mim(:,:, i));
    ml0 = zeros(size(ml));
    ml0(ml(:) == 0) = 1;
       
    % Label for Ground Glass Opacity (GGO)
    ml1 = zeros(size(ml));
    ml1(ml(:) == 1) = 1;
    ml1 = imbinarize(ml1');
       
    % Label for Pleural Effusion
    ml3 = zeros(size(ml));
    ml3(ml(:) == 3) = 1;
    ml3 = imbinarize(ml3');
        
    % Label for Consolidation
    ml2 = zeros(size(ml));
    ml2(ml(:) == 2) = 1;
    ml2 = imbinarize(ml2');
    
    
    % combine all labels 1, 2, 3 data into 1 (Abnormal Labels)
    ml12 = imbinarize(ml2 + ml1 + ml3);
    positive = ml12;
     
     % Normal Labels
     negative = logical(mlim1').*imcomplement(ml12);
     negative(negative==1)=2;
     
     
     % Overall Labels
     labels = positive+negative;
     
     
     % Augmentation
     
     orgi = image_scan;        % Original
     fhi  = flip(orgi,1);      % Flip Horizontal
     fvi  = flip(orgi,2);      % Flip Vertical
    
    
     
    % Similarly Do for labels
     
     orgl = labels;
     fhl  = flip(orgl,1);
     fvl  = flip(orgl,2);
    
     
     % Concatenate the data
     

      if i == 1
          inp = cat(3,orgi,fhi,fvi);
          lab = cat(3,orgl,fhl,fvl);
      else
          inp = cat(3,inp,orgi,fhi,fvi);
          lab = cat(3,lab,orgl,fhl,fvl);
          
      end
  
      
      % Train data 270 slices
                
end


