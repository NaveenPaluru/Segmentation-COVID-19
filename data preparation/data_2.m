% 

% Data Taken from: http://medicalsegmentation.com/covid19/
% Segmentation dataset nr. 2 (13th April) Image volumes (308 Mb) – 9 volumes, total of > 800 slices
%                                         Covid19 masks (0.3 Mb) – includes  > 350 annotated slices
%                                         Lung masks of (1.0 Mb) – includes  > 700 annotated slices

% lung masks by Johannes Hofmanninger:  https://github.com/JoHof/lungmask
% 
% Phaneendra Yalavarthy
% Created on: March 31, 2020; Updated by Naveen Paluru on: April 18, 2020
% 


% Do Some clean up

clc; clear all; close all;

% Download the files from above repositoroes and unzip them and add all the
% paths to matlab.

addpath(genpath(['./']));

% Get all the files with .nii extension

mskni = dir('rp_msk/*.nii'); lngmk= dir('rp_lung_msk/*.nii');
imnii  = dir('rp_im/*.nii' );

% Make a bounding box to ccrop the images and labels from 630x630 to
% 512x512.

r = centerCropWindow2d([630,630],[512,512]);

% Two empty mtrices for preparing test data

IMG = [];
LAB = [];

% iterate over all the 9 volumes

for file = 1:length(mskni(:))
    
    image= niftiread(strcat(imnii(file).folder,'/', imnii(file).name));  
    mask = niftiread(strcat(mskni(file).folder,'/', mskni(file).name));
    lmsk = niftiread(strcat(lngmk(file).folder,'/', lngmk(file).name));
    
    % make a directory for current volume
    
    mkdir(num2str(file));
    
    % Go to the created directory (This is just to save the images here for
    % visualization)
    
    cd(num2str(file))
    
    % iterate over all slices in the current voume
    
    for j = 1:size(image,3)   
        
        % Take one frame
        frame = squeeze(image(:,:, j));  
        
        % Lung mask (make sure it is a binary mask)
        mlim1 = squeeze(lmsk(:,:, j));
        mlim1(mlim1(:)>1) = 1;        
        
        % Make background 0figure;
        frame = frame.*double(mlim1);
        
        % It is tilted, so transpose
        image_scan_original = frame';
        image_scan_original = uint8(255 * mat2gray(image_scan_original));
        
         % Background
        ml = squeeze(mask(:,:, j));
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
        
        
        % combine all labels 1, 2, 3 data into 1 (Abnormal labels)
        ml12 = imbinarize(ml2 + ml1 + ml3);        
        pos = ml12;
        
        % Make normal labels
        neg = logical(mlim1').*imcomplement(ml12);
        neg(neg==1)=2;        
        
        % Combile both labels (0:bg, 1:Abnrml, 2:nrml)
        lab = pos+neg;
        
        
        % Now crop the image and label to 512 x512
        imag  = imcrop(image_scan_original,r);
        labl  = imcrop(lab,r);
        
        % A small check that ensures we are not involving only background
        % in the test set
        
         if length(unique(lab))>1 && sum(lab>0,'all')>100
            IMG = cat(3,IMG, imag);
            LAB = cat(3,LAB, labl);
         end
        
        % This will save all the slices of each volume into  separate folders.
        % Change the flag Visible from off to on to view images while execting.
        
        fig=figure('rend','painters','pos', [50 , 300, 1500, 600],'Visible', 'off');
        subplot(121);imshow(mat2gray(imag));
        xlabel('Image');
        subplot(122);imshow(mat2gray(labl));
        xlabel('GT');
        saveas(fig, strcat(num2str(j),'.jpg'));
        close all;       
    end
    
    % Come out of current volume directory
    cd ../
end

% Finally Save IMG and LAB in testVOL.mat file. This mat file should have the:
% IMG : 512x512x704
% LAB : 512x512x704
