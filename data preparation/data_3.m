clc; clear all; close all;

% Download the files from above repositoroes and unzip them and add all the
% paths to matlab.

addpath(genpath(['./']));

% Get all the files with .nii extension

mskni = dir('Infection_Mask/*.gz');    lngmk = dir('Lung_Mask/*.gz');    imnii  = dir('COVID-19-CT-Seg_20cases/*.gz' );


% Two empty mtrices for preparing data

IMG = [];
LAB = [];

% iterate over all the 20 volumes

for file = 1:20
    
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
        
         if file > 10
            ang = -90;
         else
            ang = 90;
        end
        
        % Lung mask (make sure it is a binary mask)
        mlim1 = squeeze(lmsk(:,:, j));
        mlim1(mlim1(:)>1) = 1;   
       
        
        % Make background 0figure;
        frame = double(frame).*double(mlim1);
        
        % It is tilted, so rotate
        image_scan_original = imrotate(frame,ang);
        image_scan_original = uint8(255 * mat2gray(image_scan_original));
        
        pos = uint8(imrotate(squeeze(mask(:,:,j)),ang));
        pos(pos(:)>1) = 1;
        pos = double(pos);
        neg = imrotate(double(mlim1),ang).*imcomplement(pos);
        neg(neg==1)=2;   
        lab = double(pos + neg);
        %unique(lab)
        
        if file >10
            
            % Make a bounding box to ccrop the images and labels from 630x630 to
            % 512x512.

            r = centerCropWindow2d([630,630],[512,512]);
        
            % Now crop the image and label to 512 x512
            imag  = imcrop(image_scan_original,r);
            labl  = imcrop(lab,r);
       
        else
            imag  = image_scan_original;
            labl  = lab;
        end

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
    name = strcat('patient',num2str(file),'.mat');
    save(name,'IMG','LAB');
    IMG = [];
    LAB = [];
end


