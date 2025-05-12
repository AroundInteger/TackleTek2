clear;clc;
%%
%fldr = "/Users/Rowan/Library/CloudStorage/OneDrive-SwanseaUniversity/Research/WIPS/Intensity Monitoring/";
%fldr = "/Users/MRBIMac/OneDrive - Swansea University/Research/WIPS/Intensity Monitoring/";
fldr = "/Users/iMacPro/Library/CloudStorage/OneDrive-SwanseaUniversity/Research/WIPS/Intensity Monitoring/";
fn = strcat(fldr,"R1_2.mp4");

%% Open VideoViewer


vR = VideoReader(fn);


% Parameters
numFrames = vR.NumFrames;
frameSize = [vR.Height, vR.Width];
%numBgFrames = min(numFrames, numFrames); % Number of frames to use for background model

% Initialize matrix to store frames

%% Read all frames

%frameStore = zeros(frameSize(1),frameSize(2),numFrames);
frameMatrix = zeros(prod(frameSize), numFrames,3);

for ii = 1:numFrames

    frame = read(vR,ii);
    %labFrame = rgb2lab(frame);
    rFrame = frame(:,:,1);
    gFrame = frame(:,:,2);
    bFrame = frame(:,:,3);
    %grayFrame = (rgb2gray(frame));
    frameMatrix(:,ii,1) = rFrame(:);
    frameMatrix(:,ii,2) = gFrame(:);
    frameMatrix(:,ii,3) = bFrame(:);

end
%%
mask_kk = false(size(frameMatrix));
for kk = 1:3
X1 = frameMatrix(:,1:numFrames-1,kk);
X2 = frameMatrix(:,2:numFrames,kk);
%% Perform truncated SVD on X1
[U, S, V] = svd(X1, 'econ');
%%
r = 1;

U_r = U(:, 1:r);
S_r = S(1:r, 1:r);
V_r = V(:, 1:r);

% Compute DMD matrix
Atilde = U_r' * X2 * V_r / S_r;

% Eigendecomposition of Atilde
[W, D] = eig(Atilde);

% DMD modes
Phi = X2 * V_r / S_r * W;

% DMD eigenvalues
lambda = diag(D);

% Compute DMD amplitudes
b = Phi \ X1(:, 1);


%% Process all frames
%outputVideoDMD = VideoWriter('output_dmd.mp4',"MPEG-4");
%open(outputVideoDMD);



for ii = 1:1:numFrames
  
    frameVec = frameMatrix(:,ii);
  
    % weights = U_k' * frameVec;
    
   
    % Reconstruct background using DMD
    bgVec = real(Phi * (b .* lambda.^(ii-1)));

    
    % Compute foreground
    fgVec = abs(frameVec - bgVec);
    
    % Reshape to image
    fgImage = reshape(fgVec, frameSize);
    fgImage = imadjust(uint8(fgImage));
    %sobelGradient = imgradient(fgImage);
    
    % Threshold to get binary mask
    mask = bwareaopen(fgImage > 200,50); % Adjust threshold as needed
    %mask = imfill(mask,'holes');

    I = fgImage;
    I(~mask) = 0;

    % if ii > 1
    % 
    % % Display results
    % figure(19)
    % subplot(3,1,1), imshow(reshape(uint8(frameVec), frameSize)), title('Original Frame');
    % subplot(3,1,2), imshow(reshape(uint8(bgVec), frameSize)), title('background Mask');
    % subplot(3,1,3), imshow(I), title('Foreground Mask');
    % drawnow;
    % 
    % dmdResult = I;
    %    %opticalFlowResult(repmat(opticalFlowMask, [1, 1, 3])) = 255; % Highlight moving objects
    % 
    % %writeVideo(outputVideoDMD, dmdResult);
    % end

    % maskJ = mask;

    mask_kk(:,ii,kk) = mask(:);
end

end
%%
for ii = 100%:1:numFrames

    bw_r = reshape(mask_kk(:,ii,1), frameSize);
    bw_g = reshape(mask_kk(:,ii,2), frameSize);
    bw_b = reshape(mask_kk(:,ii,3), frameSize);
    frameVec = reshape(frameMatrix(:,ii,:), [frameSize,3]);

    mask = or(bw_r,bw_g);

    I = frameVec;
    I(~mask) = 0;

    Ib_fM = frameMatrix(:,ii,3);
    I_b = reshape(Ib_fM, frameSize);

    mask_helmet = I_b < 100;
    figure(1);imshow(mask_helmet);
 

    figure(19)
    subplot(4,1,1), imshow(bw_r);
    subplot(4,1,2), imshow(bw_g);
    subplot(4,1,3), imshow(bw_b);
    subplot(4,1,4), imshowpair(bw_r,uint8(frameVec));


end
%%

%close(outputVideoDMD);

% This script does the following:
% 
% 1. Reads the video and sets up parameters.
% 2. Creates a matrix where each column is a flattened grayscale frame.
% 3. Performs SVD on this matrix.
% 4. Keeps only the top k principal components to create a background model.
% 5. Processes each frame of the video:
%    - Projects the frame onto the eigenspace.
%    - Reconstructs the background using this projection.
%    - Subtracts the reconstructed background from the original frame to get the foreground.
%    - Thresholds the difference to create a binary foreground mask.
%    - Displays the original frame and the foreground mask.
% 
% Key points:
% 
% - The `numBgFrames` parameter determines how many frames are used to construct the background model. Adjust this based on your video.
% - The `k` parameter determines how many principal components are kept. A larger k will capture more variation but may also include some foreground objects in the background model.
% - The threshold for creating the binary mask may need adjustment depending on your video.
% 
% This method works well for videos where the background is mostly static and the camera doesn't move. It can handle gradual lighting changes and some periodic motion in the background.
% 
% To improve this further, you could:
% 1. Update the background model periodically.
% 2. Use a more sophisticated thresholding method.
% 3. Apply morphological operations to clean up the foreground mask.

