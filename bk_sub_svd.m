clear
%%
fldr = "/Users/Rowan/Library/CloudStorage/OneDrive-SwanseaUniversity/Research/WIPS/Intensity Monitoring/";
%fldr = "/Users/MRBIMac/OneDrive - Swansea University/Research/WIPS/Intensity Monitoring/";
%fldr = "/Users/iMacPro/Library/CloudStorage/OneDrive-SwanseaUniversity/Research/WIPS/Intensity Monitoring/";
fn = strcat(fldr,"R1.mp4");

%% Open VideoViewer


v = VideoReader(fn);


% Parameters
numFrames = v.NumFrames;
frameSize = [v.Height, v.Width];
numBgFrames = min(50, numFrames); % Number of frames to use for background model


RECT = [60,960,3650,755];
for i = 1:numBgFrames
    frame = imcrop(readFrame(v),RECT);
end
%%

% Initialize matrix to store frames
frameMatrix = zeros(prod(frameSize), numBgFrames);

% Read frames and store in matrix
for i = 1:numBgFrames
    frame = readFrame(v);
    grayFrame = rgb2gray(frame);
    frameMatrix(:, i) = double(grayFrame(:));
end

% Perform SVD
[U, S, V] = svd(frameMatrix, 'econ');

%% Keep only the top k components
k = 10; % Number of principal components to keep

U_k = U(:, 1:k);
S_k = S(1:k, 1:k);
V_k = V(:, 1:k);

% Compute background model
bgModel = U_k * S_k * V_k';

s = diag(S_k);
figure(9);semilogy(s,'o')

% Reset video reader
% v.CurrentTime = 0;

%% Process all frames
for i = 20%:numFrames
    % Read frame
    % frame = readFrame(v);
    % grayFrame = rgb2gray(frame);
    
    % Reshape frame to column vector
    %frameVec = double(grayFrame(:));
    frameVec = frameMatrix(:,i);
    
    % Project frame onto eigenspace
    weights = U_k' * frameVec;
    
    % Reconstruct background
    bgVec = U_k * weights;
    
    % Compute foreground
    fgVec = abs(frameVec - bgVec);
    
    % Reshape to image
    fgImage = reshape(fgVec, frameSize);
    
    % Threshold to get binary mask
    mask = bwareaopen(fgImage > 15,20); % Adjust threshold as needed
    
    % Display results
    figure(19)
    subplot(2,1,1), imshow(grayFrame), title('Original Frame');
    subplot(2,1,2), imshow(mask), title('Foreground Mask');
    drawnow;
end


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

