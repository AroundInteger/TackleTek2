clear
%%
%fldr = "/Users/Rowan/Library/CloudStorage/OneDrive-SwanseaUniversity/Research/WIPS/Intensity Monitoring/";
%fldr = "/Users/MRBIMac/OneDrive - Swansea University/Research/WIPS/Intensity Monitoring/";
fldr = "/Users/iMacPro/Library/CloudStorage/OneDrive-SwanseaUniversity/Research/WIPS/Intensity Monitoring/";
fn = strcat(fldr,"R1.mp4");

%% Open VideoViewer


v = VideoReader(fn);

% Parameters
numFrames = v.NumFrames;
frameSize = [v.Height, v.Width];
numBgFrames = min(numFrames, numFrames); % Number of frames to use for background model
r = 10; % Rank for truncated SVD in DMD

% Initialize matrices to store frames
X1 = zeros(prod(frameSize), numBgFrames-1);
X2 = zeros(prod(frameSize), numBgFrames-1);

% Read frames and store in matrices
for i = 1:numBgFrames
    frame = readFrame(v);
    grayFrame = rgb2gray(frame);
    if i < numBgFrames
        X1(:, i) = double(grayFrame(:));
    end
    if i > 1
        X2(:, i-1) = double(grayFrame(:));
    end
end

%% Perform truncated SVD on X1
[U, S, V] = svd(X1, 'econ');
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

% Reset video reader
v.CurrentTime = 0;

%% Process all frames
for i = 1:5:numFrames
    % Read frame
    frame = readFrame(v);
    grayFrame = rgb2gray(frame);
    frameVec = double(grayFrame(:));
    
    % Reconstruct background using DMD
    bgVec = real(Phi * (b .* lambda.^(i-1)));
    
    % Compute foreground
    fgVec = abs(frameVec - bgVec);
    
    % Reshape to image
    fgImage = reshape(fgVec, frameSize);
    
    % Threshold to get binary mask
    mask = bwareaopen(fgImage > 25,200); % Adjust threshold as needed
    
    % Display results
    % subplot(3,1,1), imshow(grayFrame), title('Original Frame');
    % subplot(3,1,2), imshow(uint8(reshape(bgVec, frameSize))), title('Background');
    % subplot(3,1,3), 
    figure(2);imshow(mask), title('Foreground Mask');
    drawnow;
end


% Key improvements and differences from the SVD version:
% 
% 1. DMD Computation:
%    - We create two matrices X1 and X2, where X2 is X1 shifted by one frame.
%    - We perform a truncated SVD on X1 to reduce computational complexity.
%    - We compute the DMD matrix Atilde and its eigendecomposition.
%    - We calculate the DMD modes (Phi) and eigenvalues (lambda).
% 
% 2. Background Reconstruction:
%    - Instead of projecting each frame onto a static eigenspace, we use the DMD modes and eigenvalues to reconstruct the background for each frame.
%    - This allows for a time-evolving background model, which can capture more complex background dynamics.
% 
% 3. Visualization:
%    - We now display the original frame, the reconstructed background, and the foreground mask for better comparison.
% 
% Advantages of DMD over SVD:
% 
% 1. Captures temporal dynamics: DMD can model how the background changes over time, which is particularly useful for scenes with periodic motion or gradual changes.
% 2. Frequency information: The DMD eigenvalues provide information about the frequency of different modes, which can be used to identify and filter out certain types of background motion.
% 3. Potentially better at separating foreground and background: In some cases, DMD can more effectively separate foreground objects from a dynamic background.
% 
% Potential improvements:
% 
% 1. Adaptive thresholding: Implement a more sophisticated thresholding method that adapts to local image statistics.
% 2. Online DMD: Implement an online version of DMD that updates the background model as new frames arrive.
% 3. Sparsity-promoting DMD: Use a variant of DMD that promotes sparsity in the modes, which can lead to a more compact representation of the background.

