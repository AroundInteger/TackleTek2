clear;clc
%%
%fldr = "/Users/Rowan/Library/CloudStorage/OneDrive-SwanseaUniversity/Research/WIPS/Intensity Monitoring/";
%fldr = "/Users/MRBIMac/OneDrive - Swansea University/Research/WIPS/Intensity Monitoring/";
fldr = "/Users/iMacPro/Library/CloudStorage/OneDrive-SwanseaUniversity/Research/WIPS/Intensity Monitoring/";
fn = strcat(fldr,"R1_1.mp4");

% Read the video file
videoFile = strcat(fldr,"R1_1.mp4");
videoReader = VideoReader(videoFile);

% Create optical flow object
opticFlow = opticalFlowLK('NoiseThreshold', 0.0001);
%opticFlow = opticalFlowLKDoG('NoiseThreshold', 0.0001);
%opticFlow = opticalFlowFarneback("NumPyramidLevels",3,"NumIterations",5);

% Initialize video writer for results
outputVideoOpticalFlow = VideoWriter('output_optical_flow.mp4',"MPEG-4");
outputVideoMotionSegmentation = VideoWriter('output_motion_segmentation.mp4',"MPEG-4");
open(outputVideoOpticalFlow);
open(outputVideoMotionSegmentation);

% Parameters for motion-based segmentation
numFramesToAverage = 30; % Number of frames to average for background model
backgroundModel = [];

% Process each frame
while hasFrame(videoReader)
   frame = readFrame(videoReader);
   grayFrame = rgb2gray(frame);

   % blue helmet

   % Optical Flow Method
   flow = estimateFlow(opticFlow, grayFrame);
   mag = sqrt(flow.Vx.^2 + flow.Vy.^2);
   threshold = 0.5; % Threshold for detecting motion
   opticalFlowMask = bwareaopen(mag > threshold,2000);
   %opticalFlowMask = bwareaopen(imfill(bwareaopen(mag > threshold,1000),"holes"),10000);
   opticalFlowResult = frame;
   opticalFlowResult(repmat(opticalFlowMask, [1, 1, 3])) = 255; % Highlight moving objects

   % Motion-Based Segmentation Method
   if isempty(backgroundModel)
       backgroundModel = double(grayFrame);
   else
       backgroundModel = (backgroundModel * (numFramesToAverage - 1) + double(grayFrame)) / numFramesToAverage;
   end
   motionMask = imfill(abs(double(grayFrame) - backgroundModel) > 20,"holes"); % Threshold for detecting motion
   motionSegmentationResult = frame;
   motionSegmentationResult(repmat(motionMask, [1, 1, 3])) = 255; % Highlight moving objects

   % Write results to video
   writeVideo(outputVideoOpticalFlow, opticalFlowResult);
   writeVideo(outputVideoMotionSegmentation, motionSegmentationResult);

   % Display results for comparison
   subplot(2, 1, 1);
   imshow(opticalFlowResult);
   title('Optical Flow Result');

   subplot(2, 1, 2);
   imshow(motionSegmentationResult);
   title('Motion-Based Segmentation Result');

   drawnow;
end

% Close video writers
close(outputVideoOpticalFlow);
close(outputVideoMotionSegmentation);
% 
% disp('Processing complete. Results saved to output_optical_flow.avi and output_motion_segmentation.avi');