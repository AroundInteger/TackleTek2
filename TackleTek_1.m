clear
%%

fldr = "/Users/iMacPro/Library/CloudStorage/OneDrive-SwanseaUniversity/Research/TackleTek/";
fn = strcat(fldr,"TackleTEK.mov");

%% Open VideoViewer


vR = VideoReader(fn);
vW = VideoWriter("R1_4.mp4","MPEG-4");open(vW)
% 

%%
numFrames = vR.NumFrames;

%frames = cell(numFrames, 1);
RECT = [100,1150,3500,480];
for kk = 1:1:100

    jj = kk + 209;
    I = read(vR, jj);

    J = imcrop(I,RECT);
    figure(1);imshow(J);drawnow
    writeVideo(vW,J)

    %frames{kk} = frame;
end

close(vW)
%figure(1);imshow(J)

% %%
% detector = yolov2ObjectDetector('tiny-yolov2-coco');
% 
% for kk = 1%:numFrames
% 
%     I = rgb2gray(frames{kk});
% 
%     figure(1);imshow(I)
% 
%     detections = detect(detector, I);
% 
%     J = insertObjectAnnotation(I,'rectangle',bboxes,scores);
% 
% end
% 








%%

% for ii = 60*3.5:60*6
% % I0 = read(v,round(60*4));
% I = read(vR,ii);
% % figure(1);subplot(1,2,1)
% % imshow(I)
% 
% % [bboxes,scores] = detect(detector,I);
% % %[bboxes,scores] = peopleDetector(I);
% % 
% % J = insertObjectAnnotation(I,'rectangle',bboxes,scores);
% figure(1);%subplot(1,2,2)
% imshow(I)
% %title('Detected People and Detection Scores')
% drawnow
% 
% end