clear
%%
%fldr = "/Users/Rowan/Library/CloudStorage/OneDrive-SwanseaUniversity/Research/WIPS/Intensity Monitoring/";
fldr = "/Users/MRBIMac/OneDrive - Swansea University/Research/WIPS/Intensity Monitoring/";
%fldr = "/Users/iMacPro/Library/CloudStorage/OneDrive-SwanseaUniversity/Research/WIPS/Intensity Monitoring/";
fn = strcat(fldr,"R1_1.mp4");

%% Open VideoViewer


vR = VideoReader(fn);
%vW = VideoWriter("R1.mp4","MPEG-4");
%%
%detector = peopleDetectorACF;
%peopleDetector = vision.PeopleDetector;

dt = 1/60;

numFrames = 151;
% open(vW)
% frames = cell(numFrames, 1);
% for kk = 1:numFrames
%     jj = kk + 209;
%     frame = read(vR, jj);
% 
%     writeVideo(vW,frame)
% 
%     frames{kk} = frame;
% end
% 
% close(vW)


%%
%detector = yolov2ObjectDetector('tiny-yolov2-coco');
%RECT = [60,960,3650,755];
for kk = 1%:5:numFrames


    %I = imcrop(read(vR, kk),RECT);
    I = read(vR, kk);
    K = read(vR, kk+50);


    
    J = rgb2ycbcr(I);

    %[J,rect] = imcrop(I);
    %J = Ic;
    

    id_r = and(I(:,:,1) > 40,I(:,:,1) < 90);
    id_g = and(I(:,:,2) > 0,I(:,:,2) < 50);
    id_b = and(I(:,:,3) > 0,I(:,:,3) < 30);
    id_orange = id_r&id_g&id_b;
    figure(1);imshowpair(I,id_orange);

    id_sky = and(J(:,:,1) > 1,J(:,:,1) < 100);
    id_sky = imfill(id_sky,"holes");
    id_sky = imopen(~id_sky,strel('disk',10));

    figure(8);imshow(id_sky);


    id_g = and(J(:,:,2) > 135,J(:,:,2) < 256);

    bw_hg = bwareaopen(imclearborder(id_g),1000);
    bw_hg = imopen(bw_hg,strel('disk',5));
    bw_hg = ~bwareaopen(~bw_hg,100);
    bw_hg(id_sky) = 0;

    %figure(1);imshow()
    figure(2);
    subplot(3,1,1);imshow(I)
    subplot(3,1,2);imshow(J)
    subplot(3,1,3);imshowpair(I,bw_hg)

    %figure(8);imshow(~bw)

    % c = bwconncomp(bw);
    % rp = regionprops(c);
    % 
    % detections = detect(detector, I);
    % 
    % J = insertObjectAnnotation(I,'rectangle',bboxes,scores);

end

% %%
% 
% %detector = yolov2ObjectDetector('tiny-yolov2-coco');
% RECT = [60,960,3650,755];
% for kk = 1%:numFrames-50
% 
% 
%     I1 = imcrop(read(vR, kk),RECT);
%     I2 = imcrop(read(vR, kk+5),RECT);
% 
%     J1 = rgb2ycbcr(I1);
%     J2 = rgb2lab(I2);
% 
%     %[J,rect] = imcrop(I);
%     %J = Ic;
% 
% 
%     %id_r = and(J1(:,:,1) > 0,J1(:,:,1) < 40);
%     id_g = and(J1(:,:,2) > 140,J1(:,:,2) < 256);
%     %id_b = and(J1(:,:,3) > -40,J1(:,:,3) < -20);
% 
%     bw1 = bwareaopen(imclearborder(id_g),500);
%     figure(1);
%     subplot(3,1,1);imshow(I1)
%     subplot(3,1,2);imshow(bw1)
%     subplot(3,1,3);imshowpair(I1,bw1)
% 
%     mxJ = max(J2);
% 
%     id_r = J2(:,:,1) == max(J2(:));
%     id_g = J2(:,:,2) == max(J2(:));
%     id_b = J2(:,:,3) == max(J2(:));
% 
%     bw2 = id_r&id_g&id_b;
% 
%     %figure(1);imshow()
% 
%     figure(2);
%     subplot(3,1,1);imshow(I2)
%     subplot(3,1,2);imshow(J2)
%     subplot(3,1,3);imshowpair(I2,bw2)
% 
% 
%     %D = []
%     % c = bwconncomp(bw1);
%     % rp = regionprops(c);
% 
%     % detections = detect(detector, I);
%     % 
%     % J = insertObjectAnnotation(I,'rectangle',bboxes,scores);
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