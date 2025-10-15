%% compare_with_stl_ssim.m
% This script aligns an STL scan to two simulated deformations via
% scale + translation + PCA + ICP, then computes 3D‐SSIM between the
% experimental shape and each simulation.

clearvars; close all; clc

%% 1) Load simulated shapes
data_ng = load('output_deps_thermal_NoG.mat');
data_g  = load('output_deps_thermal_WithGravity.mat');

NP    = data_ng.NP_total;
Q1    = data_ng.Q_history(end,:);
Q2    = data_g.Q_history(end,:);
Xdef1 = reshape(Q1,3,NP)';   % No‐gravity nodes (NP×3)
Xdef2 = reshape(Q2,3,NP)';   % With‐gravity nodes

%% 2) Load STL scan (outer surface mesh)
TR     = stlread('outer_surface_by_centroid_ray.stl');
Vstl   = TR.Points;          % M×3 vertices

%% 3) Match scale and centroids
simPts = [Xdef1; Xdef2];
simBBox = max(simPts,[],1) - min(simPts,[],1);
stlBBox = max(Vstl,[],1)   - min(Vstl,[],1);
scale   = mean(simBBox ./ stlBBox);
scale = 0.195;
scale

centSim = mean(simPts,1);
centStl = mean(Vstl,1);

Vstl = (Vstl - centStl) * scale + centSim;

figure(1); hold on;
scatter3(Xdef1(:,1),Xdef1(:,2),Xdef1(:,3),20,'r','filled');
scatter3(Xdef2(:,1),Xdef2(:,2),Xdef2(:,3),20,'b','filled');
scatter3(Vstl(:,1), Vstl(:,2), Vstl(:,3),20,'k');
axis equal; legend('NoG','WithG','STL');
title('After Scale + Center'); view(3); grid on;

%% 4) Rough alignment via PCA, but enforce no‐reflection
% Center both point‐sets
Xc1 = bsxfun(@minus, Xdef1, mean(Xdef1,1));  
Xc2 = bsxfun(@minus, Vstl,    mean(Vstl,   1));

% SVD to get principal axes
[~,~,Vsim]  = svd(Xc1, 'econ');
[~,~,VstlP] = svd(Xc2, 'econ');

% Build a pure rotation, not a reflection
Rinit = Vsim * VstlP';
if det(Rinit) > 0
    % flip the third column of VstlP and rebuild
    VstlP(:,3) = -VstlP(:,3);
    Rinit       = Vsim * VstlP';
end

% Apply that rotation about the common centroid
centAll = mean([Xdef1; Xdef2; Vstl],1);
Vstl = (Rinit*(Vstl - centAll)')' + centAll;

% Visualize immediately
figure(2); clf; hold on;
scatter3(Xdef1(:,1),Xdef1(:,2),Xdef1(:,3),20,'r','filled');
scatter3(Vstl(:,1), Vstl(:,2), Vstl(:,3),20,'k');
axis equal; legend('NoG','STL');
title('After PCA Alignment (no reflection)'); view(3); grid on;


%% 4.5) Fine-tune heading (Z-axis) alignment
% Project the first principal axes of each cloud down into XY:
u = Vsim(:,1);    % sim’s first PCA direction (a 3×1 vector)
v = VstlP(:,1);   % STL’s first PCA direction
uXY = u(1:2)/norm(u(1:2));
vXY = v(1:2)/norm(v(1:2));

% Compute signed angle from vXY to uXY
% phi = atan2( det([vXY uXY]), dot(vXY,uXY) );
%           ^ 2×2 det of [vXY uXY]     ^ dot product

phi=14/180*pi

% Build a Z-rotation by phi
Rz = [ cos(phi)  -sin(phi)   0;
       sin(phi)   cos(phi)   0;
         0          0        1 ];

% Apply that extra rotation about the common centroid
Vstl = (Rz*(Vstl - centAll)')' + centAll;

% Visualize to confirm
figure(21); clf; hold on;
scatter3(Xdef1(:,1),Xdef1(:,2),Xdef1(:,3),20,'r','filled');
scatter3(Vstl(:,1), Vstl(:,2), Vstl(:,3),20,'k','filled');
axis equal; legend('NoG','STL');
title('After PCA + Z-Rotation'); view(3); grid on;


%% 5) Refine with rigid ICP (point‐to‐point)
pcSim = pointCloud(Xdef1);
pcSTL = pointCloud(Vstl);

tform = pcregistericp(pcSTL, pcSim, ...
                      'Metric','pointToPoint', ...
                      'Extrapolate', true);

Vstl = pctransform(pcSTL, tform).Location;
htmlGray = [128 128 128]/255;
figure(3); hold on;
scatter3(Xdef1(:,1),Xdef1(:,2),Xdef1(:,3),20,'k','filled');
scatter3(Vstl(:,1), Vstl(:,2), Vstl(:,3),20,'Color', htmlGray);
axis equal; legend('NoG','STL');
title('After PCA + ICP Alignment'); view(3); grid on;




%% 6) Build common voxel grid
allPts = [Xdef1; Xdef2; Vstl];
mins   = min(allPts,[],1) - 0.05*(max(allPts,[],1)-min(allPts,[],1));
maxs   = max(allPts,[],1) + 0.05*(max(allPts,[],1)-min(allPts,[],1));

N = 64;
xs = linspace(mins(1),maxs(1),N);
ys = linspace(mins(2),maxs(2),N);
zs = linspace(mins(3),maxs(3),N);
[XX,YY,ZZ] = ndgrid(xs,ys,zs);
gridPts = [XX(:),YY(:),ZZ(:)];

%% 7) Voxelize as nearest‐neighbor distance fields
Mdl1 = createns(Xdef1,'NSMethod','kdtree');
[~,D1] = knnsearch(Mdl1,gridPts); D1 = reshape(D1,[N,N,N]);

Mdl2 = createns(Xdef2,'NSMethod','kdtree');
[~,D2] = knnsearch(Mdl2,gridPts); D2 = reshape(D2,[N,N,N]);

Mdl3 = createns(Vstl,'NSMethod','kdtree');
[~,D3] = knnsearch(Mdl3,gridPts); D3 = reshape(D3,[N,N,N]);

vol1 = mat2gray(D1);
vol2 = mat2gray(D2);
vol3 = mat2gray(D3);

%% 8) Compute 3D‐SSIM maps & indices
winSize = 10;
window  = ones(winSize,winSize,winSize)/winSize^3;
K1=0.01; K2=0.03; L=1; C1=(K1*L)^2; C2=(K2*L)^2;

ssim_map = cell(2,1);
ssim_idx = zeros(2,1);
for k=1:2
    A = vol3;      % STL reference
    B = (k==1)*vol1 + (k==2)*vol2;
    muA  = convn(A,window,'same');
    muB  = convn(B,window,'same');
    sA2  = convn(A.^2,window,'same') - muA.^2;
    sB2  = convn(B.^2,window,'same') - muB.^2;
    sAB  = convn(A.*B,window,'same') - muA.*muB;
    M    = ((2*muA.*muB + C1).*(2*sAB + C2)) ./ ...
           ((muA.^2 + muB.^2 + C1).*(sA2 + sB2 + C2));
    ssim_map{k} = M;
    ssim_idx(k) = mean(M(:));
end

fprintf('SSIM(STL vs No‐Gravity) = %.4f\n', ssim_idx(1));
fprintf('SSIM(STL vs With‐Gravity) = %.4f\n', ssim_idx(2));

%% 9) Visualize SSIM slices & node coloring
sliceIdx = ceil(N/2);
titles = {'No‐Gravity','With‐Gravity'};
figure('Position',[100 100 900 500]);
for k=1:2
  subplot(2,2,(k-1)*2+1);
  imagesc(xs,ys, ssim_map{k}(:,:,sliceIdx)); axis image off;
  colormap(gca,jet); colorbar;
  title(sprintf('%s SSIM (slice)=%.3f',titles{k},ssim_idx(k)));

  subplot(2,2,(k-1)*2+2);
  Sk = interp3(xs,ys,zs,ssim_map{k}, Xdef1(:,1),Xdef1(:,2),Xdef1(:,3));
  scatter3(Xdef1(:,1),Xdef1(:,2),Xdef1(:,3),20,Sk,'filled');
  axis equal off; view(3);
  colormap(gca,jet); colorbar;
  title(sprintf('%s SSIM at nodes',titles{k}));
end
