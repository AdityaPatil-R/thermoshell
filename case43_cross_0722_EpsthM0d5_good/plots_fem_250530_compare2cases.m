clear variables
close all

%% Load both datasets
data_ng = load('output_deps_thermal_NoG.mat');         % no-gravity run
data_g  = load('output_deps_thermal_WithGravity.mat'); % with-gravity run

%% Unpack sizes & fixed nodes
NP           = data_ng.NP_total;
Efull        = data_ng.ConnectivityMatrix_line;   % assume connectivity same
hinge_quads  = data_ng.HingeQuads_order;
fixedNodes   = data_ng.fixedNodes + 1;             % convert 0→1 based

%% Trim histories to actual steps
t_ng = numel(data_ng.time_log);
t_g  = numel(data_g.time_log);
r_ng = size(data_ng.Q_history,1);
r_g  = size(data_g.Q_history,1);

% Results with No gravity
Q1      = data_ng.Q_history(r_ng,:);
strain1 = data_ng.strain_history(r_ng,:);
theta1  = data_ng.theta_history(r_ng,:);

% Results with Gravity
Q2      = data_g.Q_history(r_g,:);
strain2 = data_g.strain_history(r_g,:);
theta2  = data_g.theta_history(r_g,:);
time1   = data_ng.time_log(end);
time2   = data_g.time_log(end);

%% reshape deformed coords
Edges  = Efull(:,2:3);
if any(Edges(:)==0), Edges = Edges + 1; end
X0cols = data_ng.X0_4columns;

q1    = Q1;
q2    = Q2;
Xdef1 = reshape(q1,3,NP)';
Xdef2 = reshape(q2,3,NP)';


%% Compute per-node Euclidean differences
nodeDiffs = Xdef2 - Xdef1;                         % NP×3
distances = sqrt(sum(nodeDiffs.^2, 2));            % NP×1

% Summary statistics
meanDiff = mean(distances);
maxDiff  = max(distances);
rmsDiff  = sqrt(mean(distances.^2));

fprintf('Mean disp distance: %g\n', meanDiff);
fprintf('Max  disp distance: %g\n', maxDiff);
fprintf('RMS  disp distance: %g\n', rmsDiff);

% Optional: visualize per-node difference
figure;
scatter3(Xdef1(:,1), Xdef1(:,2), Xdef1(:,3), 20, distances, 'filled');
colorbar;  axis equal;
xlabel('X'); ylabel('Y'); zlabel('Z');
title('Nodal disp difference distance');


%% common reference
X_ref = X0cols(:,2:4);

%% Figure 1: reference + overlaid deformed
figure(1), clf, hold on
[Xr,Yr,Zr] = buildSegments(X_ref, Edges);
line(Xr, Yr, Zr, ...
     'Color','k', 'LineStyle','-', 'LineWidth',1);
[Xe1,Ye1,Ze1] = buildSegments(Xdef1, Edges);
line(Xe1, Ye1, Ze1, ...
     'Color','r', 'LineStyle','-', 'LineWidth',1.5);
[Xe2,Ye2,Ze2] = buildSegments(Xdef2, Edges);
line(Xe2, Ye2, Ze2, ...
     'Color','b', 'LineStyle','--','LineWidth',1.5);
plot3( Xdef1(fixedNodes,1), Xdef1(fixedNodes,2), Xdef1(fixedNodes,3), ...
       'go','MarkerFaceColor','g','MarkerSize',8);
plot3( Xdef2(fixedNodes,1), Xdef2(fixedNodes,2), Xdef2(fixedNodes,3), ...
       'mo','MarkerFaceColor','m','MarkerSize',8);
xlabel('x'); ylabel('y'); zlabel('z');
title('Ref & Def (noG:red, withG:blue)');
view(3); grid on; rotate3d on; box on;
hold off

%% Figure 2: strain overlay
cmap = cool(256);
str1 = strain1;
str2 = strain2;
str_min = min([str1,str2]);
str_max = max([str1,str2]);

idx1 = round(1 + (str1-str_min)/(str_max-str_min)*(size(cmap,1)-1));
idx1 = max(1,min(size(cmap,1),idx1));
col1 = cmap(idx1,:);

idx2 = round(1 + (str2-str_min)/(str_max-str_min)*(size(cmap,1)-1));
idx2 = max(1,min(size(cmap,1),idx2));
col2 = cmap(idx2,:);

figure(2), clf, hold on
for e=1:size(Edges,1)
  line(Xe1(:,e), Ye1(:,e), Ze1(:,e), ...
       'Color',col1(e,:), 'LineStyle','-', 'LineWidth',2);
  line(Xe2(:,e), Ye2(:,e), Ze2(:,e), ...
       'Color',col2(e,:), 'LineStyle','--','LineWidth',2);
end
colormap(cmap)
if str_max>str_min, caxis([str_min,str_max]), end
cb = colorbar; cb.Label.String='Axial strain \epsilon';
title('Strain (solid=noG, dashed=withG)');
view(3); grid on; rotate3d on; box on;
hold off

%% Figure 3: dihedral overlay
Hedges = hinge_quads(:,2:3);
if any(Hedges(:)==0), Hedges = Hedges + 1; end
[Xh1,Yh1,Zh1] = buildSegments(Xdef1, Hedges);
[Xh2,Yh2,Zh2] = buildSegments(Xdef2, Hedges);
th_min = min([theta1,theta2]);
th_max = max([theta1,theta2]);

idxh1 = round(1 + (theta1-th_min)/(th_max-th_min)*(size(cmap,1)-1));
idxh1 = max(1,min(size(cmap,1),idxh1));
colh1 = cmap(idxh1,:);

idxh2 = round(1 + (theta2-th_min)/(th_max-th_min)*(size(cmap,1)-1));
idxh2 = max(1,min(size(cmap,1),idxh2));
colh2 = cmap(idxh2,:);

figure(3), clf, hold on
for h=1:size(Hedges,1)
  line(Xh1(:,h), Yh1(:,h), Zh1(:,h), ...
       'Color',colh1(h,:), 'LineStyle','-', 'LineWidth',3);
  line(Xh2(:,h), Yh2(:,h), Zh2(:,h), ...
       'Color',colh2(h,:), 'LineStyle','--','LineWidth',3);
end
colormap(cmap)
if th_max>th_min, caxis([th_min,th_max]), end
cb2 = colorbar; cb2.Label.String='\theta (rad)';
title('Dihedral (solid=noG, dashed=withG)');
view(3); grid on; rotate3d on; box on;
hold off

%% Figure 4: k_s overlay
ks = data_ng.ks_array;
ks_min = min(ks); ks_max = max(ks);
idxs = round(1 + (ks-ks_min)/(ks_max-ks_min)*(size(cmap,1)-1));
idxs = max(1,min(size(cmap,1),idxs));
cols = cmap(idxs,:);

figure(4), clf, hold on
for e=1:size(Edges,1)
  line(Xe1(:,e), Ye1(:,e), Ze1(:,e), ...
       'Color',cols(e,:), 'LineStyle','-', 'LineWidth',2.5);
  line(Xe2(:,e), Ye2(:,e), Ze2(:,e), ...
       'Color',cols(e,:), 'LineStyle','--','LineWidth',2.5);
end
colormap(cmap), caxis([ks_min,ks_max])
cb4 = colorbar; cb4.Label.String='k_s';
title('k_s (solid & dashed share color)');
view(3); grid on; rotate3d on; box on;
hold off

%% Figure 5: k_b overlay
kb = data_ng.kb_array;
kb_min = min(kb); kb_max = max(kb);
idxb = round(1 + (kb-kb_min)/(kb_max-kb_min)*(size(cmap,1)-1));
idxb = max(1,min(size(cmap,1),idxb));
colb = cmap(idxb,:);

[Xh5,Yh5,Zh5] = buildSegments(Xdef1, Hedges);
figure(5), clf, hold on
for h=1:size(Hedges,1)
  line(Xh5(:,h), Yh5(:,h), Zh5(:,h), ...
       'Color',colb(h,:), 'LineStyle','-', 'LineWidth',3);
  line(Xh5(:,h), Yh5(:,h), Zh5(:,h), ...
       'Color',colb(h,:), 'LineStyle','--','LineWidth',3);
end
colormap(cmap), caxis([kb_min,kb_max])
cb5 = colorbar; cb5.Label.String='k_b';
title('k_b (solid=noG, dashed=withG)');
view(3); grid on; rotate3d on; box on; zlim([-0.01,0.01]);
hold off


%% helper at the end of the script
function [Xseg,Yseg,Zseg] = buildSegments(Xnodes, Edges)
    % Xnodes: Nnodes×3, Edges: Nedges×2
    Xseg = [Xnodes(Edges(:,1),1)'; Xnodes(Edges(:,2),1)'];
    Yseg = [Xnodes(Edges(:,1),2)'; Xnodes(Edges(:,2),2)'];
    Zseg = [Xnodes(Edges(:,1),3)'; Xnodes(Edges(:,2),3)'];
end
