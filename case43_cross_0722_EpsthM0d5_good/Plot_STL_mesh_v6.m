
% Reads an STL file, aligns/rotates it, prints mesh statistics,
% and separates outer/inner surfaces by face normals.

%% 1) Load & preprocess
TR = stlread('cross0613_4.stl');
F  = TR.ConnectivityList;
V  = TR.Points;

% Shift Z so min(Z)=0
Zmin = min(V(:,3));
V(:,3) = V(:,3) - Zmin;

% Rotate about X by +90° so the "front" lifts up
alpha = 90;  % degrees
Rx    = [1       0            0;
         0   cosd(alpha)  -sind(alpha);
         0   sind(alpha)   cosd(alpha)];
V = (Rx * V')';

%% 2) Full wireframe preview
f = figure(100);
set(f, 'Color', 'w', 'Renderer', 'opengl');
trimesh(F, V(:,1), V(:,2), V(:,3), ...
        'EdgeColor',[0.7 0.7 0.7], 'FaceColor','none', 'LineWidth',0.3);
daspect([1 1 1]); axis tight; view(45,30);
camlight headlight; material dull; rotate3d on;
xlabel('X'); ylabel('Y'); zlabel('Z');

%% 3) Mesh statistics
numNodes     = size(V,1);
numTriangles = size(F,1);
E = [F(:,[1 2]); F(:,[2 3]); F(:,[3 1])]; E = sort(E,2); E = unique(E,'rows');
numEdges = size(E,1);
edgeVecs    = V(E(:,1),:) - V(E(:,2),:);
edgeLengths = sqrt(sum(edgeVecs.^2,2));
minEdge = min(edgeLengths); avgEdge = mean(edgeLengths); maxEdge = max(edgeLengths);
v1 = V(F(:,2),:) - V(F(:,1),:); v2 = V(F(:,3),:) - V(F(:,1),:);
crossP = cross(v1,v2,2);
triAreas = 0.5 * sqrt(sum(crossP.^2,2));
minArea = min(triAreas); avgArea = mean(triAreas); maxArea = max(triAreas);
bbMin = min(V,[],1); bbMax = max(V,[],1);
fprintf('Mesh statistics:\nNodes: %d, Edges: %d, Triangles: %d\n', numNodes, numEdges, numTriangles);
fprintf('Edge [min,avg,max]: %g, %g, %g\n', minEdge, avgEdge, maxEdge);
fprintf('Area [min,avg,max]: %g, %g, %g\n', minArea, avgArea, maxArea);
fprintf('BBox X:[%g,%g], Y:[%g,%g], Z:[%g,%g]\n', bbMin(1),bbMax(1), bbMin(2),bbMax(2), bbMin(3),bbMax(3));

%% 4) Separate outer/inner by face normals
fn    = crossP;
norms = sqrt(sum(fn.^2,2));
normals = fn ./ norms;
zcomp = normals(:,3);
idx   = kmeans(zcomp,2,'Replicates',5);
mu    = accumarray(idx, zcomp, [], @mean);
outerCluster = find(mu == max(mu));
maskB = (idx == outerCluster);
F_outer = F(maskB,:);
F_inner = F(~maskB,:);

%% 5) Combined outer/inner plot
f = figure(101);
set(f, 'Name','Separated Layers','Color','w','Renderer','opengl');
hold on
trimesh(F_outer, V(:,1),V(:,2),V(:,3), 'EdgeColor','b','FaceColor','none','LineWidth',0.5);
trimesh(F_inner, V(:,1),V(:,2),V(:,3), 'EdgeColor','r','FaceColor','none','LineWidth',0.3);
hold off
daspect([1 1 1]); axis tight; view(45,30); rotate3d on;
xlabel('X'); ylabel('Y'); zlabel('Z');

%% 6) Outer surface only
f = figure(102);
set(f, 'Name','Outer Surface Only','Color','w','Renderer','opengl');
trimesh(F_outer, V(:,1),V(:,2),V(:,3), 'EdgeColor','b','FaceColor','none','LineWidth',0.5);
daspect([1 1 1]); axis tight; view(45,30); rotate3d on;
xlabel('X'); ylabel('Y'); zlabel('Z');
title('Outer Surface Only');

%% 7) Inner surface only
f = figure(103);
set(f, 'Name','Inner Surface Only','Color','w','Renderer','opengl');
trimesh(F_inner, V(:,1),V(:,2),V(:,3), 'EdgeColor','r','FaceColor','none','LineWidth',0.3);
daspect([1 1 1]); axis tight; view(45,30); rotate3d on;
xlabel('X'); ylabel('Y'); zlabel('Z');
title('Inner Surface Only');

%% 4’) Separate outer/inner by dot(normal, centroid‑ray)
% 4.1) compute mesh center
meshCenter     = mean(V,1);

% 4.2) compute face centroids
faceCentroids  = (V(F(:,1),:) + V(F(:,2),:) + V(F(:,3),:)) / 3;

% 4.3) build rays from center to centroid
rays           = faceCentroids - meshCenter;    % N×3

% 4.4) normalize your face normals (if not already unit length)
fn             = crossP;                        % un‑normalized from before
norms          = sqrt(sum(fn.^2,2));
normals        = fn ./ norms;                   % N×3

% 4.5) dot product
dp             = sum(normals .* rays, 2);       % N×1

% 4.6) mask
maskOuter2     = dp > 0;
F_outer2       = F(maskOuter2,:);
F_inner2       = F(~maskOuter2,:);

%% 5’) Plot with this new split
f = figure(104);
set(f, 'Color', 'w', 'Renderer', 'opengl');
hold on
trimesh(F_outer2, V(:,1),V(:,2),V(:,3), 'EdgeColor','b','FaceColor','none','LineWidth',0.5);
trimesh(F_inner2, V(:,1),V(:,2),V(:,3), 'EdgeColor','r','FaceColor','none','LineWidth',0.3);
hold off
daspect([1 1 1]); axis tight; view(45,30); rotate3d on;
xlabel('X'); ylabel('Y'); zlabel('Z');

%% 6) Outer surface only
f = figure(105);
set(f, 'Name','Outer Surface Only','Color','w','Renderer','opengl');
trimesh(F_outer2, V(:,1),V(:,2),V(:,3), 'EdgeColor','b','FaceColor','none','LineWidth',0.5);
daspect([1 1 1]); axis tight; view(45,30); rotate3d on;
xlabel('X'); ylabel('Y'); zlabel('Z');
title('Outer Surface Only');

%% 7) Inner surface only
f = figure(106);
set(f, 'Name','Inner Surface Only','Color','w','Renderer','opengl');
trimesh(F_inner2, V(:,1),V(:,2),V(:,3), 'EdgeColor','r','FaceColor','none','LineWidth',0.3);
daspect([1 1 1]); axis tight; view(45,30); rotate3d on;
xlabel('X'); ylabel('Y'); zlabel('Z');
title('Inner Surface Only');

%% 8) Export separated layers as new STL files
% Build cleaned vertex & face lists for writing
% Outer
uv_outer = unique(F_outer2(:));
V_outer = V(uv_outer, :);
map_outer = zeros(size(V,1),1);
map_outer(uv_outer) = 1:numel(uv_outer);
F_outer_new = map_outer(F_outer2);
% Inner
uv_inner = unique(F_inner2(:));
V_inner = V(uv_inner, :);
map_inner = zeros(size(V,1),1);
map_inner(uv_inner) = 1:numel(uv_inner);
F_inner_new = map_inner(F_inner2);

% Create triangulation objects
TR_outer2 = triangulation(F_outer_new, V_outer);
TR_inner2 = triangulation(F_inner_new, V_inner);

% Write to STL (R2019b+)
stlwrite(TR_outer2, 'outer_surface_by_centroid_ray.stl');
stlwrite(TR_inner2, 'inner_surface_by_centroid_ray.stl');

fprintf('Exported STL files: outer_surface_by_centroid_ray.stl, inner_surface_by_centroid_ray.stl\n');

