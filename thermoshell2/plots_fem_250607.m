clear variables
close all

iAxis = 1;   % 0 = no labels, no axis, etc.
iView = 2;   % 1 = x-y, 2 = x-z, 3 = y-z

%% Load
data = load('output_deps_thermal_NoG.mat');



Q_hist        = data.Q_history;           % (nSteps+1)×(3·NP)
X0cols        = data.X0_4columns;         % NP×4: [nid,x,y,z]
Efull         = data.ConnectivityMatrix_line;  % Nedges×3: [eid,n0,n1]
strainH       = data.strain_history;
thetaH        = data.theta_history;       % (nSteps+1)×Nhinges
hinge_quads   = data.HingeQuads_order;    % Nhinges×5: [hid,n0,n1,oppA,oppB]
NP            = data.NP_total;
[nSteps,~]    = size(Q_hist);
ks            = data.ks_array;
kb            = data.kb_array;
q0            = data.q_old;
time_log      = data.time_log;
fixedNodes    = data.fixedNodes;

% fixedNodes = [0 1 2 3 4 5 6 7];
fixedNodes    = fixedNodes +1;            % Convert from 0-based to 1-based numbering

%% edge list
Edges = Efull(:,2:3);
if any(Edges(:)==0), Edges = Edges + 1; end
Nedges = size(Edges,1);

%% Find first all-zero row
zeroRows     = all(Q_hist == 0, 2);   
firstZero    = find(zeroRows, 1);

if isempty(firstZero)
    % no all‑zero rows ⇒ use the full history
    t = nSteps;
else
    % step just before the first all‑zero row
    t = firstZero - 1;
end

%% Final deformed state

% q       = Q_hist(t,:);   
q       = q0(1,:);   

X_def   = reshape(q,3,NP)';            
strains = strainH(t,:);

%% Reference coords
X_ref = X0cols(:,2:4);

%% Figure 1: reference (black) + deformed (red)
figure(1), clf, hold on
[Xr,Yr,Zr] = buildSegments(X_ref, Edges);
line(Xr,Yr,Zr,'Color','k','LineWidth',1);
[Xe,Ye,Ze] = buildSegments(X_def, Edges);
line(Xe,Ye,Ze,'Color','r','LineWidth',1.5);

xlabel('x'); ylabel('y'); zlabel('z');
title(sprintf('Reference & Deformed (t=%d/%d)',t,nSteps));
view(3); grid on; rotate3d on; box on;

% X_def is NP×3: [x y z] per node
plot3( X_def(fixedNodes,1), ...
       X_def(fixedNodes,2), ...
       X_def(fixedNodes,3), ...
       'ko', ...                % black circles
       'MarkerSize',8, ...
       'MarkerFaceColor','g' ); % filled

%% Figure 2: deformed  strain
cmap    = cool(256);

str_min = min(strains);
str_max = max(strains);
if str_max~=str_min
  idxs = round(1 + (strains-str_min)/(str_max-str_min)*(size(cmap,1)-1));
  idxs = max(1,min(size(cmap,1),idxs));
else
  idxs = ones(1,Nedges);
end
colors = cmap(idxs,:);

figure(2), clf, hold on
for e = 1:Nedges
  line( Xe(:,e), Ye(:,e), Ze(:,e), ...
        'Color', colors(e,:), 'LineWidth',2.5 );
end

colormap(cmap)
caxis([str_min str_max])
hcb = colorbar;
hcb.Label.String = 'Axial strain \epsilon';


switch iView
    case 1 % x-y
        view(0, 90); % top-down (z pointing out)
    case 2 % x-z
        view(0, 0); % x-z plane (y pointing out)
    case 3 % y-z
        view(90, 0); % y-z plane (x pointing out)
    otherwise
        view(3); % default 3D view
end

if iAxis
    xlabel('x');
    ylabel('y');
    zlabel('z');
    title(sprintf('contour of \\epsilon (t=%d/%d)', t, nSteps));
    grid on; rotate3d on; box on;
    axis equal;
else
    axis off; % hides axis lines, ticks, labels, etc.
    % Optionally also hide colorbar label:
    hcb.Label.String = '';
    axis equal;
end
hold off

%% Figure 3: deformed hinges colored by dihedral θ
% extract dihedral angles at this timestep
thetas = thetaH(t,:);           % 1×Nhinges
Nhinges = numel(thetas);

% hinge‐edge node pairs
Hedges = hinge_quads(:,2:3);
if any(Hedges(:)==0), Hedges = Hedges + 1; end

% build 3D segments for each hinge
[Xh,Yh,Zh] = buildSegments(X_def, Hedges);

cmap2 = cmap;  % you can also use redblue, coolwarm, etc.
th_min = min(thetas);
th_max = max(thetas);
if th_max~=th_min
  idxs2 = round(1 + (thetas-th_min)/(th_max-th_min)*(size(cmap2,1)-1));
  idxs2 = max(1,min(size(cmap2,1),idxs2));
else
  idxs2 = ones(1,Nhinges);
end
colors2 = cmap2(idxs2,:);

figure(3), clf, hold on
for h = 1:Nhinges
  line( Xh(:,h), Yh(:,h), Zh(:,h), ...
        'Color', colors2(h,:), 'LineWidth',3 );
end

colormap(cmap2)
caxis([th_min th_max])
hcb2 = colorbar;
hcb2.Label.String = 'Dihedral angle \theta (rad)';

switch iView
    case 1 % x-y
        view(0, 90);
    case 2 % x-z
        view(0, 0);
    case 3 % y-z
        view(90, 0);
    otherwise
        view(3);
end

if iAxis
    xlabel('x');
    ylabel('y');
    zlabel('z');
    title(sprintf('Contour of \\theta (t=%d/%d)', t, nSteps));
    grid on; rotate3d on; box on;
    axis equal;
else
    axis off;
    hcb2.Label.String = '';
    axis equal;
end
hold off;

%% Figure 4: ks
cmap    = bone(256);

% str_min = min(ks);
% str_max = max(ks);

str_min = 100;
str_max = 4400;
unique(ks)

if str_max~=str_min
  idxs = round(1 + (ks-str_min)/(str_max-str_min)*(size(cmap,1)-1));
  idxs = max(1,min(size(cmap,1),idxs));
else
  idxs = ones(1,Nedges);
end
colors = cmap(idxs,:);

figure(4), clf, hold on
for e = 1:Nedges
  line( Xe(:,e), Ye(:,e), Ze(:,e), ...
        'Color', colors(e,:), 'LineWidth',2.5 );
end

% --- fixed‐node markers in black ---
% X_def is NP×3: [x y z] per node
plot3( X_def(fixedNodes,1), ...
       X_def(fixedNodes,2), ...
       X_def(fixedNodes,3), ...
       'ko', ...                % black circles
       'MarkerSize',8, ...
       'MarkerFaceColor','k' ); % filled

colormap(cmap)
caxis([str_min str_max])
hcb = colorbar;
hcb.Label.String = 'axial sfittness k_s';

switch iView
    case 1 % x-y
        view(0, 90);
    case 2 % x-z
        view(0, 0);
    case 3 % y-z
        view(90, 0);
    otherwise
        view(3);
end

if iAxis
    xlabel('x');
    ylabel('y');
    zlabel('z');
    title(sprintf('Contour of k_s (t=%d/%d)', t, nSteps));
    grid on; rotate3d on; box on;
    axis equal;
else
    axis off;
    hcb.Label.String = '';
    axis equal;
end

hold off;


%% Figure 24: ks ref config
cmap    = bone(256);

% str_min = min(ks);
% str_max = max(ks);

str_min = 100;
str_max = 4400;
unique(ks)

if str_max~=str_min
  idxs = round(1 + (ks-str_min)/(str_max-str_min)*(size(cmap,1)-1));
  idxs = max(1,min(size(cmap,1),idxs));
else
  idxs = ones(1,Nedges);
end
colors = cmap(idxs,:);

figure(24), clf, hold on
for e = 1:Nedges
  line( Xr(:,e), Yr(:,e), Zr(:,e), ...
        'Color', colors(e,:), 'LineWidth',2.5 );
end

% --- fixed‐node markers in black ---
% X_def is NP×3: [x y z] per node
plot3( X_ref(fixedNodes,1), ...
       X_ref(fixedNodes,2), ...
       X_ref(fixedNodes,3), ...
       'ko', ...                % black circles
       'MarkerSize',8, ...
       'MarkerFaceColor','k' ); % filled

colormap(cmap)
caxis([str_min str_max])
hcb = colorbar;
hcb.Label.String = 'axial sfittness k_s';

switch iView
    case 1 % x-y
        view(0, 90);
    case 2 % x-z
        view(0, 0);
    case 3 % y-z
        view(90, 0);
    otherwise
        view(3);
end

if iAxis
    xlabel('x');
    ylabel('y');
    zlabel('z');
    title(sprintf('Contour of k_s (t=%d/%d)', t, nSteps));
    grid on; rotate3d on; box on;
    axis equal;
else
    axis off;
    hcb.Label.String = '';
    axis equal;
end

hold off;


%% Figure 5: deformed hinges colored by bending stiffness k_b
% hinge‐edge node pairs (same as for θ)
Hedges = hinge_quads(:,2:3);
if any(Hedges(:)==0), Hedges = Hedges + 1; end

% build 3D segments for each hinge
[Xh5,Yh5,Zh5] = buildSegments(X_def, Hedges);

% per‐hinge kb values
kb_vals  = kb;            % (1×Nhinges)
Nhinges5 = numel(kb_vals);

% choose colormap
cmap5 = cool(256);
kb_min = min(kb_vals);
kb_max = max(kb_vals);

if kb_max ~= kb_min
  idxs5 = round(1 + (kb_vals - kb_min)/(kb_max - kb_min)*(size(cmap5,1)-1));
  idxs5 = max(1, min(size(cmap5,1), idxs5));
else
  idxs5 = ones(1, Nhinges5);
end
colors5 = cmap5(idxs5, :);

figure(5), clf, hold on
for h = 1:Nhinges5
  line( Xh5(:,h), Yh5(:,h), Zh5(:,h), ...
        'Color', colors5(h,:), 'LineWidth', 3 );
end

colormap(cmap5)
caxis([kb_min kb_max])
hcb5 = colorbar;
hcb5.Label.String = 'Bending stiffness k_b';

switch iView
    case 1 % x-y
        view(0, 90);
    case 2 % x-z
        view(0, 0);
    case 3 % y-z
        view(90, 0);
    otherwise
        view(3);
end

if iAxis
    xlabel('x');
    ylabel('y');
    zlabel('z');
    title(sprintf('Contour of k_b (t=%d/%d)', t, nSteps));
    grid on; rotate3d on; box on;
    zlim([-0.01 0.01]);
    axis equal;
else
    axis off;
    hcb5.Label.String = '';
    zlim([-0.01 0.01]);
    axis equal;
end
hold off;

%% 8) Export deformed cylinders to STL

% Use the deformed DOF positions
V = X_def;  

% Rebuild the edge list (1‑based indexing)
Edges = Efull(:,2:3);
if any(Edges(:)==0)
    Edges = Edges + 1;
end

% Compute mean edge length for cylinder radius
edgeVecs    = V(Edges(:,1),:) - V(Edges(:,2),:);
edgeLengths = sqrt(sum(edgeVecs.^2,2));
r           = 0.1 * mean(edgeLengths);  % 1% of mean edge length

nFacets = 12;      % number of sides around each cylinder

V_beam = [];
F_beam = [];

for e = 1:size(Edges,1)
    % Endpoints of this beam
    i1 = Edges(e,1);  i2 = Edges(e,2);
    p1 = V(i1,:);     p2 = V(i2,:);
    d  = p2 - p1;     L = norm(d);
    
    % Build a unit‐length cylinder along Z
    [Xc,Yc,Zc] = cylinder(r, nFacets);
    Zc = Zc * L;
    [Fc,Vc] = surf2patch(Xc, Yc, Zc, 'triangles');  
    
    % Compute rotation matrix R that sends [0 0 1] -> d/L (Rodrigues)
    vZ = [0 0 1];
    u  = d / L;
    k  = cross(vZ,u);
    if norm(k) < 1e-6
        % nearly parallel or antiparallel
        if dot(vZ,u) > 0
            R = eye(3);
        else
            R = diag([1 1 -1]);
        end
    else
        k = k / norm(k);
        ang = acos(dot(vZ,u));
        K = [    0   -k(3)  k(2);
               k(3)    0   -k(1);
              -k(2)  k(1)    0   ];
        R = eye(3) + sin(ang)*K + (1-cos(ang))*(K*K);
    end
    
    % Rotate & translate the cylinder vertices
    Vc_rot   = (R * Vc')';
    Vc_trans = Vc_rot + p1;
    
    % Accumulate into one big mesh
    idx0      = size(V_beam,1);
    V_beam    = [V_beam;   Vc_trans];
    F_beam    = [F_beam;   Fc + idx0];
end

% Create a MATLAB triangulation and write to STL
TR_beam = triangulation(F_beam, V_beam);
stlwrite(TR_beam, 'mesh_deformed.stl');

fprintf('Wrote mesh_deformed.stl with %d vertices and %d triangles\n', ...
        size(V_beam,1), size(F_beam,1));





fun_SaveSTL(X_ref, Efull, 'mesh_ref.stl');




figure(6)
bar(time_log)
xlabel('steps'); ylabel('time');


function fun_SaveSTL(X_def, Efull, filename)
%WRITEDEFORMEDSTL Creates a cylinder-based STL mesh of a deformed truss/beam structure.
%
%   writeDeformedSTL(X_def, Efull, filename)
%
%   INPUTS:
%       X_def   : NP x 3 array of deformed nodal positions
%       Efull   : NE x 3 array of element connectivity (columns 2 and 3 are nodes)
%       filename: String, name of the output STL file
%
%   OUTPUTS:
%       None (writes an STL file with name 'filename')
%
%   Example:
%       writeDeformedSTL(X_def, Efull, 'mesh_deformed.stl');

    % Use the deformed DOF positions
    V = X_def;  

    % Rebuild the edge list (1‑based indexing)
    Edges = Efull(:,2:3);
    if any(Edges(:)==0)
        Edges = Edges + 1;
    end

    % Compute mean edge length for cylinder radius
    edgeVecs    = V(Edges(:,1),:) - V(Edges(:,2),:);
    edgeLengths = sqrt(sum(edgeVecs.^2,2));
    r           = 0.1 * mean(edgeLengths);  % 1% of mean edge length

    nFacets = 12;      % number of sides around each cylinder

    V_beam = [];
    F_beam = [];

    for e = 1:size(Edges,1)
        % Endpoints of this beam
        i1 = Edges(e,1);  i2 = Edges(e,2);
        p1 = V(i1,:);     p2 = V(i2,:);
        d  = p2 - p1;     L = norm(d);

        % Build a unit-length cylinder along Z
        [Xc,Yc,Zc] = cylinder(r, nFacets);
        Zc = Zc * L;
        [Fc,Vc] = surf2patch(Xc, Yc, Zc, 'triangles');  

        % Compute rotation matrix R that sends [0 0 1] -> d/L (Rodrigues)
        vZ = [0 0 1];
        u  = d / L;
        k  = cross(vZ,u);
        if norm(k) < 1e-6
            % nearly parallel or antiparallel
            if dot(vZ,u) > 0
                R = eye(3);
            else
                R = diag([1 1 -1]);
            end
        else
            k = k / norm(k);
            ang = acos(dot(vZ,u));
            K = [    0   -k(3)  k(2);
                   k(3)    0   -k(1);
                  -k(2)  k(1)    0   ];
            R = eye(3) + sin(ang)*K + (1-cos(ang))*(K*K);
        end

        % Rotate & translate the cylinder vertices
        Vc_rot   = (R * Vc')';
        Vc_trans = Vc_rot + p1;

        % Accumulate into one big mesh
        idx0      = size(V_beam,1);
        V_beam    = [V_beam;   Vc_trans];
        F_beam    = [F_beam;   Fc + idx0];
    end

    % Create a MATLAB triangulation and write to STL
    TR_beam = triangulation(F_beam, V_beam);
    stlwrite(TR_beam, filename);

    fprintf('Wrote %s with %d vertices and %d triangles\n', ...
            filename, size(V_beam,1), size(F_beam,1));
end


%% function: build 3×Nedges arrays for coordinates
function [Xseg,Yseg,Zseg] = buildSegments(Xnodes, Edges)
    % Xnodes: Nnodes×3, Edges: Nedges×2 (1-based)
    Xseg = [ Xnodes(Edges(:,1),1)'; Xnodes(Edges(:,2),1)' ];
    Yseg = [ Xnodes(Edges(:,1),2)'; Xnodes(Edges(:,2),2)' ];
    Zseg = [ Xnodes(Edges(:,1),3)'; Xnodes(Edges(:,2),3)' ];
end
