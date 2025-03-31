function [y_pred,Z] = JSBC_BS(data3D, lidar2D, gt, ind, beta, M, nc,L)
%% prod. by caozhe@mail.nwpu.edu.cn
% Input:
%       - data3D: 3D HSI data
%       - lidar2D: 2D Lidar data
%       - nc: cluster number
% Output:
%       - y_pred: clustering labels.
% seed = 2;
% rng(seed);
[~,~,dim_hid] = size(data3D);
[~,~,dim_lidar] = size(lidar2D);
data3D = data3D./max(data3D(:));
lidar2D = lidar2D./max(lidar2D(:));

%fprintf('Superpixels number : %d\n', M);
Mdata = data3D;
Mdata(:,:,size(data3D,3)+1) = lidar2D;
tic;
[splabels] = cubseg(Mdata, M);% ERS segmentation.
%[splabels] = SLICseg(Mdata, M);% SLIC segmentation.
%[splabels] = LSCseg(Mdata, M);% LSC segmentation.
%[splabels] = SNICseg(Mdata, M);% Kmeans segmentation.
%[splabels] = Watershedseg(Mdata, M);% Kmeans segmentation.
segtime = toc;
disp(segtime);
[z_hid,A_hid,X_hid] = Generate_anchor_graph(data3D,M,splabels,dim_hid);
[z_lidar,A_lidar,X_lidar] = Generate_anchor_graph(lidar2D,M,splabels,dim_lidar);
z_hid = double(z_hid);
A_hid = double(A_hid);
X_hid = double(X_hid);
z_lidar = double(z_lidar);
A_lidar = double(A_lidar);
X_lidar = double(X_lidar);
objective_ori = 0;
delta_obj = 0;
N = size(z_hid,1);
clear data3D lidar2D Mdata
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
nV = 2;
sX = [N, M, nV];
theta = diag(1/dim_hid .* ones(1, dim_hid));
Z{1} = z_hid;
Z{2} = z_lidar;
for k = 1:nV
    J{k} = zeros(N, M);
    Q{k} = zeros(N, M);
    XA{k} = zeros(N, M);
end
eta = 1.3; mu = 10e-5; rho = 10e-5; max_mu = 10e12; max_rho = 10e12;
Isconverg = 0; maxIter = 10; iter = 1;
betaf = ones(nV, 1); 
maxkappa = 0;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
while(Isconverg == 0)  
    %%%update theta matrix
    m_k = zeros(1,dim_hid);
    for k = 1:dim_hid
        m_k(k) = sum(sum((X_hid(k,:)' - A_hid(k,:)).^2 .* Z{1}));
    end
    sum_mk = sum(1./m_k(:));
    theta = diag(1 ./ (m_k(1:dim_hid) * sum_mk));
    
    %%%undate tensor J
    for v =1:nV
        XA{v} = Z{v} - Q{v}./rho;
    end
    M_tensor = cat(3, XA{:,:});
    M_vector = M_tensor(:);
    [myj, obj] = wshrinkObj_weight_lp(M_vector, beta*betaf./rho, sX, 0, 3, 1, L);
    J_tensor = reshape(myj, sX);
    for k=1:nV
        J{k} = J_tensor(:,:,k);
    end
    clear J_tensor Q_tensor M_vector

    %%%update tensor Z

    theta_diag = diag(theta);


    X_theta = theta_diag .* X_hid;
    A_theta = theta_diag .* A_hid;


    XA1 = zeros(N, M);
    XA2 = zeros(N, M);

    for j = 1:M
        m_1 = X_theta - A_theta(:, j);
        m_2 = X_lidar - A_lidar(:, j);

        XA1(:, j) = sum(m_1.^2, 1);
        XA2(:, j) = sum(m_2.^2, 1);
    end

    XA{1} = XA1;
    XA{2} = XA2;

    for v =1:nV
        P{v} = J{v} + Q{v}./rho;
        R{v} = P{v}.*rho - XA{v};
    end
    for v = 1:nV
        for i = 1:M
            MM = R{v}(:,i);
            Z{v}(:,i) = EProjSimplex_new(MM, 1);
        end
    end

    clear P R MM
    %%%update Q
    for k = 1:nV
        Q{k} = Q{k} + mu .* (Z{k} - J{k});
    end
    mu = min(eta * mu, max_mu);
    rho = min(eta * rho, max_rho);


    objective_value = 0;

    % calculate obj
    X_proj = theta * X_hid;
    A_proj = theta * A_hid;

    for v = 1:nV
        diff_matrix = sum((reshape(X_proj, size(X_proj, 1), 1, size(X_proj, 2)) ...
            - reshape(A_proj, size(A_proj, 1), size(A_proj, 2), 1)).^2, 1);
        diff_matrix = squeeze(diff_matrix);


        Z_real = real(Z{v});


        objective_value = objective_value + sum(diff_matrix(:) .* Z_real(:));
    end

    if iter==1
        objective_ori = objective_value;
    else
        delta_obj = abs(objective_value-objective_ori)

    end

    if iter == 10
        a = 1;
    end
    if iter > 1 && (delta_obj < 1e-5 || iter >= maxIter)
        Isconverg = 1;
    end
    iter = iter + 1;
    Z_final = (Z{1} + Z{2})./2;
    Z_final(isnan(Z_final) | isinf(Z_final)) = 0;
    [u,~,~] = svds(Z_final,nc);
    u = real(u);
    y_pred = kmeans(u,nc,'MaxIter', 1000);





    end


end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [Z,A,X] = Generate_anchor_graph(data,m,splabels,dim)
     if(dim==1)
         X = reshape(data, [], dim);
         X = X';
         t = 5;
 
        A = meanInd(X, splabels(:), ones(size(X,2),m));
        [Z,~,~,~] = initZ(X, A, t);
 
     else


        X = reshape(data, [], dim);
        X = X';
        t = 5;

        A = meanInd(X, splabels(:), ones(size(X,2),m));
        [Z,~,~,~] = initZ(X, A, t);
     end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [num]=pixelNum(data,Tbase)
%% Adaptively determine the number of superpixels
    [M_1,N_1,B]=size(data);
    Y_scale=scaleForSVM(reshape(data,M_1*N_1,B));
    p=1;
    [Y_pca] = pca(Y_scale, p);
    img = im2uint8(mat2gray(reshape(Y_pca', M_1, N_1, p)));
    [m,n] = size(img);
    BW = edge(img,'log');
    ind = find(BW~=0);
    Len = length(ind);
    Ratio = Len/(m*n);
    num = fix(Ratio * Tbase);    
    
end

