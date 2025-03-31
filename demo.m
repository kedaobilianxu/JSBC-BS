clear; close all; clc;

addpath('./funs');
addpath('./Data/HSI-LiDAR-Trento');



%%

%dataType = 'Trento';   
dataType = 'MUUFL'; 
%dataType = 'Houston';  



%  M = [12];
%  L = [0.008];



M = [33];
L = [0.003];


%   M = [30];
%  L = [0.01];


[data3D, lidar2D, gt, ind, c] = loadHSI(dataType);
[nr,nc,~] = size(data3D);



for M_i = 1:size(M,2)
    for L_i = 1:size(L,2)
            tic;
            [y_pred,Z] = JSBC_BS(data3D, lidar2D, gt, ind, 0.7, M(M_i),c,L(L_i));
            running_time = toc;
            [result,PA, UA, AA, OA, Kappa] = HSI_ClusteringMeasure(gt(ind),y_pred(ind));
            results = [M(M_i), L(L_i),AA, OA, Kappa,running_time];
    end
end






resall = y_pred;
res = resall(ind);
new_res = bestMap(gt(ind),y_pred(ind));
resallmap = zeros(nr*nc,1);
for i = 1:c
    temp = new_res(find(res==i));
    if ~isempty(temp)
        resallmap(find(resall==i)) = max(temp);
    else
        resallmap(find(resall==i)) = setdiff(gt(ind),new_res);
    end
end
%map = mat2rgb_Trento(gt,nr,nc); 
map = mat2rgb_Trento(resallmap,nr,nc); 
%map = mat2rgb_MUUFL(resallmap,nr,nc); 
%map = mat2rgb_Houston(resallmap,nr,nc);  % ground truth 
imshow(map,'border','tight','initialmagnification','fit')
set(gcf,'Position',[0,0,1*nr,1*nc]);
axis off

