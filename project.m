% Eve MacDonald & Matt MacMillan

%% ---Load the data sets---
close all;

data_a = readtable('data/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv');
data_b = readtable('data/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv');
data_c = readtable('data/Friday-WorkingHours-Morning.pcap_ISCX.csv');
data_d = readtable('data/Monday-WorkingHours.pcap_ISCX.csv');
data_e = readtable('data/Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv');
data_f = readtable('data/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv');
data_g = readtable('data/Tuesday-WorkingHours.pcap_ISCX.csv');
data_h = readtable('data/Wednesday-workingHours.pcap_ISCX.csv');

%% ---Data Manipulation---
%DDoS
data_aa = table2array(data_a(:,[1:78]));
data_al = table2array(data_a(:,79));


data_ba = table2array(data_b(:,[1:78]));
data_bl = table2array(data_b(:,79));

data_ca = table2array(data_c(:,[1:78]));
data_cl = table2array(data_c(:,79));

data_da = table2array(data_d(:,[1:78]));
data_dl = table2array(data_d(:,79));

data_ea = table2array(data_e(:,[1:78]));
data_el = table2array(data_e(:,79));

data_fa = table2array(data_f(:,[1:78]));
data_fl = table2array(data_f(:,79));

data_ga = table2array(data_g(:,[1:78]));
data_gl = table2array(data_g(:,79));

data_ha = table2array(data_h(:,[1:78]));
data_hl = table2array(data_h(:,79));

allData_a = [data_aa;data_ba;data_ca;data_da;data_ea;data_fa;data_ga;data_ha];
allData_l = [data_al;data_bl;data_cl;data_dl;data_el;data_fl;data_gl;data_hl];
allData_bl = [data_al;data_bl;data_cl;data_dl;data_el;data_fl;data_gl;data_hl];

allData_t = [data_a;data_b;data_c;data_d;data_e;data_f;data_g;data_h];


for i = 1:length(allData_bl)
    if allData_bl{i} ~= "BENIGN"
        allData_bl{i} = 'MALICIOUS';
    end
end

%% Get Rid of Garbage Data
delete_rows = [];
for i = 1:size(allData_a,1)
    for j = 1:width(allData_a)
        if(isnan(allData_a(i,j))  || allData_a(i,j) == inf)
            delete_rows = [delete_rows i];
        end
    end
end
allData_a(delete_rows,:) = [];
allData_l(delete_rows,:) = [];
allData_bl(delete_rows,:) = [];

allData_t(delete_rows,:) = [];


delete_columns = [];

for i = 1:width(allData_a)
    if (length(unique(allData_a(:,i))) == 1)
        delete_columns = [delete_columns i];
    end
end

allData_a(:,delete_columns) = [];
for i = 1:length(delete_columns)
    allData_t(:,delete_columns(i)) = [];
end




%% ---SFS---
% k = 10;
% c = cvpartition(allData_l,'KFold',k);
% opts = statset('display','iter');
% fun_linear = @(XT,yT,Xt,yt)...
%     (sum(~strcmp(yt,classify(Xt,XT,yT,'linear'))));
% fun_quadratic = @(XT,yT,Xt,yt)...
%     (sum(~strcmp(yt,classify(Xt,XT,yT,'quadratic'))));



% disp('Linear SFS');
% [bestmodel_l,hist_l] = sequentialfs(fun_linear,allData_a,allData_l,'cv',c,'options',opts);
% 
% disp('\nQuadratic SFS');
% [bestmodel_q,hist_q] = sequentialfs(fun_quadratic,allData_a,allData_l,'cv',c,'options',opts);
% 
% [idx,scores] = fscmrmr(allData_t, 'Label');

%% PCA - Binary Case
[coeff_a,score_a,latent_a,tsquared_a,explained_a,mu_a] = pca(allData_a);

figure()
plot(score_a(strcmp(allData_bl,'BENIGN'),1), ...
    score_a(strcmp(allData_bl,'BENIGN'),2),'xb');
hold on
plot(score_a(strcmp(allData_bl,'MALICIOUS'),1), ...
    score_a(strcmp(allData_bl,'MALICIOUS'),2),'xr');
legend('Benign','Malicious');
title('First two dimensions of PCA')
hold off


%% PCA - Multiple 

figure()
plot(score_a(strcmp(allData_l,'Bot'),1), ...
    score_a(strcmp(allData_l,'Bot'),2),'xr');
hold on
plot(score_a(strcmp(allData_l,'DDos'),1), ...
    score_a(strcmp(allData_l,'DDos'),2),'xg');

plot(score_a(strcmp(allData_l,'DoS GoldenEye'),1), ...
    score_a(strcmp(allData_l,'DoS GoldenEye'),2),'xc');

plot(score_a(strcmp(allData_l,'DoS Hulk'),1), ...
    score_a(strcmp(allData_l,'DoS Hulk'),2),'xy');

plot(score_a(strcmp(allData_l,'DoS Slowhttptest'),1), ...
    score_a(strcmp(allData_l,'DoS Slowhttptest'),2),'xm');

plot(score_a(strcmp(allData_l,'DoS slowloris'),1), ...
    score_a(strcmp(allData_l,'DoS slowloris'),2),'xk');

plot(score_a(strcmp(allData_l,'DoS slowloris'),1), ...
    score_a(strcmp(allData_l,'DoS slowloris'),2),'x');

plot(score_a(strcmp(allData_l,'DoS slowloris'),1), ...
    score_a(strcmp(allData_l,'DoS slowloris'),2),'x');

plot(score_a(strcmp(allData_l,'FTP-Patator'),1), ...
    score_a(strcmp(allData_l,'FTP-Patator'),2),'x');

plot(score_a(strcmp(allData_l,'Heartbleed'),1), ...
    score_a(strcmp(allData_l,'Heartbleed'),2),'x');

plot(score_a(strcmp(allData_l,'Infiltration'),1), ...
    score_a(strcmp(allData_l,'Infiltration'),2),'x');

plot(score_a(strcmp(allData_l,'PortScan'),1), ...
    score_a(strcmp(allData_l,'PortScan'),2),'x');

plot(score_a(strcmp(allData_l,'FTP-Patator'),1), ...
    score_a(strcmp(allData_l,'FTP-Patator'),2),'x');

plot(score_a(strcmp(allData_l,'SSH-Patator'),1), ...
    score_a(strcmp(allData_l,'SSH-Patator'),2),'x');

plot(score_a(strcmp(allData_l,'Web Attack � Brute Force'),1), ...
    score_a(strcmp(allData_l,'Web Attack � Brute Force'),2),'x');

plot(score_a(strcmp(allData_l,'Web Attack � Sql Injection'),1), ...
    score_a(strcmp(allData_l,'Web Attack � Sql Injection'),2),'x');

plot(score_a(strcmp(allData_l,'Web Attack � XSS'),1), ...
    score_a(strcmp(allData_l,'Web Attack � XSS'),2),'x');


legend('Bot','DDos','DoS GoldenEye','DoS Hulk','DoS Slowhttptest', ...
    'DoS slowloris','FTP-Patator','Heartbleed','Infiltration','PortScan',...
    'SSH-Patator','Web Attack � Brute Force','Web Attack � Sql Injection',...
    'Web Attack � XSS');
title('First two dimensions of PCA (excluding Benign)')
hold off

%% MRMR

[idx,scores] = fscmrmr(allData_t,'Label');

%% Generate Plot
figure()
bar(scores(idx(1:15)))
title('MRMR Scores')
xlabel('Predictor rank')
ylabel('Predictor importance score')
xticklabels(strrep(allData_t.Properties.VariableNames(idx(1:15)),'_','\_'))
xtickangle(90)

%% Fishers 

[index,feature_score] = feature_rank(allData_a',allData_bl);
figure()
bar(feature_score(1:15));
title('Fisher Scores')
xlabel('Predictor rank')
ylabel('Fisher Score')
xticklabels(strrep(allData_t.Properties.VariableNames(index(1:15)),'_','\_'))
xtickangle(90)

%% Relief
[idx_rel,weights_rel] = relieff(allData_a,allData_bl,5);


