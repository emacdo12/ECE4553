% Eve MacDonald & Matt MacMillan

%% ---Load the data sets---

data_a = readtable('data\Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv');
data_b = readtable('data\Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv');
data_c = readtable('data\Friday-WorkingHours-Morning.pcap_ISCX.csv');
data_d = readtable('data\Monday-WorkingHours.pcap_ISCX.csv');
data_e = readtable('data\Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv');
data_f = readtable('data\Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv');
data_g = readtable('data\Tuesday-WorkingHours.pcap_ISCX.csv');
data_h = readtable('data\Wednesday-workingHours.pcap_ISCX.csv');

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

allData_t = [data_a;data_b;data_c;data_d;data_e;data_f;data_g;data_h];


for i = 1:length(allData_l)
    if allData_l{i} ~= "BENIGN"
        allData_l{i} = 'MALICIOUS';
    end

end

%% Get Rid of Garbage Data
for i = 1:size(allData_a)
    for j = 1:width(allData_a)
        if(isnan(allData_a(i,j))  || allData_a(i,j) == inf)
            allData_a(j,:) = [];
        end
    end
end

for i = 1:width(allData_a)
    if (length(unique(allData_a(:,i))) == 1)
        allData_a(:,i) = [];
    end
end



%% ---SFS---
k = 10;
c = cvpartition(allData_l,'KFold',k);
opts = statset('display','iter');
fun_linear = @(XT,yT,Xt,yt)...
    (sum(~strcmp(yt,classify(Xt,XT,yT,'linear'))));
fun_quadratic = @(XT,yT,Xt,yt)...
    (sum(~strcmp(yt,classify(Xt,XT,yT,'quadratic'))));

% disp('Linear SFS');
% [bestmodel_l,hist_l] = sequentialfs(fun_linear,allData_a,allData_l,'cv',c,'options',opts);
% 
% disp('\nQuadratic SFS');
% [bestmodel_q,hist_q] = sequentialfs(fun_quadratic,allData_a,allData_l,'cv',c,'options',opts);

[idx,scores] = fscmrmr(allData_t, 'Label');
