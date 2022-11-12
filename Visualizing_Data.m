clear;
clc;
close all;

%% -- Load Data --

data_a = readtable('data/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv');
data_b = readtable('data/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv');
data_c = readtable('data/Friday-WorkingHours-Morning.pcap_ISCX.csv');
data_d = readtable('data/Monday-WorkingHours.pcap_ISCX.csv');
data_e = readtable('data/Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv');
data_f = readtable('data/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv');
data_g = readtable('data/Tuesday-WorkingHours.pcap_ISCX.csv');
data_h = readtable('data/Wednesday-workingHours.pcap_ISCX.csv');

%% ---Data Manipulation---

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
deleted_rows = allData_a(delete_rows,:);
deleted_labels = allData_l(delete_rows,:);
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

deleted_features = allData_t.Properties.VariableNames(delete_columns);

allData_a(:,delete_columns) = [];
for i = 1:length(delete_columns)
    allData_t(:,delete_columns(i)) = [];
end

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

%% --Load Python Data--

correlation = readtable('Correlation.csv');
correlation_a = table2array(correlation);
information_gain = table2array(readtable('Information_gain.csv'));
Forest_Importances = table2array(readtable("Forest_Importance.csv"));
feature_nums = [1:70];

figure()
heatmap(feature_nums,feature_nums,correlation_a)

figure()
[information_gain, Index] = sort(information_gain,'descend');
bar(information_gain(1:15))
title('Information Gain');
ylabel('Score')
xlabel('Features')
xticklabels(strrep(allData_t.Properties.VariableNames(Index(1:15)),'_','\_'))
xtickangle(90)

figure()
[for_importances,f_index] = sort(Forest_Importances,'descend');
bar(for_importances(1:15))
title('Forest Feature Importance')
ylabel('Score')
xlabel('Features')
xticklabels(strrep(allData_t.Properties.VariableNames(f_index(1:15)),'_','\_'))
xtickangle(90)

