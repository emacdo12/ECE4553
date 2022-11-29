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
%% Number labels
labels_num_b = zeros(1,length(allData_l));
labels_num = zeros(1,length(allData_l));

for i = 1:length(labels_num_b)
    if allData_l(i) == "BENIGN"
        labels_num_b(i) = 1;
    else
        labels_num_b(i) = 0;
    end
end

for i = 1:length(labels_num)
    if allData_l(i) == "BENIGN"
        labels_num(i) = 1;
    elseif allData_l(i) == "Bot"
        labels_num(i) = 2;
    elseif allData_l(i) == "DDoS"
        labels_num(i) = 3;
    elseif allData_l(i) == "DoS GoldenEye"
        labels_num(i) = 4;
    elseif allData_l(i) == "DoS Hulk"
        labels_num(i) = 5;
    elseif allData_l(i) == "DoS Slowhttptest"
        labels_num(i) = 6;
    elseif allData_l(i) == "DoS slowloris"
        labels_num(i) = 7;
    elseif allData_l(i) == "FTP-Patator"
        labels_num(i) = 8;
    elseif allData_l(i) == "Heartbleed"
        labels_num(i) = 9;
    elseif allData_l(i) == "Infiltration"
        labels_num(i) = 10;
    elseif allData_l(i) == "PortScan"
        labels_num(i) = 11;
    elseif allData_l(i) == "SSH-Patator"
        labels_num(i) = 12;
    elseif allData_l(i) == "Web Attack � Brute Force"
        labels_num(i) = 13;
    elseif allData_l(i) == "Web Attack � Sql Injection"
        labels_num(i) = 14;
    else 
        labels_num(i) = 15;
    end 
end

%% MRMR

[m_idx,m_scores] = fscmrmr(allData_t,'Label');

%% Generate Plot
figure()
bar(m_scores(m_idx(1:15)))
title('MRMR Scores')
xlabel('Predictor rank')
ylabel('Predictor importance score')
xticklabels(strrep(allData_t.Properties.VariableNames(m_idx(1:15)),'_','\_'))
xtickangle(90)

%% Fishers 

[f_index,feature_score] = feature_rank(allData_a',allData_bl);
figure()
bar(feature_score(1:15));
title('Fisher Scores')
xlabel('Predictor rank')
ylabel('Fisher Score')
xticklabels(strrep(allData_t.Properties.VariableNames(f_index(1:15)),'_','\_'))
xtickangle(90)

%% --Load Python Data--

correlation = readtable('Correlation.csv');
correlation_a = table2array(correlation);
information_gain = table2array(readtable('Information_gain.csv'));
Forest_Importances = table2array(readtable("Forest_Importance.csv"));
feature_nums = [1:70];

figure()
heatmap(feature_nums,feature_nums,abs(correlation_a))

figure()
[information_gain, ig_index] = sort(information_gain,'descend');
bar(information_gain(1:15))
title('Information Gain');
ylabel('Score')
xlabel('Features')
xticklabels(strrep(allData_t.Properties.VariableNames(ig_index(1:15)),'_','\_'))
xtickangle(90)

figure()
[for_importances,f_index] = sort(Forest_Importances,'descend');
bar(for_importances(1:15))
title('Forest Feature Importance')
ylabel('Score')
xlabel('Features')
xticklabels(strrep(allData_t.Properties.VariableNames(f_index(1:15)),'_','\_'))
xtickangle(90)

%% ULDA 
W_b = my_ulda(allData_a,labels_num_b',2);
W   = my_ulda(allData_a,labels_num,15);
%writematrix(W,"ULDA_scaler.csv");

%% Visualize ULDA

ULDA = allData_a * W;

figure()
plot(ULDA(labels_num == 1,1), ...
    ULDA(labels_num == 1,2),'x');
hold on
for i = 2:15
    plot(ULDA(labels_num == i,1), ...
    ULDA(labels_num == i,2),'x');
end

legend('Bot','DDos','DoS GoldenEye','DoS Hulk','DoS Slowhttptest', ...
    'DoS slowloris','FTP-Patator','Heartbleed','Infiltration','PortScan',...
    'SSH-Patator','Web Attack � Brute Force','Web Attack � Sql Injection',...
    'Web Attack � XSS');
title('First two dimensions of ULDA Multiple Classes')
hold off

%% PCA 
[coeff_a,score_a,latent_a,tsquared_a,explained_a,mu_a] = pca(allData_a);

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


figure()
plot(score_a(strcmp(allData_bl,'BENIGN'),1), ...
    score_a(strcmp(allData_bl,'BENIGN'),2),'xb');
hold on
plot(score_a(strcmp(allData_bl,'MALICIOUS'),1), ...
    score_a(strcmp(allData_bl,'MALICIOUS'),2),'xr');
legend('Benign','Malicious');
title('First two dimensions of PCA')
hold off
