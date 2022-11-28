% my_ulda
% Function written by Christian Morrell
% Process inspired by notes from Dr. Scheme and ULDA tutorial shown here:
% https://sebastianraschka.com/Articles/2014_python_lda.html#lda-in-5-steps
% Inputs:
% dataset: N x M matrix containing data / features used for classification.
% Each row is a sample and each column is a feature.
% labels: N x 1 vector containing class labels. Each element is a label
% corresponding to a sample in dataset.
% num_classes: number of classes in dataset.
% Outputs:
% W: C-1 x M matrix, where C is the number of classes in the dataset. 
% Coefficients used to project original dataset into ULDA dimension space.
 
function W = my_ulda(dataset, labels, num_classes)
 
% 1. Compute the mean for each class
means = zeros(num_classes, width(dataset));
for i = 1:num_classes
    class_rows = labels == i;
    class_data = dataset(class_rows, :);
    means(i, :) = mean(class_data);
end
 
% 2. Compute scatter matrices for each class
S_w = 0;
for i = 1:num_classes
    class_rows = labels == i;
    class_data = dataset(class_rows, :);
    S_w = S_w + (length(class_data) - 1) * cov(class_data);
end

% 3. Compute the total within class scatter and between-class scatter
S_B = 0;
overall_mean = mean(dataset, 1);
for i = 1:num_classes
    class_rows = labels == i;
    class_data = dataset(class_rows, :);
    mean_vector = means(i, :);
    S_B = S_B + length(class_data) * (mean_vector - overall_mean) .* (mean_vector - overall_mean)';
end
 
 
% 4. Compute inverse
S_wi = inv(S_w);
 
% 5. Eigenvalues
[eigen_vectors, eigen_values] = eig(S_wi * S_B);
 
eigen_values = eigen_values(eigen_values ~= 0);
 
eig_pairs = cell(length(eigen_values), 2);
 
for i = 1:length(eigen_values)
    eig_pairs{i, 1} = abs(eigen_values(i));
    eig_pairs{i, 2} = eigen_vectors(:, i);
end
 
sorted_eig_pairs = sortrows(eig_pairs, 1, 'descend');   % sort eigenvalues from highest to lowest
 
% Set separability goal if desired
separability_goal = 0.95;
eigSum = sum(eigen_values);
explained = eigen_values./eigSum;
idx = find(cumsum(explained) > separability_goal, 1);

W = zeros(height(eigen_vectors), num_classes-1);
for i = 1:num_classes-1
    W(:, i) = cell2mat(sorted_eig_pairs(i, 2));
end
W = real(W);    % only return real part
end