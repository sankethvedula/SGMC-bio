clc; clear all; close all;
baseline  = 'mgrnnm';  %, 'rls_wnn', 'grmf', 'cmf', 'mgrnnm'
num_folds = 10;
num_reps  = 5;
cv_mode   = 'S1';
dataset   = 'ic';
iter_num  = 24000;

global_AUCs_baseline_mean  = zeros(1, num_folds);
global_AUCs_dmf_mean       = zeros(1, num_folds);
global_AUPRs_baseline_mean = zeros(1, num_folds);
global_AUPRs_dmf_mean      = zeros(1, num_folds);
global_RMSEs_baseline_mean = zeros(1, num_folds);
global_RMSEs_dmf_mean = zeros(1, num_folds);

for fold_num = 1:num_folds

    AUCs_baseline  = zeros(1, num_reps);
    AUCs_dmf       = zeros(1, num_reps);

    AUPRs_baseline = zeros(1, num_reps);
    AUPRs_dmf      = zeros(1, num_reps);
    
    RMSE_baseline  = zeros(1, num_reps);
    RMSE_dmf       = zeros(1, num_reps);
 
    for rep_num = 1:num_reps
        f_name_baseline = strcat('data_for_DMF/data_',...
                                 'fold_', num2str(fold_num), ...
                                 'rep_', num2str(rep_num), '_', ...
                                  baseline, '_',...
                                  dataset, '_', ...
                                  cv_mode, '.mat');
        s2 = load(f_name_baseline);
        y2 = s2.y2;
        Y  = s2.Y;
        y3 = s2.y3;
        Sd = s2.Sd;
        St = s2.St;
        left_out = s2.left_out;
        test_ind = s2.test_ind;
        
        % create masks
        W_train = ones(size(Y));
        W_test  = zeros(size(Y));
        
        W_train(test_ind) = 0;    % train mask
        W_test (test_ind) = 1;    % test mask
        
        
        dmf_f_name = strcat('data_for_DMF/data_',...
                         'fold_', num2str(fold_num), ...
                         'rep_', num2str(rep_num), '_', ...
                          'mgrnnm_', ...
                          dataset, '_', ...
                          cv_mode, '/Y3_',...
                          num2str(iter_num), '_SGMC.mat' );
        y3_ = load(dmf_f_name);
        y3_ = y3_.y3';
        
        rmse_baseline = rmse(y3 , Y, W_test);
        rmse_dmf      = rmse(y3_, Y, W_test);
        
        auc_baseline = calculate_auc (y3 (test_ind),Y(test_ind));
        auc_dmf      = calculate_auc (y3_(test_ind),Y(test_ind));
        
        aupr_baseline = calculate_aupr (y3 (test_ind),Y(test_ind));
        aupr_dmf      = calculate_aupr (y3_(test_ind),Y(test_ind));
        
        % fprintf([num2str(auc_baseline),'\t \t', num2str(auc_dmf), '\n']);
        
        AUCs_baseline(rep_num)  = auc_baseline;
        AUCs_dmf(rep_num)       = auc_dmf;
        
        AUPRs_baseline(rep_num) = aupr_baseline;
        AUPRs_dmf(rep_num)      = aupr_dmf;
        
        RMSE_baseline(rep_num)  = rmse_baseline;
        RMSE_dmf(rep_num)       = rmse_dmf;
        
    end
        
    global_AUCs_baseline_mean(fold_num)  = mean(AUCs_baseline);
    global_AUCs_dmf_mean(fold_num)       = mean(AUCs_dmf);
    global_AUPRs_baseline_mean(fold_num) = mean(AUPRs_baseline);
    global_AUPRs_dmf_mean(fold_num)      = mean(AUPRs_dmf);
    global_RMSEs_baseline_mean(fold_num) = mean(RMSE_baseline);
    global_RMSEs_dmf_mean(fold_num)      = mean(RMSE_dmf);
end

mean_AUC_baseline   = mean(global_AUCs_baseline_mean);
mean_AUPR_baseline  = mean(global_AUPRs_baseline_mean);
mean_RMSE_baseline  = mean(global_RMSEs_baseline_mean);

std_AUC_baseline    = std(global_AUCs_baseline_mean);
std_AUPR_baseline   = std(global_AUPRs_baseline_mean);
std_RMSE_baseline   = std(global_RMSEs_baseline_mean);

mean_AUC_dmf = mean(global_AUCs_dmf_mean);
mean_AUPR_dmf = mean(global_AUPRs_dmf_mean);
mean_RMSE_dmf = mean(global_RMSEs_dmf_mean);

std_AUC_dmf   = std(global_AUCs_dmf_mean);
std_AUPR_dmf   = std(global_AUPRs_dmf_mean);
std_RMSE_dmf   = std(global_RMSEs_dmf_mean);

disp('AUC:')
disp(['Baseline:  ', num2str(mean_AUC_baseline), '  ', num2str(std_AUC_baseline)])
disp(['SGMC:      ', num2str(mean_AUC_dmf), '  ', num2str(std_AUC_dmf)])


disp('AUPR:')
disp(['Baseline:  ', num2str(mean_AUPR_baseline), '  ', num2str(std_AUPR_baseline)])
disp(['SGMC:      ', num2str(mean_AUPR_dmf), '  ', num2str(std_AUPR_dmf)])


disp('RMSE:')
disp(['Baseline:  ', num2str(mean_RMSE_baseline), '  ', num2str(std_RMSE_baseline)])
disp(['SGMC:      ', num2str(mean_RMSE_dmf), '  ', num2str(std_RMSE_dmf)])


function [rmse_] = rmse(pred, gt, W_test)
    num_non_zero = size(find(W_test),1);
    rmse_  = sqrt(sum(sum(((W_test.*(pred - gt)).^2)))/num_non_zero);
end