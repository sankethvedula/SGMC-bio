clc; clear all; close all;
baseline  = 'mgrnnm';  %, 'rls_wnn', 'grmf', 'cmf', 'mgrnnm'
num_folds = 9;
num_reps  = 5;
cv_mode   = 'S3';
dataset   = 'nr';
iter_num  = 10000;

global_AUCs_baseline_mean  = zeros(1, num_folds);
global_AUCs_dmf_mean       = zeros(1, num_folds);

global_AUCs_baseline_std   = zeros(1, num_folds);
global_AUCs_dmf_std        = zeros(1, num_folds);

global_AUPRs_baseline_mean = zeros(1, num_folds);
global_AUPRs_dmf_mean      = zeros(1, num_folds);

global_AUPRs_baseline_std  = zeros(1, num_folds);
global_AUPRs_dmf_std       = zeros(1, num_folds);

for fold_num = 1:num_folds

    AUCs_baseline  = zeros(1, num_reps);
    AUCs_dmf       = zeros(1, num_reps);

    AUPRs_baseline = zeros(1, num_reps);
    AUPRs_dmf      = zeros(1, num_reps);
 
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
        
        
        dmf_f_name = strcat('data_for_DMF/data_',...
                         'fold_', num2str(fold_num), ...
                         'rep_', num2str(rep_num), '_', ...
                          'mgrnnm_', ...
                          dataset, '_', ...
                          cv_mode, '/Y3_',...
                          num2str(iter_num), '_SGMC.mat' );
        y3_ = load(dmf_f_name);
        y3_ = y3_.y3';
        
        auc_baseline = calculate_auc (y3 (test_ind),Y(test_ind));
        auc_dmf      = calculate_auc (y3_(test_ind),Y(test_ind));
        
        aupr_baseline = calculate_aupr (y3 (test_ind),Y(test_ind));
        aupr_dmf      = calculate_aupr (y3_(test_ind),Y(test_ind));
        
        % fprintf([num2str(auc_baseline),'\t \t', num2str(auc_dmf), '\n']);
        
        AUCs_baseline(rep_num)  = auc_baseline;
        AUCs_dmf(rep_num)       = auc_dmf;
        
        AUPRs_baseline(rep_num) = aupr_baseline;
        AUPRs_dmf(rep_num) = aupr_dmf;
        
    end
    disp(AUCs_baseline); disp(AUCs_dmf);
    global_AUCs_baseline_mean  = mean(AUCs_baseline);
    global_AUCs_dmf_mean       = mean(AUCs_dmf);
    global_AUPRs_baseline_mean = mean(AUPRs_baseline);
    global_AUPRs_dmf_mean      = mean(AUPRs_dmf);
    
    global_AUCs_baseline_std  = std(AUCs_baseline);
    global_AUCs_dmf_std       = std(AUCs_dmf);
    global_AUPRs_baseline_std = std(AUPRs_baseline);
    global_AUPRs_dmf_std      = std(AUPRs_dmf);
    
end
