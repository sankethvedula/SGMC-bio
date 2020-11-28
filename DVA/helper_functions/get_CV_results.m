function [ auc,aupr,XcROC,YcROC,XcPR,YcPR, T ] = get_CV_results(Y,n,cv_setting,predictionMethod  )
%GET_CV_RESULTS Summary of this function goes here
%   Detailed explanation goes here

global Sd Sv
            AUCs  = zeros(1,n);  AUPRs = zeros(1,n);   
            % loop over the n folds
            
            for i=1:n

               [test_ind,len]=get_test_ind( cv_setting,i,n, Y,Sd,Sv);
               
               
               f_name = strcat('./data_for_DMF/data_', num2str(i), '_', ...
                 predictionMethod, '_', num2str(cv_setting), '.mat');
               W_train = ones(size(Y));
               W_test  = zeros(size(Y));

               W_train(test_ind) = 0;    % train mask
               W_test (test_ind) = 1;    % test mask

               y2 = Y;
               y2(test_ind) = 0;
               fprintf('*');
               st=tic;
               
%                y3 = alg_template(y2, predictionMethod, test_ind, []);
%                % save for dmf
%                s1.y2 = y2;
%                s1.Y = Y;
%                s1.y3 = y3;
%                s1.Sd = preprocess_PNN(Sd, 5);
%                s1.Sv = preprocess_PNN(Sv, 5);
%                s1.test_ind = test_ind;
%                s1.omega_train = W_train;
%                s1.omega_test  = W_test;
%                save(f_name, '-struct', 's1');
%                 
                s2 = load(f_name);
                y2 = s2.y2;
                Y  = s2.Y;
                y3 = s2.y3;
                Sd = s2.Sd;
                Sv = s2.Sv;
                test_ind = s2.test_ind;

                sgmc_iter = 270000;
                y3_ = load(strcat('data_for_DMF/data_', num2str(i), ...
                            '_', predictionMethod, '_', ...
                            '1_Y3_', num2str(sgmc_iter),'_SGMC.mat'));
                y3_ = y3_.y3';

               endt= double(toc-st);

               [AUCs(i),XcROC,YcROC]  = calculate_auc (y3(test_ind),Y(test_ind));    
               [AUPRs(i),XcPR,YcPR, T] = calculate_aupr(y3(test_ind),Y(test_ind));
               %if (length(test_ind)==(round(len/n)-1)) %for 1st fold test_ind has 1 less element than the total elements in test_ind from other 9 folds..so copy 1 ele to X coordinate and Y coordinate
                   %disp('Ìn ifffff')
               %end
               
            end
            auc= mean(AUCs);    aupr= mean(AUPRs);
end

function [S,p]=preprocess_PNN(S,p)
%preprocess_PNN sparsifies S by keeping, for each drug/virus, the "p"
% nearest neighbors (NNs) and discarding the rest. 

    NN_mat = zeros(size(S));
    for j=1:length(NN_mat)
        [~,indx] = sort(S(j,:),'descend');
        indx = indx(1:p+1);     % keep drug/virus j and its "p" NNs
        NN_mat(j,indx) = 1;
    end
    NN_mat = (NN_mat+NN_mat')/2;
    S = NN_mat .* S;

end
